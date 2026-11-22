#include "MediaEngine.h"
#include "FramePacingRepair.h"
#include "DemoSource.h"
#include "RemotePlay.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/channel_layout.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}
#include <pulse/error.h>
#include <pulse/simple.h>

#include <algorithm>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace veyamux {
namespace {
using steady = std::chrono::steady_clock;
double secs(steady::duration d) { return std::chrono::duration<double>(d).count(); }
constexpr size_t kMaxQueueBytes = 64u * 1024 * 1024;
constexpr int kMaxVideoFrames = 3;

AVPixelFormat pickFormat(AVCodecContext*, const AVPixelFormat* fmts) {
    // Prefer VAAPI surfaces; otherwise the first software format.
    for (const AVPixelFormat* p = fmts; *p != AV_PIX_FMT_NONE; ++p)
        if (*p == AV_PIX_FMT_VAAPI) return *p;
    for (const AVPixelFormat* p = fmts; *p != AV_PIX_FMT_NONE; ++p) {
        const AVPixFmtDescriptor* d = av_pix_fmt_desc_get(*p);
        if (d && !(d->flags & AV_PIX_FMT_FLAG_HWACCEL)) return *p;
    }
    return AV_PIX_FMT_NONE;
}
std::string avErr(int e) {
    char b[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(e, b, sizeof b);
    return b;
}
} // namespace

void MediaEngine::Clock::set(double p) {
    std::lock_guard l(m);
    pts = p;
    wall = steady::now();
}
double MediaEngine::Clock::get() {
    std::lock_guard l(m);
    return running ? pts + secs(steady::now() - wall) : pts;
}
void MediaEngine::Clock::setRunning(bool r) {
    std::lock_guard l(m);
    const auto now = steady::now();
    if (running) pts += secs(now - wall);
    wall = now;
    running = r;
}

MediaEngine::MediaEngine(std::string shaderDir) : shaderDir_(std::move(shaderDir)) {
    av_log_set_level(AV_LOG_ERROR);
}

MediaEngine::~MediaEngine() {
    close();
    if (gpuThread_.joinable()) gpuThread_.join();
    chain_.reset();
    vk_.reset();
}

// ------------------------------------------------------------------ open / close

bool MediaEngine::openVideoDecoder(bool tryHw, std::string& error) {
    AVStream* st = fmt_->streams[vIndex_];
    const AVCodec* codec = avcodec_find_decoder(st->codecpar->codec_id);
    if (!codec) { error = "No decoder for the video stream"; return false; }
    if (vctx_) avcodec_free_context(&vctx_);
    vctx_ = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(vctx_, st->codecpar);
    vctx_->thread_count = 0;
    vctx_->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;
    vctx_->pkt_timebase = st->time_base;
    if (tryHw) {
        if (!hwDevice_) {
            const char* dev = std::getenv("VEYAMUX_VAAPI_DEVICE");
            if (av_hwdevice_ctx_create(&hwDevice_, AV_HWDEVICE_TYPE_VAAPI, dev, nullptr, 0) < 0) hwDevice_ = nullptr;
        }
        if (hwDevice_) {
            vctx_->hw_device_ctx = av_buffer_ref(hwDevice_);
            vctx_->get_format = pickFormat;
        }
    }
    int r = avcodec_open2(vctx_, codec, nullptr);
    if (r < 0 && tryHw && hwDevice_) return openVideoDecoder(false, error);
    if (r < 0) { error = "Could not open the video decoder: " + avErr(r); return false; }
    return true;
}

bool MediaEngine::open(const std::string& path, std::string& error, bool rawToGpu) {
    close();
    int r = avformat_open_input(&fmt_, path.c_str(), nullptr, nullptr);
    if (r < 0) { error = "Could not open the file: " + avErr(r); fmt_ = nullptr; return false; }
    if ((r = avformat_find_stream_info(fmt_, nullptr)) < 0) {
        error = "Could not read stream info: " + avErr(r);
        avformat_close_input(&fmt_);
        return false;
    }
    vIndex_ = av_find_best_stream(fmt_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (vIndex_ >= 0 && (fmt_->streams[vIndex_]->disposition & AV_DISPOSITION_ATTACHED_PIC)) vIndex_ = -1;
    aIndex_ = av_find_best_stream(fmt_, AVMEDIA_TYPE_AUDIO, -1, vIndex_, nullptr, 0);
    if (vIndex_ < 0 && aIndex_ < 0) {
        error = "The file contains no playable audio or video";
        avformat_close_input(&fmt_);
        return false;
    }
    startTime_ = fmt_->start_time != AV_NOPTS_VALUE ? double(fmt_->start_time) / AV_TIME_BASE : 0.0;
    info_ = {};
    info_.duration = fmt_->duration > 0 ? double(fmt_->duration) / AV_TIME_BASE : 0.0;
    if (AVDictionaryEntry* t = av_dict_get(fmt_->metadata, "title", nullptr, 0)) info_.title = t->value;
    if (info_.title.empty()) {
        std::string n = path;
        if (auto s = n.find_last_of('/'); s != std::string::npos) n = n.substr(s + 1);
        info_.title = n;
    }
    if (vIndex_ >= 0) {
        AVStream* st = fmt_->streams[vIndex_];
        vTimeBase_ = av_q2d(st->time_base);
        const char* hw = std::getenv("VEYAMUX_HWDEC");
        if (!openVideoDecoder(!(hw && std::string(hw) == "0"), error)) { avformat_close_input(&fmt_); return false; }
        info_.hasVideo = true;
        info_.width = vctx_->width;
        info_.height = vctx_->height;
        const AVRational fr = av_guess_frame_rate(fmt_, st, nullptr);
        info_.fps = fr.den ? av_q2d(fr) : 0.0;
        info_.videoCodec = avcodec_get_name(st->codecpar->codec_id);
        // straight to the GPU only with square pixels (the window shows the picture at its own aspect) and without
        // HLG (the GPU path knows PQ only); everything else keeps the CPU conversion
        const AVRational sar = st->sample_aspect_ratio.num ? st->sample_aspect_ratio : st->codecpar->sample_aspect_ratio;
        const bool square = sar.num == 0 || sar.num == sar.den;
        rawLive_ = rawToGpu && square && st->codecpar->color_trc != AVCOL_TRC_ARIB_STD_B67;
    } else {
        rawLive_ = false;
    }
    if (aIndex_ >= 0) {
        AVStream* st = fmt_->streams[aIndex_];
        aTimeBase_ = av_q2d(st->time_base);
        const AVCodec* codec = avcodec_find_decoder(st->codecpar->codec_id);
        actx_ = codec ? avcodec_alloc_context3(codec) : nullptr;
        if (actx_) {
            avcodec_parameters_to_context(actx_, st->codecpar);
            actx_->pkt_timebase = st->time_base;
            if (avcodec_open2(actx_, codec, nullptr) < 0) avcodec_free_context(&actx_);
        }
        if (actx_) {
            info_.hasAudio = true;
            info_.audioCodec = avcodec_get_name(st->codecpar->codec_id);
        } else aIndex_ = -1;
    }

    quit_ = false; eof_ = false; endedFired_ = false; playing_ = false;
    serial_ = 1; seekReq_ = -1.0; seekTarget_ = 0.0; showOne_ = true; rawReset_ = true;
    vpkts_.abort = apkts_.abort = false;
    clock_.setRunning(false);
    clock_.set(0.0);
    demux_ = std::thread([this] { demuxLoop(); });
    if (vIndex_ >= 0) {
        vdec_ = std::thread([this] { videoLoop(); });
        render_ = std::thread([this] { renderLoop(); });
    }
    if (aIndex_ >= 0) adec_ = std::thread([this] { audioLoop(); });
    return true;
}

void MediaEngine::joinAll() {
    quit_ = true;
    for (PacketQueue* q : {&vpkts_, &apkts_}) {
        std::lock_guard l(q->m);
        q->abort = true;
        q->cv.notify_all();
    }
    { std::lock_guard l(gate_m_); gate_cv_.notify_all(); }
    { std::lock_guard l(fq_m_); fq_cv_.notify_all(); }
    { std::lock_guard l(pace_m_); pace_cv_.notify_all(); }
    for (std::thread* t : {&demux_, &vdec_, &adec_, &render_, &capThread_, &capAudioThread_, &pacer_})
        if (t->joinable()) t->join();
}

void MediaEngine::close() {
    if (!fmt_ && !live_) return;
    joinAll();
    cap_.reset();
    live_ = false;
    rawLive_ = false;
    clearQueue(vpkts_);
    clearQueue(apkts_);
    { std::lock_guard l(fq_m_); frames_.clear(); }
    if (vctx_) avcodec_free_context(&vctx_);
    if (actx_) avcodec_free_context(&actx_);
    if (hwDevice_) av_buffer_unref(&hwDevice_);
    avformat_close_input(&fmt_);
    vIndex_ = aIndex_ = -1;
    playing_ = false;
}

// ------------------------------------------------------------------ queues

void MediaEngine::pushPacket(PacketQueue& q, Pkt p) {
    std::lock_guard l(q.m);
    if (p.pkt) { q.bytes += size_t(p.pkt->size); }
    if (p.kind == Pkt::Data) ++q.dataCount;
    q.q.push_back(p);
    q.cv.notify_one();
}

bool MediaEngine::popPacket(PacketQueue& q, Pkt& out) {
    std::unique_lock l(q.m);
    q.cv.wait(l, [&] { return !q.q.empty() || q.abort; });
    if (q.abort) return false;
    out = q.q.front();
    q.q.pop_front();
    if (out.pkt) q.bytes -= size_t(out.pkt->size);
    if (out.kind == Pkt::Data) --q.dataCount;
    return true;
}

void MediaEngine::clearQueue(PacketQueue& q) {
    std::lock_guard l(q.m);
    for (Pkt& p : q.q) if (p.pkt) av_packet_free(&p.pkt);
    q.q.clear();
    q.bytes = 0;
    q.dataCount = 0;
}

bool MediaEngine::openCapture(const std::string& devicePath, const CaptureMode& mode, const std::string& audioSource,
                              std::string& error, bool rawToGpu) {
    close();
    auto cap = std::make_unique<CaptureSource>();
    if (!cap->open(devicePath, mode, error)) return false;
    cap_ = std::move(cap);
    info_ = {};
    info_.title = "Capture: " + (cap_->cardName().empty() ? devicePath : cap_->cardName());
    info_.hasVideo = true;
    info_.hasAudio = !audioSource.empty();
    info_.width = cap_->width();
    info_.height = cap_->height();
    info_.fps = cap_->fps();
    info_.videoCodec = CaptureSource::isScreenCast(devicePath) ? "PipeWire" : CaptureSource::isDemo(devicePath) ? "Demo"
                    : CaptureSource::isRemotePlay(devicePath) ? "Remote Play" : fourccName(mode.fourcc);
    info_.audioCodec = audioSource.empty() ? "" : "live";
    capAudioSource_ = audioSource;
    capPath_ = devicePath;
    capMode_ = mode;
    quit_ = false; endedFired_ = false; eof_ = false; showOne_ = false;
    serial_ = 1; vpkts_.abort = apkts_.abort = false;
    { std::lock_guard l(fq_m_); frames_.clear(); }
    liveStart_ = steady::now();
    live_ = true;
    rawLive_ = rawToGpu;
    playing_ = true;
    capThread_ = std::thread([this] { captureLoop(); });
    if (rawToGpu) pacer_ = std::thread([this] { pacerLoop(); });
    if (!rawToGpu) render_ = std::thread([this] { renderLoop(); });
    if (!audioSource.empty()) capAudioThread_ = std::thread([this] { captureAudioLoop(); });
    return true;
}

void MediaEngine::captureLoop() {
    FramePacingRepair repair;
    auto lastPaced = std::chrono::steady_clock::time_point{};
    std::vector<uint8_t> rgba;
    RawFrame raw;
    int idle = 0;
    bool noSignal = false;
    // VRR pacing: the card samples the game at its fixed signal rate, so a 45 fps game arrives as pictures 1, 1 and 2
    // signal frames apart (16.7 / 16.7 / 33.3 ms at 60 Hz). Shown on arrival that judders even on a VRR display;
    // instead each picture is held until one content interval after the previous one (at most ~3/4 of an interval),
    // which spaces them evenly at the game's own rate.
    using clk = std::chrono::steady_clock;
    clk::time_point lastShown{};
    double paceMs = 0;
    // Reopens the device with the same mode: when the card is unplugged, or when it stops delivering after the
    // console changes resolution/refresh rate/HDR (many UVC cards stall until the stream is restarted).
    auto reopen = [&]() -> bool {
        cap_->close();
        std::string e;
        return cap_->open(capPath_, capMode_, e);
    };
    clk::time_point lastReopen{};
    // A shared window/screen only sends a picture when something changes, and cannot be reopened once it is gone.
    // (the demo scene behaves the same way: no reopening, and nothing to wait for while its world is built)
    // (Remote Play too: its errors are the console's own reasons, and reconnecting is the user's choice)
    const bool screenCast = CaptureSource::isScreenCast(capPath_) || CaptureSource::isDemo(capPath_) || CaptureSource::isRemotePlay(capPath_);
    while (!quit_) {
        std::string err;
        const bool toGpu = rawLive_.load();
        const auto r = toGpu ? cap_->readLatest(raw, 100, err) : cap_->read(rgba, 100, err);
        if (r == CaptureSource::Result::Error && screenCast) {
            if (cb_.onError) cb_.onError(err.empty() ? "The shared window was closed" : err);
            break;
        }
        if (r == CaptureSource::Result::Timeout && screenCast) continue;
        if (r == CaptureSource::Result::Error) {
            if (cb_.onError) cb_.onError("Capture card lost, reconnecting...");
            noSignal = true;
            while (!quit_ && !reopen()) std::this_thread::sleep_for(std::chrono::seconds(1));
            lastReopen = clk::now();
            idle = 0;
            continue;
        }
        if (r == CaptureSource::Result::Timeout) {
            if (++idle >= 10 && !noSignal) { noSignal = true; if (cb_.onError) cb_.onError("No signal from the capture card"); }
            // no picture for 3 s: restart the stream once every 3 s (recovers cards that stall on a signal change)
            if (idle >= 30 && clk::now() - lastReopen > std::chrono::seconds(3)) {
                lastReopen = clk::now();
                reopen();
                idle = 10;
            }
            continue;
        }
        idle = 0;
        if (noSignal) { noSignal = false; if (cb_.onError) cb_.onError(""); }
        if (!playing_.load()) continue;                 // paused: keep the device draining, show nothing new
        // A game slower than the signal (30 fps in 60 Hz) repeats every picture. With frame generation on, repeats are
        // dropped here so it interpolates between distinct pictures (30 -> 60) instead of between two copies.
        if (skipRepeats_.load() && (toGpu ? raw.repeat : cap_->lastWasRepeat())) continue;
        if (vrrPacing_.load()) {
            const auto now = clk::now();
            const double since = lastShown.time_since_epoch().count() ? std::chrono::duration<double, std::milli>(now - lastShown).count() : 1e9;
            const auto c = cap_->content();
            // content interval: the measured game frame rate, else a running average of the gaps between pictures
            if (c.contentFps > 1.0 && !c.staticPicture) paceMs = 1000.0 / c.contentFps;
            else if (since < 200.0) paceMs = paceMs > 0 ? 0.9 * paceMs + 0.1 * since : since;
            const double minGap = std::clamp(paceMs, 4.0, 100.0) * 0.97;
            if (paceMs > 0 && since < minGap) {
                const double wait = std::min(minGap - since, 0.75 * paceMs);
                std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(wait));
            }
            lastShown = clk::now();
        }
        if (toGpu) {
            if (pacingRepair_.load() && skipRepeats_.load() && cb_.onLiveFrame) {
                // even out a locked game's uneven showing: hand the picture to the pacer thread with its due time
                if (pacingReset_.exchange(false)) repair.reset();
                const double fps = cap_->fps() > 1.0 ? cap_->fps() : 60.0;
                const int64_t transport = int64_t(1e7 / fps);   // one capture refresh, 100 ns
                const auto captured = raw.captured.time_since_epoch().count() ? raw.captured : clk::now();
                const int64_t sample = std::chrono::duration_cast<std::chrono::nanoseconds>(captured.time_since_epoch()).count() / 100;
                const bool boundary = clk::now() - lastPaced > std::chrono::milliseconds(500);   // a stall or a pause
                const auto res = repair.apply(sample, sample, transport, boundary);
                lastPaced = clk::now();
                ++pacedTotal_;
                if (res.repaired) ++pacedRepaired_;
                // + one refresh: the grid line may lie up to one refresh before the arrival
                const auto due = captured + std::chrono::nanoseconds((res.shift + transport) * 100);
                std::lock_guard l(pace_m_);
                while (paced_.size() >= 2) paced_.pop_front();   // never pile up: latency first
                paced_.push_back({due, std::move(raw)});
                raw = RawFrame{};
                if (!paceSpare_.empty()) { raw = std::move(paceSpare_.back()); paceSpare_.pop_back(); }
                pace_cv_.notify_all();
                continue;
            }
            pacingReset_ = true;
            if (cb_.onLiveFrame) cb_.onLiveFrame(raw);   // raw now holds a used buffer to fill next time
            else raw = RawFrame{};
            continue;
        }
        auto fr = std::make_shared<Frame>();
        fr->rgba = std::move(rgba);
        fr->pool = pool_;
        rgba = pool_->take(0);   // a used buffer (keeps its capacity) for the next picture
        fr->w = cap_->width(); fr->h = cap_->height();
        fr->aspect = double(fr->w) / fr->h;
        fr->serial = serial_.load();
        std::lock_guard l(fq_m_);
        while (frames_.size() >= 2) frames_.pop_front();   // never block: latency beats completeness
        frames_.push_back(std::move(fr));
        fq_cv_.notify_all();
    }
}

// Loudness levelling: the short-term level (RMS over ~0.4 s) is pulled toward -20 dBFS with a slow gain (attack
// ~0.3 s, release ~3 s, at most +12 dB / -12 dB), then a fast peak limiter keeps it under -1 dBFS.
struct Leveler {
    float level = 0.0f, gain = 1.0f, lim = 1.0f;
    void process(int16_t* s, int frames, int rate) {
        if (frames <= 0) return;
        double sum = 0;
        for (int i = 0; i < frames * 2; ++i) sum += double(s[i]) * s[i];
        const float rms = float(std::sqrt(sum / (frames * 2)) / 32768.0);
        const float dt = float(frames) / float(rate);
        level += (rms - level) * std::min(1.0f, dt / 0.4f);
        if (level > 0.0005f) {   // not in silence: aim for -20 dBFS
            const float want = std::clamp(0.1f / level, 0.25f, 4.0f);
            const float tau = want < gain ? 0.3f : 3.0f;
            gain += (want - gain) * std::min(1.0f, dt / tau);
        }
        constexpr float ceil = 0.89f * 32767.0f;   // -1 dBFS
        for (int i = 0; i < frames * 2; ++i) {
            float v = s[i] * gain * lim;
            if (std::fabs(v) > ceil) { lim *= ceil / std::fabs(v); v = s[i] * gain * lim; }
            s[i] = int16_t(std::clamp(int(std::lround(v)), -32768, 32767));
            lim += (1.0f - lim) * 0.00005f;   // limiter release (~0.4 s at 48 kHz stereo)
        }
    }
};

static int qEnvIntOr(const char* name, int fallback) {
    const char* v = std::getenv(name);
    return v && *v ? std::atoi(v) : fallback;
}

void MediaEngine::captureAudioLoop() {
    constexpr int kRate = 48000;
    // Low-latency pass-through: 5 ms blocks in, ~20 ms queued out (was 20 / 50 ms). PipeWire (pipewire-pulse) lowers
    // its graph quantum to match. VEYAMUX_AUDIO_MS=<block ms> trades latency for robustness on busy systems.
    const int blockMs = std::clamp(qEnvIntOr("VEYAMUX_AUDIO_MS", 5), 2, 40);
    const int queueMs = std::max(20, blockMs * 4);
    pa_sample_spec ss{PA_SAMPLE_S16LE, kRate, 2};
    pa_buffer_attr rattr;
    rattr.maxlength = rattr.tlength = rattr.prebuf = rattr.minreq = uint32_t(-1);
    rattr.fragsize = uint32_t(pa_usec_to_bytes(pa_usec_t(blockMs) * 1000, &ss));
    pa_buffer_attr pattr;
    pattr.maxlength = pattr.fragsize = uint32_t(-1);
    pattr.tlength = uint32_t(pa_usec_to_bytes(pa_usec_t(queueMs) * 1000, &ss));
    pattr.prebuf = uint32_t(pa_usec_to_bytes(pa_usec_t(blockMs) * 2000, &ss));   // start playing after two blocks
    pattr.minreq = uint32_t(pa_usec_to_bytes(pa_usec_t(blockMs) * 1000, &ss));
    int err = 0;
    // "@demo": the demo scene's own sound, made here instead of recorded (paced by the playback stream)
    // ("@ps": PlayStation Remote Play's sound, received over the network)
    const bool psSound = capAudioSource_ == "@ps";
    const bool demoSound = capAudioSource_ == "@demo" || psSound;
    const char* dev = capAudioSource_ == "@default" ? nullptr : capAudioSource_.c_str();
    pa_simple* rec = demoSound ? nullptr : pa_simple_new(nullptr, "VeyAmux", PA_STREAM_RECORD, dev, "capture", &ss, nullptr, &rattr, &err);
    if (!rec && !demoSound) { if (cb_.onGpuMessage) cb_.onGpuMessage(std::string("Capture audio unavailable: ") + pa_strerror(err)); return; }
    int sinkGen = sinkGen_.load();
    std::string sink = sinkName();
    pa_simple* out = pa_simple_new(nullptr, "VeyAmux", PA_STREAM_PLAYBACK, sink.empty() ? nullptr : sink.c_str(), "capture playback", &ss, nullptr, &pattr, &err);
    if (!out && !sink.empty())   // the chosen device is gone: system default
        out = pa_simple_new(nullptr, "VeyAmux", PA_STREAM_PLAYBACK, nullptr, "capture playback", &ss, nullptr, &pattr, &err);
    Leveler leveler;
    if (!out) {
        if (cb_.onGpuMessage) cb_.onGpuMessage(std::string("Audio output unavailable: ") + pa_strerror(err));
        if (rec) pa_simple_free(rec);
        return;
    }
    std::vector<int16_t> buf(size_t(kRate / 1000 * blockMs) * 2);   // one block
    // Delay line for the audio offset (sound held back to match a picture that is delayed by processing)
    std::vector<int16_t> line;
    size_t lineHead = 0, lineFill = 0;
    int tick = 0;
    while (!quit_) {
        if (psSound) remoteplay::audio(buf.data(), int(buf.size() / 2));
        else if (demoSound) demo::audio(buf.data(), int(buf.size() / 2));
        else if (pa_simple_read(rec, buf.data(), buf.size() * 2, &err) < 0) break;
        if (!playing_.load()) { if (demoSound) std::this_thread::sleep_for(std::chrono::milliseconds(blockMs)); continue; }
        if (cb_.onAudio)   // the block started one block ago and is heard after the audio delay
            cb_.onAudio(buf.data(), int(buf.size() / 2),
                        std::chrono::steady_clock::now() - std::chrono::milliseconds(blockMs - std::max(0, audioDelayMs_.load())));
        if (normalize_.load()) leveler.process(buf.data(), int(buf.size() / 2), kRate);
        const float vol = muted_.load() ? 0.0f : volume_.load();
        if (vol != 1.0f)
            for (auto& s : buf) s = int16_t(std::clamp(int(std::lround(s * vol)), -32768, 32767));
        if (sinkGen != sinkGen_.load()) {   // output device changed
            sinkGen = sinkGen_.load();
            sink = sinkName();
            pa_simple* n = pa_simple_new(nullptr, "VeyAmux", PA_STREAM_PLAYBACK, sink.empty() ? nullptr : sink.c_str(), "capture playback", &ss, nullptr, &pattr, &err);
            if (n) { pa_simple_free(out); out = n; }
        }
        const size_t want = size_t(std::max(0, audioDelayMs_.load())) * (kRate / 1000) * 2;   // samples (both channels)
        if (want > 0) {
            if (line.size() != want + buf.size()) { line.assign(want + buf.size(), 0); lineHead = 0; lineFill = want; }
            // write the new block behind the delayed audio, then send the oldest block
            for (size_t i = 0; i < buf.size(); ++i) line[(lineHead + lineFill + i) % line.size()] = buf[i];
            for (size_t i = 0; i < buf.size(); ++i) buf[i] = line[(lineHead + i) % line.size()];
            lineHead = (lineHead + buf.size()) % line.size();
        } else if (!line.empty()) {
            line.clear();
        }
        if (pa_simple_write(out, buf.data(), buf.size() * 2, &err) < 0) break;
        // Capture and output clocks drift apart; keep the playback queue short so audio never lags the picture.
        // checked every ~0.5 s; more than three queue lengths behind: drop the backlog
        if (++tick % std::max(1, 500 / blockMs) == 0 && pa_simple_get_latency(out, &err) > pa_usec_t(queueMs) * 3000) pa_simple_flush(out, &err);
    }
    pa_simple_free(out);
    if (rec) pa_simple_free(rec);
}

// ------------------------------------------------------------------ control

void MediaEngine::play() {
    if (!isOpen()) return;
    if (!live_ && eof_ && endedFired_) seek(0.0);
    playing_ = true;
    clock_.setRunning(true);
    std::lock_guard l(gate_m_);
    gate_cv_.notify_all();
}

void MediaEngine::pause() {
    playing_ = false;
    clock_.setRunning(false);
    std::lock_guard l(gate_m_);
    gate_cv_.notify_all();
}

void MediaEngine::seek(double s) {
    if (!fmt_ || live_) return;
    s = std::max(0.0, info_.duration > 0 ? std::min(s, std::max(0.0, info_.duration - 0.05)) : s);
    seekReq_ = s;
    clock_.set(s);                       // UI shows the target immediately
    std::lock_guard l(gate_m_);
    gate_cv_.notify_all();
}

double MediaEngine::position() const {
    if (live_) return secs(steady::now() - liveStart_);
    double p = const_cast<MediaEngine*>(this)->clock_.get();
    if (info_.duration > 0) p = std::min(p, info_.duration);
    return std::max(0.0, p);
}

void MediaEngine::setEnhance(const EnhanceSettings& s) {
    { std::lock_guard l(set_m_); settings_ = s; }
    // Repeats are dropped for frame generation (interpolate between distinct pictures) and for VRR pacing (show the
    // game's own pictures, evenly spaced, and let a VRR display follow them).
    skipRepeats_ = s.frameGen != FrameGenMode::Off || s.vrr;
    vrrPacing_ = s.vrr && s.frameGen == FrameGenMode::Off;
    const bool rep = s.pacingRepair && s.frameGen != FrameGenMode::Off;
    if (rep != pacingRepair_.exchange(rep)) pacingReset_ = true;
    redraw_ = true;   // repaint the current frame with the new settings while paused
    std::lock_guard l(gate_m_);
    gate_cv_.notify_all();
}

EngineStats MediaEngine::stats() const {
    std::lock_guard l(stat_m_);
    return stats_;
}
std::string MediaEngine::gpuName() const { return gpuName_; }

// ------------------------------------------------------------------ demux

void MediaEngine::demuxLoop() {
    auto wait = [&] {
        std::unique_lock l(gate_m_);
        gate_cv_.wait_for(l, std::chrono::milliseconds(10));
    };
    auto stat = [](PacketQueue& q, size_t& bytes, size_t& count) {
        std::lock_guard l(q.m);
        bytes = q.bytes; count = q.dataCount;
    };
    while (!quit_) {
        const double s = seekReq_.exchange(-1.0);
        if (s >= 0.0) {
            const int64_t ts = int64_t((s + startTime_) * AV_TIME_BASE);
            if (av_seek_frame(fmt_, -1, ts, AVSEEK_FLAG_BACKWARD) >= 0) {
                const int ser = ++serial_;
                seekTarget_ = s;
                clearQueue(vpkts_);
                clearQueue(apkts_);
                { std::lock_guard l(fq_m_); frames_.clear(); fq_cv_.notify_all(); }
                if (vIndex_ >= 0) pushPacket(vpkts_, {nullptr, ser, Pkt::Flush});
                if (aIndex_ >= 0) pushPacket(apkts_, {nullptr, ser, Pkt::Flush});
                eof_ = false;
                endedFired_ = false;
                showOne_ = !playing_.load();
                clock_.set(s);
            }
        }
        if (eof_) { wait(); continue; }
        size_t vb = 0, vc = 0, ab = 0, ac = 0;
        stat(vpkts_, vb, vc);
        stat(apkts_, ab, ac);
        const bool vEnough = vIndex_ < 0 || vc >= 48;
        const bool aEnough = aIndex_ < 0 || ac >= 96;
        if (vb + ab > kMaxQueueBytes || (vEnough && aEnough)) { wait(); continue; }

        AVPacket* pkt = av_packet_alloc();
        const int r = av_read_frame(fmt_, pkt);
        if (r < 0) {
            av_packet_free(&pkt);
            if (r != AVERROR_EOF && r != AVERROR(EAGAIN)) {
                if (cb_.onError) cb_.onError("Read error: " + avErr(r));
            }
            if (r != AVERROR(EAGAIN)) {
                eof_ = true;
                if (vIndex_ >= 0) pushPacket(vpkts_, {nullptr, serial_.load(), Pkt::Eof});
                if (aIndex_ >= 0) pushPacket(apkts_, {nullptr, serial_.load(), Pkt::Eof});
            }
            continue;
        }
        if (pkt->stream_index == vIndex_) pushPacket(vpkts_, {pkt, serial_.load(), Pkt::Data});
        else if (pkt->stream_index == aIndex_) pushPacket(apkts_, {pkt, serial_.load(), Pkt::Data});
        else av_packet_free(&pkt);
    }
}

// ------------------------------------------------------------------ video decode

void MediaEngine::videoLoop() {
    AVFrame* f = av_frame_alloc();
    AVFrame* sw = av_frame_alloc();
    SwsContext* sws = nullptr;
    int serial = serial_.load();
    double lastPts = 0.0;
    bool hwTried = false;
    const double frameDur = info_.fps > 0 ? 1.0 / info_.fps : 1.0 / 30.0;

    SwsContext* swsRaw = nullptr;
    auto queueFrame = [&](std::shared_ptr<Frame> fr, int serial) {
        std::unique_lock l(fq_m_);
        fq_cv_.wait(l, [&] { return quit_ || int(frames_.size()) < kMaxVideoFrames || serial != serial_.load(); });
        if (quit_) return;
        if (serial == serial_.load()) frames_.push_back(std::move(fr));
        fq_cv_.notify_all();
    };
    // Straight to the GPU (rawLive_): NV12 for 8-bit 4:2:0 in limited range (VA-API gives it as is), P010 for PQ HDR,
    // RGBA for everything else (converted here as before). The GPU path assumes limited range and BT.709 from 720p up
    // (BT.601 below), as for capture cards; other video goes the RGBA way.
    auto deliverRaw = [&](AVFrame* src, double pts, int serial, int hdr) -> bool {
        const int w = src->width & ~1, h = src->height & ~1;
        const auto fmt = AVPixelFormat(src->format);
        const bool limited = src->color_range != AVCOL_RANGE_JPEG && fmt != AV_PIX_FMT_YUVJ420P;
        const bool hd = h >= 720 || w >= 1280;
        const bool matrixOk = src->colorspace == AVCOL_SPC_UNSPECIFIED || (hd ? src->colorspace == AVCOL_SPC_BT709
                              : (src->colorspace == AVCOL_SPC_BT470BG || src->colorspace == AVCOL_SPC_SMPTE170M));
        const AVPixFmtDescriptor* d = av_pix_fmt_desc_get(fmt);
        const bool yuv420 = d && !(d->flags & AV_PIX_FMT_FLAG_RGB) && d->log2_chroma_w == 1 && d->log2_chroma_h == 1;
        auto fr = std::make_shared<Frame>();
        fr->w = w; fr->h = h; fr->pts = pts; fr->serial = serial; fr->aspect = double(w) / h;
        fr->pool = pool_;
        AVPixelFormat dstFmt;
        if (hdr == 1) { dstFmt = AV_PIX_FMT_P010LE; fr->fmt = RawFrame::P010; fr->stride = w * 2; fr->hdr = 1; }
        else if (yuv420 && limited && matrixOk) { dstFmt = AV_PIX_FMT_NV12; fr->fmt = RawFrame::NV12; fr->stride = w; }
        else { dstFmt = AV_PIX_FMT_RGBA; fr->fmt = RawFrame::RGBA; fr->stride = w * 4; }
        const size_t bytes = fr->fmt == RawFrame::RGBA ? size_t(fr->stride) * h : size_t(fr->stride) * h * 3 / 2;
        fr->rgba = pool_->take(bytes);
        uint8_t* y = fr->rgba.data();
        uint8_t* uv = y + size_t(fr->stride) * h;
        if (fmt == dstFmt) {   // NV12 / P010 already (VA-API): copy the rows
            const size_t row = size_t(w) * (dstFmt == AV_PIX_FMT_P010LE ? 2 : 1);
            for (int r = 0; r < h; ++r) std::memcpy(y + size_t(r) * fr->stride, src->data[0] + size_t(r) * src->linesize[0], row);
            for (int r = 0; r < h / 2; ++r) std::memcpy(uv + size_t(r) * fr->stride, src->data[1] + size_t(r) * src->linesize[1], row);
        } else {
            swsRaw = sws_getCachedContext(swsRaw, src->width, src->height, fmt, w, h, dstFmt, SWS_BILINEAR, nullptr, nullptr, nullptr);
            if (!swsRaw) return false;
            if (dstFmt == AV_PIX_FMT_RGBA) {
                int cs = hd ? SWS_CS_ITU709 : SWS_CS_ITU601;
                if (src->colorspace == AVCOL_SPC_BT709) cs = SWS_CS_ITU709;
                else if (src->colorspace == AVCOL_SPC_BT470BG || src->colorspace == AVCOL_SPC_SMPTE170M) cs = SWS_CS_ITU601;
                else if (src->colorspace == AVCOL_SPC_BT2020_NCL || src->colorspace == AVCOL_SPC_BT2020_CL) cs = SWS_CS_BT2020;
                sws_setColorspaceDetails(swsRaw, sws_getCoefficients(cs), limited ? 0 : 1, sws_getCoefficients(SWS_CS_DEFAULT), 1, 0,
                                         1 << 16, 1 << 16);
            }
            uint8_t* dst[4] = {y, dstFmt == AV_PIX_FMT_RGBA ? nullptr : uv, nullptr, nullptr};
            int dstStride[4] = {fr->stride, dstFmt == AV_PIX_FMT_RGBA ? 0 : fr->stride, 0, 0};
            sws_scale(swsRaw, src->data, src->linesize, 0, src->height, dst, dstStride);
        }
        queueFrame(std::move(fr), serial);
        return true;
    };
    auto deliver = [&](AVFrame* src) -> bool {
        // returns false when a fatal conversion problem occurred
        double pts;
        const int64_t bt = src->best_effort_timestamp;
        pts = bt == AV_NOPTS_VALUE ? lastPts + frameDur : double(bt) * vTimeBase_ - startTime_;
        lastPts = pts;
        if (serial != serial_.load()) return true;
        if (pts < seekTarget_.load() - 0.001) return true;           // accurate seek: skip pre-roll frames
        const int w = src->width, h = src->height;
        if (w <= 0 || h <= 0) return true;
        // HDR video (PQ / HLG) keeps 16 bits per channel for the GPU's tone mapping; everything else becomes RGBA8.
        const int hdr = src->color_trc == AVCOL_TRC_SMPTE2084 ? 1 : (src->color_trc == AVCOL_TRC_ARIB_STD_B67 ? 2 : 0);
        sws = sws_getCachedContext(sws, w, h, AVPixelFormat(src->format), w, h, hdr ? AV_PIX_FMT_RGBA64LE : AV_PIX_FMT_RGBA,
                                   SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (!sws) return false;
        int cs = SWS_CS_DEFAULT;
        switch (src->colorspace) {
        case AVCOL_SPC_BT709: cs = SWS_CS_ITU709; break;
        case AVCOL_SPC_BT470BG: case AVCOL_SPC_SMPTE170M: cs = SWS_CS_ITU601; break;
        case AVCOL_SPC_BT2020_NCL: case AVCOL_SPC_BT2020_CL: cs = SWS_CS_BT2020; break;
        default: cs = (h >= 720 || w >= 1280) ? SWS_CS_ITU709 : SWS_CS_ITU601; break;
        }
        const int srcRange = src->color_range == AVCOL_RANGE_JPEG ? 1 : 0;
        sws_setColorspaceDetails(sws, sws_getCoefficients(cs), srcRange, sws_getCoefficients(SWS_CS_DEFAULT), 1, 0,
                                 1 << 16, 1 << 16);
        if (rawLive_.load()) return deliverRaw(src, pts, serial, hdr);
        auto fr = std::make_shared<Frame>();
        const int bpp = hdr ? 8 : 4;
        fr->hdr = hdr;
        fr->pool = pool_;
        fr->rgba = pool_->take(size_t(w) * h * bpp);
        uint8_t* dst[4] = {fr->rgba.data(), nullptr, nullptr, nullptr};
        int dstStride[4] = {w * bpp, 0, 0, 0};
        sws_scale(sws, src->data, src->linesize, 0, h, dst, dstStride);
        fr->w = w; fr->h = h; fr->pts = pts; fr->serial = serial;
        const AVRational sar = av_guess_sample_aspect_ratio(fmt_, fmt_->streams[vIndex_], src);
        fr->aspect = (sar.num > 0 && sar.den > 0) ? (double(w) * sar.num / sar.den) / h : double(w) / h;
        queueFrame(std::move(fr), serial);
        return true;
    };

    auto drain = [&] {
        while (!quit_) {
            int r = avcodec_receive_frame(vctx_, f);
            if (r < 0) break;
            AVFrame* use = f;
            if (f->format == AV_PIX_FMT_VAAPI) {
                av_frame_unref(sw);
                if (av_hwframe_transfer_data(sw, f, 0) < 0) {
                    // Hardware path unusable -> fall back to software decoding for the rest of the file.
                    av_frame_unref(f);
                    std::string e;
                    if (!hwTried) {
                        hwTried = true;
                        openVideoDecoder(false, e);
                        hwActive_ = false;
                    }
                    break;
                }
                av_frame_copy_props(sw, f);
                use = sw;
                hwActive_ = true;
            }
            deliver(use);
            av_frame_unref(f);
        }
    };

    while (!quit_) {
        Pkt p;
        if (!popPacket(vpkts_, p)) break;
        if (p.kind == Pkt::Flush) {
            serial = p.serial;
            avcodec_flush_buffers(vctx_);
            continue;
        }
        if (p.kind == Pkt::Eof) {
            avcodec_send_packet(vctx_, nullptr);
            drain();
            auto eos = std::make_shared<Frame>();
            eos->eos = true;
            eos->serial = serial;
            std::unique_lock l(fq_m_);
            fq_cv_.wait(l, [&] { return quit_ || int(frames_.size()) < kMaxVideoFrames + 1 || serial != serial_.load(); });
            if (serial == serial_.load()) frames_.push_back(eos);
            fq_cv_.notify_all();
            continue;
        }
        if (p.serial != serial_.load() || p.serial != serial) { av_packet_free(&p.pkt); continue; }
        int r = avcodec_send_packet(vctx_, p.pkt);
        av_packet_free(&p.pkt);
        if (r < 0 && r != AVERROR(EAGAIN) && r != AVERROR_EOF) continue;
        drain();
    }
    sws_freeContext(sws);
    sws_freeContext(swsRaw);
    av_frame_free(&f);
    av_frame_free(&sw);
}

// ------------------------------------------------------------------ audio (decode + output)

void MediaEngine::audioLoop() {
    constexpr int kRate = 48000;
    pa_simple* pa = nullptr;
    bool paFailed = false;
    Leveler fileLeveler;
    int fileSinkGen = sinkGen_.load();
    auto openPa = [&] {
        if (pa || paFailed) return;
        pa_sample_spec ss{PA_SAMPLE_S16LE, kRate, 2};
        pa_buffer_attr attr;
        attr.maxlength = uint32_t(-1);
        attr.tlength = uint32_t(pa_usec_to_bytes(60000, &ss));
        attr.prebuf = uint32_t(-1);
        attr.minreq = uint32_t(-1);
        attr.fragsize = uint32_t(-1);
        int err = 0;
        const std::string sk = sinkName();
        pa = pa_simple_new(nullptr, "VeyAmux", PA_STREAM_PLAYBACK, sk.empty() ? nullptr : sk.c_str(), "playback", &ss, nullptr, &attr, &err);
        if (!pa && !sk.empty()) pa = pa_simple_new(nullptr, "VeyAmux", PA_STREAM_PLAYBACK, nullptr, "playback", &ss, nullptr, &attr, &err);
        if (!pa) {
            paFailed = true;
            if (cb_.onGpuMessage) cb_.onGpuMessage(std::string("Audio unavailable: ") + pa_strerror(err));
        }
    };

    AVFrame* f = av_frame_alloc();
    SwrContext* swr = nullptr;
    AVChannelLayout outLayout;
    av_channel_layout_default(&outLayout, 2);
    int curFmt = -1, curRate = -1, curCh = -1;
    int serial = serial_.load();
    double lastEnd = 0.0;
    std::vector<int16_t> buf;

    auto consume = [&] {
        while (!quit_) {
            if (avcodec_receive_frame(actx_, f) < 0) break;
            const double dur = double(f->nb_samples) / std::max(1, f->sample_rate);
            const int64_t bt = f->best_effort_timestamp;
            const double pts = bt == AV_NOPTS_VALUE ? lastEnd : double(bt) * aTimeBase_ - startTime_;
            lastEnd = pts + dur;
            if (serial != serial_.load() || pts + dur < seekTarget_.load()) { av_frame_unref(f); continue; }
            if (f->format != curFmt || f->sample_rate != curRate || f->ch_layout.nb_channels != curCh || !swr) {
                swr_free(&swr);
                swr_alloc_set_opts2(&swr, &outLayout, AV_SAMPLE_FMT_S16, kRate, &f->ch_layout, AVSampleFormat(f->format),
                                    f->sample_rate, 0, nullptr);
                if (!swr || swr_init(swr) < 0) { swr_free(&swr); av_frame_unref(f); continue; }
                curFmt = f->format; curRate = f->sample_rate; curCh = f->ch_layout.nb_channels;
            }
            const int maxOut = swr_get_out_samples(swr, f->nb_samples);
            buf.resize(size_t(std::max(maxOut, 1)) * 2);
            uint8_t* o[1] = {reinterpret_cast<uint8_t*>(buf.data())};
            const int n = swr_convert(swr, o, maxOut, const_cast<const uint8_t**>(f->extended_data), f->nb_samples);
            av_frame_unref(f);
            if (n <= 0) continue;

            // Wait while paused (or until a seek/quit).
            {
                std::unique_lock l(gate_m_);
                gate_cv_.wait(l, [&] { return quit_ || playing_.load() || serial != serial_.load(); });
            }
            if (quit_ || serial != serial_.load()) continue;

            std::vector<int16_t> tap;
            if (cb_.onAudio) tap.assign(buf.begin(), buf.begin() + n * 2);
            if (normalize_.load()) fileLeveler.process(buf.data(), n, kRate);
            if (fileSinkGen != sinkGen_.load() && pa) { fileSinkGen = sinkGen_.load(); pa_simple_free(pa); pa = nullptr; }   // output device changed
            const float vol = muted_.load() ? 0.0f : volume_.load();
            if (vol != 1.0f)
                for (int i = 0; i < n * 2; ++i)
                    buf[size_t(i)] = int16_t(std::clamp(int(std::lround(buf[size_t(i)] * vol)), -32768, 32767));
            openPa();
            if (pa) {
                int err = 0;
                if (pa_simple_write(pa, buf.data(), size_t(n) * 4, &err) < 0) { pa_simple_free(pa); pa = nullptr; paFailed = true; continue; }
                const double lat = double(pa_simple_get_latency(pa, &err)) / 1e6;
                if (serial == serial_.load() && playing_.load()) clock_.set(lastEnd - lat);
                if (!tap.empty())   // the block ends 'lat' from now
                    cb_.onAudio(tap.data(), n, std::chrono::steady_clock::now() +
                                std::chrono::microseconds(int64_t((lat - double(n) / kRate) * 1e6)));
            }
        }
    };

    while (!quit_) {
        Pkt p;
        if (!popPacket(apkts_, p)) break;
        if (p.kind == Pkt::Flush) {
            serial = p.serial;
            avcodec_flush_buffers(actx_);
            if (swr) { swr_free(&swr); curFmt = -1; }
            if (pa) { int e = 0; pa_simple_flush(pa, &e); }
            continue;
        }
        if (p.kind == Pkt::Eof) {
            avcodec_send_packet(actx_, nullptr);
            consume();
            if (pa) { int e = 0; pa_simple_drain(pa, &e); }
            if (vIndex_ < 0 && !endedFired_.exchange(true)) {
                playing_ = false;
                clock_.setRunning(false);
                clock_.set(info_.duration);
                if (cb_.onEnded) cb_.onEnded();
            }
            continue;
        }
        if (p.serial != serial_.load() || p.serial != serial) { av_packet_free(&p.pkt); continue; }
        const int r = avcodec_send_packet(actx_, p.pkt);
        av_packet_free(&p.pkt);
        if (r < 0 && r != AVERROR(EAGAIN)) continue;
        consume();
    }
    if (pa) pa_simple_free(pa);
    swr_free(&swr);
    av_channel_layout_uninit(&outLayout);
    av_frame_free(&f);
}

// ------------------------------------------------------------------ render (Vulkan chain)

void MediaEngine::startGpu() {
    if (gpuStarted_.exchange(true)) return;
    gpuThread_ = std::thread([this] { ensureGpu(); });
}

void MediaEngine::ensureGpu() {
    if (const char* n = std::getenv("VEYAMUX_NO_GPU"); n && std::string(n) == "1") {
        if (cb_.onGpuMessage) cb_.onGpuMessage("GPU enhancement disabled (VEYAMUX_NO_GPU=1)");
        return;
    }
    std::string err;
    auto vk = std::make_unique<VkContext>();
    if (!vk->init(err)) {
        if (cb_.onGpuMessage) cb_.onGpuMessage("Vulkan unavailable: " + err);
        return;
    }
    auto chain = std::make_unique<EnhanceChain>(*vk, shaderDir_);
    if (!chain->init(err)) {
        if (cb_.onGpuMessage) cb_.onGpuMessage("Could not set up the enhancement chain: " + err);
        return;
    }
    gpuName_ = vk->deviceName() + (vk->isSoftware() ? " (software)" : "");
    vk_ = std::move(vk);
    chain_ = std::move(chain);
    gpuReady_ = true;
    if (cb_.onGpuMessage) cb_.onGpuMessage("GPU: " + gpuName_);
}

void MediaEngine::renderLoop() {
    if (!rawLive_.load()) startGpu();   // straight to the GPU: the window's Vulkan device does the work
    std::vector<uint8_t> out;
    // Timeline of the previous frame (render thread only): temporal passes reset across seeks and jumps.
    double prevPts = 0.0;
    int prevSerial = 0;
    auto lastArrival = steady::now();

    auto napGate = [&](double seconds) {
        std::unique_lock l(gate_m_);
        gate_cv_.wait_for(l, std::chrono::duration<double>(std::min(std::max(seconds, 0.0005), 0.02)));
    };
    auto deliver = [&](const uint8_t* px, int w, int h, double aspect) {
        if (cb_.onFrame) cb_.onFrame(px, w, h, aspect);
    };
    auto gpuFailed = [&](const std::string& err) {
        if (cb_.onGpuMessage) cb_.onGpuMessage("GPU error, falling back to passthrough: " + err);
        gpuReady_ = false;
        gpuName_.clear();
    };
    auto setStats = [&](steady::time_point t0, int ow, int oh, bool gpu) {
        std::lock_guard l(stat_m_);
        stats_.processMs = secs(steady::now() - t0) * 1000.0;
        ++stats_.rendered;
        stats_.outW = ow; stats_.outH = oh;
        stats_.gpuActive = gpu;
        stats_.upError = chain_ ? chain_->upError() : std::string();
        stats_.gpuMs = gpu && chain_ ? chain_->gpuMs() : -1.0f;
        if (gpu && chain_) stats_.gpuStages = chain_->gpuStageMs();
    };
    // Single frame (no generation). Returns true when the GPU chain produced the picture.
    auto present = [&](const std::shared_ptr<Frame>& f, float frameMs = 1000.0f / 30.0f) -> bool {
        EnhanceSettings s;
        { std::lock_guard l(set_m_); s = settings_; }
        const auto t0 = steady::now();
        bool done = false;
        int ow = f->w, oh = f->h;
        if (rawLive_.load()) {   // the window's Vulkan frame does the work (LiveVulkan), as for a capture card
            if (cb_.onLiveFrame) {
                RawFrame r;
                r.fmt = RawFrame::Fmt(f->fmt);
                r.w = f->w; r.h = f->h; r.stride = f->stride;
                r.captured = steady::now();
                r.reset = std::exchange(rawReset_, false);
                r.data = std::move(f->rgba);
                cb_.onLiveFrame(r);
                f->rgba = std::move(r.data);   // a used buffer: back to the pool with the frame
            }
            setStats(t0, ow, oh, true);
            return true;
        }
        // HDR frames always go through the GPU: they need tone mapping even with every enhancement off
        if (gpuReady_.load() && chain_ && (!s.isIdentity() || f->hdr)) {
            std::string err;
            if (chain_->process(f->rgba.data(), f->w, f->h, f->w * (f->hdr ? 8 : 4), s, out, ow, oh, err, frameMs, f->hdr)) {
                deliver(out.data(), ow, oh, f->aspect);
                done = true;
            } else gpuFailed(err);
        }
        if (!done) {
            ow = f->w; oh = f->h;
            if (f->hdr) {   // no GPU: show the high bytes (not tone-mapped - washed out, but visible)
                std::vector<uint8_t> b(size_t(f->w) * f->h * 4);
                for (size_t i = 0; i < b.size(); ++i) b[i] = f->rgba[2 * i + 1];
                deliver(b.data(), f->w, f->h, f->aspect);
            } else {
                deliver(f->rgba.data(), f->w, f->h, f->aspect);
            }
        }
        setStats(t0, ow, oh, done);
        return done;
    };
    while (!quit_) {
        if (!playing_.load() && !showOne_.load()) {
            // (straight to the GPU: LiveVulkan redraws the paused picture itself when the settings change)
            if (redraw_.exchange(false) && lastFrame_ && !rawLive_.load()) {
                if (chain_) chain_->resetHistory();
                present(lastFrame_);
            }
            napGate(0.02);
            continue;
        }

        // ------------------------------------------------------------ live capture: newest frame, no clock
        if (live_.load()) {
            std::shared_ptr<Frame> latest;
            {
                std::unique_lock l(fq_m_);
                fq_cv_.wait_for(l, std::chrono::milliseconds(20), [&] { return quit_ || !frames_.empty(); });
                if (frames_.empty()) continue;
                latest = frames_.back();
                frames_.clear();
            }
            const auto now = steady::now();
            const double dt = secs(now - lastArrival);
            lastArrival = now;
            lastFrame_ = latest;
            redraw_ = false;
            present(latest, float(dt * 1000.0));
            continue;
        }

        // ------------------------------------------------------------ file playback: audio clock
        std::shared_ptr<Frame> f;
        {
            std::unique_lock l(fq_m_);
            fq_cv_.wait_for(l, std::chrono::milliseconds(20), [&] { return quit_ || !frames_.empty(); });
            if (frames_.empty()) continue;
            f = frames_.front();
        }
        auto popFront = [&] {
            std::lock_guard l(fq_m_);
            if (!frames_.empty() && frames_.front() == f) frames_.pop_front();
            fq_cv_.notify_all();
        };
        if (f->serial != serial_.load()) { popFront(); continue; }
        if (f->eos) {
            popFront();
            playing_ = false;
            clock_.setRunning(false);
            clock_.set(info_.duration);
            showOne_ = false;
            if (!endedFired_.exchange(true) && cb_.onEnded) cb_.onEnded();
            continue;
        }
        const double gap = f->pts - prevPts;
        if (!showOne_.load()) {
            // audio offset: a positive value shows the picture earlier against the audio clock (= sound later)
            const double delay = f->pts - audioDelayMs_.load() * 0.001 - clock_.get();
            if (delay > 0.004) { napGate(delay); continue; }
            size_t pending;
            { std::lock_guard l(fq_m_); pending = frames_.size(); }
            if (f->pts - clock_.get() < -0.15 && pending > 1) {
                popFront();
                std::lock_guard l(stat_m_);
                ++stats_.dropped;
                continue;
            }
        }
        popFront();
        lastFrame_ = f;
        redraw_ = false;
        // Temporal passes (FSR upscaling, temporal denoising) must not blend across a seek or a jump in the timeline.
        if (f->serial != prevSerial || gap <= 0.0 || gap > 1.0) {
            if (chain_) chain_->resetHistory();
            rawReset_ = true;
        }
        present(f, float(std::max(gap, 0.0) * 1000.0));
        prevPts = f->pts;
        prevSerial = f->serial;
        showOne_ = false;
    }
    lastFrame_.reset();
}

// Hands paced pictures (see pacingRepair_) to the GPU path at their due time.
void MediaEngine::pacerLoop() {
    std::unique_lock l(pace_m_);
    while (!quit_) {
        if (paced_.empty()) { pace_cv_.wait_for(l, std::chrono::milliseconds(100)); continue; }
        const auto due = paced_.front().due;
        if (std::chrono::steady_clock::now() < due) {
            pace_cv_.wait_until(l, std::min(due, std::chrono::steady_clock::now() + std::chrono::milliseconds(100)));
            continue;
        }
        RawFrame f = std::move(paced_.front().frame);
        paced_.pop_front();
        l.unlock();
        if (cb_.onLiveFrame) cb_.onLiveFrame(f);   // f now holds a used buffer
        f.captureBuffer.reset();                 // a driver buffer goes back to the card
        l.lock();
        if (paceSpare_.size() < 3) paceSpare_.push_back(std::move(f));
    }
    paced_.clear();
}

double MediaEngine::pacingRepairShare() {
    if (!pacingRepair_.load()) return -1.0;
    const uint64_t t = pacedTotal_.exchange(0), r = pacedRepaired_.exchange(0);
    return t ? double(r) / double(t) : 0.0;
}

} // namespace veyamux
