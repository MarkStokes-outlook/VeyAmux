#include "CaptureSource.h"
#include "FrameLayout.h"
#include "DemoSource.h"
#include "RemotePlay.h"
#include "ScreenCast.h"
#include <cstdio>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <climits>
#include <string>
#include <dirent.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#ifndef V4L2_PIX_FMT_P010
#define V4L2_PIX_FMT_P010 v4l2_fourcc('P', '0', '1', '0')   // 10-bit 4:2:0, Y plane + interleaved CbCr, 16 bits per sample
#endif

namespace veyamux {
namespace {
int xioctl(int fd, unsigned long req, void* arg) {
    int r;
    do { r = ioctl(fd, req, arg); } while (r == -1 && errno == EINTR);
    return r;
}
// Which other process has the device open? ("name (PID n)", empty if unknown.)
std::string deviceOwner(const std::string& path) {
    char real[4096];
    if (!realpath(path.c_str(), real)) return {};
    const std::string target = real;
    const long self = long(getpid());
    DIR* proc = opendir("/proc");
    if (!proc) return {};
    std::string result;
    while (dirent* e = readdir(proc)) {
        char* end = nullptr;
        const long pid = std::strtol(e->d_name, &end, 10);
        if (!end || *end || pid <= 0 || pid == self) continue;
        const std::string fdDir = std::string("/proc/") + e->d_name + "/fd";
        DIR* fds = opendir(fdDir.c_str());
        if (!fds) continue;   // other users' processes are not readable
        bool found = false;
        while (dirent* f = readdir(fds)) {
            char link[4096];
            const ssize_t n = readlink((fdDir + "/" + f->d_name).c_str(), link, sizeof(link) - 1);
            if (n > 0) { link[n] = 0; if (target == link) { found = true; break; } }
        }
        closedir(fds);
        if (!found) continue;
        std::string name;
        if (FILE* c = std::fopen((std::string("/proc/") + e->d_name + "/comm").c_str(), "r")) {
            char buf[64] = {};
            if (std::fgets(buf, sizeof(buf), c)) name = buf;
            std::fclose(c);
            while (!name.empty() && (name.back() == '\n' || name.back() == ' ')) name.pop_back();
        }
        result = (name.empty() ? std::string("?") : name) + " (PID " + std::to_string(pid) + ")";
        break;
    }
    closedir(proc);
    return result;
}
AVPixelFormat pixFmtFor(uint32_t f) {
    switch (f) {
    case V4L2_PIX_FMT_YUYV: return AV_PIX_FMT_YUYV422;
    case V4L2_PIX_FMT_UYVY: return AV_PIX_FMT_UYVY422;
    case V4L2_PIX_FMT_NV12: return AV_PIX_FMT_NV12;
    case V4L2_PIX_FMT_P010: return AV_PIX_FMT_P010LE;
    case V4L2_PIX_FMT_YUV420: return AV_PIX_FMT_YUV420P;
    case V4L2_PIX_FMT_RGB24: return AV_PIX_FMT_RGB24;
    case V4L2_PIX_FMT_BGR24: return AV_PIX_FMT_BGR24;
    default: return AV_PIX_FMT_NONE;
    }
}
bool captureLayout(uint32_t f, PixelLayout& layout) {
    switch (f) {
    case V4L2_PIX_FMT_YUYV: case V4L2_PIX_FMT_UYVY: layout = PixelLayout::Packed422; return true;
    case V4L2_PIX_FMT_NV12: layout = PixelLayout::NV12; return true;
    case V4L2_PIX_FMT_P010: layout = PixelLayout::P010; return true;
    case V4L2_PIX_FMT_YUV420: layout = PixelLayout::I420; return true;
    case V4L2_PIX_FMT_RGB24: case V4L2_PIX_FMT_BGR24: layout = PixelLayout::RGB24; return true;
    default: return false;
    }
}
bool isMjpeg(uint32_t f) { return f == V4L2_PIX_FMT_MJPEG || f == V4L2_PIX_FMT_JPEG; }
bool supported(uint32_t f) { return isMjpeg(f) || pixFmtFor(f) != AV_PIX_FMT_NONE; }
bool compressed(uint32_t f) { return isMjpeg(f); }
} // namespace

std::string fourccName(uint32_t f) {
    switch (f) {
    case V4L2_PIX_FMT_YUYV: return "YUYV";
    case V4L2_PIX_FMT_UYVY: return "UYVY";
    case V4L2_PIX_FMT_NV12: return "NV12";
    case V4L2_PIX_FMT_P010: return "P010 (HDR)";
    case V4L2_PIX_FMT_YUV420: return "I420";
    case V4L2_PIX_FMT_RGB24: return "RGB24";
    case V4L2_PIX_FMT_BGR24: return "BGR24";
    case V4L2_PIX_FMT_MJPEG: case V4L2_PIX_FMT_JPEG: return "MJPEG";
    default: { char s[5] = {char(f), char(f >> 8), char(f >> 16), char(f >> 24), 0}; return s; }
    }
}

std::vector<CaptureDevice> listCaptureDevices() {
    std::vector<CaptureDevice> out;
    DIR* d = opendir("/dev");
    if (!d) return out;
    std::vector<std::string> names;
    while (dirent* e = readdir(d))
        if (std::strncmp(e->d_name, "video", 5) == 0) names.push_back(e->d_name);
    closedir(d);
    std::sort(names.begin(), names.end(), [](const std::string& a, const std::string& b) {
        return (a.size() != b.size()) ? a.size() < b.size() : a < b;   // video2 < video10
    });
    for (const auto& n : names) {
        const std::string path = "/dev/" + n;
        int fd = ::open(path.c_str(), O_RDWR | O_NONBLOCK);
        if (fd < 0) continue;
        v4l2_capability cap{};
        if (xioctl(fd, VIDIOC_QUERYCAP, &cap) == 0) {
            const uint32_t caps = (cap.capabilities & V4L2_CAP_DEVICE_CAPS) ? cap.device_caps : cap.capabilities;
            if ((caps & V4L2_CAP_VIDEO_CAPTURE) && (caps & V4L2_CAP_STREAMING))
                out.push_back({path, reinterpret_cast<const char*>(cap.card)});
        }
        ::close(fd);
    }
    return out;
}

std::vector<CaptureMode> listCaptureModes(const std::string& path) {
    std::vector<CaptureMode> modes;
    int fd = ::open(path.c_str(), O_RDWR | O_NONBLOCK);
    if (fd < 0) return modes;
    static const int kSizes[][2] = {{3840, 2160}, {2560, 1440}, {1920, 1200}, {1920, 1080}, {1600, 900},
                                    {1280, 720}, {1024, 768}, {800, 600}, {720, 576}, {720, 480}, {640, 480}};
    static const int kFps[] = {120, 60, 50, 30, 25, 24};
    for (uint32_t i = 0;; ++i) {
        v4l2_fmtdesc fd_{};
        fd_.index = i;
        fd_.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        if (xioctl(fd, VIDIOC_ENUM_FMT, &fd_) != 0) break;
        if (!supported(fd_.pixelformat)) continue;
        std::vector<std::pair<int, int>> sizes;
        for (uint32_t j = 0;; ++j) {
            v4l2_frmsizeenum fs{};
            fs.index = j; fs.pixel_format = fd_.pixelformat;
            if (xioctl(fd, VIDIOC_ENUM_FRAMESIZES, &fs) != 0) break;
            if (fs.type == V4L2_FRMSIZE_TYPE_DISCRETE) sizes.push_back({int(fs.discrete.width), int(fs.discrete.height)});
            else {
                for (auto& s : kSizes)
                    if (uint32_t(s[0]) >= fs.stepwise.min_width && uint32_t(s[0]) <= fs.stepwise.max_width &&
                        uint32_t(s[1]) >= fs.stepwise.min_height && uint32_t(s[1]) <= fs.stepwise.max_height)
                        sizes.push_back({s[0], s[1]});
                break;
            }
        }
        for (auto [w, h] : sizes) {
            std::vector<std::pair<uint32_t, uint32_t>> ivals;   // num, den
            for (uint32_t k = 0;; ++k) {
                v4l2_frmivalenum fi{};
                fi.index = k; fi.pixel_format = fd_.pixelformat; fi.width = uint32_t(w); fi.height = uint32_t(h);
                if (xioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, &fi) != 0) break;
                if (fi.type == V4L2_FRMIVAL_TYPE_DISCRETE) ivals.push_back({fi.discrete.numerator, fi.discrete.denominator});
                else {
                    for (int f : kFps) {
                        const double iv = 1.0 / f;
                        if (iv >= double(fi.stepwise.min.numerator) / fi.stepwise.min.denominator - 1e-9 &&
                            iv <= double(fi.stepwise.max.numerator) / fi.stepwise.max.denominator + 1e-9)
                            ivals.push_back({1, uint32_t(f)});
                    }
                    break;
                }
            }
            if (ivals.empty()) ivals.push_back({1, 30});
            for (auto [n, dn] : ivals) {
                if (n == 0 || dn == 0) continue;
                CaptureMode m;
                m.fourcc = fd_.pixelformat; m.w = w; m.h = h; m.num = n; m.den = dn;
                m.fps = double(dn) / n;
                char buf[96];
                std::snprintf(buf, sizeof buf, "%dx%d @ %.3g fps · %s", w, h, std::round(m.fps * 100) / 100, fourccName(m.fourcc).c_str());
                m.label = buf;
                modes.push_back(m);
            }
        }
    }
    ::close(fd);
    std::stable_sort(modes.begin(), modes.end(), [](const CaptureMode& a, const CaptureMode& b) {
        const bool ah = a.fps >= 50, bh = b.fps >= 50;
        if (ah != bh) return ah;
        const long pa = long(a.w) * a.h, pb = long(b.w) * b.h;
        if (pa != pb) return pa > pb;
        if (std::abs(a.fps - b.fps) > 0.5) return a.fps > b.fps;
        // HDR (P010) only when chosen on purpose: twice the USB bandwidth, and an SDR source needs no tone mapping
        const bool ap = a.fourcc == V4L2_PIX_FMT_P010, bp = b.fourcc == V4L2_PIX_FMT_P010;
        if (ap != bp) return bp;
        return !compressed(a.fourcc) && compressed(b.fourcc);
    });
    return modes;
}

// ------------------------------------------------------------------ FrameConverter

FrameConverter::FrameConverter() = default;
FrameConverter::~FrameConverter() {
    sws_freeContext(sws_);
    if (mjpeg_) avcodec_free_context(&mjpeg_);
    av_frame_free(&frame_);
    av_packet_free(&pkt_);
}

bool FrameConverter::init(uint32_t fourcc, int w, int h, int stride, std::string& error) {
    sws_freeContext(sws_); sws_ = nullptr;
    avcodec_free_context(&mjpeg_);
    av_frame_free(&frame_);
    av_packet_free(&pkt_);
    if (w <= 0 || h <= 0 || w > INT_MAX / 4 || size_t(w) * h > size_t(INT_MAX) / 4) {
        error = "Invalid capture dimensions"; return false;
    }
    PixelLayout format; FrameLayout layout;
    if (!isMjpeg(fourcc) && (!captureLayout(fourcc, format) || !describeFrame(format, w, h, stride, layout))) {
        error = "Invalid capture plane layout"; return false;
    }
    fourcc_ = fourcc; w_ = w; h_ = h; stride_ = stride;
    if (isMjpeg(fourcc)) {
        const AVCodec* c = avcodec_find_decoder(AV_CODEC_ID_MJPEG);
        if (!c) { error = "FFmpeg has no MJPEG decoder"; return false; }
        mjpeg_ = avcodec_alloc_context3(c);
        if (!mjpeg_) { error = "Could not allocate MJPEG decoder"; return false; }
        mjpeg_->thread_count = 2;
        mjpeg_->thread_type = FF_THREAD_SLICE;     // frame threading would add latency
        mjpeg_->flags |= AV_CODEC_FLAG_LOW_DELAY;
        if (avcodec_open2(mjpeg_, c, nullptr) < 0) { error = "Could not open the MJPEG decoder"; return false; }
        frame_ = av_frame_alloc();
        pkt_ = av_packet_alloc();
        if (!frame_ || !pkt_) { error = "Could not allocate MJPEG frame/packet"; return false; }
    } else if (pixFmtFor(fourcc) == AV_PIX_FMT_NONE) {
        error = "Unsupported pixel format " + fourccName(fourcc);
        return false;
    }
    return true;
}

bool FrameConverter::convert(const uint8_t* data, size_t size, std::vector<uint8_t>& rgba, std::string& error) {
    if (!data || size == 0) { error = "Empty capture frame"; return false; }
    const uint8_t* planes[4] = {nullptr, nullptr, nullptr, nullptr};
    int lines[4] = {0, 0, 0, 0};
    AVPixelFormat fmt;
    int range = fullRange_ ? 1 : 0;
    int w = w_, h = h_;
    if (mjpeg_) {
        av_packet_unref(pkt_);
        if (size > size_t(INT_MAX - AV_INPUT_BUFFER_PADDING_SIZE) || av_new_packet(pkt_, int(size)) < 0) {
            error = "Could not allocate padded MJPEG packet"; return false;
        }
        // av_new_packet owns and zeroes AV_INPUT_BUFFER_PADDING_SIZE trailing bytes.
        std::memcpy(pkt_->data, data, size);
        if (avcodec_send_packet(mjpeg_, pkt_) < 0 || avcodec_receive_frame(mjpeg_, frame_) < 0) {
            error = "Corrupt MJPEG frame";
            return false;
        }
        w = frame_->width; h = frame_->height; fmt = AVPixelFormat(frame_->format);
        if (w != w_ || h != h_) { error = "MJPEG dimensions differ from capture mode"; return false; }
        for (int i = 0; i < 4; ++i) { planes[i] = frame_->data[i]; lines[i] = frame_->linesize[i]; }
        range = (frame_->color_range == AVCOL_RANGE_MPEG) ? 0 : 1;   // JPEG is full range unless flagged
    } else {
        fmt = pixFmtFor(fourcc_);
        PixelLayout format; FrameLayout layout;
        if (!captureLayout(fourcc_, format) || !validateFrame(format, w, h, stride_, size, size, layout)) {
            error = "Incomplete or invalid capture planes"; return false;
        }
        for (size_t i = 0; i < layout.count; ++i) {
            planes[i] = data + layout.planes[i].offset;
            lines[i] = int(layout.planes[i].stride);
        }
        if (fmt == AV_PIX_FMT_RGB24 || fmt == AV_PIX_FMT_BGR24) range = 1;
    }
    sws_ = sws_getCachedContext(sws_, w, h, fmt, w, h, AV_PIX_FMT_RGBA, SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!sws_) { error = "sws_getCachedContext failed"; return false; }
    const int cs = (h >= 720 || w >= 1280) ? SWS_CS_ITU709 : SWS_CS_ITU601;
    sws_setColorspaceDetails(sws_, sws_getCoefficients(cs), range, sws_getCoefficients(SWS_CS_DEFAULT), 1, 0, 1 << 16, 1 << 16);
    rgba.resize(size_t(w) * h * 4);
    uint8_t* dst[4] = {rgba.data(), nullptr, nullptr, nullptr};
    int dstLines[4] = {w * 4, 0, 0, 0};
    sws_scale(sws_, planes, lines, 0, h, dst, dstLines);
    w_ = w; h_ = h;
    return true;
}

// ------------------------------------------------------------------ CaptureSource

CaptureSource::CaptureSource() = default;
CaptureSource::~CaptureSource() { close(); }

bool CaptureSource::open(const std::string& path, const CaptureMode& mode, std::string& error) {
    close();
    probe_.reset();
    delivered_.reset();
    rawCompare_.reset();
    newSinceDelivered_ = true;
    if (isRemotePlay(path)) {
        auto s = std::make_unique<remoteplay::Stream>();
        if (!s->open(path, error)) return false;
        w_ = s->width(); h_ = s->height(); fps_ = s->fps();
        card_ = s->name();
        fourcc_ = 0;
        ps_ = std::move(s);
        return true;
    }
    if (isDemo(path)) {
        auto d = std::make_unique<demo::Stream>();
        // "demo:<fps>[:<height>]"
        const char* colon = std::strchr(path.c_str() + 5, ':');
        if (!d->open(std::atoi(path.c_str() + 5), colon ? std::atoi(colon + 1) : 1080, error)) return false;
        demo_ = std::move(d);
        w_ = demo_->width(); h_ = demo_->height(); fps_ = demo_->fps();
        card_ = "Abyssal Garden (demo)";
        fourcc_ = 0;
        return true;
    }
    if (isScreenCast(path)) {
        int pfd = -1; unsigned node = 0;
        if (std::sscanf(path.c_str(), "pipewire:%d:%u", &pfd, &node) != 2) { error = "Bad screen share address"; return false; }
        auto sc = std::make_unique<ScreenCastStream>();
        if (!sc->open(pfd, node, error)) return false;
        sc_ = std::move(sc);
        w_ = sc_->width(); h_ = sc_->height(); fps_ = sc_->fps();
        card_ = "Window / screen";
        fourcc_ = 0;
        return true;
    }
    v4l2_format fmt{};
    // A device another process is still streaming from answers EBUSY. That is typically the previous VeyAmux still
    // shutting down after "Restart now" (the Vulkan layers can make that take a moment): wait up to 5 s for it.
    for (int attempt = 0;; ++attempt) {
        fd_ = ::open(path.c_str(), O_RDWR | O_NONBLOCK);
        if (fd_ < 0) { error = "Could not open " + path + ": " + std::strerror(errno); return false; }
        v4l2_capability cap{};
        if (xioctl(fd_, VIDIOC_QUERYCAP, &cap) == 0) card_ = reinterpret_cast<const char*>(cap.card);
        fmt = {};
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        fmt.fmt.pix.width = uint32_t(mode.w);
        fmt.fmt.pix.height = uint32_t(mode.h);
        fmt.fmt.pix.pixelformat = mode.fourcc;
        fmt.fmt.pix.field = V4L2_FIELD_ANY;
        if (xioctl(fd_, VIDIOC_S_FMT, &fmt) == 0) break;
        const int err = errno;
        ::close(fd_); fd_ = -1;
        if (err == EBUSY && attempt < 20) { usleep(250 * 1000); continue; }
        if (err == EBUSY) {
            const std::string owner = deviceOwner(path);
            error = "The capture device is in use by another program" + (owner.empty() ? std::string() : ": " + owner);
        } else {
            error = std::string("VIDIOC_S_FMT: ") + std::strerror(err);
        }
        return false;
    }
    if (fmt.fmt.pix.pixelformat != mode.fourcc) { error = "The device refused pixel format " + fourccName(mode.fourcc); close(); return false; }
    if (fmt.fmt.pix.width == 0 || fmt.fmt.pix.height == 0 || fmt.fmt.pix.width > INT_MAX / 4 ||
        fmt.fmt.pix.height > INT_MAX || fmt.fmt.pix.bytesperline > INT_MAX) {
        error = "Invalid negotiated capture dimensions or stride"; close(); return false;
    }
    w_ = int(fmt.fmt.pix.width); h_ = int(fmt.fmt.pix.height);
    fourcc_ = mode.fourcc;
    rawStride_ = int(fmt.fmt.pix.bytesperline);
    if (!rawStride_) {
        const int bpp = (fourcc_ == V4L2_PIX_FMT_NV12 || fourcc_ == V4L2_PIX_FMT_YUV420) ? 1 :
                        (fourcc_ == V4L2_PIX_FMT_RGB24 || fourcc_ == V4L2_PIX_FMT_BGR24) ? 3 : 2;
        rawStride_ = w_ * bpp;
    }

    v4l2_streamparm parm{};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe = {mode.num, mode.den};
    if (xioctl(fd_, VIDIOC_S_PARM, &parm) == 0 && parm.parm.capture.timeperframe.numerator)
        fps_ = double(parm.parm.capture.timeperframe.denominator) / parm.parm.capture.timeperframe.numerator;
    else fps_ = mode.fps;

    if (!conv_.init(mode.fourcc, w_, h_, rawStride_, error)) { close(); return false; }

    v4l2_requestbuffers req{};
    // 6: the renderer may hold two (the newest and the one on screen) without starving the card
    req.count = 6; req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; req.memory = V4L2_MEMORY_MMAP;
    if (xioctl(fd_, VIDIOC_REQBUFS, &req) != 0 || req.count < 2) { error = "The device does not support memory-mapped streaming"; close(); return false; }
    bufs_.resize(req.count);
    for (uint32_t i = 0; i < req.count; ++i) {
        v4l2_buffer b{};
        b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; b.memory = V4L2_MEMORY_MMAP; b.index = i;
        if (xioctl(fd_, VIDIOC_QUERYBUF, &b) != 0) { error = "VIDIOC_QUERYBUF failed"; close(); return false; }
        bufs_[i].length = b.length;
        bufs_[i].start = mmap(nullptr, b.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, b.m.offset);
        if (bufs_[i].start == MAP_FAILED) { bufs_[i].start = nullptr; error = "mmap failed"; close(); return false; }
        if (xioctl(fd_, VIDIOC_QBUF, &b) != 0) { error = "VIDIOC_QBUF failed"; close(); return false; }
    }
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd_, VIDIOC_STREAMON, &type) != 0) { error = std::string("VIDIOC_STREAMON: ") + std::strerror(errno); close(); return false; }
    streaming_ = true;
    map_ = std::make_shared<Mapping>();
    map_->fd = fd_;
    map_->bufs = bufs_;   // the mapping now belongs to map_ (unmapped with it)
    return true;
}

CaptureSource::Mapping::~Mapping() {
    for (auto& b : bufs) if (b.start) munmap(b.start, b.length);
}

void CaptureSource::Mapping::requeue(uint32_t index) noexcept {
    std::lock_guard l(m);
    if (!live) return;
    v4l2_buffer b{};
    b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    b.memory = V4L2_MEMORY_MMAP;
    b.index = index;
    xioctl(fd, VIDIOC_QBUF, &b);
}

void CaptureSource::close() {
    sc_.reset();
    demo_.reset();
    ps_.reset();
    if (fd_ < 0) return;
    if (map_) { std::lock_guard l(map_->m); map_->live = false; }   // buffers still held are not queued again
    if (streaming_) { int type = V4L2_BUF_TYPE_VIDEO_CAPTURE; xioctl(fd_, VIDIOC_STREAMOFF, &type); streaming_ = false; }
    if (map_) map_.reset();   // unmapped when the last held frame is gone
    else for (auto& b : bufs_) if (b.start) munmap(b.start, b.length);
    bufs_.clear();
    ::close(fd_);
    fd_ = -1;
}

namespace {
std::chrono::steady_clock::time_point captureTime(const v4l2_buffer& b) {
    // V4L2 timestamps are CLOCK_MONOTONIC on current drivers, the same clock as steady_clock on Linux
    if ((b.flags & V4L2_BUF_FLAG_TIMESTAMP_MASK) == V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC && (b.timestamp.tv_sec || b.timestamp.tv_usec))
        return std::chrono::steady_clock::time_point(std::chrono::seconds(b.timestamp.tv_sec) + std::chrono::microseconds(b.timestamp.tv_usec));
    return std::chrono::steady_clock::now();
}
} // namespace

void CaptureSource::probeRaw(const v4l2_buffer& b) {
    if ((b.flags & V4L2_BUF_FLAG_ERROR) || b.index >= bufs_.size()) return;
    const auto* src = static_cast<const uint8_t*>(bufs_[b.index].start);
    if (!src || rawStride_ <= 0) return;
    PixelLayout format; FrameLayout layout;
    if (!captureLayout(fourcc_, format) ||
        !validateFrame(format, w_, h_, rawStride_, b.bytesused, bufs_[b.index].length, layout)) return;
    const uint8_t* luma = nullptr;
    int step = 2;
    if (fourcc_ == V4L2_PIX_FMT_NV12) { luma = src; step = 1; }
    else if (fourcc_ == V4L2_PIX_FMT_YUYV) luma = src;
    else if (fourcc_ == V4L2_PIX_FMT_UYVY || fourcc_ == V4L2_PIX_FMT_P010) luma = src + 1;   // P010: high byte of 16-bit luma
    if (!luma) return;
    // every buffer (also ones dropped for latency) is compared once, in full, against the last new picture; the result
    // feeds the measured picture rate and decides whether the next frame handed out is a repeat (readLatest)
    static const bool sparse = std::getenv("VEYAMUX_DEDUP_SPARSE") != nullptr;   // the old sparse grid, to compare
    if (sparse) { probe_.feed(luma, w_, h_, rawStride_, step, captureTime(b)); return; }
    const bool isNew = rawCompare_.changed(luma, w_, h_, rawStride_, step);
    newSinceDelivered_ = newSinceDelivered_ || isNew;
    probe_.feed(luma, w_, h_, rawStride_, step, captureTime(b), isNew ? 1 : 0);
    // compressed/RGB formats are probed after conversion (probeRgba)
}

void CaptureSource::probeRgba(const std::vector<uint8_t>& rgba) {
    if (fourcc_ == V4L2_PIX_FMT_NV12 || fourcc_ == V4L2_PIX_FMT_YUYV || fourcc_ == V4L2_PIX_FMT_UYVY || fourcc_ == V4L2_PIX_FMT_P010) return;
    if (rgba.size() >= size_t(w_) * h_ * 4) probe_.feed(rgba.data() + 1, w_, h_, w_ * 4, 4, std::chrono::steady_clock::now());   // green ~ luma
}

bool CaptureSource::dequeueNewest(v4l2_buffer& held, CaptureBufferLease& lease, std::string& error) {
    bool got = false;
    for (size_t i = 0; i <= bufs_.size(); ++i) {
        v4l2_buffer b{};
        b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        b.memory = V4L2_MEMORY_MMAP;
        if (xioctl(fd_, VIDIOC_DQBUF, &b) != 0) {
            if (errno == EAGAIN) break;
            error = std::string("VIDIOC_DQBUF: ") + std::strerror(errno);
            return false;
        }
        probeRaw(b);                                    // every delivered frame, including ones dropped for latency
        lease = CaptureBufferLease{map_, b.index};      // replacing the lease returns the stale buffer
        held = b;
        got = true;
    }
    if (!got) error.clear();
    return got;
}

CaptureSource::Result CaptureSource::read(std::vector<uint8_t>& rgba, int timeoutMs, std::string& error) {
    if (ps_) {
        RawFrame f;
        const auto r = ps_->readLatest(f, timeoutMs, error);
        if (r != remoteplay::Stream::Result::Frame) return r == remoteplay::Stream::Result::Timeout ? Result::Timeout : Result::Error;
        const int fmt = int(f.fmt);
        if (f.w != psConvW_ || f.h != psConvH_ || fmt != psConvFmt_) {
            if (!psConv_.init(f.fmt == RawFrame::P010 ? V4L2_PIX_FMT_P010 : V4L2_PIX_FMT_NV12, f.w, f.h, f.stride, error)) return Result::Error;
            psConvW_ = f.w; psConvH_ = f.h; psConvFmt_ = fmt;
        }
        w_ = f.w; h_ = f.h;
        if (!psConv_.convert(f.data.data(), f.data.size(), rgba, error)) return Result::Timeout;
        lastRepeat_ = false;
        probeRgba(rgba);
        return Result::Frame;
    }
    if (demo_) {
        RawFrame f;
        const auto r = demo_->readLatest(f, timeoutMs, error);
        if (r != demo::Stream::Result::Frame) return r == demo::Stream::Result::Timeout ? Result::Timeout : Result::Error;
        lastRepeat_ = false;
        probe_.feed(f.data.data() + 1, f.w, f.h, f.stride, 4, f.captured);
        rgba.swap(f.data);
        return Result::Frame;
    }
    if (sc_) {
        RawFrame f;
        const auto r = sc_->readLatest(f, timeoutMs, error);
        if (r != ScreenCastStream::Result::Frame) return r == ScreenCastStream::Result::Timeout ? Result::Timeout : Result::Error;
        w_ = f.w; h_ = f.h;
        lastRepeat_ = false;
        if (f.data.size() >= size_t(f.stride) * f.h) probe_.feed(f.data.data() + 1, f.w, f.h, f.stride, 4, f.captured);   // green ~ luma: measured picture rate
        rgba.swap(f.data);
        return Result::Frame;
    }
    if (fd_ < 0) { error = "Capture device is closed"; return Result::Error; }
    pollfd p{fd_, POLLIN, 0};
    const int r = poll(&p, 1, timeoutMs);
    if (r == 0) return Result::Timeout;
    if (r < 0) { if (errno == EINTR) return Result::Timeout; error = std::strerror(errno); return Result::Error; }
    if (p.revents & (POLLERR | POLLHUP | POLLNVAL)) { error = "Capture device disconnected"; return Result::Error; }
    v4l2_buffer b{};
    CaptureBufferLease lease;
    std::string e;
    if (!dequeueNewest(b, lease, e)) { if (!e.empty()) { error = e; return Result::Error; } return Result::Timeout; }
    Result res = Result::Timeout;
    if (!(b.flags & V4L2_BUF_FLAG_ERROR) && b.bytesused > 0 && b.index < bufs_.size() && b.bytesused <= bufs_[b.index].length) {
        std::string ce;
        if (conv_.convert(static_cast<const uint8_t*>(bufs_[b.index].start), b.bytesused, rgba, ce)) {
            res = Result::Frame;
            probeRgba(rgba);
            lastRepeat_ = !delivered_.changed(rgba.data() + 1, w_, h_, w_ * 4, 4, true);
        }
    }
    return res;
}

CaptureSource::Result CaptureSource::readLatest(RawFrame& out, int timeoutMs, std::string& error) {
    out.captureBuffer.reset();   // return any borrowed buffer from the recycled frame
    if (ps_) {
        out.repeat = false;
        const auto r = ps_->readLatest(out, timeoutMs, error);
        if (r == remoteplay::Stream::Result::Frame) {
            w_ = out.w; h_ = out.h;
            if (out.fmt == RawFrame::NV12) probe_.feed(out.data.data(), out.w, out.h, out.stride, 1, out.captured);   // luma plane
            else probe_.feed(out.data.data() + 1, out.w, out.h, out.stride, 2, out.captured);   // P010: high byte of luma
        }
        return r == remoteplay::Stream::Result::Frame ? Result::Frame : r == remoteplay::Stream::Result::Timeout ? Result::Timeout : Result::Error;
    }
    if (demo_) {
        const auto r = demo_->readLatest(out, timeoutMs, error);
        if (r == demo::Stream::Result::Frame) probe_.feed(out.data.data() + 1, out.w, out.h, out.stride, 4, out.captured);
        return r == demo::Stream::Result::Frame ? Result::Frame : r == demo::Stream::Result::Timeout ? Result::Timeout : Result::Error;
    }
    if (sc_) {
        const auto r = sc_->readLatest(out, timeoutMs, error);
        if (r == ScreenCastStream::Result::Frame) {
            w_ = out.w; h_ = out.h;
            if (out.data.size() >= size_t(out.stride) * out.h)
                probe_.feed(out.data.data() + 1, out.w, out.h, out.stride, 4, out.captured);   // measured picture rate
        }
        return r == ScreenCastStream::Result::Frame ? Result::Frame : r == ScreenCastStream::Result::Timeout ? Result::Timeout : Result::Error;
    }
    if (fd_ < 0) { error = "Capture device is closed"; return Result::Error; }
    pollfd p{fd_, POLLIN, 0};
    const int r = poll(&p, 1, timeoutMs);
    if (r == 0) return Result::Timeout;
    if (r < 0) { if (errno == EINTR) return Result::Timeout; error = std::strerror(errno); return Result::Error; }
    if (p.revents & (POLLERR | POLLHUP | POLLNVAL)) { error = "Capture device disconnected"; return Result::Error; }
    v4l2_buffer b{};
    CaptureBufferLease lease;
    std::string e;
    if (!dequeueNewest(b, lease, e)) { if (!e.empty()) { error = e; return Result::Error; } return Result::Timeout; }
    Result res = Result::Timeout;
    out.captured = captureTime(b);
    if (!(b.flags & V4L2_BUF_FLAG_ERROR) && b.bytesused > 0 && b.index < bufs_.size() && b.bytesused <= bufs_[b.index].length) {
        const uint8_t* src = static_cast<const uint8_t*>(bufs_[b.index].start);
        const size_t len = b.bytesused;
        const bool p010 = fourcc_ == V4L2_PIX_FMT_P010;
        const bool raw = fourcc_ == V4L2_PIX_FMT_YUYV || fourcc_ == V4L2_PIX_FMT_UYVY || fourcc_ == V4L2_PIX_FMT_NV12 || p010;
        if (raw) {
            PixelLayout format; FrameLayout layout;
            if (captureLayout(fourcc_, format) && validateFrame(format, w_, h_, rawStride_, len, bufs_[b.index].length, layout)) {
                const size_t need = layout.requiredBytes;
                out.fmt = fourcc_ == V4L2_PIX_FMT_YUYV ? RawFrame::YUYV : fourcc_ == V4L2_PIX_FMT_UYVY ? RawFrame::UYVY
                        : p010 ? RawFrame::P010 : RawFrame::NV12;
                static const bool forceCopy = std::getenv("VEYAMUX_CAPTURE_COPY") != nullptr;   // the old way, to compare
                // This thread is the only one that takes leases. Count this one too, so even with pacing enabled
                // at least three buffers go back to the driver. The extra reference belongs to CaptureSource.
                const size_t leased = size_t(map_.use_count() - 1);
                if (bufs_.size() < 5 || leased > bufs_.size() - 3 || forceCopy) {
                    out.data.assign(src, src + need);
                } else {
                    lease.setPixels({src, need});
                    out.captureBuffer = std::move(lease);
                }
                out.w = w_; out.h = h_; out.stride = rawStride_;
                // luma bytes for repeat detection: NV12 every byte, YUYV every 2nd, UYVY every 2nd from 1, P010 the high bytes
                const int step = fourcc_ == V4L2_PIX_FMT_NV12 ? 1 : 2, off = (fourcc_ == V4L2_PIX_FMT_UYVY || p010) ? 1 : 0;
                static const bool sparse = std::getenv("VEYAMUX_DEDUP_SPARSE") != nullptr;
                if (sparse) out.repeat = !delivered_.changed(src + off, w_, h_, rawStride_, step, true);
                else { out.repeat = !newSinceDelivered_; newSinceDelivered_ = false; }   // see probeRaw
                res = Result::Frame;
            }
        } else {
            std::string ce;
            if (conv_.convert(src, len, out.data, ce)) {
                probeRgba(out.data);
                out.repeat = !delivered_.changed(out.data.data() + 1, w_, h_, w_ * 4, 4, true);
                out.fmt = RawFrame::RGBA;
                out.w = w_; out.h = h_; out.stride = w_ * 4;
                res = Result::Frame;
            }
        }
    }
    return res;
}

} // namespace veyamux
