#include "LiveVulkan.h"
#include <cstring>
#include <QImage>
#include <QMetaObject>
#include <QQuickWindow>
#include <QSGRendererInterface>
#include <QTimer>
#include <algorithm>
#include <QDebug>
#include <chrono>

using namespace veyamux;

LiveVulkan::LiveVulkan(QQuickWindow* window, VkContext* ctx, const QString& shaderDir, QObject* parent)
    : QObject(parent), win_(window), ctx_(ctx), chain_(std::make_unique<EnhanceChain>(*ctx, shaderDir.toStdString())) {
    connect(window, &QQuickWindow::beforeRendering, this, &LiveVulkan::onBeforeRendering, Qt::DirectConnection);
}

LiveVulkan::~LiveVulkan() {
    streamQuit_ = true;
    if (streamThread_.joinable()) streamThread_.join();
}

bool LiveVulkan::init(std::string& error) {
    if (!chain_->init(error)) return false;
    streamThread_ = std::thread([this] { streamLoop(); });
    return true;
}

void LiveVulkan::setStream(int maxHeight, std::function<void(QImage, int)> sink) {
    { std::lock_guard l(sinkM_); sink_ = std::move(sink); }
    chain_->setStreamMaxHeight(maxHeight);
}

void LiveVulkan::setRecord(int maxHeight, std::function<void(veyamux::EnhanceChain::RecFrame&&)> sink) {
    { std::lock_guard l(sinkM_); recSink_ = std::move(sink); }
    chain_->setRecordMaxHeight(maxHeight);
}

void LiveVulkan::setCompare(int mode, const QImage& a, const QImage& b) {
    auto bytes = [](const QImage& img) {
        std::vector<uint8_t> v;
        if (img.isNull()) return v;
        const QImage c = img.convertToFormat(QImage::Format_RGBA8888);
        v.resize(size_t(c.width()) * c.height() * 4);
        for (int y = 0; y < c.height(); ++y) std::memcpy(v.data() + size_t(y) * c.width() * 4, c.constScanLine(y), size_t(c.width()) * 4);
        return v;
    };
    if (mode > 0) chain_->setCompareLabels(bytes(a), a.width(), a.height(), bytes(b), b.width(), b.height());
    chain_->setCompare(mode);
}

void LiveVulkan::requestScreenshot(std::function<void(QImage)> cb) {
    { std::lock_guard l(sinkM_); shotCb_ = std::move(cb); }
    chain_->requestScreenshot();
    { std::lock_guard l(m_); redraw_ = true; }   // render the current picture again even if the source is paused
    if (win_) QMetaObject::invokeMethod(win_, "update", Qt::QueuedConnection);
}

// Collects the OBS stream copies as soon as the GPU has written them (independent of when Qt renders next).
void LiveVulkan::streamLoop() {
    std::vector<EnhanceChain::StreamFrame> frames;
    std::vector<EnhanceChain::RecFrame> rec;
    while (!streamQuit_.load()) {
        rec.clear();
        chain_->pollRecord(rec);
        if (!rec.empty()) {
            std::function<void(EnhanceChain::RecFrame&&)> rs;
            { std::lock_guard l(sinkM_); rs = recSink_; }
            if (rs) for (auto& f : rec) rs(std::move(f));
        }
        EnhanceChain::StreamFrame shot;
        if (chain_->pollScreenshot(shot)) {
            std::function<void(QImage)> cb;
            { std::lock_guard l(sinkM_); cb = std::move(shotCb_); shotCb_ = nullptr; }
            if (cb) {
                QImage img(shot.w, shot.h, QImage::Format_RGBA8888);
                for (int y = 0; y < shot.h; ++y) std::memcpy(img.scanLine(y), shot.rgba.data() + size_t(y) * shot.w * 4, size_t(shot.w) * 4);
                cb(std::move(img));
            }
        }
        frames.clear();
        chain_->pollStream(frames);
        if (!frames.empty()) {
            std::function<void(QImage, int)> sink;
            { std::lock_guard l(sinkM_); sink = sink_; }
            for (auto& f : frames) {
                if (!sink) continue;
                QImage img(f.w, f.h, QImage::Format_RGBA8888);
                for (int y = 0; y < f.h; ++y) std::memcpy(img.scanLine(y), f.rgba.data() + size_t(y) * f.w * 4, size_t(f.w) * 4);
                sink(std::move(img), f.delayMs);
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

QString LiveVulkan::lastUpError() const {
    std::lock_guard l(m_);
    return QString::fromStdString(upError_);
}

QString LiveVulkan::lastError() const {
    std::lock_guard l(m_);
    return QString::fromStdString(error_);
}

void LiveVulkan::setDisplayFit(QSize px) {
    if (px.width() == fitW_.load() && px.height() == fitH_.load()) return;
    fitW_ = px.width(); fitH_ = px.height();
    { std::lock_guard l(m_); redraw_ = true; }   // draw the current picture again at the new size
    QMetaObject::invokeMethod(win_, "update", Qt::QueuedConnection);
}

void LiveVulkan::setSettings(const EnhanceSettings& s) {
    {
        std::lock_guard l(m_);
        settings_ = s;
        redraw_ = true;           // re-run the chain on the current frame (matters while paused)
    }
    QMetaObject::invokeMethod(win_, "update", Qt::QueuedConnection);
}

void LiveVulkan::setActive(bool on) {
    {
        std::lock_guard l(m_);
        active_ = on;
        if (!on) {
            mailbox_.captureBuffer.reset();
            current_.captureBuffer.reset();
            hasNew_ = haveCurrent_ = redraw_ = false;
        }
    }
    emit outputChanged();
    QMetaObject::invokeMethod(win_, "update", Qt::QueuedConnection);
}

void LiveVulkan::submit(RawFrame& frame) {
    {
        std::lock_guard l(m_);
        const auto now = std::chrono::steady_clock::now();
        const double dt = std::chrono::duration<double, std::milli>(now - lastArrival_).count();
        lastArrival_ = now;
        if (dt > 4.0 && dt < 100.0) frameMs_ = 0.85 * frameMs_ + 0.15 * dt;
        std::swap(mailbox_, frame);         // the newest frame always wins; an unrendered older one goes back for reuse
        hasNew_ = true;
    }
    QMetaObject::invokeMethod(win_, "update", Qt::QueuedConnection);
}

void LiveVulkan::onBeforeRendering() {
    bool fresh = false;
    EnhanceSettings s;
    RawFrame frame;
    {
        std::lock_guard l(m_);
        if (!active_.load()) return;
        s = settings_;
        if (hasNew_) {
            std::swap(current_, mailbox_);
            mailbox_.captureBuffer.reset();   // the old picture is no longer needed for redraw
            hasNew_ = false;
            haveCurrent_ = true;
            redraw_ = false;
            fresh = true;
        } else if (redraw_ && haveCurrent_) {
            redraw_ = false;
            fresh = true;
        }
        // Keep the pixels alive through the staging upload if the source closes during rendering.
        if (fresh) frame = std::move(current_);
    }
    if (!fresh) {
        // FSR frame generation: the generated frame is on screen; the real one follows half a frame later
        if (chain_->fgShowRealPending() && std::chrono::steady_clock::now() >= realDue_ - std::chrono::microseconds(1500)) {
            win_->beginExternalCommands();
            auto* rif = win_->rendererInterface();
            auto* cb = static_cast<VkCommandBuffer*>(rif->getResource(win_, QSGRendererInterface::CommandListResource));
            if (cb && *cb) chain_->recordShowReal(*cb);
            win_->endExternalCommands();
            noteShown();
        }
        return;
    }

    const auto t0 = std::chrono::steady_clock::now();
    if (frame.reset) { chain_->resetHistory(); frame.reset = false; }   // a video file was seeked
    LiveInput in;
    switch (frame.fmt) {
    case RawFrame::YUYV: in.fmt = LiveInput::YUYV; break;
    case RawFrame::UYVY: in.fmt = LiveInput::UYVY; break;
    case RawFrame::NV12: in.fmt = LiveInput::NV12; break;
    case RawFrame::P010: in.fmt = LiveInput::P010; break;
    default: in.fmt = LiveInput::RGBA; break;
    }
    in.data = frame.bytes();
    in.size = frame.size();
    in.w = frame.w; in.h = frame.h; in.stride = frame.stride;
    const auto captured = frame.captured;
    in.time = captured.time_since_epoch().count() ? captured : std::chrono::steady_clock::now();

    win_->beginExternalCommands();
    auto* rif = win_->rendererInterface();
    auto* cb = static_cast<VkCommandBuffer*>(rif->getResource(win_, QSGRendererInterface::CommandListResource));
    std::string err;
    double frameMs;
    { std::lock_guard l(m_); frameMs = frameMs_; }
    s.fitW = fitW_.load(); s.fitH = fitH_.load();   // the window's pixel size for the picture (VideoItem)
    bool ok = cb && *cb && chain_->recordLive(*cb, win_->graphicsStateInfo().currentFrameSlot, in, s, err, float(frameMs));
    win_->endExternalCommands();
    {
        std::lock_guard l(m_);
        if (!haveCurrent_) return;   // the source closed while rendering
        current_ = std::move(frame);  // retain the current picture for a paused redraw
        if (!ok) {
            error_ = err.empty() ? "Live path unavailable" : err;
            static int warned = 0;
            if (warned++ < 3) qWarning() << "LiveVulkan: record failed:" << QString::fromStdString(error_);
            return;
        }
        error_.clear();
        upError_ = chain_->fgError().empty() ? chain_->upError() : "Frame generation: " + chain_->fgError();
    }
    const ImageRes& real = chain_->liveDisplay();
    const auto tEnd = std::chrono::steady_clock::now();
    lastMs_ = std::chrono::duration<double, std::milli>(tEnd - t0).count();
    // Time inside VeyAmux: from the capture driver's timestamp to the moment the frame's GPU work is queued, plus the
    // measured GPU time - i.e. until the picture is ready for the display (scan-out and the card's own delay excluded).
    {
        std::lock_guard l(m_);
        gpuMs_ = chain_->gpuMs();
        gpuStages_ = chain_->gpuStageMs();
    }
    if (captured.time_since_epoch().count()) {
        const double q = std::chrono::duration<double, std::milli>(tEnd - captured).count();
        if (q >= 0 && q < 1000) {
            const double g = std::max(0.0f, chain_->gpuMs());
            const double v = q + g, o = pipeMs_.load();
            pipeMs_ = o <= 0 ? v : o + 0.1 * (v - o);
        }
    }
    const bool recreated = chain_->liveGeneration() != seenGen_;   // images were recreated (size change)
    seenGen_ = chain_->liveGeneration();
    ++seq_;
    const uint64_t realHandle = uint64_t(reinterpret_cast<uintptr_t>(real.image));
    if (qEnvironmentVariableIsSet("VEYAMUX_DEBUG_FG")) {
        static int real = 0;
        ++real;
        if (real % 10 == 0) qInfo("LiveVulkan: %d frames, GPU %.3f ms, CPU %.2f ms, capture interval %.1f ms, output %ux%u%s%s", real,
                                     double(chain_->gpuMs()), lastMs_.load(), frameMs,
                                     chain_->liveOutput().width, chain_->liveOutput().height,
                                     chain_->upError().empty() ? "" : ", upscaling error: ", chain_->upError().c_str());
    }
    publish(realHandle, int(real.width), int(real.height), recreated);
    if (chain_->fgShowRealPending()) {
        // show the real frame half a capture interval after the generated one
        const int halfUs = int(std::clamp(frameMs * 500.0, 3000.0, 30000.0));
        realDue_ = tEnd + std::chrono::microseconds(halfUs);
        QMetaObject::invokeMethod(win_, [w = win_, ms = std::max(1, halfUs / 1000 - 1)] {
            QTimer::singleShot(ms, Qt::PreciseTimer, w, [w] { w->update(); });
        }, Qt::QueuedConnection);
    }
}

void LiveVulkan::noteShown() {
    ++shown_;
    const double t = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
    std::lock_guard l(timesM_);
    shownTimes_.push_back(t);
    while (shownTimes_.size() > 600) shownTimes_.pop_front();
}

std::vector<double> LiveVulkan::shownTimes() const {
    std::lock_guard l(timesM_);
    return {shownTimes_.begin(), shownTimes_.end()};
}

void LiveVulkan::publish(uint64_t handle, int w, int h, bool recreated) {
    noteShown();
    imgW_ = w;
    imgH_ = h;
    image_ = handle;
    if (handle != lastHandle_.load() || recreated) {
        lastHandle_ = handle;
        gen_ = gen_.load() + 1;                     // VideoItem re-wraps the image on its next sync
        emit outputChanged();
    }
}

