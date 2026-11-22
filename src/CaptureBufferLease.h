#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <utility>

namespace veyamux {

// A dequeued buffer has one owner, while the mapping is shared across frames.
class CaptureBufferLease {
public:
    struct Pool {
        virtual ~Pool() = default;
        virtual void requeue(uint32_t index) noexcept = 0;
    };

    CaptureBufferLease() = default;
    CaptureBufferLease(std::shared_ptr<Pool> pool, uint32_t index) : pool_(std::move(pool)), index_(index) {}
    ~CaptureBufferLease() { reset(); }

    CaptureBufferLease(CaptureBufferLease&&) noexcept = default;
    CaptureBufferLease& operator=(CaptureBufferLease&& other) noexcept {
        if (this != &other) {
            reset();
            pool_ = std::move(other.pool_);
            index_ = other.index_;
            pixels_ = other.pixels_;
        }
        return *this;
    }

    void setPixels(std::span<const uint8_t> pixels) { pixels_ = pixels; }
    std::span<const uint8_t> pixels() const { return pool_ ? pixels_ : std::span<const uint8_t>{}; }

    void reset() noexcept {
        pixels_ = {};
        if (auto pool = std::move(pool_)) pool->requeue(index_);
    }

private:
    std::shared_ptr<Pool> pool_;
    uint32_t index_ = 0;
    std::span<const uint8_t> pixels_;
};

} // namespace veyamux
