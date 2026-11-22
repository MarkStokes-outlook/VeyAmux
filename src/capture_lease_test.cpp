#include "CaptureSource.h"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <deque>

using veyamux::CaptureBufferLease;
using veyamux::RawFrame;

namespace {
struct TestPool : CaptureBufferLease::Pool {
    bool live = true;
    std::array<unsigned, 6> releases{};
    std::array<unsigned, 6> requeues{};
    std::array<uint8_t, 6> pixels{10, 20, 30, 40, 50, 60};

    void requeue(uint32_t index) noexcept override {
        ++releases[index];
        if (live) ++requeues[index];
    }
};

void check(bool condition, const char* message) {
    if (condition) return;
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
}

RawFrame borrow(const std::shared_ptr<TestPool>& pool, uint32_t index) {
    RawFrame frame;
    frame.captureBuffer = CaptureBufferLease{pool, index};
    frame.captureBuffer.setPixels({&pool->pixels[index], 1});
    return frame;
}

void testPresentation() {
    auto pool = std::make_shared<TestPool>();
    RawFrame mailbox, current;
    RawFrame capture = borrow(pool, 0);
    std::swap(capture, mailbox);
    check(pool->releases[0] == 0, "mailbox released its new picture before upload");

    capture = borrow(pool, 1);
    std::swap(capture, mailbox);
    capture.captureBuffer.reset();
    check(pool->requeues[0] == 1 && pool->releases[1] == 0, "replacing the mailbox must return only the dropped picture");

    std::swap(current, mailbox);
    RawFrame rendering = std::move(current);
    check(rendering.size() == 1 && *rendering.bytes() == 20, "pixels must follow the lease to the renderer");
    check(current.size() == 0, "a moved-from frame cannot expose borrowed pixels");
    current = std::move(rendering);
    check(pool->releases[1] == 0 && *current.bytes() == 20, "the paused picture must still be readable for redraw");

    mailbox = borrow(pool, 2);
    std::swap(current, mailbox);
    mailbox.captureBuffer.reset();
    check(pool->requeues[1] == 1 && pool->releases[2] == 0, "presenting a new picture must return the previous buffer once");
    current.captureBuffer.reset();
    current.captureBuffer.reset();
    check(pool->requeues[2] == 1, "repeated resets must return the current buffer only once");
}

void testPacingAndFallback() {
    auto pool = std::make_shared<TestPool>();
    std::deque<RawFrame> paced;
    paced.push_back(borrow(pool, 0));
    paced.push_back(borrow(pool, 1));
    check(pool->releases[0] == 0 && pool->releases[1] == 0, "frames must stay leased while queued for pacing");
    paced.pop_front();
    RawFrame delivered = std::move(paced.front());
    paced.pop_front();
    check(pool->requeues[0] == 1 && pool->releases[1] == 0, "dropping a paced frame must return just that buffer");
    delivered = {};
    check(pool->requeues[1] == 1, "recycling a paced frame must return its buffer");

    RawFrame copied;
    {
        CaptureBufferLease lease{pool, 2};
        lease = CaptureBufferLease{pool, 3};
        check(pool->requeues[2] == 1, "dequeueing a newer picture must release the previous lease");
        copied.data.assign(1, pool->pixels[3]);
    }
    check(pool->requeues[3] == 1 && copied.size() == 1 && *copied.bytes() == 40, "copy fallback must return the buffer and retain its pixels");
}

void testCloseDuringRender() {
    auto oldPool = std::make_shared<TestPool>();
    auto newPool = std::make_shared<TestPool>();
    std::weak_ptr<TestPool> oldMapping = oldPool;
    RawFrame rendering = borrow(oldPool, 0);
    oldPool->live = false;
    oldPool.reset();
    check(!oldMapping.expired() && *rendering.bytes() == 10, "closing the pool must preserve pixels that are still being uploaded");
    {
        const auto previous = oldMapping.lock();
        rendering.captureBuffer.reset();
        check(previous->releases[0] == 1 && previous->requeues[0] == 0, "a frame from a closed pool must skip requeueing");
    }
    check(oldMapping.expired(), "releasing the last lease must release the old mapping");
    rendering = borrow(newPool, 0);
    rendering.captureBuffer.reset();
    check(newPool->requeues[0] == 1, "the reopened pool must only receive its own buffers");
}
} // namespace

int main() {
    testPresentation();
    testPacingAndFallback();
    testCloseDuringRender();
    return 0;
}
