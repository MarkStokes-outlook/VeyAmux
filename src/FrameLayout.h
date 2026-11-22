#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace veyamux {
// Contiguous, positive-stride capture formats. Plane offsets include row padding;
// requiredBytes ends at the last active byte, not unused final-row padding.
enum class PixelLayout { RGBA, RGB24, Packed422, NV12, P010, I420 };
struct FramePlane { size_t offset = 0, stride = 0, rowBytes = 0, rows = 0; };
struct FrameLayout {
    std::array<FramePlane, 3> planes{};
    size_t count = 0, requiredBytes = 0, packedBytes = 0;
};
inline bool describeFrame(PixelLayout format, int w, int h, int stride, FrameLayout& out) {
    out = {};
    if (w <= 0 || h <= 0 || stride <= 0) return false;
    const bool planar = format == PixelLayout::NV12 || format == PixelLayout::P010 || format == PixelLayout::I420;
    if ((planar || format == PixelLayout::Packed422) && (w & 1)) return false;
    if (planar && (h & 1)) return false;
    if (format == PixelLayout::I420 && (stride & 1)) return false;
    size_t bytesPerPixel = 0;
    switch (format) {
    case PixelLayout::RGBA: bytesPerPixel = 4; break;
    case PixelLayout::RGB24: bytesPerPixel = 3; break;
    case PixelLayout::Packed422: case PixelLayout::P010: bytesPerPixel = 2; break;
    case PixelLayout::NV12: case PixelLayout::I420: bytesPerPixel = 1; break;
    default: return false;
    }
    // Bound sizes to the signed byte/stride limits used by our FFmpeg consumers.
    constexpr size_t limit = std::numeric_limits<int>::max();
    if (size_t(w) > limit / bytesPerPixel) return false;
    auto add = [&](size_t pitch, size_t row, size_t rows) {
        if (pitch < row || pitch > limit / rows) return false;
        const size_t offset = out.count ? out.planes[out.count - 1].offset +
            out.planes[out.count - 1].stride * out.planes[out.count - 1].rows : 0;
        if (offset > limit - pitch * rows || out.packedBytes > limit - row * rows) return false;
        out.planes[out.count++] = {offset, pitch, row, rows};
        out.requiredBytes = offset + pitch * (rows - 1) + row;
        out.packedBytes += row * rows;
        return true;
    };
    if (!add(size_t(stride), size_t(w) * bytesPerPixel, size_t(h))) return false;
    if (format == PixelLayout::NV12 || format == PixelLayout::P010)
        return add(size_t(stride), size_t(w) * bytesPerPixel, size_t(h / 2));
    if (format == PixelLayout::I420)
        return add(size_t(stride / 2), size_t(w / 2), size_t(h / 2)) &&
               add(size_t(stride / 2), size_t(w / 2), size_t(h / 2));
    return true;
}
inline bool validateFrame(PixelLayout format, int w, int h, int stride,
                          size_t payload, size_t mapping, FrameLayout& out) {
    return payload <= mapping && describeFrame(format, w, h, stride, out) && out.requiredBytes <= payload;
}
// Validate again at the copy boundary. Tight rows make Vulkan offsets independent
// of driver padding and fit the existing four-bytes-per-pixel staging allocation.
inline bool packFrame(PixelLayout format, int w, int h, int stride, const uint8_t* src,
                      size_t size, void* dst, size_t capacity) {
    FrameLayout layout;
    if (!src || !dst || !validateFrame(format, w, h, stride, size, size, layout) || layout.packedBytes > capacity) return false;
    auto* target = static_cast<uint8_t*>(dst);
    for (size_t p = 0; p < layout.count; ++p) {
        const auto& plane = layout.planes[p];
        for (size_t y = 0; y < plane.rows; ++y) {
            std::memcpy(target, src + plane.offset + y * plane.stride, plane.rowBytes);
            target += plane.rowBytes;
        }
    }
    return true;
}
} // namespace veyamux
