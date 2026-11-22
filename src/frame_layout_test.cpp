#include "FrameLayout.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>
using namespace veyamux;
static void require(bool condition, const char* message) {
    if (!condition) { std::cerr << message << '\n'; std::exit(1); }
}
int main() {
    for (auto format : {PixelLayout::RGBA, PixelLayout::RGB24, PixelLayout::Packed422,
                        PixelLayout::NV12, PixelLayout::P010, PixelLayout::I420}) {
        const int bpp = format == PixelLayout::RGBA ? 4 : format == PixelLayout::RGB24 ? 3 :
                        (format == PixelLayout::P010 || format == PixelLayout::Packed422) ? 2 : 1;
        for (int padding : {0, 2, 18, 1024}) {
            const int stride = 8 * bpp + padding;
            FrameLayout layout;
            require(describeFrame(format, 8, 6, stride, layout), "valid layout rejected");
            std::vector<uint8_t> input(layout.requiredBytes, 0xee), expected;
            for (size_t p = 0; p < layout.count; ++p) {
                auto plane = layout.planes[p];
                for (size_t y = 0; y < plane.rows; ++y)
                    for (size_t x = 0; x < plane.rowBytes; ++x) {
                        const uint8_t value = uint8_t(1 + p * 50 + y * 3 + x);
                        input[plane.offset + y * plane.stride + x] = value;
                        expected.push_back(value);
                    }
            }
            std::vector<uint8_t> output(expected.size() + 16, 0xa5);
            require(packFrame(format, 8, 6, stride, input.data(), input.size(), output.data(), expected.size()), "repack failed");
            require(std::equal(expected.begin(), expected.end(), output.begin()), "plane contents changed");
            require(std::all_of(output.begin() + expected.size(), output.end(), [](auto v) { return v == 0xa5; }), "staging overrun");
            std::fill(output.begin(), output.end(), 0xa5);
            require(!packFrame(format, 8, 6, stride, input.data(), input.size() - 1, output.data(), expected.size()), "short source accepted");
            require(!packFrame(format, 8, 6, stride, input.data(), input.size(), output.data(), expected.size() - 1), "short staging accepted");
            require(std::all_of(output.begin(), output.end(), [](auto v) { return v == 0xa5; }), "rejected copy modified staging");
            FrameLayout checked;
            require(!validateFrame(format, 8, 6, stride, input.size(), input.size() - 1, checked), "payload exceeding mapping accepted");
            for (size_t p = 0; p < layout.count; ++p)
                require(!validateFrame(format, 8, 6, stride, layout.planes[p].offset, input.size(), checked), "missing plane accepted");
        }
    }
    FrameLayout layout;
    require(!describeFrame(PixelLayout::RGBA, 0, 6, 32, layout), "zero width accepted");
    require(!describeFrame(PixelLayout::RGBA, 8, -1, 32, layout), "negative height accepted");
    require(!describeFrame(PixelLayout::RGBA, 8, 6, -32, layout), "negative stride accepted");
    require(!describeFrame(PixelLayout::RGBA, 8, 6, 31, layout), "short stride accepted");
    require(!describeFrame(PixelLayout::NV12, 7, 6, 8, layout), "odd chroma width accepted");
    require(!describeFrame(PixelLayout::P010, 8, 5, 16, layout), "odd chroma height accepted");
    require(!describeFrame(PixelLayout::I420, 8, 6, 9, layout), "fractional chroma stride accepted");
    require(!describeFrame(PixelLayout::RGBA, 0x7fffffff, 6, 32, layout), "overflowing row accepted");
    require(!describeFrame(PixelLayout::NV12, 8, 0x7ffffffe, 0x7fffffff, layout), "overflowing plane accepted");
    require(describeFrame(PixelLayout::NV12, 8, 6, 10, layout), "NV12 layout rejected");
    require(layout.planes[1].offset == 60 && layout.requiredBytes == 88 && layout.packedBytes == 72, "NV12 offsets incorrect");
    require(describeFrame(PixelLayout::I420, 8, 6, 10, layout), "I420 layout rejected");
    require(layout.planes[2].offset == 75 && layout.requiredBytes == 89 && layout.packedBytes == 72, "I420 offsets incorrect");
    std::cout << "Capture plane validation and staging repack tests passed\n";
}
