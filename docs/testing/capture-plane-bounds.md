# Issue #2: capture plane bounds

Branch: `fix/capture-plane-bounds`, based on issue #1's
`fix/v4l2-buffer-lease` at `56af381`.

The shared `FrameLayout` descriptor validates positive dimensions and strides,
subsampled dimensions, plane offsets, active row spans, and bounded arithmetic.
Capture checks payload size against the V4L2 mapping before probing or conversion.
The Vulkan upload validates again and packs active rows into tightly packed staging;
its copy commands use the packed stride. Padding in a driver buffer therefore
cannot cause a truncated upload or an out-of-bounds GPU copy. MJPEG decoding uses
an FFmpeg-owned packet with zero padding and rejects decoded dimensions that do
not match the negotiated capture mode.

## Automated checks

`capture-frame-layout` covers RGBA, RGB24, packed 4:2:2, NV12, P010 and I420:
tight and padded rows (including padding larger than the staging allocation),
missing final-row padding, truncated planes, payloads exceeding mappings, short
staging, invalid dimensions/strides, and integer bounds. Rejected copies must not
modify staging. The existing `capture-buffer-lease` test remains registered.

Run the dependency-free layout test locally:

```sh
g++ -std=c++20 -O2 -Wall -Wextra -Werror src/frame_layout_test.cpp -o /tmp/veyamux-frame-layout-test
/tmp/veyamux-frame-layout-test
```

The AppImage workflow runs both CTest tests and runs the layout test under
AddressSanitizer/UndefinedBehaviorSanitizer.

## Still required before closing the issue

- Full Linux build and CI execution for this branch.
- Exercise tight and padded raw uploads with Vulkan validation enabled; confirm
  no copy-range, row-length or offset diagnostics for RGBA, YUYV/UYVY, NV12 and P010.
- Exercise valid, truncated and corrupt MJPEG input through the actual decoder,
  including repeated decoder initialization and mismatched decoded dimensions.
- Repeat the physical capture lifecycle checks from issue #1. Its hardware
  verification remains pending; this branch does not mark issue #1 complete.

Local validation so far: optimized layout and buffer-lease tests pass. The local
machine lacks FFmpeg/Vulkan development packages and the ASan runtime, so neither
a full build nor sanitizer/Vulkan validation has been claimed locally.
