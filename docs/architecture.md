# rs-face Architecture

A from-scratch Rust face detection library with **zero runtime dependencies**.
This document maps the source tree to the conceptual modules and explains how
data flows through the system.

## Source tree

```
src/
├── lib.rs              # Library entrypoint — re-exports the public API.
├── main.rs             # CLI: argument parsing + pipeline invocation.
├── integral.rs         # Integral image (regular, squared, rotated).
├── detector.rs         # Multi-scale sliding window + NMS (with spatial-bucket opt).
├── pipeline.rs         # Multi-threaded source → workers → sink pipeline.
├── output.rs           # Hand-rolled PNG writer + JSON manifest writer.
├── haar/               # AdaBoost cascade + Haar features.
│   ├── mod.rs
│   ├── feature.rs      # 5 feature families + custom-rect layout.
│   ├── cascade.rs      # Cascade struct, EvalCache, save/load (.rfcf).
│   └── params.rs       # Bundled demo cascade (smoke-test only).
├── cnn/                # Optional 24×24 CNN detector.
│   └── mod.rs          # Conv→ReLU→Pool→FC→Sigmoid with `_into` scratch API.
├── image/              # 8-bit Gray/RGB types, codec, PNG encoder.
│   ├── mod.rs          # GrayImage / RgbImage + resize (area, bilinear).
│   ├── codec.rs        # PGM/PPM (test/fixture format).
│   └── png.rs          # Zero-dep PNG encoder (for output frames).
├── source/             # Frame source trait + concrete impls.
│   ├── mod.rs
│   ├── image_seq.rs    # PNG/PGM/JPG files in a directory.
│   ├── http.rs         # HTTP(S) image stream.
│   ├── ffmpeg_pipe.rs  # `ffmpeg … -f rawvideo` subprocess.
│   └── synthetic.rs    # Test://N synthetic test pattern.
├── pool/               # Small worker-pool scratch buffer.
│   └── mod.rs
├── gpu/                # Optional OpenCL backend.
│   └── mod.rs          # Squared-integral + variance pre-filter + full cascade on GPU.
└── bin/                # Debug binaries.
    ├── bench_detect.rs # CLI benchmark.
    ├── cnn_train.rs    # Train CNN on synthetic data.
    └── debug_cascade.rs# Per-stage cascade decision trace.
```

## Data flow

```
                ┌────────────────────────────────────────────────────────┐
                │  FrameSource (image_seq / http / ffmpeg / synthetic)  │
                └──────────────────────┬─────────────────────────────────┘
                                       │ next_frame() -> Option<Frame>
                                       ▼
        ┌──────────────────────────────────────────────────────────────┐
        │              Dispatcher (round-robin w/ backpressure)        │
        └─┬───────────────────┬───────────────────┬────────────────────┘
          │                   │                   │
          ▼                   ▼                   ▼
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │ worker 0    │     │ worker 1    │ ... │ worker N-1  │
   │ Detector+   │     │ Detector+   │     │ Detector+   │
   │ EvalCache   │     │ EvalCache   │     │ EvalCache   │
   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
          │ result(seq)       │                   │
          └─────────┬─────────┴───────────────────┘
                    ▼
        ┌────────────────────────────────┐
        │  Sink (in-order reassembly)    │
        │  write_annotated_png(...)      │
        │  write_manifest(...)           │
        └────────────────────────────────┘
```

## Per-frame worker hot path

The single-threaded per-frame hot path inside `Detector::detect` is:

```
for level in pyramid:
    integral, squared = build_integral_images(level)        # ~10ms for 1080p
    rotated = build_rotated_integral(level)                 # ~5ms
    EvalCache.clear()
    for (x, y) in stride x stride:
        if !passes_variance(integral, squared, x, y, ...): continue  # O(1) reject
        cascade.classify(integral, rotated, x, y, cache)   # early-reject stages
    non_max_suppression(all_detections)                     # spatial-bucket opt
```

The two costliest steps are (a) integral image construction (saturates memory
bandwidth) and (b) the per-window feature evaluation inside the cascade.
Variance pre-filtering is critical because it rejects >95% of windows in real
images before the (much more expensive) cascade runs.

## EvalCache

`EvalCache` is a per-worker scratch buffer that de-duplicates feature
responses within a single window. A typical OpenCV cascade has many weak
features referencing the same underlying Haar feature, so evaluating each one
separately would be wasted work. `EvalCache` uses a generation counter
"tombstone" trick to make `clear()` O(1) — see `cascade.rs`.

## GPU backend

`gpu/mod.rs` is best-effort: it tries to `dlopen` `libOpenCL.so` at runtime.
If unavailable, the detector silently falls back to CPU. Three GPU paths exist:

1. `compute_integral_dual` — regular + squared integral in one pass.
2. `variance_prefilter` — one work-item per (x, y) window, O(1) variance test.
3. `detect_windows` — full cascade on the GPU (only worth it at `stride == 1`).

The GPU fast-path is only invoked for images ≥ 500×500 — below that the
kernel launch + PCIe transfer overhead beats the CPU path.

## Why zero deps?

The project's stated constraint is that the library links only against libc
and libm. Every codec, container parser, JSON writer, and image resize routine
is hand-rolled. This is documented in `README.md` and is a hard requirement
on PR review.

## Honesty

This project is an exercise in classical CV. Real-face detection works
within the constraints documented in the README, but the bundled CNN
weights are placeholders and the rotated Haar features evaluate to 0
(the rotated integral's Rust implementation is non-trivial to keep
correct under the borrow checker; we still use it for the few rotated
features in OpenCV's frontalface cascade, just with 0 contribution).
