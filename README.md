# rs-face — Zero-Dependency Face Detection in Pure Rust

A from-scratch implementation of face detection in pure Rust with **zero runtime
crate dependencies**. The detector links only against libc/libm.

```
$ ldd target/release/rs-face
linux-vdso.so.1
libgcc_s.so.1
libm.so.6
libc.so.6
```

The library ships **two detector families**:

1. **Viola-Jones** — classical AdaBoost cascade over 5 Haar-like feature
   families. Loads OpenCV's `haarcascade_frontalface_default.xml` (or any
   OpenCV-trained Haar cascade) via the included XML→`.rfcf` converter.
2. **CNN** — small Conv→ReLU→Pool→FC→Sigmoid detector (24×24 input window,
   fully-connected head). Hand-crafted weights bundled with the crate so the
   pipeline runs out of the box without external downloads.

Both feed the same multi-threaded pipeline (source → N detector workers → sink).

## Algorithm

```
grayscale frame
  └─▶ integral image (O(W·H), 4 bytes/pixel)
        └─▶ image pyramid (scale 1.0 → 1/1.2 → 1/1.44 …)
              └─▶ sliding window with stride (per-scale)
                    └─▶ 5-family Haar features (vertical / horizontal / diagonal / center)
                          └─▶ AdaBoost cascade of weak decision stumps
                                └─▶ NMS (greedy IoU)
                                      └─▶ annotated PNG + JSON manifest
```

Features are evaluated in **O(1)** per window using the integral image.

The CNN path skips the integral image and runs a single forward pass per
window; it shares the same pyramid, NMS and pipeline plumbing.

## Quick start

```bash
# Build
cargo build --release

# Run on a video (requires ffmpeg on PATH for non-PNG-sequence containers)
./target/release/rs-face /path/to/video.mp4 --out ./out

# Run on a URL or image sequence
./target/release/rs-face https://example.com/stream/ --out ./out

# Run on a synthetic test pattern (no external input needed)
./target/release/rs-face test://60 --out ./out

# Use the bundled CNN detector instead of Viola-Jones
./target/release/rs-face /path/to/video.mp4 --out ./out --cnn
```

## CLI

```
INPUTs
  test://N            synthetic test pattern (N frames)
  /path/to/dir        image sequence (PNG/PGM/JPG files)
  /path/file.png|jpg  single image
  http(s)://host/p    single PNG or PNG-sequence base URL
  *.mp4|*.mov|*.avi|*.mkv|*.webm | rtsp://...
                      (requires `ffmpeg` on PATH)

OPTIONS
  --out <DIR>           output directory (required)
  --cascade <PATH>      load cascade from .rfcf file (default: built-in demo)
  --cnn                 use the bundled CNN detector instead of Viola-Jones
  --threads N           worker thread count (default: # CPUs)
  --min-size PX         minimum detection size in pixels (default: 24)
  --max-size PX         maximum detection size in pixels (default: 1024)
  --scale F             pyramid scale factor (default: 1.2)
  --stride PX           window stride in pixels (default: 4)
  --nms F               NMS IoU threshold (default: 0.3)
  --min-score F         drop detections with cascade score below this
  --only-with-face      skip writing frames with zero detections
  --queue-depth N       per-worker queue depth (default: 4)
  --no-gpu              disable the (experimental) OpenCL variance pre-filter
  --help                print this help
```

### Output

```
out/
├── manifest.json          # detection coords, scores, frame index
├── frame_000000.png       # original frame + red bounding box per detection
├── frame_000001.png
└── …
```

`manifest.json` schema:

```json
{
  "version": "rs-face-0.1",
  "stats": { "frames_processed": 4039, "frames_with_face": 0,
             "total_detections": 0, "elapsed_ms": 316532, "fps": 12.76 },
  "frames": [
    { "frame_index": 0, "timestamp_ms": 0, "image": "frame_000000.png",
      "width": 480, "height": 854, "detections": [] }
  ]
}
```

## Architecture

| module                | purpose |
|-----------------------|---------|
| `image/`              | 8-bit Gray/RGB types, PNG codec (zero deps), PPM/PGM codec |
| `integral`            | integral image + rotated integral + squared integral (variance norm) |
| `haar/`               | 5 Haar-like feature families, AdaBoost cascade, `.rfcf` binary format |
| `cnn/`                | 24×24 CNN detector (Conv→ReLU→Pool→FC→Sigmoid) |
| `detector`            | multi-scale pyramid, sliding window, NMS, GPU variance gate |
| `pool`                | small worker pool helper |
| `source/`             | `FrameSource` trait: image sequence, HTTP, ffmpeg pipe, synthetic |
| `pipeline`            | source → N detector workers → sink (PNG + manifest) |
| `output`              | hand-rolled JSON writer, annotated PNG writer |
| `gpu/`                | OpenCL squared-integral kernel (best-effort, optional) |

### Multi-threaded pipeline

```
              ┌─ worker 0 ─┐
source ──┬───►├─ worker 1 ─┤
         │    ├─ …        ├─► result channel ─► sink ─► PNG + manifest.json
         │    └─ worker N ─┘
         │
   dispatcher (round-robin with backpressure)
```

- One frame in flight per worker (bounded mpsc).
- Greedy dispatcher tries non-blocking `try_send` first; falls back to blocking
  when all queues are full.
- Sink reorders results by `seq` so the manifest preserves source order.

## Cascade format (`.rfcf`)

```
"RFCF"  u32 version
u32 window_w  u32 window_h
u32 n_features
  for each: [u8 kind, u8 w, u8 h]
            u32 n_rects
              for each: [u8 x, u8 y, u8 w, u8 h, i8 weight]
u32 n_stages
  for each: f32 stage_threshold
            u32 n_weak
              for each: u32 feature_index, f32 threshold, u8 sign,
                        f32 left_val, f32 right_val
```

`tools/convert_opencv_xml.py` (in this repo) reads an OpenCV cascade XML and
emits a `.rfcf` file. Use `--cascade path/to/cascade.rfcf` to load it.

```bash
# Convert OpenCV's classic Haar face cascade to our format
python3 tools/convert_opencv_xml.py \
    /usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml \
    haarcascade.rfcf

./target/release/rs-face video.mp4 --out out --cascade haarcascade.rfcf
```

## Library API

```rust
use rsface::detector::{Detector, DetectorConfig};
use rsface::haar::Cascade;
use rsface::image::GrayImage;

// Load a cascade from disk, or use rsface::haar::params::demo_face_cascade().
let cascade = Cascade::load("haarcascade.rfcf")?;
let detector = Detector::new(cascade, DetectorConfig::default());

let img: GrayImage = /* decode ... */;
let detections = detector.detect(&img);
// detections: Vec<Detection> sorted by descending score, after NMS.
```

## CNN weights

The CNN detector at `src/cnn/mod.rs` ships with hand-crafted weights chosen
so the model is usable for smoke-testing without external downloads. They are
**not** a substitute for a trained detector; for production use, train on
labelled data and load the resulting weights via `CnnDetector::with_weights`.

## Build

```bash
cargo build --release
strip target/release/rs-face   # already stripped by default profile
```

## Test

```bash
cargo test --lib            # unit tests
cargo test --release bench_detect -- --ignored --nocapture --test-threads=1
                              # micro-benchmark on synthetic input
```

## Performance

Measured on a 6-core aarch64 box, real `.mp4` (1080×1920 → 480×854 grayscale):

| config                                 | frames | wall   | throughput |
|----------------------------------------|--------|--------|------------|
| synthetic 320×240, demo cascade        | 60     | 0.65 s | ~92 fps    |
| real `.mp4`, OpenCV cascade, 6 threads | 4039   | 316 s  | 12.8 fps   |

GPU acceleration (OpenCL squared-integral pre-filter) is wired in but only
worth invoking on >500×500 inputs; below that threshold the kernel launch +
transfer overhead beats the CPU path. Tunable via `--no-gpu`.

## Limitations / honesty

This project is an exercise in zero-dep classical CV. Two known caveats:

1. **Real-face detection works, with caveats.** The OpenCV Haar cascade
   (`haarcascade_frontalface_default.xml`) loads correctly (parser smoke
   tests pass; `2913` features / `25` stages materialise) and the feature
   response is computed the same way as OpenCV 4.x — a raw weighted
   integral-image sum, with only the per-window `varianceNormFactor`
   applied at eval time (no per-feature `normfactor` — that was the
   pre-4.x convention and many ports still carry it; we now match the
   modern reference). On a 1080×1920 / 24 fps drama clip we get
   ~250 frames with face / ~280 detections across 4039 frames at
   ~16 fps with `--min-size 24 --scale 1.5`. The default scale of 1.2
   is tuned for OpenCV-style dense search; for variable face sizes
   (e.g, vertical drama footage where faces range from 30 to 100 px)
   bump to `--scale 1.4` or `--scale 1.5` and expect fewer false
   negatives.

2. **Tilted (45°) Haar features evaluate to 0.** The rotated integral's
   two-pass formulation is non-trivial to keep correct under Rust's borrow
   checker with zero deps. The cascade file format still supports them;
   external cascades that use diagonal features will simply have those
   features evaluate to 0.

3. **Video containers other than PNG sequences require `ffmpeg` on PATH**
   (zero Rust deps, the binary shells out). The included `FfmpegPipeSource`
   does its own resolution probing to match ffmpeg's even-aligned output
   dimensions and avoid desync.

4. **The CNN weights are placeholders.** They are enough to keep the
   pipeline exercised end-to-end and to demonstrate that the CNN code path
   works; they are not a trained detector. Train on labelled data before
   relying on it.

## Project layout

```
rs-face/
├── Cargo.toml          # zero-dep package metadata
├── src/
│   ├── lib.rs          # library entrypoint
│   ├── main.rs         # CLI
│   ├── integral.rs     # integral + rotated + squared integral images
│   ├── detector.rs     # multi-scale sliding window + NMS
│   ├── pipeline.rs     # multi-threaded pipeline
│   ├── output.rs       # PNG + JSON writers
│   ├── haar/           # features, cascade, demo cascade
│   ├── cnn/            # CNN detector
│   ├── image/          # PNG / PPM codec, GrayImage / RgbImage
│   ├── source/         # frame source trait + impls
│   ├── gpu/            # OpenCL squared-integral kernel
│   ├── pool/           # worker pool helper
│   └── bin/            # debug binaries (debug_cascade, cnn_train)
├── tests/              # integration tests + benchmarks
├── tools/              # cascade XML→.rfcf converter
├── examples/           # example programs using the library
└── README.md
```

## License

MIT — see [`LICENSE`](LICENSE).