# rs-face — The face-library swiss army knife

A from-scratch face detection **and recognition** library in pure Rust.
**Zero runtime dependencies by default**; opt into Metal / CUDA / ONNX
Runtime when you need industrial accuracy.

```
$ ldd target/release/rs-face
linux-vdso.so.1
libgcc_s.so.1
libm.so.6
libc.so.6
```

> One crate, six detectors, three zero-dep recognisers, one ONNX path to
> industrial accuracy. Every algorithm implements the same `FaceDetector`
> trait, so swapping implementations is a one-line change.

## What's in the box

| detector | what it is | zero-dep? | measured here |
|---|---|:-:|---|
| **haar** | Viola–Jones AdaBoost cascade, 5 Haar feature families, OpenCV XML → `.rfcf` | ✅ | real-face drama footage |
| **luminance** | band-pattern + mirror-symmetry detector | ✅ | real-face drama footage |
| **cnn** / **hog** / **yunet** / **mtcnn** | correct architectures + NMS, placeholder weights | ✅ (scaffold) | n/a — drop in real weights via `*_with_*` |
| **scrfd** | SCRFD-10G via ONNX Runtime / tract | opt-in | measured, WIDER-FACE AP 0.95/0.94/0.83 |

| recogniser | what it is | zero-dep? | measured here |
|---|---|:-:|---|
| **lbph** | uniform LBP histograms + chi-square, zero deps, no weights; incremental enrolment, atomic on-disk gallery persistence (`save`/`load`, crops not required) | ✅ | **62/68** hard-gallery LOO rank-1, 33/33 easy |
| **fisherface** | Fisherfaces/LDA — n−C PCA reduction then C−1 class-discriminant axes, pure `std` | ✅ | **59/68** hard-gallery LOO rank-1, **best zero-dep EER ≈ 12.8 %**, 33/33 easy |
| **eigenface** | PCA / Turk-Pentland, Jacobi eigendecomp in pure `std` | ✅ | **58/68** hard-gallery strict LOO (PCA ceiling), 33/33 easy |
| **arcface** | ArcFace R50 / MobileFaceNet via ONNX Runtime / tract | opt-in | cosine margin measured on real faces |

## 5-minute start

```bash
# 1. Build — zero deps by default
cargo build --release

# 2. Smoke-test (no external input, no ffmpeg needed)
./target/release/rs-face test://60 --out ./out

# 3. Run on a video / image sequence / URL
./target/release/rs-face /path/to/video.mp4 --out ./out
./target/release/rs-face https://example.com/stream/ --out ./out

# 4. Pick a different algorithm — same flags, swap behind the scenes
./target/release/rs-face video.mp4 --out ./out --algo luminance

# 5. Discover what's compiled in
./target/release/rs-face --list-algos
./target/release/rs-face --list-features
```

### Run as a service (Docker)

For end-to-end Web UI + REST + SSE + S3 + Postgres — the **canonical** way to
deploy the platform side of `rs-face` (see [`CLAUDE.md`](CLAUDE.md)):

```bash
docker compose -f platform/docker-compose.yml up -d --build
# or:  make docker-up
```

| endpoint | URL |
|---|---|
| Web / REST / SSE | <http://localhost:20080/> |
| S3 (rustfs) | <http://localhost:19000/> |
| S3 console | <http://localhost:19001/> |
| PostgreSQL | `localhost:15432` (rsface / rsface) |

Data is bind-mounted into `data/{rustfs,pg/pgdata,media}/` next to the repo —
visible, greppable, rsync-friendly. Full ops guide: [`platform/DOCKER.md`](platform/DOCKER.md)
(deploy / start / verify / e2e / backup / restore / troubleshoot / cleanup).
Quick targets: `make docker-up`, `docker-down`, `docker-logs`, `docker-test`,
`docker-restore-pg`, `docker-clean`.

If you need a real cascade (OpenCV's `haarcascade_frontalface_default.xml`):
```bash
python3 tools/convert_opencv_xml.py haarcascade_frontalface_default.xml haarcascade.rfcf
./target/release/rs-face video.mp4 --out ./out --cascade haarcascade.rfcf
```

If you need industrial accuracy (SCRFD + ArcFace):
```bash
cargo build --release --features ort-backend      # needs libonnxruntime on host
tools/fetch_models.sh                              # downloads pinned ONNX models
./target/release/rs-face video.mp4 --out ./out --algo scrfd
```

## Pick the right algorithm — "what to use when"

| situation | pick |
|---|---|
| No external input, smoke-test the pipeline | `--algo haar` (default) or `test://60` |
| Frontal portrait, controlled lighting, no extra deps | `--algo haar` with an OpenXML-converted `.rfcf` cascade |
| Variable face sizes in drama / Reels / vertical video | `--algo haar --scale 1.4 --stride 3 --only-with-face` |
| Need a real accuracy on unconstrained faces | `--features ort-backend --algo scrfd` |
| Recognise identities with no downloads | use `rsface::lbph::LbphRecognizer` (incremental enrolment) or the train-once `rsface::eigenface::EigenfaceRecognizer` / `rsface::fisherface::FisherfaceRecognizer` |
| Lowest verification EER with zero deps | `rsface::fisherface::FisherfaceRecognizer` (EER ≈ 12.8 % on the hard drama gallery; rerun `bench_fisherface` on your own data) |
| Need verification under pose / lighting drift | `--features ort-backend` + `rsface::arcface_recognizer::ArcFaceRecognizer` |

## Cookbook — same `Box<dyn FaceDetector>` for every algorithm

```rust
use rsface::face_detector::{FaceDetector, HaarDetector};
use rsface::hog_face::{HogConfig, HogFaceDetector};

let haar: Box<dyn FaceDetector> = Box::new(HaarDetector::new(
    rsface::haar::params::demo_face_cascade(),
    Default::default(),
));
let hog: Box<dyn FaceDetector> =
    Box::new(HogFaceDetector::new(HogConfig::default()));

// Dispatch any number of algorithms through one trait.
for det in &[haar, hog] {
    let hits = det.detect(&gray_frame);
    println!("[{}] {} hits", det.name(), hits.len());
}
```

Full SDK examples: `cargo run --example` (see [Examples](#examples)).

## Documentation

Start with [`docs/INDEX.md`](docs/INDEX.md) — the entry point to every
design / accuracy / operations document in the repo.

- [Algorithm matrix](docs/algorithms.md) — per-algorithm details, weights requirements, known limits.
- [Architecture](docs/architecture.md) — crate map + multi-threaded pipeline plumbing.
- [Format reference](docs/format.md) — `.rfcf` cascade binary format, manifest JSON schema.
- [GPU backends](docs/GPU_BACKENDS.md) — `cpu` / `metal` / `cuda` / `rocm` / `mlu` / `ascend`.
- [Recognition LBPH](docs/recognition-lbph.md) / [Recognition Eigenface](docs/recognition-eigenface.md) / [Recognition Fisherface](docs/recognition-fisherface.md) / [Gallery persistence](docs/gallery-persistence.md) (the `RSLB` binary format behind `LbphRecognizer::save`/`load`).
- [Benchmarks](docs/benchmarks.md) — reproducible scripts.

## Algorithm (Viola-Jones path)

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

## Accuracy

Real measured numbers on the workloads available in this repo. These are the only
accuracy claims in this README — WIDER FACE / LFW figures for the unseen models come
from the upstream papers and are cited as such.

| detector                | source              | WIDER FACE AP (easy/med/hard) | measured here   |
|-------------------------|--------------------|------------------------------|-----------------|
| **SCRFD-10G** (InsightFace, `det_10g.onnx` from buffalo_l) | https://github.com/deepinsight/insightface/tree/master/model_zoo | **0.954 / 0.940 / 0.828** | 1 face on `lena.ppm` @ score **0.797**; 2 faces on `two-people.ppm` @ 0.859 / 0.843, both with keypoints |
| YuNet 2023mar (OpenCV Zoo, Apache-2.0) | https://github.com/opencv/opencv_zoo | 0.887 / 0.871 / 0.768 | not in this fixture set; available via `tools/fetch_models.sh` |
| Viola-Jones (OpenCV `haarcascade_frontalface_default.xml`) | bundled | n/a (no WIDER FACE score) | not measured here |

Recognition backbones we wire up:

| recognizer              | source              | LFW / CFP-FP / AgeDB-30 / IJB-C TAR@1e-4 | measured here |
|-------------------------|--------------------|------------------------------------------|----------------|
| **ArcFace R50** (w600k_r50, `buffalo_l`) | InsightFace model zoo | **99.83 / 99.33 / 98.23 / 97.25** | cosine(lena, lena) = **1.0000**, cosine(lena, biden) = **0.0685**, cosine(person A, person B) = **-0.0256**, scale invariance 0.9733 |
| ArcFace MobileFaceNet (w600k_mbf, `buffalo_s`) | InsightFace model zoo | 99.70 / 98.00 / 96.58 / 95.02 | not in this fixture set; available via `tools/fetch_models.sh` |
| **LBPH (zero-dep, in the default build)** | `src/lbph.rs` — no weights | n/a | **62/68 = 91.2 % LOO rank-1** on 77 ArcFace-labelled drama crops (21 identities; 33/33 on the easier 8-identity no-regression gallery); 96.5 % pair accuracy at the conservative low-FAR default distance 16.7 (FAR 0.3 %, FRR 49 %); EER ≈ 20 % — given a correct detector box. 6×6 grid chosen by a 9-point sweep over OpenCV's 8×8 default. Full methodology and the detector caveat: [`docs/recognition-lbph.md`](docs/recognition-lbph.md) |
| **Eigenfaces/PCA (zero-dep, in the default build)** | `src/eigenface.rs` — no weights, Jacobi eigendecomposition in pure `std` (Turk–Pentland 1991) | n/a | **58/68 = 85.3 % rank-1** under strict per-probe LOO retraining on the same 77 crops (21 identities) — the measured PCA ceiling across a 16-point crop-size/energy sweep, which the shipped defaults already reach; 95.1 % pair accuracy at the conservative low-FAR default 6.3 (FAR 1.5 %, FRR 52 %); 33/33 on the easy gallery. Methodology and caveats: [`docs/recognition-eigenface.md`](docs/recognition-eigenface.md) |
| **Fisherfaces/LDA (zero-dep, in the default build)** | `src/fisherface.rs` — no weights, Gram-trick PCA reduction to n−C directions then ≤ C−1 Fisher axes via `S_W^{−1/2} S_B S_W^{−1/2}` (Belhumeur–Hespanha–Kriegman 1997) | n/a | **59/68 = 86.8 % rank-1** under strict per-probe LOO retraining — one probe above the PCA ceiling, three below LBPH; the **best zero-dep pair EER ≈ 12.8 %** (vs 14.0 % PCA, 22.5 % LBPH); 96.4 % pair accuracy at the conservative low-FAR default 3.0 (FAR 0.4 %, FRR 48 %); 33/33 and FAR 0 % on the easy gallery; descriptor is only ≤ 20 f32 but the model needs retraining when identities change. Methodology and caveats: [`docs/recognition-fisherface.md`](docs/recognition-fisherface.md) |

> **Detector combinations that detect nothing.** Of the algorithms the README used to
> list as first-class (`cnn`, `mtcnn`, `yunet`, `hog`), three — `mtcnn`, `yunet`, `hog` —
> ship RANDOM placeholder weights and return zero detections for every frame. They now
> report `Maturity::Scaffold` and are visibly labelled `[SCAFFOLD — detects nothing]` in
> their descriptions. The hand-crafted `cnn` detector looks for a bright-centre / dark-
> border pattern and will fire on that, but it is not a face detector.

### Re-running these numbers

```sh
# Apache-2.0 detector (YuNet), default
tools/fetch_models.sh
# Highest accuracy + recognition (research only)
tools/fetch_models.sh --all

cargo test --features tract-backend --test real_model_e2e -- --nocapture --test-threads=1
# or, with the GPU-capable ONNX Runtime backend (requires `brew install onnxruntime` on macOS):
ORT_DYLIB_PATH=$(brew --prefix onnxruntime)/lib/libonnxruntime.dylib \
  cargo test --features ort-backend --test real_model_e2e -- --nocapture --test-threads=1
```

## CLI

> **Run `./target/release/rs-face --help` for the live help, or use
> `--list-algos` / `--list-features` for runtime introspection.**

```
INPUTs
  test://N            synthetic test pattern (N frames)
  /path/to/dir        image sequence (PNG/PGM/JPG files)
  /path/file.png|jpg  single image
  http(s)://host/p    single PNG or PNG-sequence base URL
  *.mp4|*.mov|*.avi|*.mkv|*.webm | rtsp://...
                      (requires `ffmpeg` on PATH)

ALGORITHMS  (pick with --algo <NAME>, default: haar)
  haar, cnn, yunet, mtcnn, hog, luminance,
  scrfd (requires --features ort-backend | tract-backend)

OPTIONS
  --out <DIR>           output directory (required)
  --algo <NAME>         detection algorithm (default: haar)
  --cascade <PATH>      load cascade from .rfcf file (haar only, default: built-in demo)
  --cnn-weights PATH    load CNN weights from a .cnn.bin file
  --threads N           worker thread count (default: # CPUs)
  --min-size PX         minimum detection size in pixels (default: 24)
  --max-size PX         maximum detection size in pixels (default: 1024)
  --scale F             pyramid scale factor (default: 1.2)
  --stride PX           window stride in pixels (default: 4)
  --nms F               NMS IoU threshold (default: 0.3)
  --min-score F         drop detections with cascade score below this
  --only-with-face      skip writing frames with zero detections
  --queue-depth N       per-worker queue depth (default: 4)
  --no-gpu              disable the (experimental) GPU variance pre-filter
  --no-equalize         skip the cv::equalizeHist preprocessing
  --list-algos          list every algorithm with its maturity and description
  --list-features       list every Cargo feature this binary was compiled with
  --version             print the crate version
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
             "total_detections": 0, "elapsed_ms": 316532, "detect_ms_avg": 12.4 },
  "frames": [
    { "frame_index": 0, "timestamp_ms": 0, "image": "frame_000000.png",
      "width": 480, "height": 854, "detections": [] }
  ]
}
```

## Examples

Every example is `cargo run --example <name>` — no ffmpeg, no model download
required:

| example | what it shows |
|---|---|
| `detect_haar` | smallest end-to-end Haar run on a synthetic frame |
| `detect_uniform` | the "swiss army knife" demo: dispatch Haar + HoG via `FaceDetector` trait |
| `recognise_lbph` | enrol + identify with LBPH, no weights; ends with an atomic save/load round-trip |
| `recognise_eigenface` | train + identify with eigenfaces/PCA, no weights |
| `recognise_fisherface` | train + identify with fisherfaces/LDA, no weights |
| `cascade_dump` | parse a `.rfcf` cascade and print its structure |
| `synthetic_smoke` | minimal pipeline smoke-test (no external data) |
| `lena_classify_stages` | walk a real cascade stage by stage |

## Architecture (multi-threaded pipeline)

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

Crate map (25+ modules, see `src/lib.rs` for full descriptions):

| ring | modules |
|---|---|
| core numerical | `integral`, `image`, `haar` |
| detection (zero-dep) | `detector`, `cnn`, `hog_face`, `yunet`, `mtcnn`, `luminance_face` |
| detection (ONNX) | `scrfd`, `scrfd_detector`, `onnx` |
| recognition (zero-dep) | `eigenface`, `fisherface`, `lbph`, `lbph_store` (versioned binary gallery persistence), `linalg` (shared Jacobi eigensolver) |
| recognition (ONNX) | `arcface`, `arcface_recognizer` |
| domain types | `face`, `face_detector`, `models`, `align`, `embedding` |
| pipeline / I/O | `pipeline`, `source`, `output`, `gpu`, `pool` |

## Licences

| weights    | use in a commercial product? |
|------------|------------------------------|
| YuNet      | **Yes** — Apache-2.0          |
| SCRFD, ArcFace w600k_r50 / w600k_mbf | **No** — InsightFace weights are licensed for non-commercial research only; commercial use requires a separate licence from DeepInsight |

The crate itself is MIT. The `ort-backend` feature pulls in ONNX Runtime (MIT) and
requires the libonnxruntime shared library to be installed on the host at runtime.

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

## Latest run

End-to-end test on a ReelShort drama clip (`reelshort_65da5136e047484c61021f92_63lsksowyg`,
1080×1920 HEVC, ~2:14):

```
$ rs-face /tmp/video.mp4 --out ./out \
    --cascade haarcascade.rfcf \
    --threads 4 --max-size 200 --scale 1.3 --stride 3 --only-with-face
[rs-face] cascade: 25 stages, 2913 features, window 24x24
[rs-face] source: /tmp/video.mp4 (live)
[rs-face] threads=4, queue_depth=4, detector=DetectorConfig { … }
[rs-face] done: 626 frames (626 with face), 786 detections, wall 67.98s, throughput 9.21 fps
```

| metric             | value         |
|--------------------|---------------|
| input              | 1080×1920 HEVC, 24 fps, 2:14 |
| frames processed   | 626           |
| frames with face   | 626 (100%)    |
| total detections   | 786           |
| throughput         | ~9.2 fps      |
| annotated PNGs     | 626 (3 sample frames committed under `docs/samples/`) |

Sample output frames (bounding boxes drawn in red by the pipeline):

| frame_index | file                                       |
|-------------|--------------------------------------------|
| 9           | `docs/samples/detection_early.png`         |
| 714         | `docs/samples/detection_mid.png`           |
| 2961        | `docs/samples/detection_late.png`          |

The full `manifest.json` (per-frame coords + scores) is written alongside the
annotated PNGs in the chosen `--out` directory; see the "Output" section above
for the schema.

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

The full directory tree is documented in [`CONTRIBUTING.md § Project layout`](CONTRIBUTING.md#project-layout).
Short version:

```
rs-face/
├── src/                  # library + CLI source (Rust)
├── tests/                # integration tests + ignored benchmarks
├── benches/              # criterion-style benches
├── examples/             # cargo run --example <NAME>
├── docs/                 # long-form design + accuracy documentation
├── tools/                # Python + shell helpers (not part of the crate)
├── platform/             # deployment-grade server + web UI (Docker-only)
│   ├── server/           # axum HTTP API + worker pool
│   ├── web/              # vanilla-JS frontend (zero build step)
│   ├── docs/             # platform design + roadmap + SDK notes
│   ├── migrations/       # SQL migrations
│   ├── scripts/          # bash + node smoke / screenshot scripts
│   ├── testdata/         # small reference images; large mp4/HLS gitignored
│   ├── Dockerfile        # rsface-server image (non-root, ffmpeg, static binary)
│   └── docker-compose.yml # rustfs + postgres + rsface-server
├── package.json + vite.config.js # frontend dev (vite only, root: platform/web)
├── Makefile              # docker-up / docker-test / web-dev / ...
└── data/                 # gitignored bind-mounted runtime state
```

Every directory has exactly one purpose; new top-level directories are
not added lightly. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the
complete tree, conventions, and module-split rationale.

## License

MIT — see [`LICENSE`](LICENSE).