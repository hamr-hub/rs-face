# Changelog

All notable changes to `rs-face` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — industrial recognition (SCRFD + ArcFace via ONNX)
- **SCRFD-10G face detector** (`src/scrfd.rs`, `src/scrfd_detector.rs`):
  letterbox preprocessing, anchor-free stride-8/16/32 decoding (no half-stride
  offset, matching InsightFace exactly), keypoint decoding, and post-processing.
  0.954 / 0.940 / 0.828 WIDER FACE easy/medium/hard AP upstream.
- **ArcFace R50 recognizer** (`src/arcface.rs`, `src/arcface_recognizer.rs`,
  `src/align.rs`, `src/embedding.rs`, `src/face.rs`): 5-landmark similarity
  transform alignment, 512-d embeddings, enrollment/identification gallery,
  and cosine matching with a configurable threshold.
- **Two real ONNX inference backends** (`src/onnx/`), both optional so the
  default build stays zero-dependency:
  - `ort-backend` — ONNX Runtime via dynamic loading (`ORT_DYLIB_PATH`), the
    route to GPU (CoreML / CUDA / TensorRT / DirectML / ROCm behind separate
    features).
  - `tract-backend` — pure-Rust CPU inference for single-static-binary deploys.
  - All 16 real-model integration tests pass under either backend.
- **Model registry with pinned SHA-256 digests** (`src/models.rs`): weights are
  verified before inference, with a zero-dependency SHA-256 implementation and
  typed licence tiers (research-only vs Apache-2.0) queryable at runtime.
- `tools/fetch_models.sh` to download YuNet (Apache-2.0) and the buffalo_l
  bundle (SCRFD + ArcFace, research-only) with digest verification.
- `bench-accuracy` bin (`benches/accuracy.rs`): end-to-end latency and
  same/different-identity cosine distributions on real weights; writes
  `docs/bench-results.md` and prints threshold guidance.
- Real-face PPM fixtures (`lena`, `biden`, `two-people`) and
  `tests/real_model_e2e.rs` — 16 integration tests covering integrity gates,
  geometry, blank-frame quietness, gallery round trips, discrimination, and
  scale invariance.
- Placeholder-weight detectors (`mtcnn`, `yunet`, `hog`) now report
  `Maturity::Scaffold` and are visibly labelled as detecting nothing.

### Changed
- README gains measured accuracy tables, a licences section (research-only vs
  commercial), and instructions to reproduce the numbers.
- `docs/benchmarks.md` documents both the InsightFace default threshold (0.36,
  TAR@FAR=1e-4) and the measured distribution midpoint (≈0.53 on the repo
  fixtures), with the formula for choosing your own.

### Added
- End-to-end face detection run on a 1080×1920 / 24 fps drama clip (ReelShort
  `reelshort_65da5136e047484c61021f92_63lsksowyg`, ~2:14, HEVC). Result:
  **626 / 626 frames with face (100 %)**, 786 total detection boxes,
  ~9.2 fps end-to-end on a 6-core aarch64 box.
- `docs/samples/` directory with three sample annotated PNGs from that run
  (early / mid / late frames).

### Documentation
- `CHANGELOG.md` (this file).
- `README.md` "Latest run" section linking to the sample frames and listing
  the run's measured throughput / detection counts.

## [0.1.0] — 2026-08-13

### Added
- Initial release: zero-dependency Viola-Jones face detector in pure Rust.
- Five-feature-family Haar cascade, OpenCV `haarcascade_frontalface_default.xml`
  loader via the included `.rfcf` binary format and XML→`.rfcf` converter.
- 24×24 CNN detector (Conv→ReLU→Pool→FC→Sigmoid) with bundled placeholder
  weights so the CNN code path is exercisable without external downloads.
- Multi-threaded pipeline (`source → N detector workers → sink`) with
  greedy NMS, optional OpenCL squared-integral pre-filter, and a hand-rolled
  JSON manifest + annotated PNG writer.
- Frame sources: image sequence, HTTP, ffmpeg pipe, synthetic test pattern.
- CI workflow (`.github/workflows/ci.yml`): build + test on Ubuntu and macOS,
  `cargo fmt --check`, `cargo clippy -- -D warnings`.

### Notes
- `Cargo.lock` is committed because this crate ships as a binary-first
  project. Consumers should treat the lockfile as informative when depending
  on the library.
- Tilted (45°) Haar features evaluate to 0 (rotated integral kept correct
  under Rust's borrow checker with zero deps is non-trivial; the cascade
  format still supports them).
- CNN bundled weights are placeholders — train on labelled data and load
  via `CnnDetector::with_weights` for production use.
