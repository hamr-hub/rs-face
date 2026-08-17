# Changelog

All notable changes to `rs-face` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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