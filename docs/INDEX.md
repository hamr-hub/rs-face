# rs-face — Documentation index

Welcome. This page is the entry point to every design / accuracy / operations
document in the repo. If you arrived via a search hit and want to know "where
do I start?", read in the order below.

> **Library convention** — every public module also carries a top-level doc
> comment. Start there for the API surface, come here for the *why*.

---

## 0. The 5-minute tour

| you want to … | read this |
|---|---|
| run the CLI on a video / URL / synthetic test | [`README.md § Quick start`](../README.md#quick-start) |
| embed `rsface` as a Rust dependency | [`README.md § Library API`](../README.md#library-api) — and the module-level `//!` block in [`src/lib.rs`](../src/lib.rs) |
| understand the algorithm matrix and "what to use when" | [`docs/algorithms.md`](algorithms.md) |
| see the pipeline plumbing diagram | [`docs/architecture.md`](architecture.md) |
| reproduce the published accuracy numbers | [`docs/benchmarks.md`](benchmarks.md) |
| add a new algorithm behind the uniform trait | [`src/face_detector.rs`](../src/face_detector.rs) — every detector implements `FaceDetector` |
| ship on macOS / iOS (Metal) or Linux/Windows (CUDA) | [`docs/GPU_BACKENDS.md`](GPU_BACKENDS.md) |
| ship industrial accuracy with ONNX | [`docs/GPU_BACKENDS.md` § ONNX backends](GPU_BACKENDS.md#onnx-backends) + [`tools/fetch_models.sh`](../tools/fetch_models.sh) |

---

## 1. Per-algorithm deep dives

Recognition (zero-dep, no external weights):

- [`docs/recognition-lbph.md`](recognition-lbph.md) — uniform LBP histograms, chi-square distance; 62/68 rank-1 on the 77-crop / 21-identity hard drama gallery (33/33 easy), 6×6 grid chosen by a 9-point sweep.
- [`docs/recognition-eigenface.md`](recognition-eigenface.md) — PCA / Turk-Pentland eigenfaces, Jacobi eigendecomposition in pure `std`, strict per-probe LOO retraining; 58/68 = the measured PCA ceiling.
- [`docs/recognition-fisherface.md`](recognition-fisherface.md) — Fisherfaces/LDA (Belhumeur 1997): n−C PCA reduction then ≤ C−1 class-discriminant axes, pseudo-inverse whitening; 59/68 rank-1 and the best zero-dep pair EER (≈ 12.8 %).
- [`docs/gallery-persistence.md`](gallery-persistence.md) — zero-dep binary LBPH gallery format (`RSLB` v1): bit-exact f32 descriptors, full decode validation, atomic save; survives restarts without the original crops.
- [`docs/recognition-video.md`](recognition-video.md) — **video-level identification** across one or many videos: tracker + single-linkage clusterer + cross-video re-id (`src/video_id.rs`, `examples/identify_short_drama.rs`).

Detection (zero-dep):

- [`docs/algorithms.md`](algorithms.md) — algorithm-by-algorithm breakdown of every detector the CLI accepts, weights requirements, and known limitations.

Detection (ONNX, opt-in via `ort-backend` or `tract-backend`):

- The pre/post-processing for SCRFD lives in [`src/scrfd.rs`](../src/scrfd.rs); for ArcFace in [`src/arcface.rs`](../src/arcface.rs). Both are backend-agnostic; the forward pass is chosen at compile time.

---

## 2. Architecture, format, performance

| doc | what it covers |
|---|---|
| [`docs/architecture.md`](architecture.md) | crate map, multi-threaded pipeline plumbing, the 5-detector uniform trait story. |
| [`docs/algorithms.md`](algorithms.md) | per-algorithm reference + **algorithm picker matrix** (start here when picking `--algo`). |
| [`docs/format.md`](format.md) | binary `.rfcf` cascade format, manifest JSON schema, ONNX model registry digests. |
| [`docs/benchmarks.md`](benchmarks.md) | reproducible bench scripts and what they measure. |
| [`docs/BENCHMARK_BASELINE.md`](BENCHMARK_BASELINE.md) | the pinned baseline numbers — every regression report compares to this. |
| [`docs/CPU_VS_GPU_REPORT.md`](CPU_VS_GPU_REPORT.md) | when GPU helps, when it doesn't, and why small images regress. |
| [`docs/GPU_BACKENDS.md`](GPU_BACKENDS.md) | `cpu` / `metal` / `cuda` / `rocm` / `mlu` / `ascend` — what's wired vs scaffold. |
| [`docs/CASCADE_FIX.md`](CASCADE_FIX.md) | history of the OpenCV XML → `.rfcf` parser bugs and the canonical converter. |

---

## 3. Measured numbers (reproducible)

- [`docs/bench-results.md`](bench-results.md) — detection accuracy / throughput on real footage.
- [`docs/bench-results-lbph.md`](bench-results-lbph.md) — LBPH rank-1, pair accuracy, EER on the labelled galleries.
- [`docs/bench-results-eigenface.md`](bench-results-eigenface.md) — eigenfaces rank-1, pair accuracy, FAR/FRR, size/energy sweep.
- [`docs/bench-results-fisherface.md`](bench-results-fisherface.md) — fisherfaces rank-1, pair accuracy, FAR/FRR, raw/equalised + size sweep.

Re-run from a clean tree:

```bash
cargo test --lib                                       # unit tests
cargo bench                                            # benches/perf_compare.rs
cargo test --features tract-backend --test real_model_e2e -- --nocapture --test-threads=1
tools/fetch_models.sh                                  # downloads pinned ONNX models (Y/N prompt)
```

---

## 4. Platform layer (`platform/`)

`platform/` is a thin deployment wrapper around the library: HTTP API, job
queue, Postgres-backed persistence, Prometheus `/metrics`, and a vanilla-JS
web UI for comparing the six detectors side by side.

- [`platform/README.md`](../platform/README.md) — what it is, how to run, port map (20080/19000/15432).
- [`platform/MINIMUM_CONFIG.md`](../platform/MINIMUM_CONFIG.md) — smallest viable box.
- [`platform/DOCKER_SIZING.md`](../platform/DOCKER_SIZING.md) — sizing for the bundled compose stack.
- [`platform/PROFILE.md`](../platform/PROFILE.md) — performance profile from real load tests.
- [`platform/CHANGELOG_CNN.md`](../platform/CHANGELOG_CNN.md) — CNN-related platform changes.

---

## 5. Repo hygiene

- [`CHANGELOG.md`](../CHANGELOG.md) — release notes (Keep a Changelog format).
- [`CONTRIBUTING.md`](../CONTRIBUTING.md) — dev setup, coding conventions, PR rules.
- [`LICENSE`](../LICENSE) — MIT.
- [`docs/samples/`](samples/) — annotated PNGs from real-world detection runs.