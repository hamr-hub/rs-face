//! # rsface — Zero-dependency face detection & recognition in pure Rust
//!
//! `rsface` is a swiss-army-knife face library: it bundles classical CV
//! (Viola–Jones Haar cascades, integral images, LBP/PCA/LDA recognisers),
//! a small trainable CNN detector, and an opt-in industrial ONNX path
//! (SCRFD detector + ArcFace recogniser via `ort` or `tract`).
//!
//! All default-build algorithms ship with **zero runtime dependencies**.
//! GPU and ONNX paths are feature-gated — pick what you need:
//!
//! ```toml
//! # smallest: pure CPU, classical CV, ~handful of KB on disk
//! rsface = "0.2"
//!
//! # add GPU (Metal on macOS, CUDA on Linux/Windows)
//! rsface = { version = "0.2", features = ["metal-backend"] }
//! rsface = { version = "0.2", features = ["cuda-backend"] }
//!
//! # add industrial accuracy (ONNX Runtime; needs `libonnxruntime`)
//! rsface = { version = "0.2", features = ["ort-backend"] }
//!
//! # add industrial accuracy without C++ runtime (pure Rust, CPU only)
//! rsface = { version = "0.2", features = ["tract-backend"] }
//! ```
//!
//! ## Module map
//!
//! The library is organised in three concentric rings: **core types**
//! (`no_std`-friendly for the hot numerical kernels), **detectors &
//! recognisers** (always-available CPU implementations behind a uniform
//! [`FaceDetector`] trait), and **I/O & orchestration** (`std` only,
//! multi-threaded [`Pipeline`], frame sources, writers).
//!
//! ### Core numerical primitives
//! - [`integral`]   : integral (summed-area), rotated, and squared-integral tables; the O(1) trick behind Haar / variance-norm.
//! - [`haar`]       : five Haar-like feature families, AdaBoost cascade, OpenCV XML → `.rfcf` loader.
//! - [`image`]      : 8-bit `GrayImage` / `RgbImage`, hand-rolled PNG codec, PPM/PGM codec.
//!
//! ### Detection algorithms (zero-dep)
//! - [`detector`]            : multi-scale sliding window + NMS, pipeline-agnostic.
//! - [`cnn`]                 : 24×24 Conv→ReLU→Pool→FC→Sigmoid CNN detector with a zero-dep trainer (starter weights; train your own).
//! - [`luminance_face`]      : band-pattern + mirror-symmetry detector (no weights, classical CV).
//!
//! ### Detection algorithms (ONNX, opt-in via `ort-backend` / `tract-backend`)
//! - [`scrfd`]               : InsightFace SCRFD-10G pre/post-processing.
//! - [`scrfd_detector`]      : end-to-end SCRFD detector using either ONNX backend.
//! - [`onnx`]                : backend-agnostic forward pass; `ort` (GPU-capable, C++ runtime) vs `tract` (pure Rust, CPU only).
//!
//! ### Recognition (zero-dep)
//! - [`eigenface`]           : PCA / Turk-Pentland eigenfaces, Jacobi eigendecomposition in pure `std`.
//! - [`fisherface`]          : Fisherfaces / LDA (Belhumeur–Hespanha–Kriegman): n−C PCA reduction then C−1 class-discriminant axes; best zero-dep pair EER.
//! - [`lbph`]                : uniform Local Binary Patterns histograms + chi-square distance.
//! - [`lbph_store`]          : zero-dep binary gallery persistence (save/load across restarts, no crops needed).
//! - [`subspace_store`]      : trained-model persistence for eigenfaces/fisherfaces (mean, axes, projections).
//!
//! ### Recognition (ONNX, opt-in)
//! - [`arcface`]             : ArcFace R50 / MobileFaceNet pre/post-processing (alignment, L2-norm).
//! - [`arcface_recognizer`]  : end-to-end embedding extractor + gallery matcher.
//!
//! ### Domain types & glue
//! - [`face_detector`]       : the [`FaceDetector`] trait — every algorithm implements it.
//! - [`recognizer`]          : the [`FaceRecognizer`] trait — LBPH/eigenfaces/Fisherfaces dispatch uniformly; shared [`Recognition`] outcome.
//! - [`face`]                : `Detection`, `Face`, landmark-aware types shared by detection + recognition.
//! - [`models`]              : model registry with pinned SHA-256 digests for downloaded weights.
//! - [`align`]               : landmark-based face alignment (112×112 canonical) for ArcFace.
//! - [`embedding`]           : `FaceEmbedding` (f32 vector, L2-normalised) + matcher.
//!
//! ### Pipeline / orchestration (`std` only)
//! - [`pipeline`]            : source → N detector workers → sink; threading, backpressure, manifest writer.
//! - [`source`]              : `FrameSource` trait; image sequence, HTTP MJPEG, optional ffmpeg pipe, synthetic test pattern.
//! - [`output`]              : hand-rolled JSON manifest writer, annotated PNG writer.
//! - [`gpu`]                 : GPU acceleration: hand-rolled dlopen OpenCL driver (zero-dep, Linux/BSD) + `GpuBackend` trait; real Metal/CUDA adapters behind the `metal-backend` / `cuda-backend` cargo features; macOS/Windows compile a no-op stub.
//! - [`pool`]                : small worker-pool helper (used by the pipeline).
//!
//! ## Cookbook (5-line recipes)
//!
//! Detect faces in a single image:
//!
//! ```
//! use rsface::detector::{Detector, DetectorConfig};
//! use rsface::haar::Cascade;
//! use rsface::image::GrayImage;
//!
//! let cascade = rsface::haar::params::demo_face_cascade();
//! let detector = Detector::new(cascade, DetectorConfig::default());
//! let gray = GrayImage::new(8, 8);   // 8x8 placeholder
//! let hits  = detector.detect(&gray);
//! ```
//!
//! Build a uniform detector (any algorithm, same call shape):
//!
//! ```
//! use rsface::face_detector::{FaceDetector, HaarDetector};
//! use rsface::haar::params;
//! use rsface::detector::DetectorConfig;
//! let det = HaarDetector::new(params::demo_face_cascade(), DetectorConfig::default());
//! assert_eq!(det.name(), "haar");
//! ```
//!
//! Train a zero-dep recogniser on a small gallery:
//!
//! ```
//! use rsface::lbph::{LbphConfig, LbphRecognizer};
//! let mut rec = LbphRecognizer::new(LbphConfig::default());
//! // rec.enroll("alice", &alice_crop);
//! // rec.enroll("bob",   &bob_crop);
//! // let outcome = rec.identify_crop(&probe);
//! ```
//!
//! Run the multi-threaded pipeline:
//!
//! ```no_run
//! use std::path::Path;
//! use rsface::haar::params::demo_face_cascade;
//! use rsface::pipeline::{Pipeline, PipelineConfig};
//! use rsface::source;
//!
//! let mut src = source::open("test://60").unwrap();
//! let stats = Pipeline::run(
//!     src.as_mut(),
//!     demo_face_cascade(),
//!     Path::new("./out"),
//!     PipelineConfig::default(),
//! ).unwrap();
//! println!(
//!     "{} frames, {} with face, {} detections in {} ms",
//!     stats.frames_processed,
//!     stats.frames_with_face,
//!     stats.total_detections,
//!     stats.elapsed_ms,
//! );
//! ```
//!
//! ## Maturity labels
//!
//! The README is the source of truth — see its "Accuracy" table for the
//! measured vs paper numbers per algorithm. In short:
//!
//! | family            | default build | what extra you must do |
//! |-------------------|---------------|------------------------|
//! | `haar`, `luminance`, `lbph`, `eigenface`, `fisherface` | ✅ Production-grade out of the box | nothing |
//! | `cnn`                                    | 🟡 Trainable toy net (starter weights only) | train with `cnn_train`, load via `--cnn-weights` |
//! | `scrfd`, `arcface`                       | ⛔ Opt-in (`ort-backend` / `tract-backend`) | download ONNX models; `tools/fetch_models.sh` does this |
//!
//! ## `no_std` & threading
//!
//! Core types (`GrayImage`, `Cascade`, `Detector` feature tables, `LbphConfig`)
//! avoid `alloc` where practical. The threading primitives in [`pipeline`],
//! [`pool`], and [`source`] require the `std` feature, which is on by default.
//!
//! ## License
//!
//! The crate is MIT. The `ort-backend` feature pulls in ONNX Runtime (MIT) and
//! requires `libonnxruntime` on the host. The crate ships no opaque weight
//! blobs: the Haar cascade is embedded as human-readable parameters, the CNN
//! starter weights are trainable via the `cnn_train` binary, and ONNX models
//! are fetched on demand through the SHA-256-pinned registry in [`models`]
//! (InsightFace weights are non-commercial research-only; YuNet is
//! Apache-2.0).

#![allow(clippy::too_many_arguments)] // Pipeline knobs are independently tuned; bundling them hides call sites.
#![allow(clippy::type_complexity)] // Detector/Cascade generics are spelled out in public APIs; not worth a type alias.
#![allow(clippy::result_large_err)] // PipelineError carries the source path for the UI to surface; not boxed.
#![allow(clippy::identity_op)] // 0-index placeholder math in CNN kernel index calcs; harmless.
#![allow(clippy::erasing_op)] // Same reason: the CNN scaffold uses 0 * N terms that are clearly placeholders.
#![allow(clippy::manual_div_ceil)] // Readability: written as `(a + b - 1) / b` for parity with OpenCV refs.
#![allow(clippy::manual_is_multiple_of)] // Avoid pulling the unstable div_rem helper.
#![allow(clippy::manual_range_contains)] // Readability: `x >= 1 && x <= 10` reads clearer in numerical kernels.
#![allow(clippy::manual_saturating_arithmetic)]
// Pipeline hot path; explicit branches are measurably faster.
// `clippy::manual_checked_ops` was renamed/removed in newer clippy; the
// `#[allow(unknown_lints)]` keeps this line working across toolchain
// versions (stable emits "unknown lint" warning otherwise — `-D warnings`
// in CI would turn that into a failure).
#![allow(unknown_lints)]
#![allow(clippy::manual_checked_ops)] // `(a / b)` written as `checked_div` reads worse in image-size math.
#![allow(clippy::unnecessary_cast)] // `u16 as u16` is sometimes emitted by cfg-gated code paths.
#![allow(clippy::io_other_error)] // PipelineError -> io::Error::new uses Display string for surfacing.
#![allow(clippy::mut_from_ref)] // OpenCL FFI returns *mut opaque; the wrapper holds the same ptr through an aliasing layer.
#![allow(clippy::redundant_closure)] // `|x| f(x)` is sometimes clearer than bare `f` for type inference.
#![allow(clippy::needless_range_loop)] // `for i in 0..n` indexing is intentional in numerical kernels.
#![allow(clippy::collapsible_if)] // Nested `if`s are clearer when each branch has a descriptive comment.
#![allow(clippy::unnecessary_map_or)] // `map_or(false, |x| ...)` reads more directly than `is_some_and` in variance checks.
#![allow(clippy::while_let_loop)] // `loop { match it.next() { ... } }` is clearer when the body has multiple branches.
#![allow(clippy::new_without_default)] // Detector constructors take config by value; `new(config)` is the canonical entry point.
#![allow(clippy::needless_collect)]
// Intentional Vec build for the next step.
// OpenCL loader caches dlopen handles in `static mut LIB`/`LIB_HANDLE`; access is gated by unsafe.
#![allow(static_mut_refs)]
#![allow(unused_parens)] // `cargo fmt` produces parens around some assignments; harmless.
#![allow(unused_unsafe)] // The OpenCL wrapper uses `unsafe {}` blocks defensively even where the call is itself unsafe.
#![allow(dead_code)] // Feature-gated GPU/ONNX backends and parity helpers keep primitives some builds don't call.
#![allow(unused_variables)] // Same reason as dead_code: code paths compiled only behind a feature flag.

pub mod align;
pub mod arcface;
pub mod arcface_recognizer;
pub(crate) mod binio;
pub mod cnn;
pub mod detector;
pub mod eigenface;
pub mod embedding;
pub mod face;
pub mod face_detector;
pub mod fisherface;
pub mod gpu;
pub mod haar;
pub mod image;
pub mod integral;
pub mod lbph;
pub mod lbph_store;
pub(crate) mod linalg;
pub mod luminance_face;
pub mod models;
pub mod onnx;
pub mod output;
pub mod pipeline;
pub mod pool;
pub mod recognizer;
pub mod scrfd;
pub mod scrfd_detector;
pub mod source;
pub mod subspace_store;

pub use detector::{Detection, Detector};
pub use face_detector::{FaceDetector, HaarDetector};
pub use haar::Cascade;
pub use image::GrayImage;
pub use luminance_face::{LuminanceConfig, LuminanceFaceDetector};
pub use pipeline::{Pipeline, PipelineConfig, PipelineStats};
pub use recognizer::{FaceRecognizer, IncrementalRecognizer, Recognition};
