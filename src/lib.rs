//! rsface — Zero-dependency Viola-Jones face detection in pure Rust.
//!
//! Modules:
//! - [`image`]   : grayscale/RGB image types, PNG + PPM encode, JPEG decode (DCT-free baseline subset).
//! - [`integral`]: integral image (summed-area table).
//! - [`haar`]    : Haar-like features, AdaBoost cascade classifier.
//! - [`detector`]: multi-scale sliding window + non-maximum suppression.
//! - [`pipeline`]: multi-threaded decode / detect / write pipeline.
//! - [`source`]  : frame sources (image sequence, HTTP MJPEG, optional ffmpeg pipe).
//! - [`output`]  : PNG writer + JSON manifest writer.
//!
//! The library is `no_std`-friendly for the core types (integral, haar, detector)
//! but uses `std` for I/O and threading.

#![allow(clippy::too_many_arguments)] // Pipeline knobs are independently tuned; bundling them hides call sites.
#![allow(clippy::type_complexity)] // Detector/Cascade generics are spelled out in public APIs; not worth a type alias.
#![allow(clippy::result_large_err)] // PipelineError carries the source path for the UI to surface; not boxed.
#![allow(clippy::identity_op)] // 0-index placeholder math in CNN kernel index calcs; harmless.
#![allow(clippy::erasing_op)] // Same reason: the CNN scaffold uses 0 * N terms that are clearly placeholders.
#![allow(clippy::manual_div_ceil)] // Readability: written as `(a + b - 1) / b` for parity with OpenCV refs.
#![allow(clippy::manual_checked_div)] // Same reason; many image-size math sites do the explicit form.
#![allow(clippy::manual_is_multiple_of)] // Avoid pulling the unstable div_rem helper.
#![allow(clippy::manual_range_contains)] // Readability: `x >= 1 && x <= 10` reads clearer in numerical kernels.
#![allow(clippy::manual_saturating_arithmetic)] // Pipeline hot path; explicit branches are measurably faster.
#![allow(clippy::unnecessary_cast)] // `u16 as u16` is sometimes emitted by cfg-gated code paths.
#![allow(clippy::io_other_error)] // PipelineError -> io::Error::new uses Display string for surfacing.
#![allow(clippy::mut_from_ref)] // OpenCL FFI returns *mut opaque; the wrapper holds the same ptr through an aliasing layer.
#![allow(clippy::redundant_closure)] // `|x| f(x)` is sometimes clearer than bare `f` for type inference.
#![allow(clippy::while_immutable_condition)] // Hand-written `while` loops over `Vec::iter()` are clearer than `.for_each`.
#![allow(clippy::needless_range_loop)] // `for i in 0..n` indexing is intentional in numerical kernels.
#![allow(unused_parens)] // `cargo fmt` produces parens around some assignments; harmless.
#![allow(unused_unsafe)] // The OpenCL wrapper uses `unsafe {}` blocks defensively even where the call is itself unsafe.
#![allow(dead_code)] // Dummy CNN/HoG/YuNet/MTCNN scaffolds ship with all primitives even if a few are unreferenced.
#![allow(unused_variables)] // Same reason as dead_code; placeholder code paths.

pub mod cnn;
pub mod detector;
pub mod face_detector;
pub mod gpu;
pub mod haar;
pub mod hog_face;
pub mod image;
pub mod integral;
pub mod mtcnn;
pub mod output;
pub mod pipeline;
pub mod pool;
pub mod source;
pub mod yunet;

pub use detector::{Detection, Detector};
pub use face_detector::FaceDetector;
pub use haar::Cascade;
pub use hog_face::{HogConfig, HogFaceDetector};
pub use image::GrayImage;
pub use mtcnn::{MtcnnConfig, MtcnnDetector};
pub use pipeline::{Pipeline, PipelineConfig, PipelineStats};
pub use yunet::{YunetConfig, YunetDetector};
