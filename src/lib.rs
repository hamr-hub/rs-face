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

pub mod image;
pub mod integral;
pub mod haar;
pub mod detector;
pub mod pipeline;
pub mod source;
pub mod output;
pub mod gpu;
pub mod pool;
pub mod cnn;

pub use detector::{Detection, Detector};
pub use haar::Cascade;
pub use image::GrayImage;
pub use pipeline::{Pipeline, PipelineConfig, PipelineStats};
