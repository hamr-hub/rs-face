//! Unified `FaceDetector` trait — every algorithm in `rs-face` core implements
//! this so platform code and tests can dispatch polymorphically.
//!
//! All detectors work on the core's `GrayImage` and produce `Detection`s from
//! `detector.rs` (pixel-space bbox + score). They are CPU-only, allocation-light,
//! and `!Sync` (detector scratch buffers are owned per-thread, see `cnn::CnnScratch`).

use crate::detector::Detection;
use crate::image::GrayImage;

/// Trait every face detector must implement. Implementors should:
/// 1. Be cheap to construct (`YunetDetector::new()` is enough to run).
/// 2. Never panic on empty / uniform input (smoke-tested via `*_no_panic` tests).
/// 3. Return detections sorted by descending score (so NMS in callers is sane).
pub trait FaceDetector: Send {
    /// Run the detector on a single grayscale frame. Returns `Vec<Detection>`.
    fn detect(&self, img: &GrayImage) -> Vec<Detection>;

    /// Short lowercase name used in `RSFACE_ALGO` env var, `/api/config`, and
    /// SSE `algo` field. Must be unique across all detectors.
    fn name(&self) -> &'static str;

    /// Optional: detector description for the Web UI compare card.
    fn description(&self) -> &'static str {
        ""
    }
}
