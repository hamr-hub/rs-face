//! Compile-time bundled face cascade.
//!
//! [`bundled_frontalface_cascade`] returns the classical OpenCV
//! `haarcascade_frontalface_default` Viola–Jones cascade (25 stages,
//! 2 913 features, 24×24 window), converted once from OpenCV's XML to the
//! compact `.rfcf` v2 format and embedded into the binary via
//! `include_bytes!`. It is what makes a zero-argument rs-face build detect
//! *real* faces out of the box — no model download, no third-party runtime.
//!
//! Provenance and license (Apache-2.0) are documented in
//! `assets/NOTICE.md`; the exact SHA-256 of the source XML is recorded there
//! for reproducibility. Re-generate with
//! `python3 tools/convert_opencv_xml.py`.

use super::cascade::Cascade;

/// The embedded `.rfcf` bytes live in `assets/`; ~118 KiB on disk, which
/// buys production-quality face detection with zero runtime setup.
static BUNDLED_RFCF: &[u8] = include_bytes!("../../assets/haarcascade_frontalface_default.rfcf");

/// Parse and return the bundled OpenCV frontal-face cascade.
///
/// Parsing is a one-time ~microsecond-cost allocation; the returned
/// [`Cascade`] is cheap to [`Clone`] (the detector does this per pipeline
/// worker), so callers should keep one instance rather than re-parsing per
/// frame.
///
/// # Panics
///
/// Only if the embedded bytes are corrupt — impossible for a file compiled
/// in from a version-controlled artifact, hence the `expect`.
pub fn bundled_frontalface_cascade() -> Cascade {
    let mut bytes = BUNDLED_RFCF;
    Cascade::from_reader(&mut bytes).expect("bundled haarcascade_frontalface_default.rfcf is valid")
}
