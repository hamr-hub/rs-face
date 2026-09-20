//! Cross-algorithm compatibility test: run every available zero-dep detector
//! against the same synthetic face image and assert that the schema of the
//! returned detections is identical, none of them panic, and every box fits
//! within the input image. This is the single most useful regression gate
//! for "did we silently break one of the detectors?".
//!
//! Requires both `detector-haar` and `detector-luminance` (and `detector-cnn`
//! only for the optional CNN sanity check, gated inside the test body).
//!
//! The CNN detector is opt-in because its template weights are slow on a
//! fresh build; we skip it unless `RSFACE_RUN_CNN_COMPAT=1` is set.

use rsface::face::Detection;
use rsface::face_detector::FaceDetector;
use rsface::image::GrayImage;

/// Build a 200x160 synthetic image with a clearly visible circular
/// bright "face-like" blob in the middle. Not a real face, but the kind
/// of input every classical detector agrees is the right shape.
fn synth_face_image() -> GrayImage {
    let (w, h) = (200usize, 160usize);
    let mut img = GrayImage::new(w, h);
    let (cx, cy) = (100.0f32, 80.0f32);
    let r = 35.0f32;
    for y in 0..h {
        for x in 0..w {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let d = (dx * dx + dy * dy).sqrt();
            let v = if d < r {
                220
            } else if d < r * 1.4 {
                160
            } else {
                30
            };
            img[(x, y)] = v;
        }
    }
    img
}

/// Every detector must produce a Detection-list whose boxes fit inside
/// the input image and whose scores are finite floats. The exact number
/// of detections is detector-specific and is NOT asserted here.
fn assert_schema_ok(name: &str, img: &GrayImage, dets: &[Detection]) {
    let (w, h) = (img.width(), img.height());
    for (i, d) in dets.iter().enumerate() {
        assert!(
            d.x + d.w <= w,
            "[{name}] detection {i} right edge {} > image width {w}",
            d.x + d.w
        );
        assert!(
            d.y + d.h <= h,
            "[{name}] detection {i} bottom edge {} > image height {h}",
            d.y + d.h
        );
        assert!(
            d.score.is_finite(),
            "[{name}] detection {i} has non-finite score: {}",
            d.score
        );
        assert!(
            d.w > 0 && d.h > 0,
            "[{name}] detection {i} has zero-area box",
        );
    }
}

#[cfg(feature = "detector-haar")]
#[test]
fn haar_detector_runs_on_synthetic_face() {
    use rsface::detector::{Detector, DetectorConfig};
    use rsface::haar::params::demo_face_cascade;

    let img = synth_face_image();
    let det = Detector::new(demo_face_cascade(), DetectorConfig::default());
    let dets = det.detect(&img);
    assert_schema_ok("haar", &img, &dets);
}

#[cfg(feature = "detector-luminance")]
#[test]
fn luminance_detector_runs_on_synthetic_face() {
    use rsface::luminance_face::{LuminanceConfig, LuminanceFaceDetector};

    let img = synth_face_image();
    let det = LuminanceFaceDetector::new(LuminanceConfig::default());
    let dets = det.detect(&img);
    assert_schema_ok("luminance", &img, &dets);
}

#[cfg(feature = "detector-cnn")]
#[test]
fn cnn_detector_runs_on_synthetic_face_opt_in() {
    // Template weights are slow on a fresh build; this test is opt-in via
    // RSFACE_RUN_CNN_COMPAT=1 to keep the default suite fast.
    if std::env::var("RSFACE_RUN_CNN_COMPAT").ok().as_deref() != Some("1") {
        eprintln!(
            "algo_compat: cnn skipped (set RSFACE_RUN_CNN_COMPAT=1 to run; \
             template weights are slow on a fresh build)"
        );
        return;
    }
    use rsface::cnn::{CnnConfig, CnnDetector};

    let img = synth_face_image();
    let det = CnnDetector::new(CnnConfig::default());
    let mut buf = vec![0.0f32; img.width() * img.height()];
    for (i, &p) in img.as_slice().iter().enumerate() {
        buf[i] = p as f32 / 255.0;
    }
    let cnn_dets = det.detect(&buf, img.width(), img.height());
    // Convert CnnDetection -> Detection (lossy but schema-equivalent).
    let dets: Vec<Detection> = cnn_dets
        .into_iter()
        .map(|d| Detection {
            x: d.x,
            y: d.y,
            w: d.w,
            h: d.h,
            score: d.confidence,
        })
        .collect();
    assert_schema_ok("cnn", &img, &dets);
}

#[cfg(feature = "detector-haar")]
#[test]
fn haar_on_zero_sized_image_returns_empty_not_panic() {
    use rsface::detector::{Detector, DetectorConfig};
    use rsface::haar::params::demo_face_cascade;

    for &(w, h) in &[(0usize, 0usize), (10, 0), (0, 10), (1, 1)] {
        let img = GrayImage::new(w, h);
        let det = Detector::new(demo_face_cascade(), DetectorConfig::default());
        let dets = det.detect(&img);
        assert!(
            dets.is_empty(),
            "expected empty on {w}x{h}, got {}",
            dets.len()
        );
    }
}

#[cfg(feature = "detector-luminance")]
#[test]
fn luminance_on_zero_sized_image_returns_empty_not_panic() {
    use rsface::luminance_face::{LuminanceConfig, LuminanceFaceDetector};

    for &(w, h) in &[(0usize, 0usize), (10, 0), (0, 10), (1, 1)] {
        let img = GrayImage::new(w, h);
        let det = LuminanceFaceDetector::new(LuminanceConfig::default());
        let dets = det.detect(&img);
        assert!(
            dets.is_empty(),
            "expected empty on {w}x{h}, got {}",
            dets.len()
        );
    }
}

#[cfg(feature = "detector-haar")]
#[test]
fn haar_detector_name_matches_brand_string() {
    use rsface::detector::{Detector, DetectorConfig};
    use rsface::face_detector::HaarDetector;
    use rsface::haar::params::demo_face_cascade;

    let det = HaarDetector::new(demo_face_cascade(), DetectorConfig::default());
    assert_eq!(det.name(), "haar");
}

#[cfg(feature = "detector-luminance")]
#[test]
fn luminance_detector_name_matches_brand_string() {
    use rsface::face_detector::FaceDetector;
    use rsface::luminance_face::{LuminanceConfig, LuminanceFaceDetector};

    let det = LuminanceFaceDetector::new(LuminanceConfig::default());
    assert_eq!(det.name(), "luminance");
}
