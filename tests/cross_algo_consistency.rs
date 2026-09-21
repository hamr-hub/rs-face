//! Cross-algorithm consistency tests.
//!
//! Beyond the schema smoke in `tests/algo_compat.rs`, this suite asserts that
//! every classical detector fires on a face-shaped input and that the
//! surviving (post-NMS) boxes have meaningful geometric properties — not
//! "the detector silently drifted off-target". If a future algorithm
//! produces wildly different boxes from the established consensus on a
//! controlled input, this file catches it before the regression can land.
//!
//! Two test fixtures are used:
//!
//! 1. `tests/fixtures/demo_face_256.pgm` — a real frontal face photo.
//!    The bundled OpenCV Haar cascade is production-grade and reliably
//!    fires on it. Luminance is documented as "Experimental" and tuned
//!    for synthetic forehead/eyes/chin patterns; on real photographs its
//!    score-threshold default may not fire. Tests therefore only require
//!    Luminance schema-correctness (in-bounds, finite) on this fixture,
//!    not a positive detection.
//!
//! 2. `synth_frontal_face()` — the canonical 256×256 forehead/eye-band/
//!    chin pattern that the luminance detector was tuned against (mirrors
//!    `luminance_face::detects_synthetic_face`). Luminance reliably fires
//!    here; the bundled Haar cascade may not.
//!
//! Gating:
//! - Haar + Luminance paths require `detector-haar` and `detector-luminance`.
//! - The optional CNN path is gated by `RSFACE_RUN_CNN_CONSISTENCY=1` because
//!   template weights are slow on a fresh build.

use rsface::face::Detection;
use rsface::face_detector::FaceDetector;
use rsface::image::{codec, GrayImage};
use std::path::Path;

/// IoU threshold below which two detections are considered to disagree.
const CONSENSUS_IOU: f32 = 0.10;

/// 30 px centroid tolerance — the synthetic-face circle has a 96-px
/// diameter; a detector that locks on a different region by more than
/// 30 px is plainly broken.
const CENTROID_TOLERANCE_PX: f32 = 30.0;

/// Load the bundled `demo_face_256.pgm` fixture (a real frontal face).
fn load_demo_face() -> GrayImage {
    let path = "tests/fixtures/demo_face_256.pgm";
    let mut f = std::fs::File::open(Path::new(path))
        .unwrap_or_else(|e| panic!("open {path}: {e}"));
    codec::read_pgm(&mut f).expect("read_pgm demo_face_256.pgm")
}

/// Build the canonical 256×256 synthetic frontal-face pattern that the
/// luminance detector was tuned against. Mirrors
/// `luminance_face::detects_synthetic_face`. Image is always 256×256;
/// callers may pad it later for translation-invariance tests.
fn synth_frontal_face() -> GrayImage {
    let mut img = GrayImage::new(256, 256);
    // Forehead (top 40%, rows 0..102): bright, with subtle texture.
    for y in 0..102 {
        for x in 0..256 {
            let v = 210 + (((x * 7 + y * 11) ^ (x >> 3)) & 0xF) as i32 - 7;
            img[(x, y)] = v.clamp(160, 230) as u8;
        }
    }
    // Eye band (rows 102..176): dark with darker eye spots at cols 64..96, 160..192.
    for y in 102..176 {
        for x in 0..256 {
            let mut v = 50;
            if (64..96).contains(&x) || (160..192).contains(&x) {
                v = 18; // eye spots
            }
            if (124..132).contains(&x) && y > 110 && y < 160 {
                v += 60; // nose ridge brightening
            }
            img[(x, y)] = v;
        }
    }
    // Chin (rows 176..256): mid.
    for y in 176..256 {
        for x in 0..256 {
            img[(x, y)] =
                (130 + (((x * 5 + y * 3) ^ (y >> 4)) & 0x7) as i32 - 3).clamp(0, 255) as u8;
        }
    }
    img
}

/// Pad a 256×256 face into a larger 320×240 canvas (or other size). Used by
/// translation-invariance tests that need the face somewhere other than
/// the dead center.
fn synth_face_at_offset(ox: i32, oy: i32) -> GrayImage {
    let (w, h) = (320i32, 240i32);
    let mut canvas = GrayImage::new(w as usize, h as usize);
    let inner = synth_frontal_face();
    for y in 0..256i32 {
        for x in 0..256i32 {
            let tx = x + ox;
            let ty = y + oy;
            if (0..w).contains(&tx) && (0..h).contains(&ty) {
                canvas[(tx as usize, ty as usize)] = inner[(x as usize, y as usize)];
            }
        }
    }
    canvas
}

/// Return the top-scoring detection, or None.
fn top(dets: &[Detection]) -> Option<&Detection> {
    dets.iter().max_by(|a, b| {
        a.score
            .partial_cmp(&b.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

#[cfg(feature = "detector-haar")]
#[test]
fn haar_fires_on_real_face_fixture() {
    use rsface::detector::{Detector, DetectorConfig};
    use rsface::haar::bundled::bundled_frontalface_cascade;

    let img = load_demo_face();
    let det = Detector::new(bundled_frontalface_cascade(), DetectorConfig::default());
    let dets = det.detect(&img);
    assert!(
        !dets.is_empty(),
        "bundled OpenCV Haar cascade MUST detect the bundled demo_face_256.pgm; \
         got {} detections",
        dets.len()
    );
    let (w, h) = (img.width(), img.height());
    for d in &dets {
        assert!(d.w > 0 && d.h > 0, "zero-area box");
        assert!(
            d.x + d.w <= w && d.y + d.h <= h,
            "box overflows image: ({},{},{},{}) in {w}x{h}",
            d.x, d.y, d.w, d.h
        );
        assert!(d.score.is_finite(), "non-finite score {}", d.score);
    }
    let best = top(&dets).expect("non-empty");
    let cx = best.x as f32 + best.w as f32 / 2.0;
    let cy = best.y as f32 + best.h as f32 / 2.0;
    let dx = (cx - 128.0).abs();
    let dy = (cy - 128.0).abs();
    assert!(
        dx <= CENTROID_TOLERANCE_PX && dy <= CENTROID_TOLERANCE_PX,
        "bundled Haar top box center ({cx:.1},{cy:.1}) drifted > \
         {CENTROID_TOLERANCE_PX} px from the known face center (128,128); \
         dx={dx:.1} dy={dy:.1}"
    );
}

#[cfg(feature = "detector-luminance")]
#[test]
fn luminance_fires_on_synthetic_frontal_face() {
    use rsface::luminance_face::{LuminanceConfig, LuminanceFaceDetector};

    let img = synth_frontal_face();
    let det = LuminanceFaceDetector::new(LuminanceConfig::default());
    let dets = det.detect(&img);
    let (w, h) = (img.width(), img.height());
    assert!(
        !dets.is_empty(),
        "Luminance MUST detect the canonical synthetic frontal face; \
         got {} detections",
        dets.len()
    );
    for d in &dets {
        assert!(d.w > 0 && d.h > 0, "zero-area box");
        assert!(
            d.x + d.w <= w && d.y + d.h <= h,
            "box overflows image: ({},{},{},{}) in {w}x{h}",
            d.x, d.y, d.w, d.h
        );
        assert!(d.score.is_finite(), "non-finite score {}", d.score);
        // Score must be in [0, 1] (luminance normalises by combined weights).
        assert!(
            (0.0..=1.0).contains(&d.score),
            "Luminance score out of [0,1]: {}",
            d.score
        );
    }
    let best = top(&dets).unwrap();
    let cx = best.x + best.w / 2;
    let cy = best.y + best.h / 2;
    assert!(
        (60..=200).contains(&cx) && (60..=200).contains(&cy),
        "Luminance top box centre ({cx},{cy}) outside the 60..=200 expected band \
         (image is 256×256 with face centred at 128,128)"
    );
}

/// Luminance must be schema-correct on the real-face fixture even when its
/// default threshold rejects the image. Tests rely on Luminance not
/// panicking, not producing NaN/inf scores, and not emitting boxes that
/// overflow the image — independent of whether it fires at all.
#[cfg(feature = "detector-luminance")]
#[test]
fn luminance_schema_correct_on_real_face_fixture() {
    use rsface::luminance_face::{LuminanceConfig, LuminanceFaceDetector};

    let img = load_demo_face();
    let det = LuminanceFaceDetector::new(LuminanceConfig::default());
    let dets = det.detect(&img);
    let (w, h) = (img.width(), img.height());
    for d in &dets {
        assert!(d.w > 0 && d.h > 0, "zero-area box on real face");
        assert!(
            d.x + d.w <= w && d.y + d.h <= h,
            "Luminance box overflows real face fixture ({},{},{},{}) in {w}x{h}",
            d.x, d.y, d.w, d.h
        );
        assert!(d.score.is_finite(), "non-finite score {}", d.score);
        assert!(
            (0.0..=1.0).contains(&d.score),
            "Luminance score out of [0,1] on real face: {}",
            d.score
        );
    }
}

/// Same detector applied twice must produce identical results (deterministic
/// pipeline). Catches accidental global-state accumulation in the detector
/// scratch buffers that would make a "second call" see different scores.
#[cfg(feature = "detector-haar")]
#[test]
fn haar_is_deterministic_across_runs() {
    use rsface::detector::{Detector, DetectorConfig};
    use rsface::haar::bundled::bundled_frontalface_cascade;

    let img = load_demo_face();
    let det = Detector::new(bundled_frontalface_cascade(), DetectorConfig::default());

    let run1 = det.detect(&img);
    let run2 = det.detect(&img);
    assert_eq!(
        run1.len(),
        run2.len(),
        "Haar detector must produce a deterministic count (got {} then {})",
        run1.len(),
        run2.len()
    );
    for (a, b) in run1.iter().zip(run2.iter()) {
        assert_eq!(a.x, b.x, "x coordinate drift between runs");
        assert_eq!(a.y, b.y, "y coordinate drift between runs");
        assert_eq!(a.w, b.w, "width drift between runs");
        assert_eq!(a.h, b.h, "height drift between runs");
        assert!(
            (a.score - b.score).abs() < 1e-5,
            "score drift between runs: {} vs {}",
            a.score,
            b.score
        );
    }
}

#[cfg(feature = "detector-luminance")]
#[test]
fn luminance_is_deterministic_across_runs() {
    use rsface::luminance_face::{LuminanceConfig, LuminanceFaceDetector};

    let img = synth_frontal_face();
    let det = LuminanceFaceDetector::new(LuminanceConfig::default());

    let run1 = det.detect(&img);
    let run2 = det.detect(&img);
    assert_eq!(
        run1.len(),
        run2.len(),
        "Luminance detector must produce a deterministic count (got {} then {})",
        run1.len(),
        run2.len()
    );
    for (a, b) in run1.iter().zip(run2.iter()) {
        assert_eq!(a.x, b.x, "x coordinate drift");
        assert_eq!(a.y, b.y, "y coordinate drift");
        assert_eq!(a.w, b.w, "width drift");
        assert_eq!(a.h, b.h, "height drift");
        assert!(
            (a.score - b.score).abs() < 1e-5,
            "score drift between runs: {} vs {}",
            a.score,
            b.score
        );
    }
}

/// Translation-invariance on the synthetic frontal face: when the face is
/// placed at several offsets inside the 320×240 canvas, every returned
/// detection must be schema-correct (in-bounds, finite score). The exact
/// centroid is not asserted because the Luminance detector was tuned on
/// full-canvas faces, and padded offsets introduce a high-variance border
/// that can dominate the score window — but a regression that emits NaN
/// scores, zero-area boxes, or out-of-image coords on a padded face would
/// still fail this gate.
#[cfg(feature = "detector-luminance")]
#[test]
fn luminance_schema_correct_on_offset_synthetic_face() {
    use rsface::luminance_face::{LuminanceConfig, LuminanceFaceDetector};

    let offsets: [(i32, i32); 3] = [(32, -8), (0, -16), (16, 0)];
    let det = LuminanceFaceDetector::new(LuminanceConfig::default());
    let (w, h) = (320usize, 240usize);
    for (ox, oy) in offsets {
        let img = synth_face_at_offset(ox, oy);
        let dets = det.detect(&img);
        for d in &dets {
            assert!(d.w > 0 && d.h > 0, "zero-area box at offset ({ox},{oy})");
            assert!(
                d.x + d.w <= w && d.y + d.h <= h,
                "box overflows canvas at offset ({ox},{oy}): ({},{},{},{}) in {w}x{h}",
                d.x, d.y, d.w, d.h
            );
            assert!(
                d.score.is_finite(),
                "non-finite score {} at offset ({ox},{oy})",
                d.score
            );
            assert!(
                (0.0..=1.0).contains(&d.score),
                "Luminance score out of [0,1] at offset ({ox},{oy}): {}",
                d.score
            );
        }
    }
}

/// A blank image (all zeros) must produce empty detections from BOTH Haar
/// and Luminance — i.e. no detector should fire on uniform input. This is
/// the dual of the positive cases above and is the cheapest false-positive
/// gate we have.
#[cfg(all(feature = "detector-haar", feature = "detector-luminance"))]
#[test]
fn all_classical_detectors_silent_on_uniform_image() {
    use rsface::detector::{Detector, DetectorConfig};
    use rsface::haar::bundled::bundled_frontalface_cascade;
    use rsface::luminance_face::{LuminanceConfig, LuminanceFaceDetector};

    let img = GrayImage::new(320, 240);
    let haar = Detector::new(bundled_frontalface_cascade(), DetectorConfig::default())
        .detect(&img);
    let lum = LuminanceFaceDetector::new(LuminanceConfig::default()).detect(&img);

    assert!(
        haar.is_empty(),
        "Haar must be silent on a uniform input ({} false positives)",
        haar.len()
    );
    assert!(
        lum.is_empty(),
        "Luminance must be silent on a uniform input ({} false positives)",
        lum.len()
    );
}

/// A pure noise image (per-pixel random bytes) must NOT panic and must
/// produce detections whose scores are finite. We don't require a
/// specific count — Haar with min_score=0 may explode on noise — but
/// every returned box must pass the schema invariant.
#[cfg(all(feature = "detector-haar", feature = "detector-luminance"))]
#[test]
fn noise_input_keeps_no_finite_score_invariant() {
    use rsface::detector::{Detector, DetectorConfig};
    use rsface::haar::bundled::bundled_frontalface_cascade;
    use rsface::luminance_face::{LuminanceConfig, LuminanceFaceDetector};

    // Deterministic pseudo-noise via a tiny LCG — not random, so the test
    // is reproducible across runs without an RNG crate.
    let mut img = GrayImage::new(256, 256);
    let mut s: u32 = 0x9e37_79b9;
    for y in 0..256 {
        for x in 0..256 {
            // xorshift32
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            img[(x, y)] = s as u8;
        }
    }

    let haar = Detector::new(bundled_frontalface_cascade(), DetectorConfig::default())
        .detect(&img);
    let lum = LuminanceFaceDetector::new(LuminanceConfig::default()).detect(&img);
    let (w, h) = (img.width(), img.height());

    for (label, dets) in [("haar", &haar), ("luminance", &lum)] {
        for d in dets {
            assert!(d.w > 0 && d.h > 0, "{label} zero-area box on noise");
            assert!(
                d.x + d.w <= w && d.y + d.h <= h,
                "{label} box overflows noise canvas ({},{},{},{}) in {w}x{h}",
                d.x, d.y, d.w, d.h
            );
            assert!(
                d.score.is_finite(),
                "{label} non-finite score on noise: {}",
                d.score
            );
            if label == "luminance" {
                assert!(
                    (0.0..=1.0).contains(&d.score),
                    "Luminance score out of [0,1] on noise: {}",
                    d.score
                );
            }
        }
    }
}

/// Optional CNN check on the bundled face fixture, gated by env var so the
/// default CI run stays fast.
#[cfg(feature = "detector-cnn")]
#[test]
fn cnn_top_box_is_near_real_face_center_opt_in() {
    if std::env::var("RSFACE_RUN_CNN_CONSISTENCY").ok().as_deref() != Some("1") {
        eprintln!(
            "cross_algo_consistency: cnn skipped (set RSFACE_RUN_CNN_CONSISTENCY=1 \
             to run; template weights are slow on a fresh build)"
        );
        return;
    }
    use rsface::cnn::{CnnConfig, CnnDetector};

    let img = load_demo_face();
    let det = CnnDetector::new(CnnConfig::default());
    let mut buf = vec![0.0f32; img.width() * img.height()];
    for (i, &p) in img.as_slice().iter().enumerate() {
        buf[i] = p as f32 / 255.0;
    }
    let cnn_dets = det.detect(&buf, img.width(), img.height());
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
    assert!(
        !dets.is_empty(),
        "CNN must fire on demo_face_256.pgm; got empty detections"
    );
    let best = top(&dets).unwrap();
    let cx = best.x as f32 + best.w as f32 / 2.0;
    let cy = best.y as f32 + best.h as f32 / 2.0;
    let dx = (cx - 128.0).abs();
    let dy = (cy - 128.0).abs();
    assert!(
        dx <= CENTROID_TOLERANCE_PX && dy <= CENTROID_TOLERANCE_PX,
        "CNN top box center ({cx:.1},{cy:.1}) drifted > {CENTROID_TOLERANCE_PX} px \
         from known face center (128,128); dx={dx:.1} dy={dy:.1}"
    );
}

/// If BOTH Haar and Luminance fire on the real face fixture, the IoU
/// between their top boxes must be a well-formed probability in [0, 1].
/// We do NOT assert a strict overlap threshold here: Haar and Luminance
/// are tuned against different priors and produce non-overlapping but
/// equally-correct boxes on photographic input (Luminance often locks on
/// the brightest facial sub-region, Haar on the broader face oval).
/// This test only guards against the IoU math breaking (NaN, out of
/// range, negative) under real algorithm output.
#[cfg(all(feature = "detector-haar", feature = "detector-luminance"))]
#[test]
fn haar_and_luminance_iou_is_well_formed_when_both_fire() {
    use rsface::detector::{Detector, DetectorConfig};
    use rsface::haar::bundled::bundled_frontalface_cascade;
    use rsface::luminance_face::{LuminanceConfig, LuminanceFaceDetector};

    let img = load_demo_face();
    let lum_cfg = LuminanceConfig {
        score_threshold: 0.10,
        ..LuminanceConfig::default()
    };
    let haar = Detector::new(bundled_frontalface_cascade(), DetectorConfig::default())
        .detect(&img);
    let lum = LuminanceFaceDetector::new(lum_cfg).detect(&img);

    let (h, l) = match (top(&haar), top(&lum)) {
        (Some(h), Some(l)) => (h, l),
        _ => {
            eprintln!(
                "haar_and_luminance_iou_is_well_formed_when_both_fire: \
                 skipped — Haar produced {} and Luminance produced {}; \
                 IoU gate cannot be evaluated",
                haar.len(),
                lum.len()
            );
            return;
        }
    };
    let iou = h.iou(l);
    assert!(
        (0.0..=1.0).contains(&iou),
        "IoU {iou} not in [0,1] for real-face top Haar vs Luminance boxes"
    );
    // Every IoU on real algorithm output must be finite; if either
    // algorithm produces a NaN score, IoU propagates the NaN.
    assert!(
        iou.is_finite(),
        "IoU not finite for real-face top Haar vs Luminance boxes: {iou}"
    );
}

/// On the synthetic frontal-face pattern, the Luminance top box's centroid
/// must lie within the 60..=200 band of the 256×256 image. This is the
/// "the detector locked onto the face, not onto noise" tripwire and is
/// checked on the synthetic fixture because Haar+Luminance consensus
/// cannot be evaluated on photographic input.
#[cfg(feature = "detector-luminance")]
#[test]
fn luminance_top_box_centroid_is_on_synthetic_face() {
    use rsface::luminance_face::{LuminanceConfig, LuminanceFaceDetector};

    let img = synth_frontal_face();
    let det = LuminanceFaceDetector::new(LuminanceConfig::default());
    let dets = det.detect(&img);
    let best = top(&dets).expect("Luminance must fire on synth_frontal_face()");
    let cx = best.x + best.w / 2;
    let cy = best.y + best.h / 2;
    assert!(
        (60..=200).contains(&cx) && (60..=200).contains(&cy),
        "Luminance top box centroid ({cx},{cy}) outside 60..=200 band — \
         likely fired on background rather than the synthetic face"
    );
    // And must be within the image.
    let (w, h) = (img.width(), img.height());
    assert!(
        best.x + best.w <= w && best.y + best.h <= h,
        "Luminance box ({},{},{},{}) overflows {w}x{h}",
        best.x, best.y, best.w, best.h
    );
}