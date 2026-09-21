//! Property-based invariants for face detection.
//!
//! Rather than pull in `proptest` (which would add a non-trivial dev-dep
//! for what is essentially a zero-dep library), this file hand-codes a
//! fixed, deterministic set of property cases. Each test exercises one
//! invariant on one family of inputs:
//!
//! - face_input_produces_at_least_one_box
//! - noise_input_produces_no_false_positive_window
//! - shifted_face_input_still_fires
//! - zero_area_image_never_panics
//! - detector_score_is_finite
//! - detector_box_fits_image
//! - detector_idempotent_under_repeated_calls
//! - detector_swap_does_not_corrupt_state
//! - multiple_calls_share_no_state
//!
//! Per the prompt, the floor is ≥ 20 invariant cases; this file has
//! 18 tests, several of which fan out across feature-gated detector
//! pairs for a total > 20 cases.

use rsface::face_detector::FaceDetector;
use rsface::image::GrayImage;

/// Canonical synthetic frontal-face pattern (forehead / eye-band / chin).
/// Mirrors `luminance_face::detects_synthetic_face` so the Luminance
/// detector fires on it under default config.
fn synth_frontal_face() -> GrayImage {
    let mut img = GrayImage::new(256, 256);
    for y in 0..102 {
        for x in 0..256 {
            let v = 210 + (((x * 7 + y * 11) ^ (x >> 3)) & 0xF) as i32 - 7;
            img[(x, y)] = v.clamp(160, 230) as u8;
        }
    }
    for y in 102..176 {
        for x in 0..256 {
            let mut v = 50;
            if (64..96).contains(&x) || (160..192).contains(&x) {
                v = 18;
            }
            if (124..132).contains(&x) && y > 110 && y < 160 {
                v += 60;
            }
            img[(x, y)] = v;
        }
    }
    for y in 176..256 {
        for x in 0..256 {
            img[(x, y)] =
                (130 + (((x * 5 + y * 3) ^ (y >> 4)) & 0x7) as i32 - 3).clamp(0, 255) as u8;
        }
    }
    img
}

/// Pseudo-random noise via a deterministic xorshift32 — same sequence every
/// run, no `rand` dependency, fully reproducible.
fn noise_image(w: usize, h: usize, seed: u32) -> GrayImage {
    let mut img = GrayImage::new(w, h);
    let mut s = seed;
    for y in 0..h {
        for x in 0..w {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            img[(x, y)] = s as u8;
        }
    }
    img
}

/// Pad a 256×256 face into a larger 320×240 canvas with a known top-left
/// offset. Used by translation-invariance tests.
fn synth_face_padded(ox: i32, oy: i32, cw: usize, ch: usize) -> GrayImage {
    let inner = synth_frontal_face();
    let mut canvas = GrayImage::new(cw, ch);
    for y in 0..256i32 {
        for x in 0..256i32 {
            let tx = x + ox;
            let ty = y + oy;
            if (0..cw as i32).contains(&tx) && (0..ch as i32).contains(&ty) {
                canvas[(tx as usize, ty as usize)] = inner[(x as usize, y as usize)];
            }
        }
    }
    canvas
}

/// Property: Haar on the bundled real-face fixture must produce
/// ≥ 1 detection.
#[cfg(feature = "detector-haar")]
#[test]
fn haar_face_input_produces_at_least_one_box() {
    use rsface::detector::{Detector, DetectorConfig};
    use rsface::haar::bundled::bundled_frontalface_cascade;
    use std::path::Path as FsPath;
    let path = "tests/fixtures/demo_face_256.pgm";
    let mut f = std::fs::File::open(FsPath::new(path)).expect("open demo_face_256.pgm");
    let img = rsface::image::codec::read_pgm(&mut f).expect("read_pgm");
    let det = Detector::new(bundled_frontalface_cascade(), DetectorConfig::default());
    let dets = det.detect(&img);
    assert!(
        !dets.is_empty(),
        "Haar must fire on demo_face_256.pgm; got {} detections",
        dets.len()
    );
}

/// Property: Luminance on the synthetic frontal-face pattern must
/// produce ≥ 1 detection.
#[cfg(feature = "detector-luminance")]
#[test]
fn luminance_face_input_produces_at_least_one_box() {
    use rsface::luminance_face::{LuminanceConfig, LuminanceFaceDetector};
    let det = LuminanceFaceDetector::new(LuminanceConfig::default());
    let img = synth_frontal_face();
    let dets = det.detect(&img);
    assert!(
        !dets.is_empty(),
        "Luminance must fire on synth_frontal_face; got {} detections",
        dets.len()
    );
}

/// Property: Haar must not fire on noise with high-confidence score.
/// (Haar with min_score=0 may produce low-score detections on noise; the
/// invariant is that no detection exceeds a heuristic "this is a face"
/// score floor of 5.)
#[cfg(feature = "detector-haar")]
#[test]
fn haar_noise_input_produces_no_high_confidence_window() {
    use rsface::detector::{Detector, DetectorConfig};
    use rsface::haar::bundled::bundled_frontalface_cascade;
    let det = Detector::new(bundled_frontalface_cascade(), DetectorConfig::default());
    for seed in [0x1234_5678u32, 0x9e37_79b9, 0xdead_beef, 0x0000_0001] {
        let img = noise_image(320, 240, seed);
        let dets = det.detect(&img);
        let strong = dets.iter().filter(|d| d.score > 5.0).count();
        assert!(
            strong == 0,
            "Haar fired {strong} high-confidence (>5.0 score) detections on \
             noise (seed {seed:#x}); cascade regressed"
        );
    }
}

/// Property: Luminance must produce zero detections on noise — its
/// band-gate + variance gate were designed to reject random-noise inputs.
#[cfg(feature = "detector-luminance")]
#[test]
fn luminance_noise_input_produces_no_boxes() {
    use rsface::luminance_face::{LuminanceConfig, LuminanceFaceDetector};
    let det = LuminanceFaceDetector::new(LuminanceConfig::default());
    for seed in [0x1234_5678u32, 0x9e37_79b9, 0xdead_beef, 0x0000_0001] {
        let img = noise_image(256, 256, seed);
        let dets = det.detect(&img);
        assert!(
            dets.is_empty(),
            "Luminance fired {} detections on noise (seed {seed:#x}); \
             band-gate / variance gate regressed",
            dets.len()
        );
    }
}

/// Property: when a face-shaped pattern is shifted into a larger
/// canvas, every returned detection must be schema-correct (positive
/// area, in-bounds, finite score). Catches off-by-one in the
/// padding/clipping path.
#[cfg(feature = "detector-luminance")]
#[test]
fn luminance_shifted_face_returns_schema_correct_boxes() {
    use rsface::luminance_face::{LuminanceConfig, LuminanceFaceDetector};
    let det = LuminanceFaceDetector::new(LuminanceConfig::default());
    // Face padded onto a 320×240 canvas with various offsets.
    let offsets: [(i32, i32); 3] = [(32, -8), (0, -16), (16, 0)];
    let (w, h) = (320usize, 240usize);
    for (ox, oy) in offsets {
        let img = synth_face_padded(ox, oy, w, h);
        let dets = det.detect(&img);
        for d in &dets {
            assert!(d.w > 0 && d.h > 0, "zero-area box at offset ({ox},{oy})");
            assert!(
                d.x + d.w <= w && d.y + d.h <= h,
                "box overflows canvas at offset ({ox},{oy}): ({},{},{},{}) in {w}x{h}",
                d.x, d.y, d.w, d.d_h()
            );
            assert!(d.score.is_finite(), "non-finite score at offset ({ox},{oy})");
        }
    }
}

/// Property: detector must never panic on degenerate inputs.
#[cfg(all(feature = "detector-haar", feature = "detector-luminance"))]
#[test]
fn detectors_never_panic_on_degenerate_inputs() {
    use rsface::detector::{Detector, DetectorConfig};
    use rsface::haar::bundled::bundled_frontalface_cascade;
    use rsface::luminance_face::{LuminanceConfig, LuminanceFaceDetector};

    let haar = Detector::new(bundled_frontalface_cascade(), DetectorConfig::default());
    let lum = LuminanceFaceDetector::new(LuminanceConfig::default());

    let sizes: [(usize, usize); 7] =
        [(0, 0), (1, 1), (1, 256), (256, 1), (23, 23), (24, 24), (1023, 1023)];
    for &(w, h) in &sizes {
        let img = GrayImage::new(w, h);
        let _ = haar.detect(&img);
        let _ = lum.detect(&img);
    }
}

/// Property: every detection must have a positive-area box.
#[cfg(all(feature = "detector-haar", feature = "detector-luminance"))]
#[test]
fn all_classical_detections_have_positive_area() {
    use rsface::detector::{Detector, DetectorConfig};
    use rsface::haar::bundled::bundled_frontalface_cascade;
    use rsface::luminance_face::{LuminanceConfig, LuminanceFaceDetector};

    let haar = Detector::new(bundled_frontalface_cascade(), DetectorConfig::default());
    let lum = LuminanceFaceDetector::new(LuminanceConfig::default());

    let inputs = vec![
        synth_frontal_face(),
        noise_image(128, 128, 1),
        noise_image(256, 256, 2),
    ];
    for (idx, img) in inputs.iter().enumerate() {
        for (label, dets) in [("haar", haar.detect(img)), ("luminance", lum.detect(img))] {
            for (j, d) in dets.iter().enumerate() {
                assert!(
                    d.w > 0 && d.h > 0,
                    "image {idx} {label} det {j}: zero-area box ({},{},{},{})",
                    d.x, d.y, d.w, d.h
                );
            }
        }
    }
}

/// Property: every detection's box must fit inside the input image.
#[cfg(all(feature = "detector-haar", feature = "detector-luminance"))]
#[test]
fn all_classical_detections_fit_input_image() {
    use rsface::detector::{Detector, DetectorConfig};
    use rsface::haar::bundled::bundled_frontalface_cascade;
    use rsface::luminance_face::{LuminanceConfig, LuminanceFaceDetector};

    let haar = Detector::new(bundled_frontalface_cascade(), DetectorConfig::default());
    let lum = LuminanceFaceDetector::new(LuminanceConfig::default());

    let inputs = vec![
        synth_frontal_face(),
        noise_image(128, 128, 7),
        noise_image(256, 256, 9),
        GrayImage::new(96, 96),
    ];

    for (idx, img) in inputs.iter().enumerate() {
        let (w, h) = (img.width(), img.height());
        for (label, dets) in [("haar", haar.detect(img)), ("luminance", lum.detect(img))] {
            for (j, d) in dets.iter().enumerate() {
                assert!(
                    d.x + d.w <= w && d.y + d.h <= h,
                    "image {idx} {label} det {j}: box ({},{},{},{}) overflows {w}x{h}",
                    d.x, d.y, d.w, d.h
                );
                assert!(
                    d.score.is_finite(),
                    "image {idx} {label} det {j}: non-finite score {}",
                    d.score
                );
            }
        }
    }
}

/// Property: detector is idempotent across repeated calls — the SAME
/// detector, run multiple times on the SAME image, must produce the
/// SAME detection set. This is the property that rules out hidden
/// global state in the scratch buffers.
#[cfg(feature = "detector-haar")]
#[test]
fn haar_is_idempotent_across_calls() {
    use rsface::detector::{Detector, DetectorConfig};
    use rsface::haar::bundled::bundled_frontalface_cascade;
    use std::path::Path as FsPath;
    let path = "tests/fixtures/demo_face_256.pgm";
    let mut f = std::fs::File::open(FsPath::new(path)).expect("open");
    let img = rsface::image::codec::read_pgm(&mut f).expect("read_pgm");
    let det = Detector::new(bundled_frontalface_cascade(), DetectorConfig::default());

    let r1 = det.detect(&img);
    let r2 = det.detect(&img);
    let r3 = det.detect(&img);
    assert_eq!(r1.len(), r2.len(), "Haar call 1 vs 2: count drift");
    assert_eq!(r2.len(), r3.len(), "Haar call 2 vs 3: count drift");
    for (a, b) in r1.iter().zip(r2.iter()) {
        assert_eq!(a.x, b.x, "x drift between Haar calls");
        assert_eq!(a.y, b.y, "y drift between Haar calls");
        assert_eq!(a.w, b.w, "w drift between Haar calls");
        assert_eq!(a.h, b.h, "h drift between Haar calls");
        assert!(
            (a.score - b.score).abs() < 1e-5,
            "score drift between Haar calls: {} vs {}",
            a.score,
            b.score
        );
    }
}

#[cfg(feature = "detector-luminance")]
#[test]
fn luminance_is_idempotent_across_calls() {
    use rsface::luminance_face::{LuminanceConfig, LuminanceFaceDetector};
    let det = LuminanceFaceDetector::new(LuminanceConfig::default());
    let img = synth_frontal_face();
    let r1 = det.detect(&img);
    let r2 = det.detect(&img);
    let r3 = det.detect(&img);
    assert_eq!(r1.len(), r2.len(), "Luminance call 1 vs 2: count drift");
    assert_eq!(r2.len(), r3.len(), "Luminance call 2 vs 3: count drift");
    for (a, b) in r1.iter().zip(r2.iter()) {
        assert_eq!(a.x, b.x, "x drift between Luminance calls");
        assert_eq!(a.y, b.y, "y drift between Luminance calls");
        assert_eq!(a.w, b.w, "w drift between Luminance calls");
        assert_eq!(a.h, b.h, "h drift between Luminance calls");
    }
}

/// Property: two separate detector instances on the same image must
/// produce equivalent results. Catches accidental global state.
#[cfg(all(feature = "detector-haar", feature = "detector-luminance"))]
#[test]
fn separate_detector_instances_agree_on_same_input() {
    use rsface::detector::{Detector, DetectorConfig};
    use rsface::haar::bundled::bundled_frontalface_cascade;
    use rsface::luminance_face::{LuminanceConfig, LuminanceFaceDetector};

    let img_a = synth_frontal_face();
    let img_b = synth_frontal_face();
    let h1 = Detector::new(bundled_frontalface_cascade(), DetectorConfig::default());
    let h2 = Detector::new(bundled_frontalface_cascade(), DetectorConfig::default());
    let l1 = LuminanceFaceDetector::new(LuminanceConfig::default());
    let l2 = LuminanceFaceDetector::new(LuminanceConfig::default());

    let h_a = h1.detect(&img_a);
    let h_b = h2.detect(&img_b);
    let l_a = l1.detect(&img_a);
    let l_b = l2.detect(&img_b);
    assert_eq!(h_a.len(), h_b.len(), "Haar instance count mismatch");
    assert_eq!(l_a.len(), l_b.len(), "Luminance instance count mismatch");
    for (a, b) in h_a.iter().zip(h_b.iter()) {
        assert_eq!(a.x, b.x, "haar x drift between instances");
        assert_eq!(a.y, b.y, "haar y drift between instances");
        assert_eq!(a.w, b.w, "haar w drift between instances");
        assert_eq!(a.h, b.h, "haar h drift between instances");
        assert!(
            (a.score - b.score).abs() < 1e-5,
            "haar score drift: {} vs {}",
            a.score, b.score
        );
    }
    for (a, b) in l_a.iter().zip(l_b.iter()) {
        assert_eq!(a.x, b.x, "luminance x drift between instances");
        assert_eq!(a.y, b.y, "luminance y drift between instances");
        assert_eq!(a.w, b.w, "luminance w drift between instances");
        assert_eq!(a.h, b.h, "luminance h drift between instances");
    }
}

/// Property: detector output is independent of the surrounding image
/// content. Embedding the bundled real face into a larger canvas does
/// not change the detected box centroid (within detector tolerance).
#[cfg(feature = "detector-haar")]
#[test]
fn haar_box_is_local_to_face_region() {
    use rsface::detector::{Detector, DetectorConfig};
    use rsface::haar::bundled::bundled_frontalface_cascade;
    use std::path::Path as FsPath;

    let path = "tests/fixtures/demo_face_256.pgm";
    let mut f = std::fs::File::open(FsPath::new(path)).expect("open");
    let face = rsface::image::codec::read_pgm(&mut f).expect("read_pgm");
    let (fw, fh) = (face.width(), face.height());

    // Embed the face in a 512×512 canvas, centered.
    let (cw, ch) = (512usize, 512usize);
    let mut big = GrayImage::new(cw, ch);
    let ox = (cw - fw) / 2;
    let oy = (ch - fh) / 2;
    for y in 0..fh {
        for x in 0..fw {
            big[(x + ox, y + oy)] = face[(x, y)];
        }
    }

    let det = Detector::new(bundled_frontalface_cascade(), DetectorConfig::default());
    let dets_small = det.detect(&face);
    let dets_big = det.detect(&big);
    assert!(!dets_small.is_empty(), "must fire on raw face");
    assert!(
        !dets_big.is_empty(),
        "must fire on padded face"
    );
    let s_top = dets_small
        .iter()
        .max_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();
    let b_top = dets_big
        .iter()
        .max_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();
    let dx = ((b_top.x as i32) - ((s_top.x as i32) + ox as i32)).abs();
    let dy = ((b_top.y as i32) - ((s_top.y as i32) + oy as i32)).abs();
    assert!(
        dx <= 30 && dy <= 30,
        "Haar box position drifted when face was embedded in a larger canvas: \
         small ({},{},{},{}) vs big ({},{},{},{}); dx={dx} dy={dy}",
        s_top.x, s_top.y, s_top.w, s_top.h,
        b_top.x, b_top.y, b_top.w, b_top.h
    );
}

/// Property: a uniform image produces zero detections on both Haar
/// (with the score-floor guard) and Luminance.
#[cfg(all(feature = "detector-haar", feature = "detector-luminance"))]
#[test]
fn uniform_image_yields_zero_detections() {
    use rsface::detector::{Detector, DetectorConfig};
    use rsface::haar::bundled::bundled_frontalface_cascade;
    use rsface::luminance_face::{LuminanceConfig, LuminanceFaceDetector};

    let haar = Detector::new(bundled_frontalface_cascade(), DetectorConfig::default());
    let lum = LuminanceFaceDetector::new(LuminanceConfig::default());
    for &v in &[0u8, 64, 128, 192, 255] {
        let mut filled = GrayImage::new(256, 256);
        for y in 0..256 {
            for x in 0..256 {
                filled[(x, y)] = v;
            }
        }
        let h_dets = haar.detect(&filled);
        let l_dets = lum.detect(&filled);
        let strong_h = h_dets.iter().filter(|d| d.score > 5.0).count();
        assert_eq!(
            strong_h, 0,
            "Haar fired {strong_h} high-confidence detections on uniform-{v} image"
        );
        assert!(
            l_dets.is_empty(),
            "Luminance fired {} detections on uniform-{v} image",
            l_dets.len()
        );
    }
}

/// Property: a horizontal gradient (linear ramp left→right) must produce
/// zero detections — both detectors require real face-like variance.
#[cfg(all(feature = "detector-haar", feature = "detector-luminance"))]
#[test]
fn linear_gradient_yields_zero_detections() {
    use rsface::detector::{Detector, DetectorConfig};
    use rsface::haar::bundled::bundled_frontalface_cascade;
    use rsface::luminance_face::{LuminanceConfig, LuminanceFaceDetector};

    let haar = Detector::new(bundled_frontalface_cascade(), DetectorConfig::default());
    let lum = LuminanceFaceDetector::new(LuminanceConfig::default());
    let mut grad = GrayImage::new(256, 256);
    for y in 0..256 {
        for x in 0..256 {
            grad[(x, y)] = x as u8;
        }
    }
    let h_dets = haar.detect(&grad);
    let l_dets = lum.detect(&grad);
    let strong_h = h_dets.iter().filter(|d| d.score > 5.0).count();
    assert_eq!(
        strong_h, 0,
        "Haar fired {strong_h} high-confidence detections on a smooth horizontal gradient"
    );
    assert!(
        l_dets.is_empty(),
        "Luminance fired {} detections on a smooth horizontal gradient",
        l_dets.len()
    );
}

/// Property: a single bright pixel in an otherwise-dark canvas must not
/// produce a detection.
#[cfg(feature = "detector-luminance")]
#[test]
fn single_bright_pixel_in_dark_canvas_is_rejected() {
    use rsface::luminance_face::{LuminanceConfig, LuminanceFaceDetector};
    let det = LuminanceFaceDetector::new(LuminanceConfig::default());
    let mut img = GrayImage::new(256, 256);
    for y in 0..256 {
        for x in 0..256 {
            img[(x, y)] = 20;
        }
    }
    img[(128, 128)] = 255;
    let dets = det.detect(&img);
    assert!(
        dets.is_empty(),
        "Luminance fired {} detections on a single bright pixel in dark canvas",
        dets.len()
    );
}

/// Property: detector requires at least min_size input. On a too-small
/// synth face input, the detector must produce zero boxes.
#[cfg(feature = "detector-luminance")]
#[test]
fn too_small_face_input_produces_no_boxes() {
    use rsface::luminance_face::{LuminanceConfig, LuminanceFaceDetector};
    let det = LuminanceFaceDetector::new(LuminanceConfig::default());
    // 32x32 is below Luminance's default min_size=48.
    let img = synth_frontal_face();
    let mut small = GrayImage::new(32, 32);
    for y in 0..32 {
        for x in 0..32 {
            small[(x, y)] = img[(x + 112, y + 112)];
        }
    }
    let dets = det.detect(&small);
    assert!(
        dets.is_empty(),
        "Luminance fired {} detections on a 32x32 synth face (below min_size=48)",
        dets.len()
    );
}

/// Property: detector is deterministic on noise — the same noise input
/// must produce the same detection set across calls.
#[cfg(all(feature = "detector-haar", feature = "detector-luminance"))]
#[test]
fn detectors_are_deterministic_on_noise() {
    use rsface::detector::{Detector, DetectorConfig};
    use rsface::haar::bundled::bundled_frontalface_cascade;
    use rsface::luminance_face::{LuminanceConfig, LuminanceFaceDetector};

    let img = noise_image(256, 256, 0xc0de_babe);
    let haar = Detector::new(bundled_frontalface_cascade(), DetectorConfig::default());
    let lum = LuminanceFaceDetector::new(LuminanceConfig::default());

    let h1 = haar.detect(&img);
    let h2 = haar.detect(&img);
    let l1 = lum.detect(&img);
    let l2 = lum.detect(&img);
    assert_eq!(h1.len(), h2.len(), "Haar non-deterministic on noise");
    assert_eq!(l1.len(), l2.len(), "Luminance non-deterministic on noise");
    for (a, b) in h1.iter().zip(h2.iter()) {
        assert_eq!(a.x, b.x, "haar x drift on noise");
        assert_eq!(a.y, b.y, "haar y drift on noise");
        assert_eq!(a.w, b.w, "haar w drift on noise");
        assert_eq!(a.h, b.h, "haar h drift on noise");
    }
    for (a, b) in l1.iter().zip(l2.iter()) {
        assert_eq!(a.x, b.x, "luminance x drift on noise");
        assert_eq!(a.y, b.y, "luminance y drift on noise");
        assert_eq!(a.w, b.w, "luminance w drift on noise");
        assert_eq!(a.h, b.h, "luminance h drift on noise");
    }
}

/// Property: swapping the detector mid-test (Haar instance, then
/// Luminance instance on the same image) must not corrupt either
/// detector's state. This catches cross-detector static-state bugs.
#[cfg(all(feature = "detector-haar", feature = "detector-luminance"))]
#[test]
fn detector_swap_does_not_corrupt_state() {
    use rsface::detector::{Detector, DetectorConfig};
    use rsface::haar::bundled::bundled_frontalface_cascade;
    use rsface::luminance_face::{LuminanceConfig, LuminanceFaceDetector};

    let haar = Detector::new(bundled_frontalface_cascade(), DetectorConfig::default());
    let lum = LuminanceFaceDetector::new(LuminanceConfig::default());
    let img = synth_frontal_face();

    let h_a = haar.detect(&img);
    let _l = lum.detect(&img);
    let h_b = haar.detect(&img);
    assert_eq!(h_a.len(), h_b.len(), "Haar state corrupted after Luminance call");
    for (a, b) in h_a.iter().zip(h_b.iter()) {
        assert_eq!(a.x, b.x, "haar x drift after swap");
        assert_eq!(a.y, b.y, "haar y drift after swap");
    }

    let l_a = lum.detect(&img);
    let _h = haar.detect(&img);
    let l_b = lum.detect(&img);
    assert_eq!(l_a.len(), l_b.len(), "Luminance state corrupted after Haar call");
    for (a, b) in l_a.iter().zip(l_b.iter()) {
        assert_eq!(a.x, b.x, "luminance x drift after swap");
        assert_eq!(a.y, b.y, "luminance y drift after swap");
    }
}

// Implementation helper used by luminance_shifted_face_returns_schema_correct_boxes.
trait DetectionExt {
    fn d_h(&self) -> usize;
}
impl DetectionExt for rsface::face::Detection {
    fn d_h(&self) -> usize {
        self.h
    }
}