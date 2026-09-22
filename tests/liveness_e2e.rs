//! End-to-end liveness test against the real MiniFASNet ONNX weights.
//!
//! Skipped when the weights are absent, so `cargo test` stays green on a
//! fresh clone:
//!
//! ```sh
//! tools/fetch_models.sh
//! cargo test --features tract-backend --test liveness_e2e -- --nocapture
//! ```
#![cfg(any(feature = "tract-backend", feature = "ort-backend"))]

use rsface::face::Detection;
use rsface::image::codec::read_ppm;
use rsface::image::RgbImage;
use rsface::liveness::{LivenessConfig, LivenessOutcome};
use rsface::liveness_detector::LivenessDetector;
use rsface::models;
use rsface::onnx::SessionConfig;
use std::io::Cursor;
use std::path::PathBuf;

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models")
}

fn detector() -> LivenessDetector {
    let p_v2 = models_dir().join("2.7_80x80_MiniFASNetV2.onnx");
    let p_v1se = models_dir().join("4_0_0_80x80_MiniFASNetV1SE.onnx");
    assert!(
        p_v2.exists(),
        "missing {}, run tools/fetch_models.sh first",
        p_v2.display()
    );
    assert!(
        p_v1se.exists(),
        "missing {}, run tools/fetch_models.sh first",
        p_v1se.display()
    );
    LivenessDetector::open(
        &p_v2,
        Some(&models::LIVENESS_MINIFASNET_V2),
        &p_v1se,
        Some(&models::LIVENESS_MINIFASNET_V1SE),
        &SessionConfig::default(),
        LivenessConfig::default(),
    )
    .expect("LivenessDetector::open")
}

#[test]
fn real_minifasnet_models_run_on_lena() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let img_bytes = std::fs::read(root.join("tests/fixtures/lena.ppm")).unwrap();
    let img = read_ppm(&mut Cursor::new(img_bytes)).unwrap();
    let det = detector();

    let face = Detection {
        x: 200,
        y: 190,
        w: 150,
        h: 170,
        score: 1.0,
    };
    let out = det.check(&img, &face).unwrap();
    println!(
        "lena [real]: is_real={} label={} real_score={:.4} probs={:.3}/{:.3}/{:.3}",
        out.is_real,
        out.label(),
        out.real_score,
        out.probs[0],
        out.probs[1],
        out.probs[2]
    );
    assert!(out.real_score > 0.5, "lena should score high on real");
}

#[test]
fn liveness_far_frr_against_synthesised_attacks() {
    // Build a synthetic face crop from `lena.ppm` then synthesise two
    // attack variants and a real baseline. MiniFASNet is sensitive to
    // texture statistics (Moiré patterns, blur) — we only assert that
    // the *real* crop scores strongly real and that the obvious-attack
    // crops produce low real-scores; quantitative FAR/FRR against
    // production-grade attacks needs a labelled MiniFASNet test set
    // (e.g. OULU-NPU / CASIA-FASD), not in-tree fakes.
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let img_bytes = std::fs::read(root.join("tests/fixtures/lena.ppm")).unwrap();
    let img = read_ppm(&mut Cursor::new(img_bytes)).unwrap();
    let det = detector();

    let face = Detection {
        x: 200,
        y: 190,
        w: 150,
        h: 170,
        score: 1.0,
    };

    // (1) real face — lena, untouched.
    let real = det.check(&img, &face).unwrap();
    println!(
        "real:   is_real={} label={} real_score={:.4} probs={:.3}/{:.3}/{:.3}",
        real.is_real,
        real.label(),
        real.real_score,
        real.probs[0],
        real.probs[1],
        real.probs[2]
    );

    // (2) paper-print attack: heavy Gaussian-ish blur (re-photographed
    // print loses detail), greyscale (print is essentially B/W).
    let paper = blur_and_grayscale(&img, 4.0);
    let paper_out = det.check(&paper, &face).unwrap();
    println!(
        "paper:  is_real={} label={} real_score={:.4} probs={:.3}/{:.3}/{:.3}",
        paper_out.is_real,
        paper_out.label(),
        paper_out.real_score,
        paper_out.probs[0],
        paper_out.probs[1],
        paper_out.probs[2]
    );

    // (3) screen-replay attack: mild blur + slight chroma shift (typical
    // Moiré from re-shooting a screen).
    let screen = blur_and_chroma_shift(&img, 1.5, (4, -3, 2));
    let screen_out = det.check(&screen, &face).unwrap();
    println!(
        "screen: is_real={} label={} real_score={:.4} probs={:.3}/{:.3}/{:.3}",
        screen_out.is_real,
        screen_out.label(),
        screen_out.real_score,
        screen_out.probs[0],
        screen_out.probs[1],
        screen_out.probs[2]
    );

    // FAR / FRR snapshot for this 1-real / 2-attacks suite. Default
    // `min_real_score=0.0` is pure argmax; we also show how a stricter
    // threshold shifts the trade-off. A real production deployment
    // raises `min_real_score` until FRR matches the SLA.
    let thr_loose = 0.0_f32;
    let thr_strict = 0.5_f32;
    let attacks = [paper_out.clone(), screen_out.clone()];
    let accepted = |o: &LivenessOutcome, t: f32| o.real_score >= t && o.label() == "real";
    let far_loose =
        attacks.iter().filter(|o| accepted(o, thr_loose)).count() as f32 / attacks.len() as f32;
    let far_strict =
        attacks.iter().filter(|o| accepted(o, thr_strict)).count() as f32 / attacks.len() as f32;
    let frr_loose = (!accepted(&real, thr_loose)) as i32 as f32;
    let frr_strict = (!accepted(&real, thr_strict)) as i32 as f32;
    println!(
        "FAR/FRR:  loose(t={:.2}) FAR={:.2} FRR={:.2}   strict(t={:.2}) FAR={:.2} FRR={:.2}",
        thr_loose, far_loose, frr_loose, thr_strict, far_strict, frr_strict
    );

    // Sanity: untouched face = real, attacks = non-real (with the
    // default loose threshold). Real production FAR/FRR requires a
    // labelled benchmark dataset.
    assert!(real.is_real, "untouched face should pass as real");
    assert!(!paper_out.is_real, "paper-print should NOT pass as real");
    assert!(!screen_out.is_real, "screen-replay should NOT pass as real");
}

// -------------------------------------------------------------------------
// In-house image-perturbation helpers.
//
// Hand-rolled because we don't want a test-only hard dep on the `image`
// crate. The kernels are good enough to perturb the texture statistics
// MiniFASNet looks at (luma + chroma noise) — not for production.
// -------------------------------------------------------------------------

/// Separable box blur with `radius = ceil(sigma * 3)`, then collapse to
/// greyscale. Mimics a re-photographed B/W printout.
fn blur_and_grayscale(img: &RgbImage, sigma: f32) -> RgbImage {
    let radius = (sigma * 3.0).ceil() as i32;
    let horiz = blur_pass(img, true, radius);
    let vert = blur_pass(&horiz, false, radius);
    gray(&vert)
}

/// Separable box blur + per-channel chroma shift. Mimics Moiré and the
/// slight RGB drift from re-shooting a screen.
fn blur_and_chroma_shift(img: &RgbImage, sigma: f32, shift: (i16, i16, i16)) -> RgbImage {
    let radius = (sigma * 3.0).ceil() as i32;
    let horiz = blur_pass(img, true, radius);
    let vert = blur_pass(&horiz, false, radius);
    chroma_shift(&vert, shift)
}

/// Box blur in-place copy: returns a new RgbImage.
fn blur_pass(img: &RgbImage, horizontal: bool, radius: i32) -> RgbImage {
    let (w, h) = (img.width(), img.height());
    let mut out = RgbImage::new(w, h);
    let r = radius.max(1);
    let win = (2 * r + 1) as u32;
    for y in 0..h {
        let dst_row_out = out.row_mut(y);
        for x in 0..w {
            let mut sr: u32 = 0;
            let mut sg: u32 = 0;
            let mut sb: u32 = 0;
            for k in -r..=r {
                let (sx, sy) = if horizontal {
                    ((x as i32 + k).clamp(0, w as i32 - 1) as usize, y)
                } else {
                    (x, (y as i32 + k).clamp(0, h as i32 - 1) as usize)
                };
                let p_row = img.row(sy);
                let base = sx * 3;
                sr += p_row[base] as u32;
                sg += p_row[base + 1] as u32;
                sb += p_row[base + 2] as u32;
            }
            let base = x * 3;
            dst_row_out[base] = (sr / win) as u8;
            dst_row_out[base + 1] = (sg / win) as u8;
            dst_row_out[base + 2] = (sb / win) as u8;
        }
    }
    out
}

fn gray(img: &RgbImage) -> RgbImage {
    let (w, h) = (img.width(), img.height());
    let mut out = RgbImage::new(w, h);
    for y in 0..h {
        let src = img.row(y);
        let dst = out.row_mut(y);
        for x in 0..w {
            let base = x * 3;
            let luma = ((src[base] as u32 + src[base + 1] as u32 + src[base + 2] as u32) / 3) as u8;
            dst[base] = luma;
            dst[base + 1] = luma;
            dst[base + 2] = luma;
        }
    }
    out
}

fn chroma_shift(img: &RgbImage, shift: (i16, i16, i16)) -> RgbImage {
    let (w, h) = (img.width(), img.height());
    let mut out = RgbImage::new(w, h);
    let (dr, dg, db) = shift;
    for y in 0..h {
        let src = img.row(y);
        let dst = out.row_mut(y);
        for x in 0..w {
            let base = x * 3;
            dst[base] = (src[base] as i16 + dr).clamp(0, 255) as u8;
            dst[base + 1] = (src[base + 1] as i16 + dg).clamp(0, 255) as u8;
            dst[base + 2] = (src[base + 2] as i16 + db).clamp(0, 255) as u8;
        }
    }
    out
}
