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
use rsface::liveness::LivenessConfig;
use rsface::liveness_detector::LivenessDetector;
use rsface::models;
use rsface::onnx::SessionConfig;
use std::io::Cursor;
use std::path::PathBuf;

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models")
}

#[test]
fn real_minifasnet_models_run_on_lena() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let p_v2 = models_dir().join("2.7_80x80_MiniFASNetV2.onnx");
    let p_v1se = models_dir().join("4_0_0_80x80_MiniFASNetV1SE.onnx");
    if !p_v2.exists() || !p_v1se.exists() {
        eprintln!("SKIP: run tools/fetch_models.sh first");
        return;
    }

    let img_bytes = std::fs::read(root.join("tests/fixtures/lena.ppm")).unwrap();
    let img = read_ppm(&mut Cursor::new(img_bytes)).unwrap();

    let detector = LivenessDetector::open(
        &p_v2,
        Some(&models::LIVENESS_MINIFASNET_V2),
        &p_v1se,
        Some(&models::LIVENESS_MINIFASNET_V1SE),
        &SessionConfig::default(),
        LivenessConfig::default(),
    )
    .unwrap();
    assert_eq!(detector.num_heads(), 2);

    let face = Detection {
        x: 200,
        y: 190,
        w: 150,
        h: 170,
        score: 1.0,
    };
    let out = detector.check(&img, &face).unwrap();
    println!(
        "liveness: is_real={} label={} real_score={:.4} probs={:.3}/{:.3}/{:.3}",
        out.is_real,
        out.label(),
        out.real_score,
        out.probs[0],
        out.probs[1],
        out.probs[2]
    );
}
