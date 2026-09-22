//! Real-model smoke test for the platform liveness wrapper.
//!
//! Runs only when the platform was built with `--features liveness` and the
//! two MiniFASNet graphs are present in `models/` (fetch via
//! `tools/fetch_models.sh`); otherwise it skips, so a default `cargo test`
//! stays green on a fresh checkout.
#![cfg(feature = "liveness")]

use rsface::face::Detection;
use rsface::image::codec::read_ppm;
use rsface::image::RgbImage;
use rsface_platform::config::Config;
use rsface_platform::liveness::Liveness;
use std::io::Cursor;
use std::path::PathBuf;

#[test]
fn platform_liveness_classifies_lena_as_real() {
    let platform_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = platform_dir.join("..");
    let p_v2 = repo_root.join("models/2.7_80x80_MiniFASNetV2.onnx");
    let p_v1se = repo_root.join("models/4_0_0_80x80_MiniFASNetV1SE.onnx");
    if !p_v2.exists() || !p_v1se.exists() {
        eprintln!("skipping: liveness models absent (run tools/fetch_models.sh)");
        return;
    }

    let mut cfg = Config::from_env();
    cfg.liveness_enabled = true;
    cfg.liveness_models_dir = repo_root.join("models");

    let liveness = Liveness::open(&cfg).expect("Liveness::open returned None with models present");

    let img_bytes = std::fs::read(repo_root.join("tests/fixtures/lena.ppm")).unwrap();
    let rgb: RgbImage = read_ppm(&mut Cursor::new(img_bytes)).unwrap();

    let det = Detection {
        x: 200,
        y: 190,
        w: 150,
        h: 170,
        score: 1.0,
    };
    let verdict = liveness.check(&rgb, &det).expect("liveness verdict");
    println!(
        "platform liveness [lena]: is_real={} label={} real_score={:.4}",
        verdict.is_real, verdict.label, verdict.real_score
    );
    assert!(verdict.is_real, "lena should be classified as real");
    assert!(verdict.real_score > 0.5);
}
