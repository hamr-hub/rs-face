//! Real-model smoke test for the platform liveness wrapper.
//!
//! Runs only when the platform was built with `--features liveness` and the
//! two MiniFASNet graphs are present in `models/` (fetch via
//! `tools/fetch_models.sh`); otherwise it skips, so a default `cargo test`
//! stays green on a fresh checkout.
#![cfg(feature = "liveness")]

use rsface::image::codec::read_ppm;
use rsface_platform::config::Config;
use rsface_platform::gallery::GalleryState;
use rsface_platform::jobs::DetectorKind;
use std::io::Cursor;
use std::path::PathBuf;

#[tokio::test]
async fn platform_liveness_enforcement_allows_real_face() {
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
    cfg.liveness_enforce = true;
    cfg.liveness_models_dir = repo_root.join("models");
    cfg.cascade_path = repo_root.join("src/weights/haarcascade_frontalface_default.rfcf");

    let state = GalleryState::load(rsface_platform::persist::Db { pool: None }, &cfg).await;
    assert!(state.has_liveness(), "liveness backend should be loaded");

    let img_bytes = std::fs::read(repo_root.join("tests/fixtures/lena.ppm")).unwrap();
    let rgb = read_ppm(&mut Cursor::new(img_bytes.clone())).unwrap();
    let gray = read_ppm(&mut Cursor::new(img_bytes)).unwrap().to_gray();

    let cascade = rsface::haar::Cascade::load(&cfg.cascade_path)
        .expect("cascade load; point RSFACE_CASCADE at the bundled rfcf");
    let detector = DetectorKind::Haar(rsface::detector::Detector::new(
        cascade,
        rsface::detector::DetectorConfig::default(),
    ));

    let faces = state.recognize(detector, &rgb, &gray, 3).await;
    assert!(!faces.is_empty(), "Haar should find the Lena face");
    for face in &faces {
        let verdict = face.liveness.as_ref().expect("liveness verdict present");
        println!(
            "face block: blocked={} is_real={} label={} real_score={:.4} matches={}",
            face.blocked,
            verdict.is_real,
            verdict.label,
            verdict.real_score,
            face.matches.len()
        );
        assert!(verdict.is_real, "lena should be classified as real");
        assert!(!face.blocked, "enforcement must not block a real face");
    }
}

#[tokio::test]
async fn platform_liveness_enforcement_blocks_print_attack() {
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
    cfg.liveness_enforce = true;
    cfg.liveness_models_dir = repo_root.join("models");
    cfg.cascade_path = repo_root.join("src/weights/haarcascade_frontalface_default.rfcf");

    let state = GalleryState::load(rsface_platform::persist::Db { pool: None }, &cfg).await;

    let img_bytes = std::fs::read(repo_root.join("tests/fixtures/lena.ppm")).unwrap();
    let real_rgb = read_ppm(&mut Cursor::new(img_bytes)).unwrap();
    let attack_rgb = blur_grayscale(&real_rgb, 4);
    let attack_gray = attack_rgb.to_gray();

    let cascade = rsface::haar::Cascade::load(&cfg.cascade_path).unwrap();
    let detector = DetectorKind::Haar(rsface::detector::Detector::new(
        cascade,
        rsface::detector::DetectorConfig::default(),
    ));

    let faces = state
        .recognize(detector, &attack_rgb, &attack_gray, 3)
        .await;
    assert!(!faces.is_empty(), "blurred lena still has a detectable box");
    let blocked_count = faces.iter().filter(|f| f.blocked).count();
    println!("attack faces={} blocked={}", faces.len(), blocked_count);
    assert!(
        blocked_count > 0,
        "a print attack must be blocked under enforcement"
    );
    for face in faces.iter().filter(|f| f.blocked) {
        assert!(
            face.matches.is_empty(),
            "blocked faces must return no matches"
        );
    }
}

/// Separable box blur then grayscale — a stand-in for a re-photographed print.
fn blur_grayscale(img: &rsface::image::RgbImage, radius: usize) -> rsface::image::RgbImage {
    use rsface::image::RgbImage;
    let (w, h) = (img.width(), img.height());
    let mut tmp = RgbImage::new(w, h);
    let mut out = RgbImage::new(w, h);
    let r = radius as isize;
    for y in 0..h {
        for x in 0..w {
            let (mut sr, mut sg, mut sb, mut n) = (0u32, 0u32, 0u32, 0u32);
            for k in -r..=r {
                let sx = (x as isize + k).clamp(0, w as isize - 1) as usize;
                let base = (y * w + sx) * 3;
                sr += img.as_slice()[base] as u32;
                sg += img.as_slice()[base + 1] as u32;
                sb += img.as_slice()[base + 2] as u32;
                n += 1;
            }
            let base = (y * w + x) * 3;
            tmp.as_mut_slice()[base] = (sr / n) as u8;
            tmp.as_mut_slice()[base + 1] = (sg / n) as u8;
            tmp.as_mut_slice()[base + 2] = (sb / n) as u8;
        }
    }
    for y in 0..h {
        for x in 0..w {
            let (mut sr, mut sg, mut sb, mut n) = (0u32, 0u32, 0u32, 0u32);
            for k in -r..=r {
                let sy = (y as isize + k).clamp(0, h as isize - 1) as usize;
                let base = (sy * w + x) * 3;
                sr += tmp.as_slice()[base] as u32;
                sg += tmp.as_slice()[base + 1] as u32;
                sb += tmp.as_slice()[base + 2] as u32;
                n += 1;
            }
            let (vr, vg, vb) = (sr / n, sg / n, sb / n);
            let luma = ((vr + vg + vb) / 3) as u8;
            let base = (y * w + x) * 3;
            let dst = out.as_mut_slice();
            dst[base] = luma;
            dst[base + 1] = luma;
            dst[base + 2] = luma;
        }
    }
    out
}
