//! Inspection helper: dump every detector's output on the canonical test
//! images. Run with:
//!   cargo test --test inspect_algos -- --ignored --nocapture
//!
//! Used only to bootstrap the golden-evaluation labels — every other test
//! path runs in CI without `--ignored`, so this file is inert by default.

use rsface::detector::{Detector, DetectorConfig};
use rsface::face_detector::FaceDetector;
use rsface::haar::bundled::bundled_frontalface_cascade;
use rsface::image::{codec, GrayImage, RgbImage};
use rsface::luminance_face::{LuminanceConfig, LuminanceFaceDetector};
use std::path::Path;

fn load(path: &str) -> GrayImage {
    let p = Path::new(path);
    let mut f = std::fs::File::open(p).expect("open");
    let ext = p.extension().and_then(|s| s.to_str()).unwrap_or("");
    match ext {
        "pgm" => codec::read_pgm(&mut f).expect("read_pgm"),
        "ppm" => {
            let rgb: RgbImage = codec::read_ppm(&mut f).expect("read_ppm");
            rgb.to_gray()
        }
        _ => panic!("unsupported ext {ext}"),
    }
}

#[test]
#[ignore]
fn inspect_demo_face_256() {
    let img = load("tests/fixtures/demo_face_256.pgm");
    let (w, h) = (img.width(), img.height());
    println!("\n== tests/fixtures/demo_face_256.pgm ({w}x{h}) ==");
    let haar = Detector::new(bundled_frontalface_cascade(), DetectorConfig::default());
    let haar_dets = haar.detect(&img);
    println!("  haar ({}):", haar_dets.len());
    for d in &haar_dets {
        println!(
            "    x={} y={} w={} h={} score={:.3}",
            d.x, d.y, d.w, d.h, d.score
        );
    }
    let lum = LuminanceFaceDetector::new(LuminanceConfig::default());
    let lum_dets = lum.detect(&img);
    println!("  luminance ({}):", lum_dets.len());
    for d in &lum_dets {
        println!(
            "    x={} y={} w={} h={} score={:.3}",
            d.x, d.y, d.w, d.h, d.score
        );
    }
}