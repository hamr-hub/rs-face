//! Minimal library usage example.
//!
//! Runs the bundled demo cascade on a synthetic frame and prints the number
//! of detections. Useful as a smoke test in CI without external image data.
//!
//! Run with:
//!   cargo run --release --example synthetic_smoke

use rsface::detector::{Detector, DetectorConfig};
use rsface::haar::params::demo_face_cascade;
use rsface::image::GrayImage;

fn main() {
    // 120x120 frame with a bright-center face-like pattern.
    let mut img = GrayImage::new(120, 120);
    for y in 0..120 {
        for x in 0..120 {
            let v = if y < 20 {
                20
            } else if y < 40 && (40..80).contains(&x) {
                200
            } else if y < 100 && (40..80).contains(&x) {
                220
            } else {
                20
            };
            img[(x, y)] = v;
        }
    }

    let detector = Detector::new(demo_face_cascade(), DetectorConfig::default());
    let detections = detector.detect(&img);

    println!("detections: {}", detections.len());
    for d in &detections {
        println!(
            "  box ({}, {}) {}x{}  score={:.3}",
            d.x, d.y, d.w, d.h, d.score
        );
    }
}
