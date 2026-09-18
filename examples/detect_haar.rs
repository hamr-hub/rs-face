//! Detect faces in a synthetic frame with the bundled Haar cascade.
//!
//! This is the smallest end-to-end example in the crate — no external image
//! data, no model download, no ffmpeg. It draws a "face-like" pattern into a
//! 120×120 grayscale buffer, runs the demo cascade, and prints the boxes it
//! found.
//!
//! Run with:
//!   cargo run --release --example detect_haar

use rsface::detector::{Detector, DetectorConfig};
use rsface::haar::params::demo_face_cascade;
use rsface::image::GrayImage;

fn main() {
    // 1. Build a synthetic 120×120 grayscale "face" — bright forehead, eyes,
    //    nose bridge and mouth area on a dark surround.
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

    // 2. Build the detector with the bundled demo cascade.
    let detector = Detector::new(demo_face_cascade(), DetectorConfig::default());

    // 3. Run detection.
    let detections = detector.detect(&img);

    println!("detections: {}", detections.len());
    for d in &detections {
        println!(
            "  box ({}, {}) {}x{}  score={:.3}",
            d.x, d.y, d.w, d.h, d.score
        );
    }
}