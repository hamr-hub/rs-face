//! Dispatch the same frame through every algorithm — the "swiss army knife"
//! demo.
//!
//! Every detector in `rsface` implements the `FaceDetector` trait so that
//! benchmark / comparison / platform code can dispatch polymorphically. This
//! example shows the simplest possible use: build one of each algorithm that
//! has real CPU implementation behind it, run them all on the same synthetic
//! frame, and print the `name()` plus count.
//!
//! Run with:
//!   cargo run --release --example detect_uniform

use rsface::face_detector::{FaceDetector, HaarDetector};
use rsface::hog_face::{HogConfig, HogFaceDetector};
use rsface::image::GrayImage;
use rsface::lbph::{LbphConfig, LbphRecognizer};

fn make_synthetic_face() -> GrayImage {
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
    img
}

fn main() {
    let img = make_synthetic_face();

    // Each detector implements FaceDetector; `name()` is the algorithm tag.
    // Haar needs a Cascade — we use the bundled demo.
    let haar = HaarDetector::new(
        rsface::haar::params::demo_face_cascade(),
        Default::default(),
    );
    // HOG ships its own scaffold config; the bundled weights are placeholder,
    // so this one will report zero detections on any input — that is *expected*
    // and is exactly what the Maturity::Scaffold label exists to communicate.
    let hog = HogFaceDetector::new(HogConfig::default());

    // To prove the trait dispatch really is uniform, store them in a heterogeneous
    // Vec<dyn FaceDetector>. In production code this is what the platform
    // layer does for "compare all algorithms on this frame".
    let detectors: Vec<(&'static str, Box<dyn FaceDetector>)> =
        vec![(haar.name(), Box::new(haar)), (hog.name(), Box::new(hog))];

    for (registered_name, det) in &detectors {
        let hits = det.detect(&img);
        println!(
            "[{:>6}]  detections={:>3}  description=\"{}\"",
            registered_name,
            hits.len(),
            registered_name
        );
    }

    // Bonus: prove the LBPH recognizer compiles and is wired into the same crate.
    // We can't usefully identify on a synthetic face, but we can prove the API:
    let mut rec = LbphRecognizer::new(LbphConfig::default());
    rec.enroll("nobody", &img);
    let outcome = rec.identify_crop(&img);
    println!("[  lbph]  recognition={:?}", outcome);
}
