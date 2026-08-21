//! Byte-level regression dump: runs the demo-cascade detector over a set of
//! deterministic synthetic images and prints every detection field to stdout
//! so a snapshot diff catches any numeric drift introduced by optimizations.
//!
//! Usage: cargo test --release --test regression_dump -- --nocapture > before.txt

use rsface::detector::{Detector, DetectorConfig};
use rsface::haar::params::demo_face_cascade;
use rsface::image::GrayImage;

fn build_synthetic(w: usize, h: usize) -> GrayImage {
    let mut img = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let spot = |cx: f32, cy: f32, r: f32, v: u8| -> u8 {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let d = (dx * dx + dy * dy).sqrt();
                if d < r {
                    v
                } else {
                    20
                }
            };
            let v = spot(w as f32 * 0.25, h as f32 * 0.30, 60.0, 200)
                .max(spot(w as f32 * 0.70, h as f32 * 0.40, 90.0, 220))
                .max(spot(w as f32 * 0.50, h as f32 * 0.75, 45.0, 200));
            img[(x, y)] = v;
        }
    }
    img
}

fn dump(img: &GrayImage, cfg: DetectorConfig, tag: &str) {
    let det = Detector::new(demo_face_cascade(), cfg);
    let dets = det.detect(img);
    println!("== {tag}: {} detections", dets.len());
    for d in &dets {
        println!(
            "{tag} x={} y={} w={} h={} score={:e} bits={}",
            d.x,
            d.y,
            d.w,
            d.h,
            d.score,
            d.score.to_bits()
        );
    }
}

#[test]
fn regression_dump() {
    let cfg = DetectorConfig {
        use_gpu: false,
        ..DetectorConfig::default()
    };
    for (w, h) in [(64, 64), (120, 120), (320, 240), (640, 480)] {
        let img = build_synthetic(w, h);
        dump(&img, cfg.clone(), &format!("default-{w}x{h}"));
    }
    // Variance prefilter disabled + stride 2 to exercise more of the scan space.
    let mut cfg2 = cfg.clone();
    cfg2.variance_threshold = u64::MAX;
    cfg2.window_stride = 2;
    let img = build_synthetic(320, 240);
    dump(&img, cfg2, "novar-stride2-320x240");

    // Equalized path.
    let mut cfg3 = cfg.clone();
    cfg3.equalize_hist = true;
    let img = build_synthetic(200, 160);
    dump(&img, cfg3, "equalized-200x160");
}
