//! Benchmark the detector on a synthetic image. Run with:
//!   `cargo test --release bench_detect -- --nocapture --test-threads=1`

use rsface::detector::{Detector, DetectorConfig};
use rsface::haar::params::demo_face_cascade;
use rsface::image::GrayImage;
use std::time::Instant;

fn build_synthetic(w: usize, h: usize) -> GrayImage {
    let mut img = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            // Three "face-like" bright spots at different scales.
            let spot = |cx: f32, cy: f32, r: f32, v: u8| -> u8 {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let d = (dx * dx + dy * dy).sqrt();
                if d < r { v } else { 20 }
            };
            let v = spot(w as f32 * 0.25, h as f32 * 0.30, 60.0, 200)
                  .max(spot(w as f32 * 0.70, h as f32 * 0.40, 90.0, 220))
                  .max(spot(w as f32 * 0.50, h as f32 * 0.75, 45.0, 200));
            img[(x, y)] = v;
        }
    }
    img
}

#[test]
#[ignore]
fn bench_640x480() {
    let img = build_synthetic(640, 480);
    let det = Detector::new(demo_face_cascade(), DetectorConfig::default());
    // Warm up.
    let _ = det.detect(&img);
    let iters = 20;
    let t0 = Instant::now();
    let mut total = 0;
    for _ in 0..iters {
        total += det.detect(&img).len();
    }
    let dt = t0.elapsed();
    println!("640x480  iters={}  total_dets={}  total={:?}  per_frame={:?}  fps={:.2}",
        iters, total, dt, dt / iters, iters as f32 / dt.as_secs_f32());
}

#[test]
#[ignore]
fn bench_1920x1080() {
    let img = build_synthetic(1920, 1080);
    let det = Detector::new(demo_face_cascade(), DetectorConfig::default());
    let _ = det.detect(&img);
    let iters = 5;
    let t0 = Instant::now();
    let mut total = 0;
    for _ in 0..iters {
        total += det.detect(&img).len();
    }
    let dt = t0.elapsed();
    println!("1920x1080 iters={}  total_dets={}  total={:?}  per_frame={:?}  fps={:.2}",
        iters, total, dt, dt / iters, iters as f32 / dt.as_secs_f32());
}