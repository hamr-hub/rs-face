//! Standalone benchmark binary.
//! `cargo run --release --bin bench_detect -- [w] [h] [iters]`

use rsface::detector::{Detector, DetectorConfig};
use rsface::haar::params::demo_face_cascade;
use rsface::image::GrayImage;
use std::time::Instant;

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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let w: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(640);
    let h: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(480);
    let iters: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(20);

    eprintln!("[bench] building {w}x{h} synthetic");
    let img = build_synthetic(w, h);
    let det = Detector::new(demo_face_cascade(), DetectorConfig::default());
    // Warm up.
    let _ = det.detect(&img);
    let t0 = Instant::now();
    let mut total = 0;
    for _ in 0..iters {
        total += det.detect(&img).len();
    }
    let dt = t0.elapsed();
    let per_frame = dt / iters as u32;
    let fps = iters as f32 / dt.as_secs_f32();
    println!(
        "{w}x{h}  iters={iters}  total_dets={total}  total={:?}  per_frame={:?}  fps={:.2}",
        dt, per_frame, fps
    );
}
