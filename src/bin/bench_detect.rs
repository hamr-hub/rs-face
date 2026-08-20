//! Standalone benchmark binary.
//! `cargo run --release --bin bench_detect -- [w] [h] [iters]`
//!
//! Same clippy allow set as src/lib.rs (binary targets don't inherit the
//! lib's crate-level allows).
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::identity_op)]
#![allow(clippy::erasing_op)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::manual_saturating_arithmetic)]
#![allow(clippy::manual_checked_ops)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::io_other_error)]
#![allow(clippy::mut_from_ref)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::collapsible_match)]
#![allow(clippy::unnecessary_map_or)]
#![allow(clippy::while_let_loop)]
#![allow(clippy::new_without_default)]
#![allow(clippy::needless_collect)]
#![allow(clippy::unused_self)]
#![allow(clippy::needless_borrow)]
#![allow(clippy::single_match)]
#![allow(clippy::no_effect)]
#![allow(clippy::ptr_arg)]
#![allow(unused_parens)]
#![allow(unused_variables)]
#![allow(dead_code)]
#![allow(unused_imports)]

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
