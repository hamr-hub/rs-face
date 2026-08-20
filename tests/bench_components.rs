//! Component-level micro-benchmarks. Run with:
//!   `cargo test --release --test bench_components -- --ignored --nocapture --test-threads=1`
//!
//! Each test isolates one hot path so we can see where time is going
//! without needing a profiler.

use rsface::detector::non_max_suppression;
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

fn time_it<F: FnMut()>(label: &str, iters: usize, mut f: F) -> f64 {
    // Warm up.
    for _ in 0..3 { f(); }
    let t0 = Instant::now();
    for _ in 0..iters { f(); }
    let dt = t0.elapsed();
    let per = dt.as_secs_f64() / iters as f64;
    println!("{label:35} iters={iters:4} per={:8.3} ms", per * 1e3);
    per
}

#[test]
#[ignore]
fn bench_integral_components() {
    std::env::set_var("RSFACE_TIMING", "1");
    bench_integral_components_inner();
    std::env::remove_var("RSFACE_TIMING");
}

fn bench_integral_components_inner() {
    let img640 = build_synthetic(640, 480);
    let img1080 = build_synthetic(1920, 1080);
    println!("\n--- integral components (release build) ---");

    for (label, img) in [("640x480", &img640), ("1920x1080", &img1080)] {
        println!("\n[{label}]");
        time_it("IntegralImage::from_gray", 30, || {
            let _ = rsface::integral::IntegralImage::from_gray(img);
        });
        time_it("SquaredIntegralImage::from_gray", 30, || {
            let _ = rsface::integral::SquaredIntegralImage::from_gray(img);
        });
        time_it("RotatedIntegralImage::from_gray", 30, || {
            let _ = rsface::integral::RotatedIntegralImage::from_gray(img);
        });
        time_it("resize_area 1920->640", 30, || {
            let _ = img.resize_area(640, 360);
        });
    }
}

#[test]
#[ignore]
fn bench_nms() {
    std::env::set_var("RSFACE_TIMING", "1");
    bench_nms_inner();
    std::env::remove_var("RSFACE_TIMING");
}

fn bench_nms_inner() {
    use rsface::detector::DetectorConfig;
    use rsface::detector::Detector;
    let img = build_synthetic(1920, 1080);
    let det = Detector::new(demo_face_cascade(), DetectorConfig::default());
    let dets = det.detect(&img);
    println!("\n--- NMS over {} raw detections ---", dets.len());
    time_it("non_max_suppression", 30, || {
        let _ = non_max_suppression(dets.clone(), 0.3);
    });
}

#[test]
#[ignore]
fn bench_classify_window() {
    use rsface::haar::EvalCache;
    use rsface::integral::RotatedIntegralImage;
    let img = build_synthetic(1920, 1080);
    let ii = rsface::integral::IntegralImage::from_gray(&img);
    let ri = RotatedIntegralImage::from_gray(&img);
    let c = demo_face_cascade();
    // Pick a window that's likely to pass several stages.
    let mut cache = EvalCache::new(c.features.len());
    println!("\n--- cascade::classify single window ---");
    time_it("classify(face region)", 500, || {
        let _ = c.classify(&ii, Some(&ri), 200, 200, &mut cache);
    });
}

#[test]
#[ignore]
fn bench_pyramid_full() {
    std::env::set_var("RSFACE_TIMING", "1");
    bench_pyramid_full_inner();
    std::env::remove_var("RSFACE_TIMING");
}

fn bench_pyramid_full_inner() {
    use rsface::image::GrayImage;
    use rsface::detector::{Detector, DetectorConfig};
    use rsface::haar::params::demo_face_cascade;
    // Per-level resize cost for a 1920x1080 input — shows where time
    // actually goes inside the pyramid construction.
    let img = build_synthetic(1920, 1080);
    let mut cur = img.clone();
    println!("\n--- resize_area per pyramid level (1920x1080) ---");
    for level in 0..25 {
        let cw = cur.width();
        let ch = cur.height();
        let nw = ((cw as f32) / 1.2).round().max(24.0) as usize;
        let nh = ((ch as f32) / 1.2).round().max(24.0) as usize;
        if nw >= cw || nh >= ch { break; }
        time_it(&format!("level {level:2}: {cw}x{ch} -> {nw}x{nh}"), 5, || {
            let _ = cur.resize_area(nw, nh);
        });
        cur = cur.resize_area(nw, nh);
    }
    let img640 = build_synthetic(640, 480);
    let img1080 = build_synthetic(1920, 1080);
    let det = Detector::new(demo_face_cascade(), DetectorConfig::default());
    println!("\n--- full detector ---");
    time_it("Detector::detect 640x480", 10, || {
        let _ = det.detect(&img640);
    });
    time_it("Detector::detect 1920x1080", 5, || {
        let _ = det.detect(&img1080);
    });
}
