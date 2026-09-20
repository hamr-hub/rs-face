//! Focused 640x480 single-image latency benchmark.
//!
//! Measures end-to-end `Detector::detect` on a deterministic synthetic
//! grayscale fixture for the bundled OpenCV Haar cascade. The fixture
//! contains three "face-like" bright spots at different positions and
//! scales so the cascade has to do real work (variance pre-filter pass
//! followed by full evaluation on the survivors).
//!
//! Run:
//!   cargo bench --bench detect_640x480
//!
//! Output: criterion reports median ns/iter + per-iter detect_ms.
//! Criterion's harness is `false` (see Cargo.toml) so this is the same
//! pattern as `benches/perf_compare.rs`.

use rsface::detector::{Detector, DetectorConfig};
use rsface::haar::Cascade;
use rsface::image::GrayImage;
use std::time::Instant;

/// Deterministic synthetic 640x480 grayscale "image" — three bright
/// face-like spots at different positions and scales plus a noisy
/// background. Stable across runs/targets.
fn synth_face_640x480() -> GrayImage {
    let (w, h) = (640usize, 480usize);
    let mut img = GrayImage::new(w, h);
    // LCG background
    let mut s: u32 = 0x1234_5678;
    let spot = |cx: f32, cy: f32, r: f32, v: u8, dx: f32, dy: f32| -> u8 {
        let dist = (dx * dx + dy * dy).sqrt();
        if dist < r {
            v
        } else {
            20
        }
    };
    for y in 0..h {
        for x in 0..w {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let n = ((s >> 24) as u8).max(20);
            let v = spot(
                w as f32 * 0.25,
                h as f32 * 0.30,
                60.0,
                220,
                x as f32,
                y as f32,
            )
            .max(spot(
                w as f32 * 0.70,
                h as f32 * 0.40,
                90.0,
                230,
                x as f32,
                y as f32,
            ))
            .max(spot(
                w as f32 * 0.50,
                h as f32 * 0.75,
                45.0,
                215,
                x as f32,
                y as f32,
            ));
            // Background noise blends with the spot highlights.
            img[(x, y)] = v.max(n);
        }
    }
    img
}

/// Load the real `haarcascade_frontalface_default.rfcf` shipped under
/// `src/weights/`. Path is computed from `CARGO_MANIFEST_DIR` so the
/// benchmark is location-independent.
fn load_cascade() -> Cascade {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/weights/haarcascade_frontalface_default.rfcf");
    Cascade::load(&p).expect("bundled OpenCV cascade must load")
}

fn time_run(det: &Detector, img: &GrayImage, n_iters: usize) -> (f64, usize) {
    // Warm-up.
    let _ = det.detect(img);
    let mut samples_ms = Vec::with_capacity(n_iters);
    let mut last_dets = 0usize;
    for _ in 0..n_iters {
        let t0 = Instant::now();
        let r = det.detect(img);
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        samples_ms.push(dt);
        last_dets = r.len();
    }
    samples_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = samples_ms[samples_ms.len() / 2];
    (median, last_dets)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let n_iters: usize = std::env::var("RSFACE_BENCH_RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let preset = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| "default".to_string());

    let cascade = load_cascade();
    let img = synth_face_640x480();

    let cfg = match preset.as_str() {
        "fast" => DetectorConfig::fast(),
        "accurate" => DetectorConfig::accurate(),
        _ => DetectorConfig::default(),
    };

    let det = Detector::new(cascade.clone(), cfg.clone());
    println!(
        "[detect_640x480] preset={preset} cascade={} stages, {} features, window {}x{}",
        cascade.num_stages(),
        cascade.num_features(),
        cascade.window_w,
        cascade.window_h
    );
    println!(
        "[detect_640x480] image={}x{} iters={}",
        img.width(),
        img.height(),
        n_iters
    );

    let (median_ms, n_dets) = time_run(&det, &img, n_iters);
    let fps = 1000.0 / median_ms.max(1e-9);
    println!(
        "[detect_640x480] median per-frame: {:.2} ms ({:.1} fps)  detections={}",
        median_ms, fps, n_dets
    );
}
