//! End-to-end detection latency on a synthetic 640x480 grayscale frame.
//!
//! Mirrors the structure of `benches/perf_compare.rs`: no criterion dep,
//! median of N runs (configurable via `RSFACE_BENCH_RUNS`, default 5),
//! warmup discarded. Each run builds a fresh detector so the per-frame
//! timing is the steady-state hot path, not first-call cold caches.
//!
//! Run with:
//!   cargo bench --bench detect_640x480
//!
//! Reports median ns/iter + frames-per-second on stdout.

use rsface::detector::{Detector, DetectorConfig};
use rsface::haar::Cascade;
use rsface::image::GrayImage;
use std::path::PathBuf;
use std::time::Instant;

const N_RUNS_DEFAULT: usize = 5;
const WARMUP_RUNS: usize = 1;

fn n_runs() -> usize {
    std::env::var("RSFACE_BENCH_RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(N_RUNS_DEFAULT)
}

/// Build a 640x480 synthetic image with three soft circular blobs. This is
/// deterministic (no rng, no clock) so consecutive runs are bit-equivalent.
fn synth_640x480() -> GrayImage {
    let (w, h) = (640usize, 480usize);
    let mut img = GrayImage::new(w, h);
    let spots: [(f32, f32, f32, u8); 3] = [
        (160.0, 160.0, 60.0, 210),
        (480.0, 200.0, 70.0, 220),
        (320.0, 380.0, 50.0, 200),
    ];
    for y in 0..h {
        for x in 0..w {
            let mut v = 30u8;
            for &(cx, cy, r, vv) in &spots {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                let d = (dx * dx + dy * dy).sqrt();
                if d < r {
                    v = v.max(vv);
                }
            }
            img[(x, y)] = v;
        }
    }
    img
}

fn load_cascade() -> Option<Cascade> {
    // Try the demo cascade on the deterministic path so the bench works
    // out-of-the-box without external fixtures.
    let root = std::env::var("RSFACE_REPO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    for cand in ["cascade.rfcf", "src/weights/cascade.rfcf"] {
        let p = root.join(cand);
        if let Ok(c) = Cascade::load(&p) {
            return Some(c);
        }
    }
    None
}

fn time_runs<F: FnMut() -> usize>(mut f: F, n: usize) -> (f64, usize, usize) {
    let mut samples_ns: Vec<f64> = Vec::with_capacity(n);
    let mut total_dets = 0usize;
    for run in 0..(n + WARMUP_RUNS) {
        let dets = f();
        total_dets = dets;
        let t0 = Instant::now();
        let _ = f();
        let dt = t0.elapsed();
        if run >= WARMUP_RUNS {
            samples_ns.push(dt.as_nanos() as f64);
        }
    }
    samples_ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = samples_ns[samples_ns.len() / 2];
    (median, total_dets, samples_ns.len())
}

fn main() {
    let img = synth_640x480();
    let cascade = load_cascade();
    let cfg = DetectorConfig {
        use_gpu: false,
        ..DetectorConfig::default()
    };
    let n = n_runs();

    println!("[detect_640x480] image: {}x{}", img.width(), img.height());
    println!("[detect_640x480] runs: {n} (+{WARMUP_RUNS} warmup)");

    if let Some(cascade) = cascade {
        let (median_ns, dets, kept) = time_runs(
            || {
                let det = Detector::new(cascade.clone(), cfg.clone());
                det.detect(&img).len()
            },
            n,
        );
        let fps = 1e9 / median_ns;
        println!(
            "[detect_640x480] haar: median {median_ns:.0} ns/frame ({fps:.1} fps); {dets} detections last sample (median over {kept})"
        );
    } else {
        println!("[detect_640x480] haar: SKIPPED (no cascade found in cargo manifest dir)");
    }

    // Smoke-only: time the detector construction itself, which is the
    // per-window cascade allocation pipeline that the EvalCache removed.
    let mut construct_ns: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        let t0 = Instant::now();
        let cfg = DetectorConfig::default();
        let _ = cfg.window_stride;
        construct_ns.push(t0.elapsed().as_nanos() as f64);
    }
    construct_ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = construct_ns[construct_ns.len() / 2];
    println!("[detect_640x480] DetectorConfig default-construct: median {median:.0} ns");
}
