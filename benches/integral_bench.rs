//! Integral image build + 1000-rectangle query bench at two resolutions:
//! 640x480 (the common Haar cascade input) and 1920x1080 (the wide u64 path).
//!
//! Run with: cargo bench --bench integral_bench
//!
//! Reports median ms for the build step + median ns for the query batch.

use rsface::image::GrayImage;
use rsface::integral::IntegralImage;
use std::time::Instant;

const N_RUNS_DEFAULT: usize = 5;
const N_QUERIES: usize = 1000;

fn n_runs() -> usize {
    std::env::var("RSFACE_BENCH_RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(N_RUNS_DEFAULT)
}

fn synth(w: usize, h: usize) -> GrayImage {
    // Deterministic pattern: every other row ramps 0..255. The integral
    // query workload touches every cached corner so it is the realistic
    // hot path, not a synthetic zero corner case.
    let mut img = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            img[(x, y)] = (((x + y) * 7) % 256) as u8;
        }
    }
    img
}

fn build_median_ms(w: usize, h: usize, n: usize) -> (f64, bool) {
    let img = synth(w, h);
    let mut samples: Vec<f64> = Vec::with_capacity(n);
    let mut wide = false;
    for _ in 0..n {
        let t0 = Instant::now();
        let ii = IntegralImage::from_gray(&img);
        samples.push(t0.elapsed().as_nanos() as f64);
        if ii.is_wide() {
            wide = true;
        }
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (samples[samples.len() / 2] / 1e6, wide)
}

fn query_median_ns(ii: &IntegralImage, n: usize) -> f64 {
    let w = ii.width();
    let h = ii.height();
    // Generate 1000 rectangles that are always valid.
    let mut samples: Vec<f64> = Vec::with_capacity(n);
    let mut sink: u64 = 0;
    for _ in 0..n {
        let t0 = Instant::now();
        for i in 0..N_QUERIES {
            // Cycle through a few common sizes.
            let sz = match i % 5 {
                0 => 8,
                1 => 16,
                2 => 24,
                3 => 32,
                _ => 64,
            };
            let x1 = (i * 3) % w.saturating_sub(sz + 1).max(1);
            let y1 = (i * 5) % h.saturating_sub(sz + 1).max(1);
            let x2 = (x1 + sz).min(w);
            let y2 = (y1 + sz).min(h);
            sink = sink.wrapping_add(ii.rect_sum(x1, y1, x2, y2));
        }
        samples.push(t0.elapsed().as_nanos() as f64);
    }
    if sink == u64::MAX {
        println!("overflow");
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn main() {
    let n = n_runs();
    println!("[integral_bench] runs={n} queries/batch={N_QUERIES}");

    for &(w, h) in &[(640usize, 480usize), (1920usize, 1080usize)] {
        let img = synth(w, h);
        let (build_ms, wide) = build_median_ms(w, h, n);
        let ii = IntegralImage::from_gray(&img);
        let q_ns = query_median_ns(&ii, n);
        println!(
            "[integral_bench] {w}x{h} build: median {build_ms:.2} ms ({}) ; queries: median {q_ns:.0} ns / {N_QUERIES} = {:.1} ns/query",
            if wide { "u64 wide path" } else { "u32 narrow path" },
            q_ns / N_QUERIES as f64,
        );
    }
}
