//! Targeted micro-benchmarks for the SIMD-friendly hot paths added in
//! the `perf/simd-hotpath` sprint:
//!
//!  * `IntegralImage::from_gray` — u32 fused 1-pass build with the
//!    SSE2 / NEON row-prefix + vertical-add kernels. Run at 640×480
//!    (the detection-default scale) and 1920×1080 (the wide u32 path).
//!
//! Reads `RSFACE_BENCH_RUNS` for the iteration count (default 5) so the
//! same binary drives a CI smoke (1 run) and a developer spot-check
//! (5+ runs).
//!
//! Run with:
//!   cargo bench --bench perf_simd

use rsface::image::GrayImage;
use rsface::integral::IntegralImage;
use std::hint::black_box;
use std::time::Instant;

const N_RUNS_DEFAULT: usize = 5;

fn n_runs() -> usize {
    std::env::var("RSFACE_BENCH_RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(N_RUNS_DEFAULT)
}

/// Deterministic per-resolution pattern: every other row ramps 0..255
/// so the corner reads exercise a realistic mix of values (not all-0
/// or all-255, which would hit an awkward u32 wraparound edge case
/// that disproportionately helps or hurts the SIMD path).
fn synth(w: usize, h: usize) -> GrayImage {
    let mut img = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            img[(x, y)] = (((x + y) * 7) % 256) as u8;
        }
    }
    img
}

fn median_ms<F: FnMut()>(mut f: F, n: usize) -> f64 {
    let mut samples = Vec::with_capacity(n);
    for _ in 0..n {
        let t = Instant::now();
        f();
        samples.push(t.elapsed().as_nanos() as f64);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2] / 1e6
}

/// Build the u32 narrow integral image (SSE2/NEON row-prefix +
/// vertical-add). Reports median ms over N runs plus a fingerprint
/// sink so the build isn't dead-code-eliminated.
fn bench_integral_build(w: usize, h: usize, n: usize) -> f64 {
    let img = synth(w, h);
    median_ms(
        || {
            let ii = IntegralImage::from_gray(black_box(&img));
            // Touch the table size so the build isn't dead-code-eliminated.
            black_box(ii.width() * ii.height());
        },
        n,
    )
}

/// Variance pre-filter micro-benchmark: build the (sum, sum_sq) pair
/// for a stride=4 sliding window over a 640×480 narrow integral image.
/// Compares the in-line cascade path's per-iteration cost.
fn bench_variance_prefilter(w: usize, h: usize, n: usize) -> f64 {
    let img = synth(w, h);
    let ii = IntegralImage::from_gray(&img);
    let sq = rsface::integral::SquaredIntegralImage::from_gray(&img);
    let stride = ii.width() + 1;
    let win = 22usize;
    median_ms(
        || {
            let mut sink: u64 = 0;
            let mut y = 1;
            while y + win < h {
                let mut x = 1;
                while x + win < w {
                    let s = ii.rect_sum(x, y, x + win, y + win);
                    let ss = sq.rect_sum_sq(x, y, x + win, y + win);
                    sink = sink.wrapping_add(s).wrapping_add(ss);
                    x += 4;
                }
                y += 4;
            }
            black_box(sink);
        },
        n,
    )
}

fn main() {
    let n = n_runs();
    println!("[perf_simd] runs={n}");
    println!(
        "[perf_simd] integral_build 640x480: median {:.2} ms",
        bench_integral_build(640, 480, n)
    );
    println!(
        "[perf_simd] integral_build 1920x1080: median {:.2} ms",
        bench_integral_build(1920, 1080, n)
    );
    println!(
        "[perf_simd] variance_prefilter 640x480: median {:.2} ms",
        bench_variance_prefilter(640, 480, n)
    );
}
