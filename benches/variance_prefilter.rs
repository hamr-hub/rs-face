//! Focused micro-benchmark of the variance pre-filter hot loop, comparing
//! the scalar `passes_variance_sums_fast` path against the new 4-wide
//! SIMD `passes_variance_mask_4` path on the same (sum, sum_sq) inputs.
//!
//! The detector's per-window cost splits roughly into:
//!   * 8 integral-image corner reads  (cheap, hardware-prefetched)
//!   * the variance-decision arithmetic (4 multiplies + 1 compare)
//!   * the cascade eval (only when variance passes — the bulk of the cost)
//!
//! This bench measures just the second bullet: how long it takes to make
//! the variance decision for `w * h` windows worth of pre-loaded
//! (sum, sum_sq) pairs. Both kernels walk the same data, so any
//! per-iteration difference is purely the SIMD arithmetic + masking
//! overhead.
//!
//! Run with:
//!   cargo bench --bench variance_prefilter -- --warm-up-time 1 --measurement-time 2
//!
//! Reports median ms per kernel at 640×480.

use rsface::haar::params::demo_face_cascade;
use rsface::image::GrayImage;
use rsface::integral::{IntegralImage, SquaredIntegralImage};
use std::hint::black_box;
use std::time::Instant;

const N_RUNS_DEFAULT: usize = 5;

fn n_runs() -> usize {
    std::env::var("RSFACE_BENCH_RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(N_RUNS_DEFAULT)
}

fn synth(w: usize, h: usize) -> GrayImage {
    let mut img = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            img[(x, y)] = (((x + y) * 7) % 256) as u8;
        }
    }
    img
}

/// Pre-compute (sum, sum_sq) for every 24×24 window in the image, in
/// row-major order. We do this ONCE per run so the bench measures only
/// the variance decision, not the rectangle reads.
fn precompute_sums(img: &GrayImage) -> (Vec<u64>, Vec<u64>) {
    let ii = IntegralImage::from_gray(img);
    let sq = SquaredIntegralImage::from_gray(img);
    let win = 24usize;
    let w = img.width();
    let h = img.height();
    let nw = w.saturating_sub(win) + 1;
    let nh = h.saturating_sub(win) + 1;
    let mut sums = vec![0u64; nw * nh];
    let mut sum_sqs = vec![0u64; nw * nh];
    for y in 0..nh {
        for x in 0..nw {
            sums[y * nw + x] = ii.rect_sum(x, y, x + win, y + win);
            sum_sqs[y * nw + x] = sq.rect_sum_sq(x, y, x + win, y + win);
        }
    }
    (sums, sum_sqs)
}

fn median_ms<F: FnMut()>(mut f: F, n: usize) -> f64 {
    let mut samples = Vec::with_capacity(n);
    for _ in 0..n {
        let t = Instant::now();
        f();
        samples.push(t.elapsed().as_nanos() as f64);
    }
    samples.sort_by(|a, b| a.partial_cmp(&b).unwrap());
    samples[n / 2] / 1e6
}

fn main() {
    let _cascade = demo_face_cascade(); // sanity: cascade is not used but keeps the dep linked
    let n = n_runs();
    let img = synth(640, 480);
    let (sums, sum_sqs) = precompute_sums(&img);
    let n_pixels = (22u64 * 22u64) as u64;
    let n_pixels_sq = n_pixels * n_pixels;
    let thr = 200u64;
    let len = sums.len();
    println!("[variance_prefilter] windows per frame: {len}");

    // Scalar path: one window at a time, integer arithmetic, branch.
    let scalar_ms = median_ms(
        || {
            let mut sink: u64 = 0;
            for i in 0..len {
                let pass = SquaredIntegralImage::passes_variance_sums_fast(
                    sums[i],
                    sum_sqs[i],
                    n_pixels,
                    n_pixels_sq,
                    thr,
                );
                if pass {
                    sink = sink.wrapping_add(1);
                }
            }
            black_box(sink);
        },
        n,
    );

    // SIMD path: 4 windows at a time, f64 lane math, branchless mask.
    let n_pixels_f64 = n_pixels as f64;
    let thr_n_sq_f64 = (thr as f64) * (n_pixels_sq as f64);
    let simd_ms = median_ms(
        || {
            let mut sink: u32 = 0;
            let chunks = len / 4;
            for c in 0..chunks {
                let i = c * 4;
                let mask = SquaredIntegralImage::passes_variance_mask_4(
                    [sums[i], sums[i + 1], sums[i + 2], sums[i + 3]],
                    [sum_sqs[i], sum_sqs[i + 1], sum_sqs[i + 2], sum_sqs[i + 3]],
                    n_pixels_f64,
                    thr_n_sq_f64,
                );
                sink = sink.wrapping_add(mask.count_ones());
            }
            // Tail
            for i in (chunks * 4)..len {
                let pass = SquaredIntegralImage::passes_variance_sums_fast(
                    sums[i],
                    sum_sqs[i],
                    n_pixels,
                    n_pixels_sq,
                    thr,
                );
                if pass {
                    sink = sink.wrapping_add(1);
                }
            }
            black_box(sink);
        },
        n,
    );

    println!(
        "[variance_prefilter] scalar (1 window/iter): median {:.2} ms",
        scalar_ms
    );
    println!(
        "[variance_prefilter] simd   (4 windows/iter): median {:.2} ms",
        simd_ms
    );
    let speedup = scalar_ms / simd_ms;
    println!("[variance_prefilter] speedup: {:.2}x", speedup);
}
