//! IoU throughput bench: time 1000 IoU calls per iteration across a mix of
//! disjoint, partial-overlap, nested, and degenerate boxes.
//!
//! Run with: cargo bench --bench iou_bench
//!
//! Reports median ns for the 1000-call batch.

use rsface::face::{Detection, FaceDetection};
use std::time::Instant;

const BATCH: usize = 1000;
const N_RUNS_DEFAULT: usize = 5;

fn n_runs() -> usize {
    std::env::var("RSFACE_BENCH_RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(N_RUNS_DEFAULT)
}

fn int_pairs() -> Vec<(Detection, Detection)> {
    // Mix of disjoint, partial-overlap, nested, degenerate cases.
    let boxes: Vec<Detection> = vec![
        Detection { x: 0, y: 0, w: 20, h: 20, score: 0.0 },
        Detection { x: 5, y: 5, w: 20, h: 20, score: 0.0 },
        Detection { x: 100, y: 100, w: 30, h: 30, score: 0.0 },
        Detection { x: 0, y: 0, w: 0, h: 0, score: 0.0 },
        Detection { x: 10, y: 10, w: 100, h: 100, score: 0.0 },
        Detection { x: 20, y: 20, w: 80, h: 80, score: 0.0 },
        Detection { x: 0, y: 0, w: 1, h: 1, score: 0.0 },
        Detection { x: 50, y: 50, w: 10, h: 10, score: 0.0 },
    ];
    let mut pairs = Vec::with_capacity(boxes.len() * boxes.len());
    for a in &boxes {
        for b in &boxes {
            pairs.push((a.clone(), b.clone()));
        }
    }
    pairs
}

fn f32_pairs() -> Vec<(FaceDetection, FaceDetection)> {
    let boxes = vec![
        FaceDetection { x1: 0.0, y1: 0.0, x2: 20.0, y2: 20.0, score: 0.0, landmarks: None },
        FaceDetection { x1: 5.5, y1: 5.5, x2: 25.5, y2: 25.5, score: 0.0, landmarks: None },
        FaceDetection { x1: 100.0, y1: 100.0, x2: 130.0, y2: 130.0, score: 0.0, landmarks: None },
        FaceDetection { x1: 0.0, y1: 0.0, x2: 0.0, y2: 0.0, score: 0.0, landmarks: None },
        FaceDetection { x1: -3.0, y1: -7.0, x2: 10.0, y2: 10.0, score: 0.0, landmarks: None },
        FaceDetection { x1: 10.5, y1: 10.5, x2: 90.5, y2: 90.5, score: 0.0, landmarks: None },
    ];
    let mut pairs = Vec::with_capacity(boxes.len() * boxes.len());
    for a in &boxes {
        for b in &boxes {
            pairs.push((*a, *b));
        }
    }
    pairs
}

fn run_batch<F: FnMut(&(Detection, Detection)) -> f32>(
    pairs: &[(Detection, Detection)],
    n: usize,
    mut f: F,
) -> f64 {
    // Pad the batch up to BATCH entries by cycling the pair list.
    let mut samples_ns: Vec<f64> = Vec::with_capacity(n);
    let mut sink = 0.0f32;
    for _ in 0..n {
        let t0 = Instant::now();
        let mut acc = 0.0f32;
        for i in 0..BATCH {
            let p = &pairs[i % pairs.len()];
            acc += f(p);
        }
        samples_ns.push(t0.elapsed().as_nanos() as f64);
        sink += acc;
    }
    // Prevent the optimiser from deleting the work.
    if sink.is_nan() {
        println!("nan sink");
    }
    samples_ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples_ns[samples_ns.len() / 2]
}

fn run_batch_f32<F: FnMut(&(FaceDetection, FaceDetection)) -> f32>(
    pairs: &[(FaceDetection, FaceDetection)],
    n: usize,
    mut f: F,
) -> f64 {
    let mut samples_ns: Vec<f64> = Vec::with_capacity(n);
    let mut sink = 0.0f32;
    for _ in 0..n {
        let t0 = Instant::now();
        let mut acc = 0.0f32;
        for i in 0..BATCH {
            let p = &pairs[i % pairs.len()];
            acc += f(p);
        }
        samples_ns.push(t0.elapsed().as_nanos() as f64);
        sink += acc;
    }
    if sink.is_nan() {
        println!("nan sink");
    }
    samples_ns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples_ns[samples_ns.len() / 2]
}

fn main() {
    let n = n_runs();
    let ipairs = int_pairs();
    let fpairs = f32_pairs();
    println!("[iou_bench] batch={BATCH} runs={n}");

    let int_ns = run_batch(&ipairs, n, |(a, b)| a.iou(b));
    println!(
        "[iou_bench] Detection::iou: median {int_ns:.0} ns / {BATCH} calls = {:.2} ns/call",
        int_ns / BATCH as f64
    );

    let f32_ns = run_batch_f32(&fpairs, n, |(a, b)| a.iou(b));
    println!(
        "[iou_bench] FaceDetection::iou: median {f32_ns:.0} ns / {BATCH} calls = {:.2} ns/call",
        f32_ns / BATCH as f64
    );
}