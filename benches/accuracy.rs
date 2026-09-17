//! `bin/bench-accuracy` — measured throughput AND pair-pair similarity distributions.
//!
//! Runs the full SCRFD -> ArcFace pipeline against real-face fixtures and reports:
//!
//!   * Per-stage latency (detect, embed) on the real model with the available backend.
//!   * Cosine similarity distributions for same-identity and different-identity pairs,
//!     which are the actual inputs to a threshold decision.
//!
//! Run with:
//!
//! ```sh
//! tools/fetch_models.sh --all
//! cargo run --release --features ort-backend --bin bench-accuracy
//! # or:
//! cargo run --release --features tract-backend --bin bench-accuracy
//! ```
//!
//! Skipped (not failed) when the model weights are absent.

use std::path::PathBuf;

fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models")
}

fn require(file: &str) -> Option<PathBuf> {
    let p = model_dir().join(file);
    if p.exists() {
        Some(p)
    } else {
        eprintln!(
            "SKIP: {} not found; run tools/fetch_models.sh --all first",
            p.display()
        );
        None
    }
}

fn preferred_backend() -> Option<rsface::onnx::Backend> {
    use rsface::onnx::Backend;
    let avail = Backend::available();
    [Backend::Ort, Backend::Tract]
        .into_iter()
        .find(|b| avail.contains(b))
}

fn load_ppm(name: &str) -> Option<rsface::image::RgbImage> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(format!("{name}.ppm"));
    if !p.exists() {
        return None;
    }
    let mut f = std::fs::File::open(&p).expect("open fixture");
    Some(rsface::image::codec::read_ppm(&mut f).expect("decode ppm"))
}

fn fmt_stats(s: &mut String, name: &str, samples: &[f64], unit: &str) {
    if samples.is_empty() {
        s.push_str(&format!("  {name}: no samples\n"));
        return;
    }
    let mut sorted: Vec<f64> = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let p50 = sorted[n / 2];
    // p95 = element at position 95% of N. Saturating-guard against the empty case is
    // redundant here (we already returned) but cheap.
    let p95_idx = (n.saturating_mul(95) / 100).min(n - 1);
    let p95 = sorted[p95_idx];
    let mean = sorted.iter().sum::<f64>() / n as f64;
    let min = sorted[0];
    let max = sorted[n - 1];
    s.push_str(&format!(
        "  {name}: n={n} mean={mean:.2}{unit} p50={p50:.2}{unit} p95={p95:.2}{unit} min={min:.2} max={max:.2}\n"
    ));
}

/// Run the detector on a fixture and return the top-scoring landmarked detection, if any.
fn top_landmark(
    det: &rsface::scrfd_detector::ScrfdDetector,
    name: &str,
) -> Option<(rsface::image::RgbImage, rsface::face::Landmarks)> {
    let img = load_ppm(name)?;
    let dets = det.detect_rgb(&img).expect("detect");
    let top = dets
        .into_iter()
        .filter(|d| d.landmarks.is_some())
        .max_by(|a, b| a.score.total_cmp(&b.score))?;
    Some((img, top.landmarks.unwrap()))
}

fn main() {
    let mut out = String::new();
    let backend = match preferred_backend() {
        Some(b) => b,
        None => {
            eprintln!(
                "No inference backend compiled in; build with --features ort-backend \
                 or --features tract-backend"
            );
            return;
        }
    };
    out.push_str(&format!("backend: {}\n", backend.as_str()));

    let (Some(dpath), Some(rpath)) = (require("det_10g.onnx"), require("w600k_r50.onnx")) else {
        return;
    };

    let cfg = rsface::onnx::SessionConfig::default()
        .with_backend(backend)
        .with_threads(1);

    let det = match rsface::scrfd_detector::ScrfdDetector::open(
        &dpath,
        Some(rsface::models::find("scrfd_10g_kps").unwrap()),
        &cfg,
        rsface::scrfd::ScrfdConfig::default(),
    ) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("load detector: {e}");
            return;
        }
    };
    let rec = match rsface::arcface_recognizer::ArcFaceRecognizer::open(
        &rpath,
        Some(rsface::models::find("arcface_w600k_r50").unwrap()),
        &cfg,
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("load recogniser: {e}");
            return;
        }
    };

    let Some((img, lms)) = top_landmark(&det, "lena") else {
        eprintln!("No landmarked face in lena.ppm");
        return;
    };

    // Warm-up so the first call's allocator cost does not dominate the mean.
    for _ in 0..3 {
        let _ = det.detect_rgb(&img).unwrap();
        let _ = rec.embed(&img, &lms).unwrap();
    }

    const N: usize = 30;
    let mut det_total = Vec::with_capacity(N);
    let mut embed_total = Vec::with_capacity(N);
    for _ in 0..N {
        let t0 = std::time::Instant::now();
        let _ = det.detect_rgb(&img).unwrap();
        det_total.push(t0.elapsed().as_secs_f64() * 1000.0);

        let t0 = std::time::Instant::now();
        let _ = rec.embed(&img, &lms).unwrap();
        embed_total.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    let dets = det.detect_rgb(&img).unwrap();
    out.push_str(&format!(
        "fixture: lena.ppm {}x{}, {} in frame\n",
        img.width(),
        img.height(),
        dets.len()
    ));
    out.push_str("\n-- latency (ms) --\n");
    fmt_stats(&mut out, "detect (SCRFD+postprocess)", &det_total, "");
    fmt_stats(&mut out, "embed (ArcFace)", &embed_total, "");

    let mut same: Vec<f64> = Vec::new();
    let mut diff: Vec<f64> = Vec::new();
    let e_self = rec.embed(&img, &lms).unwrap();
    for _ in 0..20 {
        // Embedding::cosine returns f32; widen to f64 so the bench prints with the
        // resolution the cosine values actually have.
        let c = e_self.cosine(&rec.embed(&img, &lms).unwrap()).unwrap() as f64;
        same.push(c);
    }
    if let Some((other_img, other_lms)) = top_landmark(&det, "biden") {
        for _ in 0..20 {
            let c = e_self
                .cosine(&rec.embed(&other_img, &other_lms).unwrap())
                .unwrap() as f64;
            diff.push(c);
        }
    }
    out.push_str("\n-- cosine similarity --\n");
    fmt_stats(&mut out, "same identity", &same, "");
    fmt_stats(&mut out, "different identity", &diff, "");

    if !same.is_empty() && !diff.is_empty() {
        let mean_same: f64 = same.iter().copied().sum::<f64>() / same.len() as f64;
        let mean_diff: f64 = diff.iter().copied().sum::<f64>() / diff.len() as f64;
        let margin = mean_same - mean_diff;
        out.push_str(&format!("  margin (mean same - mean diff): {margin:.4}\n"));

        let min_same = same.iter().copied().fold(f64::INFINITY, f64::min);
        let max_diff = diff.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let recommended = (min_same + max_diff) / 2.0;
        out.push_str(&format!(
            "  recommended threshold (midpoint worst-same/best-diff): {recommended:.4}\n"
        ));
        let default_threshold: f64 = rsface::embedding::MatchConfig::default().threshold as f64;
        let note = if (recommended - default_threshold).abs() < 0.10 {
            "within 0.10 of measured midpoint"
        } else {
            "differs from measured midpoint by > 0.10 -- consider recalibrating"
        };
        out.push_str(&format!(
            "  default MatchConfig threshold: {default_threshold:.2} ({note})\n"
        ));
    }

    let path = std::env::var("BENCH_OUT").unwrap_or_else(|_| "docs/bench-results.md".into());
    if let Some(parent) = std::path::Path::new(&path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&path, &out).expect("write bench output");
    println!("{out}");
    println!("Wrote results to {path}");
}
