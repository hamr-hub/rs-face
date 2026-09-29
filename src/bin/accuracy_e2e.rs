//! End-to-end detection + recognition accuracy harness.
//!
//! Runs the SCRFD+ArcFace pipeline on the platform's processed test video
//! (`platform/testdata/face-walking.mp4` by default) and reports concrete
//! per-frame numbers:
//!
//!   * per-frame face counts (mean, p95, max)
//!   * cross-frame cosine similarity for the dominant track (i.e. how
//!     stable is the embedding of "the same person" over time)
//!   * per-frame latency (detect + align + embed)
//!
//! Skipped (not failed) when model weights are absent, or when the test
//! video is missing.
//!
//! Run with:
//!   cargo run --release --features ort-backend --bin accuracy-e2e
//! or:
//!   cargo run --release --features ort-backend --bin accuracy-e2e -- \
//!     --video path/to/another.mp4 --max-frames 300

#![cfg(any(feature = "ort-backend", feature = "tract-backend"))]

use std::path::PathBuf;
use std::time::Instant;

use rsface::align::norm_crop;
use rsface::arcface_recognizer::ArcFaceRecognizer;
use rsface::embedding::{Embedding, Gallery, MatchConfig};
use rsface::face::{FaceDetection, Landmarks};
use rsface::face_detector::FaceDetector;
use rsface::image::RgbImage;
use rsface::models;
use rsface::scrfd::ScrfdConfig;
use rsface::scrfd_detector::ScrfdDetector;
use rsface::source::open as open_source;

fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models")
}

fn require(file: &str) -> Option<PathBuf> {
    let p = model_dir().join(file);
    if p.exists() {
        Some(p)
    } else {
        eprintln!("SKIP: {} not found; run tools/fetch_models.sh --all", p.display());
        None
    }
}

fn preferred_backend() -> rsface::onnx::Backend {
    use rsface::onnx::Backend;
    let avail = Backend::available();
    for b in [Backend::Ort, Backend::Tract] {
        if avail.contains(&b) {
            return b;
        }
    }
    panic!("no inference backend enabled; build with --features ort-backend or tract-backend");
}

fn collect_rgb_frames(path: &PathBuf, max_frames: u64) -> Vec<RgbImage> {
    let path_str = match path.to_str() {
        Some(s) => s,
        None => {
            eprintln!("non-utf8 path: {}", path.display());
            return Vec::new();
        }
    };
    let mut source = match open_source(path_str) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("open_source({}): {}", path.display(), e);
            return Vec::new();
        }
    };
    let mut frames = Vec::with_capacity(max_frames as usize);
    for _ in 0..max_frames {
        match source.next_frame() {
            Ok(Some(frame)) => match frame.rgb {
                Some(rgb) => frames.push((*rgb).clone()),
                None => {
                    // Frame produced only gray; lift to RGB.
                    frames.push(RgbImage::from_gray(&frame.gray))
                }
            },
            Ok(None) => break,
            Err(e) => {
                eprintln!("next_frame: {}", e);
                break;
            }
        }
    }
    frames
}

/// Align + embed a single face detection using its landmarks.
fn embed_det(rec: &ArcFaceRecognizer, rgb: &RgbImage, det: &FaceDetection) -> Option<Embedding> {
    let lms = det.landmarks.as_ref()?;
    let crop = norm_crop(rgb, lms)?;
    rec.embed(&crop, lms).ok()
}

#[derive(Default)]
struct RunStats {
    frames: u64,
    frames_with_face: u64,
    total_detections: u64,
    detect_ms: Vec<f64>,
    embed_ms: Vec<f64>,
    detect_counts: Vec<u64>,
    /// Top-confidence embedding per frame (the "dominant" face).
    dominant_embeddings: Vec<Embedding>,
    /// All embeddings per frame for offline analysis.
    all_per_frame: Vec<Vec<Embedding>>,
}

fn pctile(v: &[f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((s.len() as f64 * q).floor() as usize).min(s.len() - 1);
    s[idx]
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut video: PathBuf = PathBuf::from("platform/testdata/face-walking.mp4");
    let mut max_frames: u64 = 60;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--video" => video = PathBuf::from(args.next().expect("--video <PATH>")),
            "--max-frames" => {
                max_frames = args
                    .next()
                    .expect("--max-frames <N>")
                    .parse()
                    .expect("u64")
            }
            "--help" | "-h" => {
                println!("usage: accuracy-e2e [--video <PATH>] [--max-frames <N>]");
                return;
            }
            other => {
                eprintln!("unknown arg: {other}");
                return;
            }
        }
    }

    if !video.exists() {
        eprintln!("video not found: {}; skip", video.display());
        return;
    }
    let (Some(dpath), Some(rpath)) = (require("det_10g.onnx"), require("w600k_r50.onnx")) else {
        return;
    };

    let backend = preferred_backend();
    eprintln!("backend: {}", backend.as_str());

    let session_cfg = rsface::onnx::SessionConfig::default()
        .with_backend(backend)
        .with_threads(1);

    let det = match ScrfdDetector::open(
        &dpath,
        models::find("scrfd_10g_kps"),
        &session_cfg,
        ScrfdConfig::default(),
    ) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("SCRFD load failed: {e}");
            return;
        }
    };
    let rec = match ArcFaceRecognizer::open(
        &rpath,
        models::find("arcface_w600k_r50"),
        &session_cfg,
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("ArcFace load failed: {e}");
            return;
        }
    };

    eprintln!("loading frames from {}", video.display());
    let frames = collect_rgb_frames(&video, max_frames);
    if frames.is_empty() {
        eprintln!("no frames decoded");
        return;
    }
    eprintln!("frames: {}", frames.len());

    let mut st = RunStats::default();
    let mut gallery = Gallery::new(MatchConfig {
        threshold: 0.45,
        min_margin: 0.0,
    });

    // Warm up (allocator + first-call caches).
    let _ = det.detect_faces_rgb(&frames[0]);

    for (i, rgb) in frames.iter().enumerate() {
        let t0 = Instant::now();
        let dets = det.detect_faces_rgb(rgb);
        st.detect_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        st.detect_counts.push(dets.len() as u64);

        if !dets.is_empty() {
            st.frames_with_face += 1;
            st.total_detections += dets.len() as u64;

            let embed_start = Instant::now();
            let mut per_frame = Vec::with_capacity(dets.len());
            // Pick the top-scoring face as "the dominant person".
            let mut best_det: Option<&FaceDetection> = None;
            for d in &dets {
                if best_det.map_or(true, |b| d.score > b.score) {
                    best_det = Some(d);
                }
            }
            if let Some(d) = best_det {
                if let Some(emb) = embed_det(&rec, rgb, d) {
                    st.dominant_embeddings.push(emb.clone());
                    per_frame.push(emb);
                }
            }
            st.embed_ms.push(embed_start.elapsed().as_secs_f64() * 1000.0);
            st.all_per_frame.push(per_frame);
        }
        st.frames += 1;
        if i % 10 == 0 {
            eprintln!(
                "frame {}/{}\tfaces: {}\tlatency: {:.0}ms detect / {:.0}ms embed",
                i,
                frames.len(),
                dets.len(),
                st.detect_ms.last().copied().unwrap_or(0.0),
                st.embed_ms.last().copied().unwrap_or(0.0),
            );
        }
    }

    let mut summary = String::new();
    summary.push_str(&format!("# Accuracy e2e — {}\n", video.display()));
    summary.push_str(&format!("frames decoded: {}\n", frames.len()));
    summary.push_str(&format!("frames with face: {}\n", st.frames_with_face));
    summary.push_str(&format!("total detections: {}\n", st.total_detections));

    summary.push_str(&format!(
        "detect latency ms: mean={:.0} p50={:.0} p95={:.0}\n",
        if st.detect_ms.is_empty() {
            0.0
        } else {
            st.detect_ms.iter().sum::<f64>() / st.detect_ms.len() as f64
        },
        pctile(&st.detect_ms, 0.5),
        pctile(&st.detect_ms, 0.95),
    ));
    summary.push_str(&format!(
        "embed latency ms:  mean={:.0} p50={:.0} p95={:.0}\n",
        if st.embed_ms.is_empty() {
            0.0
        } else {
            st.embed_ms.iter().sum::<f64>() / st.embed_ms.len() as f64
        },
        pctile(&st.embed_ms, 0.5),
        pctile(&st.embed_ms, 0.95),
    ));

    let counts = &st.detect_counts;
    summary.push_str(&format!(
        "detections per frame: mean={:.2} max={} over {} frames\n",
        counts.iter().sum::<u64>() as f64 / counts.len().max(1) as f64,
        counts.iter().copied().max().unwrap_or(0),
        counts.len()
    ));

    // Cross-frame cosine for the dominant embedding — proxy for re-id stability.
    let mut prev: Option<&Embedding> = None;
    let mut cos: Vec<f64> = Vec::new();
    for e in &st.dominant_embeddings {
        if let Some(p) = prev {
            if let Some(c) = p.cosine(e) {
                cos.push(c as f64);
            }
        }
        prev = Some(e);
    }
    summary.push_str(&format!(
        "cross-frame same-identity cosine (re-id proxy):\n  n={} mean={:.3} p50={:.3} min={:.3} max={:.3}\n",
        cos.len(),
        if cos.is_empty() {
            0.0
        } else {
            cos.iter().sum::<f64>() / cos.len() as f64
        },
        pctile(&cos, 0.5),
        if cos.is_empty() {
            0.0
        } else {
            cos.iter().cloned().fold(f64::INFINITY, f64::min)
        },
        if cos.is_empty() {
            0.0
        } else {
            cos.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        },
    ));

    // Add a synthetic enrollment to test recognition accuracy:
    // take the dominant embedding from frame 0 and query it against itself,
    // plus an artificial "different person" vector.
    if let Some(first) = st.dominant_embeddings.first() {
        gallery.enroll("track_a", first.clone());
        // Build a synthetic "different person" by negating the first 64 dims.
        let mut diff = first.as_slice().to_vec();
        for d in diff.iter_mut().take(64) {
            *d = -*d;
        }
        let diff_emb = match Embedding::from_raw(&diff) {
            Some(e) => e,
            None => {
                println!("{summary}");
                return;
            }
        };
        gallery.enroll("track_b", diff_emb.clone());

        let outcome_self = gallery.identify(first);
        let outcome_diff = gallery.identify(&diff_emb);
        summary.push_str(&format!("self-match: {outcome_self:?}\n"));
        summary.push_str(&format!("different-match: {outcome_diff:?}\n"));
    }

    print!("{summary}");
}
