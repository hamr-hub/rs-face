//! M1 quantitative evaluation harness for rs-face (detection recall + 1:N identification).
//!
//! Independent of the rs-face crate: consumes only its public API
//! (ScrfdDetector + ArcFaceRecognizer + Gallery). Sequential, single-threaded,
//! one image at a time to stay within the 8 GB Jetson memory budget.
//!
//! Run (from repo root):
//!   ORT_DYLIB_PATH=data/eval_m1/libonnxruntime.so \
//!     cargo run --release --manifest-path tools/eval_m1/Cargo.toml -- \
//!     --models-dir models --split data/eval_m1/split --out data/eval_m1/results.json

use rsface::arcface_recognizer::ArcFaceRecognizer;
use rsface::embedding::Gallery;
use rsface::face::FaceDetection;
use rsface::image::RgbImage;
use rsface::models;
use rsface::onnx::SessionConfig;
use rsface::scrfd::ScrfdConfig;
use rsface::scrfd_detector::ScrfdDetector;
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(Deserialize)]
struct Entry {
    file: String,
    identity: String,
}

#[derive(Deserialize)]
struct Manifest {
    gallery: Vec<Entry>,
    probes: Vec<Entry>,
}

fn spec_for(file: &str) -> &'static models::ModelSpec {
    models::REGISTRY
        .iter()
        .find(|s| s.file_name == file)
        .unwrap_or_else(|| panic!("model {file} missing from registry"))
}

fn load_rgb(root: &Path, rel: &str) -> RgbImage {
    // Manifest entries may carry either a path relative to `root` or a full
    // path (this split stores the full data/... prefix). Resolve whichever
    // exists so we never double-prefix root.
    let as_rel = root.join(rel);
    let p = if as_rel.exists() { as_rel } else { Path::new(rel).to_path_buf() };
    let bytes = std::fs::read(&p).unwrap_or_else(|e| {
        eprintln!("cannot read {p:?}: {e}");
        std::process::exit(2);
    });
    decode_jpeg_rgb(&bytes).unwrap_or_else(|e| {
        eprintln!("cannot decode {p:?}: {e}");
        std::process::exit(2);
    })
}

use rsface::image::jpeg::decode_jpeg_rgb;

/// Largest-score detection (LFW frames contain exactly one labelled face).
fn best_detection(dets: &[FaceDetection]) -> Option<&FaceDetection> {
    dets.iter().fold(None, |acc: Option<&FaceDetection>, d| match acc {
        Some(a) if a.score >= d.score => Some(a),
        _ => Some(d),
    })
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let mut models_dir = PathBuf::from("models");
    let mut split_dir = PathBuf::from("data/eval_m1/split");
    let mut out_path = PathBuf::from("data/eval_m1/results.json");
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--models-dir" => {
                models_dir = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--split" => {
                split_dir = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--out" => {
                out_path = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            other => {
                eprintln!("unknown arg {other}");
                return ExitCode::from(2);
            }
        }
    }

    let det_spec = spec_for("det_10g.onnx");
    let rec_spec = spec_for("w600k_r50.onnx");
    let session_cfg = SessionConfig::default();
    let detector = ScrfdDetector::open(
        &models_dir.join("det_10g.onnx"),
        Some(det_spec),
        &session_cfg,
        ScrfdConfig::default(),
    )
    .expect("load SCRFD");
    let recognizer = ArcFaceRecognizer::open(
        &models_dir.join("w600k_r50.onnx"),
        Some(rec_spec),
        &session_cfg,
    )
    .expect("load ArcFace");

    let manifest: Manifest = serde_json::from_slice(
        &std::fs::read(split_dir.join("manifest.json")).expect("read manifest"),
    )
    .expect("parse manifest");

    // ---- enroll gallery ---------------------------------------------------
    // Use rank directly (not Gallery::identify) so thresholds never affect metrics.
    let mut gallery = Gallery::default();
    let mut gallery_records = Vec::new();
    for e in &manifest.gallery {
        let img = load_rgb(&split_dir, &e.file);
        let dets = detector.detect_rgb(&img).expect("detect");
        let rec = if let Some(d) = best_detection(&dets) {
            match d.landmarks {
                Some(lms) => match recognizer.embed(&img, &lms) {
                    Ok(emb) => {
                        gallery.enroll(e.identity.clone(), emb);
                        serde_json::json!({"file": e.file, "identity": e.identity,
                            "detected": true, "score": d.score, "embedded": true})
                    }
                    Err(err) => serde_json::json!({"file": e.file, "identity": e.identity,
                        "detected": true, "score": d.score, "embedded": false, "error": err.to_string()}),
                },
                None => serde_json::json!({"file": e.file, "identity": e.identity,
                    "detected": true, "score": d.score, "embedded": false, "error": "no landmarks"}),
            }
        } else {
            serde_json::json!({"file": e.file, "identity": e.identity,
                "detected": false, "embedded": false, "error": "no detection"})
        };
        gallery_records.push(rec);
    }

    // ---- probes ------------------------------------------------------------
    let mut probe_records = Vec::new();
    let mut detected = 0usize;
    let mut embedded = 0usize;
    let mut top1_hits = 0usize;
    let mut top5_hits = 0usize;
    let mut failures = Vec::new();

    for e in &manifest.probes {
        let img = load_rgb(&split_dir, &e.file);
        let dets = detector.detect_rgb(&img).expect("detect");
        let Some(d) = best_detection(&dets) else {
            probe_records.push(serde_json::json!({"file": e.file, "identity": e.identity,
                "detected": false, "embedded": false}));
            failures.push(serde_json::json!({"file": e.file, "identity": e.identity,
                "reason": "detection miss"}));
            continue;
        };
        detected += 1;
        let Some(lms) = d.landmarks else {
            probe_records.push(serde_json::json!({"file": e.file, "identity": e.identity,
                "detected": true, "score": d.score, "embedded": false}));
            failures.push(serde_json::json!({"file": e.file, "identity": e.identity,
                "reason": "no landmarks"}));
            continue;
        };
        let emb = match recognizer.embed(&img, &lms) {
            Ok(emb) => emb,
            Err(err) => {
                failures.push(serde_json::json!({"file": e.file, "identity": e.identity,
                    "reason": format!("embed failed: {err}")}));
                continue;
            }
        };
        embedded += 1;
        let ranked = gallery.rank(&emb);
        let top5: Vec<&(String, f32)> = ranked.iter().take(5).collect();
        let top1_label = top5[0].0.clone();
        let in_top1 = top1_label == e.identity;
        let in_top5 = top5.iter().any(|(l, _)| l == &e.identity);
        if in_top1 {
            top1_hits += 1;
        }
        if in_top5 {
            top5_hits += 1;
        }
        if !in_top1 {
            let top5_json: Vec<serde_json::Value> = top5
                .iter()
                .map(|(l, s)| serde_json::json!([l, s]))
                .collect();
            failures.push(serde_json::json!({"file": e.file, "identity": e.identity,
                "predicted": top1_label, "similarity": top5[0].1,
                "top5": top5_json}));
        }
        let top5_json: Vec<serde_json::Value> = top5
            .iter()
            .map(|(l, s)| serde_json::json!([l, s]))
            .collect();
        probe_records.push(serde_json::json!({"file": e.file, "identity": e.identity,
            "detected": true, "score": d.score, "embedded": true,
            "top1_correct": in_top1, "top5_correct": in_top5,
            "top1": [top1_label, top5[0].1],
            "top5": top5_json}));
    }

    // detection recall across ALL evaluated images (gallery + probes)
    let total_imgs = manifest.gallery.len() + manifest.probes.len();
    let gallery_detected = gallery_records
        .iter()
        .filter(|r| r["detected"].as_bool().unwrap_or(false))
        .count();
    let detection_hits = gallery_detected + detected;

    let summary = serde_json::json!({
        "people_in_gallery": gallery.len(),
        "gallery_images": manifest.gallery.len(),
        "probe_images": manifest.probes.len(),
        "detection": {
            "images_evaluated": total_imgs,
            "detected": detection_hits,
            "recall": detection_hits as f64 / total_imgs as f64,
        },
        "identification": {
            "probes_embedded": embedded,
            "top1_hits": top1_hits,
            "top5_hits": top5_hits,
            "top1": top1_hits as f64 / embedded as f64,
            "top5": top5_hits as f64 / embedded as f64,
        },
        "gallery_records": gallery_records,
        "probe_records": probe_records,
        "failures": failures,
    });

    std::fs::write(&out_path, serde_json::to_vec_pretty(&summary).unwrap()).expect("write results");
    println!("{}", serde_json::to_string_pretty(&summary["detection"]).unwrap());
    println!("{}", serde_json::to_string_pretty(&summary["identification"]).unwrap());
    println!("wrote {out_path:?}");
    ExitCode::SUCCESS
}
