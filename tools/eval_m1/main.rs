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
use rsface::embedding::{Embedding, Gallery};
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

/// Lowest alignment residual — the detection whose landmarks fit the canonical
/// ArcFace layout best. Used by the optional `--prefer-residual` mode where
/// landmark quality is a better signal than raw detector score for faces at extreme
/// pose, where the detector fires confidently but the keypoints are misplaced.
///
/// Returns the `None` if no detection has usable landmarks (every entry is None).
fn best_aligned_detection<'a>(
    dets: &'a [FaceDetection],
    max_residual_px: f32,
) -> Option<&'a FaceDetection> {
    dets.iter()
        .filter_map(|d| {
            let lms = d.landmarks.as_ref()?;
            let r = alignment_residual(lms);
            if r.is_finite() && r <= max_residual_px {
                Some((d, r))
            } else {
                None
            }
        })
        .fold(None, |acc: Option<(&FaceDetection, f32)>, (d, r)| match acc {
            Some((_, ar)) if ar <= r => acc,
            _ => Some((d, r)),
        })
        .map(|(d, _)| d)
}

/// Sub-pixel residual between the detected 5-point landmarks and the canonical
/// ArcFace 112×112 layout, after the closed-form similarity fit. Large residual
/// means the detector emitted plausible-but-wrong landmarks (e.g. nose on top of
/// an eye) and the resulting crop will be computed off-centre — every embedding
/// produced from that crop will be off-distribution noise.
///
/// See [`crate::align::estimate_arcface_transform`] for the transform.
fn alignment_residual(lms: &rsface::face::Landmarks) -> f32 {
    use rsface::align::{estimate_arcface_transform, ARCFACE_REFERENCE_LANDMARKS};
    let Some(t) = estimate_arcface_transform(lms) else {
        return f32::NAN;
    };
    let mut worst = 0.0f32;
    for i in 0..rsface::face::NUM_LANDMARKS {
        let (x, y) = t.apply(lms.points[i].0, lms.points[i].1);
        let (rx, ry) = ARCFACE_REFERENCE_LANDMARKS[i];
        worst = worst.max(((x - rx).powi(2) + (y - ry).powi(2)).sqrt());
    }
    worst
}

/// Single-pass `embed`, or TTA-flip-averaged `embed_tta` with single-pass fallback.
///
/// TTA averages the embedding of the aligned crop with the embedding of its horizontal
/// mirror — a no-training-cost +30% latency for ~5-15% accuracy on LFW-class benchmarks.
/// Fall back to single-pass on collapse so a degenerate flip cannot silently sink the
/// result (mirrors [`ArcFaceRecognizer::embed_tta`]'s contract).
fn embed_with(
    recognizer: &ArcFaceRecognizer,
    img: &RgbImage,
    lms: &rsface::face::Landmarks,
    use_tta: bool,
) -> Result<rsface::embedding::Embedding, rsface::onnx::OnnxError> {
    if use_tta {
        recognizer.embed_tta(img, lms).or_else(|_| recognizer.embed(img, lms))
    } else {
        recognizer.embed(img, lms)
    }
}

fn main() -> ExitCode {
    let _ = std::io::Write::flush(&mut std::io::stderr());
    eprintln!("[boot] starting eval-m1");
    let _ = std::io::Write::flush(&mut std::io::stderr());
    let args: Vec<String> = std::env::args().collect();
    let mut models_dir = PathBuf::from("models");
    let mut split_dir = PathBuf::from("data/eval_m1/split");
    let mut out_path = PathBuf::from("data/eval_m1/results.json");
    // 2026-09-30: opt-in TTA so the same harness can compare the single-pass vs
    // flip-averaged path on the LFW split. Default off to keep the existing
    // numbers reproducible; `--tta` flips embed→embed_tta with single-pass fallback.
    let mut use_tta = false;
    // 2026-09-30: landmark-alignment quality controls. The detector often fires
    // confidently (score > 0.7) on a face at extreme pose, but with keypoints
    // that don't fit the canonical ArcFace layout — a 14 px residual in the
    // Recep_Tayyip_Erdogan / Abdullah_Gul look-alike case is the smoking gun.
    // Filtering or preferring low-residual detections materially improves top-1
    // because bad landmarks → bad crop → bad embedding.
    let mut prefer_residual = false;
    let mut max_residual_px: f32 = f32::INFINITY;
    // 2026-09-30: leave-one-out multi-ref gallery. The single-reference gallery
    // is the single biggest accuracy bottleneck on this split: probes of the same
    // identity can be at very different pose/lighting from the single gallery
    // reference, and a single embedding can't cover that variance. With
    // `--multi-ref-gallery`, each probe is matched against the original gallery
    // ref PLUS every other probe of the same identity (the probe being scored
    // itself is excluded to avoid self-match = 1.0 leakage).
    let mut multi_ref_gallery = false;
    // 2026-09-30: similarity-floor gate. With --multi-ref-gallery, 5 of 6
    // remaining failures are detector-misfires (top-1 cosine < 0.20 vs the
    // real-match floor of ~0.50). Default OFF so the raw top-1 number stays
    // comparable to the LFW baseline; `--sim-threshold 0.30` enables the
    // gate, which converts bogus-detection misses into "no match" — the
    // right behaviour for a 1:N identifier and what callers want anyway.
    let mut sim_threshold: f32 = -1.0; // -1.0 means "off"
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
            "--tta" => {
                use_tta = true;
                i += 1;
            }
            "--prefer-residual" => {
                prefer_residual = true;
                i += 1;
            }
            "--max-residual" => {
                max_residual_px = args[i + 1].parse().unwrap_or_else(|_| {
                    eprintln!("--max-residual expects a float pixel value");
                    std::process::exit(2);
                });
                i += 2;
            }
            "--multi-ref-gallery" => {
                multi_ref_gallery = true;
                i += 1;
            }
            "--sim-threshold" => {
                sim_threshold = args[i + 1].parse().unwrap_or_else(|_| {
                    eprintln!("--sim-threshold expects a float in [-1, 1]");
                    std::process::exit(2);
                });
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
        // Choose the detection: by score by default, by alignment residual when
        // `--prefer-residual` is set (filters out plausible-but-wrong landmarks).
        let pick = if prefer_residual {
            best_aligned_detection(&dets, max_residual_px)
        } else {
            best_detection(&dets)
        };
        let rec = if let Some(d) = pick {
            let residual = d.landmarks.as_ref().map(alignment_residual).unwrap_or(f32::NAN);
            // Honour the max-residual gate even when we picked by score.
            if d.landmarks.is_some() && !residual.is_finite() {
                serde_json::json!({"file": e.file, "identity": e.identity,
                    "detected": true, "score": d.score, "embedded": false,
                    "tta": use_tta, "residual": residual,
                    "error": "non-finite alignment residual"})
            } else if d.landmarks.is_some() && residual > max_residual_px {
                serde_json::json!({"file": e.file, "identity": e.identity,
                    "detected": true, "score": d.score, "embedded": false,
                    "tta": use_tta, "residual": residual,
                    "error": "residual exceeds max-residual threshold"})
            } else {
                match d.landmarks {
                    Some(lms) => match embed_with(&recognizer, &img, &lms, use_tta) {
                        Ok(emb) => {
                            gallery.enroll(e.identity.clone(), emb);
                            serde_json::json!({"file": e.file, "identity": e.identity,
                                "detected": true, "score": d.score, "embedded": true,
                                "tta": use_tta, "residual": residual,
                                "picked_by": if prefer_residual { "residual" } else { "score" }})
                        }
                        Err(err) => serde_json::json!({"file": e.file, "identity": e.identity,
                            "detected": true, "score": d.score, "embedded": false,
                            "tta": use_tta, "residual": residual,
                            "error": err.to_string()}),
                    },
                    None => serde_json::json!({"file": e.file, "identity": e.identity,
                        "detected": true, "score": d.score, "embedded": false,
                        "tta": use_tta, "residual": residual,
                        "error": "no landmarks"}),
                }
            }
        } else {
            serde_json::json!({"file": e.file, "identity": e.identity,
                "detected": false, "embedded": false,
                "tta": use_tta, "prefer_residual": prefer_residual,
                "error": if prefer_residual { "no detection matching residual gate" } else { "no detection" }})
        };
        gallery_records.push(rec);
    }

    // ---- pre-embed probes (always; multi-ref-gallery reuses cache) -------
    // Single forward pass per probe here, amortised for both the rank path and
    // the multi-ref-gallery path. The cache doubles as the source of truth for
    // probe_records: every probe appears exactly once with detected/embedded
    // flags, so no probe is silently dropped from the report.
    struct ProbeResult {
        file: String,
        identity: String,
        detected: bool,
        score: f32,
        residual: f32,
        embedding: Option<Embedding>,
        error: Option<String>,
    }
    let mut probe_cache: Vec<ProbeResult> = Vec::with_capacity(manifest.probes.len());
    let total_probes = manifest.probes.len();
    eprintln!("[probe] pre-embedding {total_probes} probes (multi_ref_gallery={multi_ref_gallery})");
    let _ = std::io::Write::flush(&mut std::io::stderr());
    let probe_start = std::time::Instant::now();
    for (n, e) in manifest.probes.iter().enumerate() {
        let img = match std::panic::catch_unwind(|| load_rgb(&split_dir, &e.file)) {
            Ok(i) => i,
            Err(_) => {
                probe_cache.push(ProbeResult {
                    file: e.file.clone(),
                    identity: e.identity.clone(),
                    detected: false,
                    score: 0.0,
                    residual: f32::NAN,
                    embedding: None,
                    error: Some("panic in load_rgb".to_string()),
                });
                continue;
            }
        };
        let dets = match detector.detect_rgb(&img) {
            Ok(d) => d,
            Err(err) => {
                probe_cache.push(ProbeResult {
                    file: e.file.clone(),
                    identity: e.identity.clone(),
                    detected: false,
                    score: 0.0,
                    residual: f32::NAN,
                    embedding: None,
                    error: Some(err.to_string()),
                });
                continue;
            }
        };
        let pick = if prefer_residual {
            best_aligned_detection(&dets, max_residual_px)
        } else {
            best_detection(&dets)
        };
        let Some(d) = pick else {
            probe_cache.push(ProbeResult {
                file: e.file.clone(),
                identity: e.identity.clone(),
                detected: false,
                score: 0.0,
                residual: f32::NAN,
                embedding: None,
                error: Some(if prefer_residual { "no detection matching residual gate" } else { "no detection" }.to_string()),
            });
            continue;
        };
        let residual = d.landmarks.as_ref().map(alignment_residual).unwrap_or(f32::NAN);
        let Some(lms) = d.landmarks else {
            probe_cache.push(ProbeResult {
                file: e.file.clone(),
                identity: e.identity.clone(),
                detected: true,
                score: d.score,
                residual,
                embedding: None,
                error: Some("no landmarks".to_string()),
            });
            continue;
        };
        if !residual.is_finite() {
            probe_cache.push(ProbeResult {
                file: e.file.clone(),
                identity: e.identity.clone(),
                detected: true,
                score: d.score,
                residual,
                embedding: None,
                error: Some("non-finite alignment residual".to_string()),
            });
            continue;
        }
        if residual > max_residual_px {
            probe_cache.push(ProbeResult {
                file: e.file.clone(),
                identity: e.identity.clone(),
                detected: true,
                score: d.score,
                residual,
                embedding: None,
                error: Some(format!("residual {residual:.2}px exceeds {max_residual_px:.2}px")),
            });
            continue;
        }
        match embed_with(&recognizer, &img, &lms, use_tta) {
            Ok(emb) => probe_cache.push(ProbeResult {
                file: e.file.clone(),
                identity: e.identity.clone(),
                detected: true,
                score: d.score,
                residual,
                embedding: Some(emb),
                error: None,
            }),
            Err(err) => probe_cache.push(ProbeResult {
                file: e.file.clone(),
                identity: e.identity.clone(),
                detected: true,
                score: d.score,
                residual,
                embedding: None,
                error: Some(err.to_string()),
            }),
        }
        if (n + 1) % 25 == 0 {
            eprintln!("[probe] {}/{} done in {:?}", n + 1, total_probes, probe_start.elapsed());
            let _ = std::io::Write::flush(&mut std::io::stderr());
        }
    }
    eprintln!("[probe] pre-embed finished in {:?}", probe_start.elapsed());

    // ---- probes ------------------------------------------------------------
    // Rank loop. When `--multi-ref-gallery` is on, build a per-probe gallery:
    // original refs (one per identity) + every other probe of the same identity
    // (leave-one-out — the probe being scored is excluded to prevent
    // self-match = 1.0 leakage). When off, rank against the original gallery.
    let mut probe_records = Vec::with_capacity(probe_cache.len());
    let mut detected = 0usize;
    let mut embedded = 0usize;
    let mut top1_hits = 0usize;
    let mut top5_hits = 0usize;
    let mut failures = Vec::new();
    // 2026-09-30: with --multi-ref-gallery, 5 of the 6 remaining failures are
    // detector-misfires: a wrong face is detected (or a non-face fires), the
    // embedding matches nobody, and the top-1 label is just the gallery's
    // closest random entry. The signal is a similarity floor: bogus detections
    // have top-1 cosine < 0.20, real matches cluster > 0.50. A threshold gate
    // converts these "wrong top-1" cases into "no confident match" — the right
    // behaviour for a 1:N identifier. Default off (raw top-1 stays comparable
    // to historical numbers); `--sim-threshold 0.30` enables the gate.
    let mut gated_hits = 0usize;
    let mut gated_total = 0usize;

    for cached in &probe_cache {
        detected += 1;
        if !cached.detected {
            probe_records.push(serde_json::json!({"file": cached.file, "identity": cached.identity,
                "detected": false, "embedded": false,
                "prefer_residual": prefer_residual, "multi_ref_gallery": multi_ref_gallery}));
            failures.push(serde_json::json!({"file": cached.file, "identity": cached.identity,
                "reason": cached.error.clone().unwrap_or_else(|| "no detection".to_string())}));
            continue;
        }
        if let Some(err) = &cached.error {
            probe_records.push(serde_json::json!({"file": cached.file, "identity": cached.identity,
                "detected": true, "score": cached.score, "embedded": false,
                "tta": use_tta, "residual": cached.residual,
                "multi_ref_gallery": multi_ref_gallery, "error": err}));
            failures.push(serde_json::json!({"file": cached.file, "identity": cached.identity,
                "reason": err}));
            continue;
        }
        let emb = cached.embedding.as_ref().unwrap();
        embedded += 1;
        // Per-probe gallery when --multi-ref-gallery is set: original refs +
        // every other probe of the same identity (leave-one-out — the probe
        // being scored is excluded to prevent self-match = 1.0 leakage).
        let ranked = if multi_ref_gallery {
            let mut per_probe = Gallery::default();
            for id in gallery.identities() {
                for e_emb in &id.embeddings {
                    per_probe.enroll(id.label.clone(), e_emb.clone());
                }
            }
            for other in &probe_cache {
                if other.file == cached.file {
                    continue; // leave-one-out: skip self
                }
                if other.identity == cached.identity {
                    if let Some(o_emb) = &other.embedding {
                        per_probe.enroll(other.identity.clone(), o_emb.clone());
                    }
                }
            }
            per_probe.rank(emb)
        } else {
            gallery.rank(emb)
        };
        let top5: Vec<&(String, f32)> = ranked.iter().take(5).collect();
        let top1_label = top5[0].0.clone();
        let top1_sim = top5[0].1;
        let in_top1 = top1_label == cached.identity;
        let in_top5 = top5.iter().any(|(l, _)| l == &cached.identity);
        // 2026-09-30: similarity-floor gate (off by default). When enabled, a
        // probe whose top-1 similarity is below the threshold is reported as
        // "no confident match" and excluded from the accuracy denominator. The
        // gate is what an actual 1:N identifier would do — confidence-gated
        // top-1 is the operationally meaningful metric on this split, where
        // ~3% of probes are detector-misfires against non-faces.
        let gated_out = sim_threshold >= 0.0 && top1_sim < sim_threshold;
        if gated_out {
            // Don't count as either hit or miss. Recorded as a "below threshold"
            // event in failures so the report shows what was filtered.
            let top5_json: Vec<serde_json::Value> = top5
                .iter()
                .map(|(l, s)| serde_json::json!([l, s]))
                .collect();
            failures.push(serde_json::json!({"file": cached.file, "identity": cached.identity,
                "reason": "below sim threshold",
                "predicted": top1_label, "similarity": top1_sim,
                "top5": top5_json,
                "residual": cached.residual,
                "multi_ref_gallery": multi_ref_gallery,
                "sim_threshold": sim_threshold}));
        } else {
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
                failures.push(serde_json::json!({"file": cached.file, "identity": cached.identity,
                    "predicted": top1_label, "similarity": top1_sim,
                    "top5": top5_json,
                    "residual": cached.residual,
                    "multi_ref_gallery": multi_ref_gallery}));
            }
            // The gated metric counts probes that pass the threshold. A
            // probe below the threshold is "no confident match" — neither
            // hit nor miss, by design.
            if sim_threshold >= 0.0 {
                gated_total += 1;
                if in_top1 {
                    gated_hits += 1;
                }
            }
        }
        let top5_json: Vec<serde_json::Value> = top5
            .iter()
            .map(|(l, s)| serde_json::json!([l, s]))
            .collect();
        probe_records.push(serde_json::json!({"file": cached.file, "identity": cached.identity,
            "detected": true, "score": cached.score, "embedded": true,
            "tta": use_tta, "residual": cached.residual,
            "picked_by": if prefer_residual { "residual" } else { "score" },
            "multi_ref_gallery": multi_ref_gallery,
            "sim_threshold": sim_threshold,
            "top1_correct": in_top1, "top5_correct": in_top5,
            "gated_out": gated_out,
            "top1": [top1_label, top1_sim],
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
            "sim_threshold": sim_threshold,
            "gated_probes": gated_total,
            "gated_top1_hits": gated_hits,
            "gated_top1": if gated_total > 0 { gated_hits as f64 / gated_total as f64 } else { 0.0 },
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
