//! Side-by-side comparison of recognition accuracy with vs. without test-time
//! augmentation (TTA).
//!
//! Loads the LFW-style manifest at `data/eval_m1/split/manifest.json`, runs the
//! SCRFD+ArcFace pipeline twice on every probe — once with the single-pass
//! [`ArcFaceRecognizer::embed`] and once with the flip-averaged
//! [`ArcFaceRecognizer::embed_tta`] — and prints per-mode top-1/top-5 plus a
//! per-probe delta on the cases that flip between correct and wrong.
//!
//! The single-pass results match what `tools/eval_m1` already produces, so
//! anything this binary surfaces as "TTA fixed N cases / broke M cases" is the
//! marginal lift attributable to the augmentation alone.
//!
//! Run with:
//!   ORT_DYLIB_PATH=data/eval_m1/libonnxruntime.so \
//!     cargo run --release --features ort-backend --bin accuracy-tta
//!
//! Skipped (not failed) when model weights or the LFW split are absent.

#![cfg(any(feature = "ort-backend", feature = "tract-backend"))]

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use rsface::arcface_recognizer::ArcFaceRecognizer;
use rsface::embedding::{Embedding, Gallery};
use rsface::face::FaceDetection;
use rsface::image::RgbImage;
use rsface::models;
use rsface::onnx::SessionConfig;
use rsface::scrfd::ScrfdConfig;
use rsface::scrfd_detector::ScrfdDetector;

#[derive(Clone)]
struct Entry {
    file: String,
    identity: String,
}

struct Manifest {
    gallery: Vec<Entry>,
    probes: Vec<Entry>,
}

/// Minimal manifest parser tailored to `data/eval_m1/split/manifest.json`:
/// two arrays (`gallery`, `probes`), each an array of `{ "file": "...", "identity": "..." }`.
///
/// Pretty-printed JSON means each object's keys live on separate lines, so we
/// walk char-by-char and accumulate a per-object buffer between `{` and `}`.
/// Avoids pulling in serde for this single read.
fn parse_manifest(text: &str) -> Manifest {
    let mut out = Manifest {
        gallery: Vec::new(),
        probes: Vec::new(),
    };
    // 1) Find each section ("gallery" / "probes") by scanning for `"key":` followed by `[`.
    // 2) Within a section, walk every `{ ... }` object and pull the two string
    //    fields we care about. The walker tolerates nested braces (none here)
    //    by counting depth.
    let bytes = text.as_bytes();
    let mut i = 0;
    let mut current_section: Option<&'static str> = None;
    while i < bytes.len() {
        // Section header: look for `"<name>":` then check the next non-space is `[`.
        if bytes[i] == b'"' {
            if let Some(end_quote) = find_byte(bytes, i + 1, b'"') {
                let key = &text[i + 1..end_quote];
                if matches!(key, "gallery" | "probes") {
                    let after = skip_ws(bytes, end_quote + 1);
                    if after < bytes.len() && bytes[after] == b':' {
                        let after_colon = skip_ws(bytes, after + 1);
                        if after_colon < bytes.len() && bytes[after_colon] == b'[' {
                            current_section = Some(if key == "gallery" {
                                "gallery"
                            } else {
                                "probes"
                            });
                            i = after_colon + 1;
                            continue;
                        }
                    }
                }
                i = end_quote + 1;
                continue;
            }
        }

        // Inside a section: each top-level `{ ... }` object is one entry.
        if current_section.is_some() && bytes[i] == b'{' {
            // Walk to matching `}`, tracking nested braces.
            let mut depth = 1i32;
            let mut j = i + 1;
            while j < bytes.len() && depth > 0 {
                match bytes[j] {
                    b'{' => depth += 1,
                    b'}' => depth -= 1,
                    b'"' => {
                        // Skip string literal so braces inside don't fool us.
                        j = find_byte(bytes, j + 1, b'"').unwrap_or(bytes.len());
                    }
                    _ => {}
                }
                if depth == 0 {
                    break;
                }
                j += 1;
            }
            let obj_text = &text[i + 1..j];
            let file = extract_str(obj_text, "file");
            let identity = extract_str(obj_text, "identity");
            if let (Some(file), Some(identity)) = (file, identity) {
                let entry = Entry { file, identity };
                if current_section == Some("gallery") {
                    out.gallery.push(entry);
                } else {
                    out.probes.push(entry);
                }
            }
            i = j + 1;
            continue;
        }

        i += 1;
    }
    out
}

fn skip_ws(bytes: &[u8], mut i: usize) -> usize {
    while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b'\t' || bytes[i] == b'\n' || bytes[i] == b'\r') {
        i += 1;
    }
    i
}

fn find_byte(bytes: &[u8], from: usize, target: u8) -> Option<usize> {
    let mut k = from;
    while k < bytes.len() {
        if bytes[k] == target {
            return Some(k);
        }
        // Don't walk past a closing brace at this depth — caller handles.
        if bytes[k] == b'}' {
            return None;
        }
        k += 1;
    }
    None
}

fn extract_str(text: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\":");
    let start = text.find(&needle)? + needle.len();
    let rest = text[start..].trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn model_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models")
}

fn split_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/eval_m1/split")
}

fn preferred_backend() -> rsface::onnx::Backend {
    use rsface::onnx::Backend;
    let avail = Backend::available();
    [Backend::Ort, Backend::Tract]
        .into_iter()
        .find(|b| avail.contains(b))
        .expect("no inference backend enabled; build with --features ort-backend or tract-backend")
}

fn load_rgb(root: &Path, rel: &str) -> RgbImage {
    let as_rel = root.join(rel);
    let p = if as_rel.exists() {
        as_rel
    } else {
        Path::new(rel).to_path_buf()
    };
    let bytes = std::fs::read(&p).unwrap_or_else(|e| {
        eprintln!("cannot read {p:?}: {e}");
        std::process::exit(2);
    });
    rsface::image::jpeg::decode_jpeg_rgb(&bytes).unwrap_or_else(|e| {
        eprintln!("cannot decode {p:?}: {e}");
        std::process::exit(2);
    })
}

fn best_detection(dets: &[FaceDetection]) -> Option<&FaceDetection> {
    dets.iter().fold(None, |acc: Option<&FaceDetection>, d| match acc {
        Some(a) if a.score >= d.score => Some(a),
        _ => Some(d),
    })
}

struct ProbeResult {
    file: String,
    identity: String,
    /// (label, similarity, in_top1, in_top5) — single-pass mode.
    single: (String, f32, bool, bool),
    /// Same fields for TTA mode.
    tta: (String, f32, bool, bool),
}

fn topk_correct(ranked: &[(String, f32)], identity: &str, k: usize) -> (String, f32, bool, bool) {
    let top1 = ranked.first().cloned().unwrap_or_default();
    let in_top1 = top1.0 == identity;
    let in_top5 = ranked.iter().take(k).any(|(l, _)| l == identity);
    (top1.0, top1.1, in_top1, in_top5)
}

fn main() -> ExitCode {
    // CLI: --max-gallery N  --max-probes N  --probe-start N  --probe-contains a,b,c
    let mut max_gallery: Option<usize> = None;
    let mut max_probes: Option<usize> = None;
    let mut probe_start: usize = 0;
    let mut probe_filter: Option<Vec<String>> = None;
    {
        let mut args = std::env::args().skip(1);
        while let Some(a) = args.next() {
            match a.as_str() {
                "--max-gallery" => {
                    max_gallery = args.next().and_then(|s| s.parse().ok());
                }
                "--max-probes" => {
                    max_probes = args.next().and_then(|s| s.parse().ok());
                }
                "--probe-start" => {
                    probe_start = args.next().and_then(|s| s.parse().ok()).unwrap_or(0);
                }
                "--probe-contains" => {
                    let raw = args.next().unwrap_or_default();
                    probe_filter = Some(
                        raw.split(',')
                            .map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty())
                            .collect(),
                    );
                }
                "--help" | "-h" => {
                    eprintln!(
                        "usage: accuracy-tta [--max-gallery N] [--max-probes N] [--probe-start N] \
                         [--probe-contains A,B,C]"
                    );
                    return ExitCode::SUCCESS;
                }
                other => {
                    eprintln!("unknown arg: {other}");
                    return ExitCode::from(2);
                }
            }
        }
    }

    let models_dir = model_dir();
    let split = split_dir();
    let det_path = models_dir.join("det_10g.onnx");
    let rec_path = models_dir.join("w600k_r50.onnx");
    if !det_path.exists() || !rec_path.exists() || !split.join("manifest.json").exists() {
        eprintln!(
            "SKIP: missing model weights ({det_path:?} or {rec_path:?}) or split ({:?}); \
             run tools/fetch_models.sh --all and tools/eval_m1/make_jpeg_corpus.py",
            split
        );
        return ExitCode::SUCCESS;
    }

    let backend = preferred_backend();
    eprintln!("backend: {}", backend.as_str());

    let session_cfg = SessionConfig::default().with_backend(backend).with_threads(1);
    let det_spec = models::REGISTRY
        .iter()
        .find(|s| s.file_name == "det_10g.onnx")
        .expect("det_10g spec");
    let rec_spec = models::REGISTRY
        .iter()
        .find(|s| s.file_name == "w600k_r50.onnx")
        .expect("w600k_r50 spec");
    let detector = ScrfdDetector::open(&det_path, Some(det_spec), &session_cfg, ScrfdConfig::default())
        .expect("load SCRFD");
    let recognizer =
        ArcFaceRecognizer::open(&rec_path, Some(rec_spec), &session_cfg).expect("load ArcFace");

    let manifest_text = std::fs::read_to_string(split.join("manifest.json")).expect("read manifest");
    let manifest = parse_manifest(&manifest_text);
    eprintln!(
        "manifest: {} gallery entries, {} probe entries",
        manifest.gallery.len(),
        manifest.probes.len()
    );
    if let Some(first) = manifest.gallery.first() {
        eprintln!("first gallery: {} -> {}", first.identity, first.file);
    }

    // ---- gallery (single-pass embeddings — keep the comparison fair) ----
    let mut gallery = Gallery::default();
    let gallery_entries: Vec<Entry> = match max_gallery {
        Some(n) => manifest.gallery.iter().take(n).cloned().collect(),
        None => manifest.gallery.clone(),
    };
    let t_gallery_start = std::time::Instant::now();
    for (gi, e) in gallery_entries.iter().enumerate() {
        let t0 = std::time::Instant::now();
        let img = load_rgb(&split, &e.file);
        let t1 = std::time::Instant::now();
        let dets = detector.detect_rgb(&img).expect("detect");
        let t2 = std::time::Instant::now();
        if let Some(d) = best_detection(&dets) {
            if let Some(lms) = d.landmarks.as_ref() {
                if let Ok(emb) = recognizer.embed(&img, lms) {
                    gallery.enroll(e.identity.clone(), emb);
                }
            }
        }
        let t3 = std::time::Instant::now();
        if gi < 3 || gi % 10 == 0 {
            eprintln!(
                "gallery {}/{}: load={:.0}ms detect={:.0}ms embed={:.0}ms (total {:.0}ms)",
                gi + 1,
                gallery_entries.len(),
                t1.duration_since(t0).as_secs_f64() * 1000.0,
                t2.duration_since(t1).as_secs_f64() * 1000.0,
                t3.duration_since(t2).as_secs_f64() * 1000.0,
                t3.duration_since(t0).as_secs_f64() * 1000.0,
            );
        }
    }
    eprintln!(
        "gallery done in {:.1}s",
        t_gallery_start.elapsed().as_secs_f64()
    );

    // ---- probes ----
    let probe_entries: Vec<Entry> = if let Some(filter) = probe_filter.as_ref() {
        manifest
            .probes
            .iter()
            .filter(|e| {
                filter.iter().any(|needle| {
                    e.file.contains(needle) || e.identity.contains(needle)
                })
            })
            .cloned()
            .collect()
    } else {
        match max_probes {
            Some(n) => manifest
                .probes
                .iter()
                .skip(probe_start)
                .take(n)
                .cloned()
                .collect(),
            None => manifest.probes.iter().skip(probe_start).cloned().collect(),
        }
    };
    let mut results = Vec::with_capacity(probe_entries.len());
    let t_probes_start = std::time::Instant::now();
    for (pi, e) in probe_entries.iter().enumerate() {
        let img = load_rgb(&split, &e.file);
        let dets = detector.detect_rgb(&img).expect("detect");
        let Some(d) = best_detection(&dets) else {
            continue;
        };
        let Some(lms) = d.landmarks.as_ref() else {
            continue;
        };
        // Single-pass embed + rank.
        let single_emb: Embedding = match recognizer.embed(&img, lms) {
            Ok(e) => e,
            Err(_) => continue,
        };
        // TTA embed (flip-averaged). On collapse, fall back to single-pass so a
        // degenerate TTA does not artificially sink one branch's metrics.
        let tta_emb: Embedding = match recognizer.embed_tta(&img, lms) {
            Ok(e) => e,
            Err(_) => single_emb.clone(),
        };
        let ranked_s = gallery.rank(&single_emb);
        let ranked_t = gallery.rank(&tta_emb);
        results.push(ProbeResult {
            file: e.file.clone(),
            identity: e.identity.clone(),
            single: topk_correct(&ranked_s, &e.identity, 5),
            tta: topk_correct(&ranked_t, &e.identity, 5),
        });
        if pi % 20 == 0 {
            eprintln!(
                "probe {}/{} (elapsed {:.0}s)",
                pi + 1,
                probe_entries.len(),
                t_probes_start.elapsed().as_secs_f64(),
            );
        }
    }
    eprintln!(
        "probes done in {:.1}s",
        t_probes_start.elapsed().as_secs_f64()
    );

    let n = results.len();
    let s_top1 = results.iter().filter(|r| r.single.2).count();
    let t_top1 = results.iter().filter(|r| r.tta.2).count();
    let s_top5 = results.iter().filter(|r| r.single.3).count();
    let t_top5 = results.iter().filter(|r| r.tta.3).count();

    // Per-probe detail (always — small set, useful when scanning failures).
    println!();
    println!("# per-probe detail ({} entries)", n);
    for r in &results {
        let name = r.file.rsplit('/').next().unwrap_or(&r.file);
        let s_mark = if r.single.2 { '✓' } else { '✗' };
        let t_mark = if r.tta.2 { '✓' } else { '✗' };
        println!(
            "  {s_mark}{t_mark} {}: truth={} | single: {} ({:.3}) | tta: {} ({:.3})",
            name, r.identity, r.single.0, r.single.1, r.tta.0, r.tta.1,
        );
    }

    println!();
    println!("# accuracy-tta — LFW eval ({} probes)", n);
    println!();
    println!("                top-1            top-5");
    println!(
        "single-pass:   {:>4}/{} ({:5.1}%)   {:>4}/{} ({:5.1}%)",
        s_top1,
        n,
        100.0 * s_top1 as f64 / n.max(1) as f64,
        s_top5,
        n,
        100.0 * s_top5 as f64 / n.max(1) as f64,
    );
    println!(
        "tta (flip):    {:>4}/{} ({:5.1}%)   {:>4}/{} ({:5.1}%)",
        t_top1,
        n,
        100.0 * t_top1 as f64 / n.max(1) as f64,
        t_top5,
        n,
        100.0 * t_top5 as f64 / n.max(1) as f64,
    );
    println!(
        "delta top-1:   {:+} probes ({:+.2} pp)",
        t_top1 as i64 - s_top1 as i64,
        100.0 * (t_top1 as i64 - s_top1 as i64) as f64 / n.max(1) as f64
    );
    println!(
        "delta top-5:   {:+} probes ({:+.2} pp)",
        t_top5 as i64 - s_top5 as i64,
        100.0 * (t_top5 as i64 - s_top5 as i64) as f64 / n.max(1) as f64
    );

    // Per-probe deltas: which cases flipped between correct and wrong?
    let mut fixed: Vec<&ProbeResult> = results
        .iter()
        .filter(|r| !r.single.2 && r.tta.2)
        .collect();
    let mut broken: Vec<&ProbeResult> = results
        .iter()
        .filter(|r| r.single.2 && !r.tta.2)
        .collect();
    fixed.sort_by(|a, b| b.tta.1.partial_cmp(&a.tta.1).unwrap_or(std::cmp::Ordering::Equal));
    broken.sort_by(|a, b| b.tta.1.partial_cmp(&a.tta.1).unwrap_or(std::cmp::Ordering::Equal));
    println!();
    println!("TTA flipped {} probe(s) wrong → right:", fixed.len());
    for r in &fixed {
        println!(
            "  + {}: {} -> {} (sim {:.3} -> {:.3})",
            r.file.rsplit('/').next().unwrap_or(&r.file),
            r.identity,
            r.tta.0,
            r.single.1,
            r.tta.1,
        );
    }
    println!("TTA flipped {} probe(s) right → wrong:", broken.len());
    for r in &broken {
        println!(
            "  - {}: {} -> {} (sim {:.3} -> {:.3})",
            r.file.rsplit('/').next().unwrap_or(&r.file),
            r.identity,
            r.tta.0,
            r.single.1,
            r.tta.1,
        );
    }
    ExitCode::SUCCESS
}