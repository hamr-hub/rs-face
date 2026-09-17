//! Offline ground-truth preparation for the zero-dep LBPH accuracy bench.
//!
//! This binary is **not** part of the zero-dependency story: it requires the
//! `ort-backend` feature and real SCRFD + ArcFace weights. It is an offline labelling
//! tool. For every converted frame (`*.ppm`, or stored `*.png`) under the input
//! directory it:
//!
//!   1. detects the largest face with SCRFD-10G,
//!   2. embeds that face with ArcFace R50 and groups frames into identities by cosine
//!      similarity (incremental clustering),
//!   3. writes a plain **bounding-box grayscale crop** (120x120, square, +30% margin)
//!      using only image operations the zero-dep crate itself ships — no keypoint
//!      alignment — so the LBPH bench measures a recogniser fed by "a correct box",
//!      which is the honest isolated-recogniser condition.
//!
//! Output filenames encode the ArcFace-derived ground-truth label:
//!
//! ```text
//! <out>/id00__<source-dir>__<frame-stem>.pgm
//! <out>/labels.csv
//! ```
//!
//! Usage:
//!
//! ```sh
//! tools/lbph_prep.sh                       # converts frames, then runs this
//! # or directly:
//! cargo run --release --features ort-backend --bin prep_lbph_crops -- \
//!     out/lbph_frames out/lbph_crops
//! ```

use std::collections::BTreeMap;
use std::fs;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use rsface::embedding::Embedding;
use rsface::image::codec;
use rsface::image::png;
use rsface::image::{GrayImage, RgbImage};
use rsface::onnx::Backend;
use rsface::scrfd::ScrfdConfig;

/// ArcFace cosine joining threshold. Same-identity drama-frame pairs typically stay
/// well above this; measured fixture margin was same≈1.0 vs diff≈0.07.
const CLUSTER_SIM: f32 = 0.40;
/// Minimum detected face box in original-frame pixels.
const MIN_FACE_PX: f32 = 80.0;
/// Extra margin around the SCRFD box for the LBPH crop (OpenCV LBPH convention).
const CROP_EXPAND: f32 = 1.30;
/// LBPH normalised crop size; keep in sync with `LbphConfig::default().face_size`.
const CROP_SIZE: usize = 120;
/// Detector score floor for frames accepted into the evaluation set.
const DET_SCORE_MIN: f32 = 0.5;

struct Args {
    input: PathBuf,
    output: PathBuf,
}

fn parse_args() -> Result<Args, String> {
    let positional: Vec<String> = std::env::args().skip(1).collect();
    match positional.as_slice() {
        [input, output] => Ok(Args {
            input: PathBuf::from(input),
            output: PathBuf::from(output),
        }),
        _ => Err("usage: prep_lbph_crops <frames_dir> <crops_dir>".to_string()),
    }
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    if !args.input.is_dir() {
        eprintln!("input directory {} not found", args.input.display());
        std::process::exit(2);
    }
    prepare_output_dir(&args.output);

    let (det, rec) = match load_models() {
        Ok(x) => x,
        Err(msg) => {
            eprintln!("{msg}");
            std::process::exit(2);
        }
    };

    let frames = collect_frames(&args.input);
    println!(
        "scanning {} frames under {}",
        frames.len(),
        args.input.display()
    );

    // Clusters hold ArcFace embeddings + provenance for auditing.
    let mut clusters: Vec<Cluster> = Vec::new();
    let mut written = Vec::new();
    let mut skipped = 0usize;

    for path in &frames {
        let img = match read_rgb(path) {
            Some(img) => img,
            None => continue,
        };
        let dets = det.detect_rgb(&img).expect("SCRFD detect");
        let Some(face) = dets
            .into_iter()
            .filter(|d| d.score >= DET_SCORE_MIN)
            .filter(|d| d.width().min(d.height()) >= MIN_FACE_PX)
            .filter(|d| d.landmarks.is_some())
            .max_by(|a, b| a.score.total_cmp(&b.score))
        else {
            // No landmarked face large/confident enough to label from this frame.
            skipped += 1;
            continue;
        };
        let lms = face.landmarks.unwrap();

        let emb = rec.embed(&img, &lms).expect("ArcFace embed");
        let (cid, join_sim) = assign_cluster(&mut clusters, &emb);

        let gray = box_crop_gray(&img, &face);
        let source = path
            .parent()
            .and_then(|p| p.file_name())
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "frame".to_string());
        let stem = path.file_stem().unwrap().to_string_lossy().to_string();
        let file = format!("id{cid:02}__{source}__{stem}.pgm");
        write_pgm(&args.output.join(&file), &gray);
        written.push(WrittenCrop {
            file: file.clone(),
            cluster: cid,
            source,
            frame: stem,
            score: face.score,
            join_sim,
        });
    }

    write_labels_csv(&args.output.join("labels.csv"), &written);
    print_audit(&args.output, &clusters, &written, frames.len(), skipped);

    if clusters.iter().filter(|c| c.members.len() >= 2).count() < 2 {
        eprintln!(
            "\nWARNING: fewer than 2 clusters with >= 2 crops; the LBPH bench needs \
             multiple identities with repeated frames to be meaningful."
        );
    }
}

struct WrittenCrop {
    file: String,
    cluster: usize,
    source: String,
    frame: String,
    score: f32,
    join_sim: f32,
}

struct Cluster {
    members: Vec<Embedding>,
}

/// Assign `emb` to the cluster whose nearest member is most similar; create a new
/// cluster when the best similarity is below `CLUSTER_SIM`. Returns (cluster id, the
/// similarity that justified the assignment — `None`-equivalent 1.0 for a new cluster).
fn assign_cluster(clusters: &mut Vec<Cluster>, emb: &Embedding) -> (usize, f32) {
    let mut best: Option<(usize, f32)> = None;
    for (cid, c) in clusters.iter().enumerate() {
        for m in &c.members {
            let sim = emb.cosine(m).expect("same embedding dim");
            if best.map_or(true, |(_, bs)| sim > bs) {
                best = Some((cid, sim));
            }
        }
    }
    match best {
        Some((cid, sim)) if sim >= CLUSTER_SIM => {
            clusters[cid].members.push(emb.clone());
            (cid, sim)
        }
        _ => {
            clusters.push(Cluster {
                members: vec![emb.clone()],
            });
            (clusters.len() - 1, 1.0)
        }
    }
}

/// Square, edge-clamped crop around the box centre with `CROP_EXPAND` margin,
/// converted to gray and bilinear-resized to the LBPH face size. Only zero-dep image
/// primitives are used (pixel copy, `RgbImage::to_gray`, `resize_bilinear`).
fn box_crop_gray(img: &RgbImage, face: &rsface::face::FaceDetection) -> GrayImage {
    let cx = (face.x1 + face.x2) / 2.0;
    let cy = (face.y1 + face.y2) / 2.0;
    let side = face.width().max(face.y2 - face.y1) * CROP_EXPAND;
    let half = side / 2.0;

    let x0 = (cx - half).round().max(0.0) as usize;
    let y0 = (cy - half).round().max(0.0) as usize;
    let x1 = ((cx + half).round() as usize).min(img.width());
    let y1 = ((cy + half).round() as usize).min(img.height());
    let (w, h) = (x1.saturating_sub(x0).max(1), y1.saturating_sub(y0).max(1));

    let mut square = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let sx = (x0 + x).min(img.width() - 1);
            let sy = (y0 + y).min(img.height() - 1);
            let src = &img.row(sy)[sx * 3..sx * 3 + 3];
            let dst = square.row_mut(y);
            dst[x * 3..x * 3 + 3].copy_from_slice(src);
        }
    }
    let gray = square.to_gray();
    if w == CROP_SIZE && h == CROP_SIZE {
        gray
    } else {
        gray.resize_bilinear(CROP_SIZE, CROP_SIZE)
    }
}

fn load_models() -> Result<
    (
        rsface::scrfd_detector::ScrfdDetector,
        rsface::arcface_recognizer::ArcFaceRecognizer,
    ),
    String,
> {
    let available = Backend::available();
    if !available.contains(&Backend::Ort) {
        return Err(
            "ort backend unavailable at runtime (set ORT_DYLIB_PATH to a libonnxruntime \
             shared library)."
                .to_string(),
        );
    }
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models");
    let dpath = model_dir.join("det_10g.onnx");
    let rpath = model_dir.join("w600k_r50.onnx");
    if !dpath.exists() || !rpath.exists() {
        return Err(format!(
            "model weights missing under {} — run tools/fetch_models.sh --all",
            model_dir.display()
        ));
    }
    let cfg = rsface::onnx::SessionConfig::default()
        .with_backend(Backend::Ort)
        .with_threads(1);
    let det = rsface::scrfd_detector::ScrfdDetector::open(
        &dpath,
        Some(rsface::models::find("scrfd_10g_kps").unwrap()),
        &cfg,
        ScrfdConfig::default().with_min_face_size(MIN_FACE_PX),
    )
    .map_err(|e| format!("load detector: {e}"))?;
    let rec = rsface::arcface_recognizer::ArcFaceRecognizer::open(
        &rpath,
        Some(rsface::models::find("arcface_w600k_r50").unwrap()),
        &cfg,
    )
    .map_err(|e| format!("load recogniser: {e}"))?;
    Ok((det, rec))
}

/// Frames the lab pipeline can decode without a JPEG decoder: PPM is the normal path
/// (`tools/lbph_prep.sh` converts the source JPGs with Pillow); uncompressed/stored PNG
/// also works via the crate's native decoder.
fn collect_frames(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    walk_frames(root, &mut out);
    out.sort();
    out
}

fn walk_frames(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("read frames dir") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            walk_frames(&path, out);
        } else if path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "ppm" | "png"))
        {
            out.push(path);
        }
    }
}

fn read_rgb(path: &Path) -> Option<RgbImage> {
    let f = fs::File::open(path).ok()?;
    let mut r = std::io::BufReader::new(f);
    let decoded = match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("ppm") => codec::read_ppm(&mut r),
        Some("png") => png::decode_to_rgb(&mut r),
        _ => return None,
    };
    match decoded {
        Ok(img) => Some(img),
        Err(e) => {
            eprintln!("skip {}: {e}", path.display());
            None
        }
    }
}

fn write_pgm(path: &Path, img: &GrayImage) {
    let f = fs::File::create(path).expect("create crop");
    codec::write_pgm(&mut BufWriter::new(f), img).expect("write pgm");
}

/// Refuse to run into a populated directory: stale crops from a previous clustering
/// would silently contaminate the bench.
fn prepare_output_dir(dir: &Path) {
    if let Ok(entries) = fs::read_dir(dir) {
        if entries.flatten().next().is_some() {
            eprintln!(
                "output directory {} is not empty; remove it (or pick a new path) and rerun",
                dir.display()
            );
            std::process::exit(2);
        }
    }
    fs::create_dir_all(dir).expect("create crops dir");
}

fn write_labels_csv(path: &Path, rows: &[WrittenCrop]) {
    let mut csv = String::from("crop,cluster,source,frame,det_score,join_cosine\n");
    for r in rows {
        csv.push_str(&format!(
            "{},id{:02},{},{},{:.4},{:.4}\n",
            r.file, r.cluster, r.source, r.frame, r.score, r.join_sim
        ));
    }
    fs::write(path, csv).expect("write labels.csv");
}

fn print_audit(
    out_dir: &Path,
    clusters: &[Cluster],
    rows: &[WrittenCrop],
    frames: usize,
    dropped: usize,
) {
    println!("\nframes scanned: {frames}");
    println!("frames without an accepted landmarked face: {dropped}");
    println!("crops written: {}", rows.len());
    println!(
        "identities (clusters at cosine >= {CLUSTER_SIM}): {}",
        clusters.len()
    );

    // Per-cluster audit: size, sources represented, exemplar frames, and the minimum
    // cosine between members — a low value means a likely bad merge worth inspecting.
    for (cid, c) in clusters.iter().enumerate() {
        let members: Vec<&WrittenCrop> = rows.iter().filter(|r| r.cluster == cid).collect();
        let mut sources: BTreeMap<&str, usize> = BTreeMap::new();
        for m in &members {
            *sources.entry(m.source.as_str()).or_default() += 1;
        }
        let min_join = members
            .iter()
            .map(|m| m.join_sim)
            .fold(f32::INFINITY, f32::min);
        let src: Vec<String> = sources.iter().map(|(s, n)| format!("{s}:{n}")).collect();
        println!(
            "  id{cid:02}: n={} sources=[{}] min_join_cos={min_join:.3} exemplars=[{}]",
            members.len(),
            src.join(","),
            members
                .iter()
                .take(3)
                .map(|m| m.frame.as_str())
                .collect::<Vec<_>>()
                .join(","),
        );
    }
    println!(
        "\nlabels written to {}",
        out_dir.join("labels.csv").display()
    );
}
