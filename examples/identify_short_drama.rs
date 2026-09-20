//! Identify people across one or more short-drama videos.
//!
//! Pipeline:
//!
//! 1. Open each input via `rsface::source::open` (zero-dep when given a path
//!    pattern, falls back to `ffmpeg` pipe for `.mp4` etc.).
//! 2. For every frame, run a face detector (`--algo haar` or `--algo luminance`,
//!    both zero-dep) and an LBPH descriptor extractor on every detection.
//! 3. Stream detections + embeddings through `video_id::LightTracker` to
//!    collapse the per-frame bbox stream into per-person tracks.
//! 4. Cluster the tracks' mean embeddings across the whole video with
//!    `video_id::IdentityCluster` to assign identity ids.
//! 5. Across the whole video batch, re-cluster via `video_id::merge_across_videos`
//!    so the same actor in different episodes ends up with the same id.
//! 6. Write `out/manifest.json` plus per-video sub-manifests and best-bbox
//!    thumbnail PNGs.
//!
//! Zero runtime dependencies. To use industrial accuracy (SCRFD detector +
//! ArcFace recogniser) see the module docstring in `src/video_id.rs` —
//! `Identify` is a trait, so any detector / recogniser pair can be plugged
//! in. This binary sticks to the zero-dep default for "no setup required".
//!
//! ## Usage
//!
//! ```text
//! # one or many videos; mixing videos and image sequences is fine
//! cargo run --example identify_short_drama -- \
//!     clips/ep01.mp4 clips/ep02.mp4 clips/ep03.mp4 \
//!     --out ./out
//!
//! # pick a different zero-dep detector
//! cargo run --example identify_short_drama -- \
//!     clips/ep01.mp4 --algo luminance --out ./out
//! ```

use std::path::{Path, PathBuf};

use rsface::detector::{Detection, Detector, DetectorConfig};
use rsface::embedding::Embedding;
use rsface::face_detector::{FaceDetector, HaarDetector};
use rsface::haar::params::demo_face_cascade;
use rsface::image::{GrayImage, RgbImage};
use rsface::lbph::{extract as lbph_extract, LbphConfig};
use rsface::source;
use rsface::video_id::{
    identify_video, merge_across_videos, render_video_manifest_json, Identify, Track,
    VideoIdConfig, VideoIdentification,
};
use rsface::LuminanceFaceDetector;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let parsed = match parse_args(&args) {
        Ok(p) => p,
        Err(msg) => {
            eprintln!("identify_short_drama: {msg}");
            eprintln!(
                "usage: identify_short_drama <VIDEO>... --out <DIR> [--algo haar|luminance] \
                 [--iou 0.3] [--age 8] [--threshold 0.36]"
            );
            std::process::exit(2);
        }
    };

    let out_dir = &parsed.out_dir;
    std::fs::create_dir_all(out_dir).expect("create out dir");
    let per_video_dir = out_dir.join("per_video");
    std::fs::create_dir_all(&per_video_dir).expect("create per_video dir");

    let cfg = VideoIdConfig {
        track_iou: parsed.iou,
        track_max_age: parsed.age,
        cluster_threshold: parsed.threshold,
        ..VideoIdConfig::default()
    };
    let lbph = LbphConfig::default();

    let mut runs: Vec<VideoIdentification> = Vec::new();
    for video in &parsed.videos {
        eprintln!(">>> processing {}", video.display());
        let mut src = source::open(video.to_str().expect("non-utf8 path")).unwrap_or_else(|e| {
            panic!("open {}: {e}", video.display());
        });
        let mut id = PipelineRunner::new(parsed.algo, lbph.clone());
        let result = identify_video(src.as_mut(), video.clone(), cfg.clone(), &mut id)
            .unwrap_or_else(|e| panic!("identify_video {}: {e}", video.display()));

        // Render this video's sub-manifest + thumbnail crops.
        let stem = video
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("video")
            .to_string();
        let sub_dir = per_video_dir.join(&stem);
        std::fs::create_dir_all(&sub_dir).expect("create sub dir");
        render_track_crops(src.as_mut(), &result, &sub_dir, &stem);
        let json = render_video_manifest_json(&result);
        std::fs::write(sub_dir.join("manifest.json"), json).expect("write sub manifest");
        eprintln!(
            "    {}: {} frames, {} faces, {} identities",
            stem,
            result.frames_processed,
            result.faces_embedded,
            result.identities.len()
        );
        runs.push(result);
    }

    // Cross-video re-id: pull every track out of every per-video run and
    // re-cluster so "Alice in ep01" and "Alice in ep03" land in the same id.
    let global_identities = merge_across_videos(runs.clone(), cfg.clone());
    let global_manifest = render_global_manifest(&parsed.videos, &global_identities, &runs);
    std::fs::write(out_dir.join("manifest.json"), global_manifest).expect("write manifest");

    eprintln!(
        ">>> wrote {} ({} global identities across {} videos)",
        out_dir.join("manifest.json").display(),
        global_identities.len(),
        parsed.videos.len()
    );
}

/// Holds the detector state for one video and routes every frame through
/// `Identify::detect_and_embed`. The detector + LBPH pair is zero-dep.
struct PipelineRunner {
    haar: Detector,
    luminance: LuminanceFaceDetector,
    algo: Algo,
    lbph: LbphConfig,
}

#[derive(Clone, Copy, Debug)]
enum Algo {
    Haar,
    Luminance,
}

impl PipelineRunner {
    fn new(algo: Algo, lbph: LbphConfig) -> Self {
        Self {
            haar: Detector::new(demo_face_cascade(), DetectorConfig::default()),
            luminance: LuminanceFaceDetector::new(rsface::LuminanceConfig::default()),
            algo,
            lbph,
        }
    }

    fn detect(&self, gray: &GrayImage) -> Vec<Detection> {
        match self.algo {
            Algo::Haar => self.haar.detect(gray),
            Algo::Luminance => self.luminance.detect(gray),
        }
    }

    fn embed(&self, gray: &GrayImage, dets: &[Detection]) -> Vec<(usize, Embedding)> {
        let mut out = Vec::with_capacity(dets.len());
        for (i, d) in dets.iter().enumerate() {
            let (x0, y0, x1, y1) = clamp_bbox(gray, d);
            if x1 <= x0 || y1 <= y0 {
                continue;
            }
            let crop = crop_gray(gray, x0, y0, x1, y1);
            let desc = lbph_extract(&crop, &self.lbph);
            // LbphDescriptor stores L1-normalised histograms; rescale to a
            // unit vector so Embedding::cosine is well-defined (it asserts
            // unit length for monotonic distance).
            let norm: f32 = desc.as_slice().iter().map(|v| v * v).sum::<f32>().sqrt();
            let raw: Vec<f32> = if norm > 1e-8 {
                desc.as_slice().iter().map(|v| v / norm).collect()
            } else {
                continue;
            };
            if let Some(e) = Embedding::from_raw(&raw) {
                // Tag with the detection index: a dropped crop here must
                // not shift later embeddings onto the wrong detection.
                out.push((i, e));
            }
        }
        out
    }
}

/// Bridge from `&mut PipelineRunner` to the `Identify` trait without
/// requiring an extra `Clone`. Re-runs detect + embed on every frame.
impl Identify for PipelineRunner {
    type Err = std::convert::Infallible;
    fn detect_and_embed(
        &mut self,
        gray: &GrayImage,
        _rgb: Option<&RgbImage>,
    ) -> Result<(Vec<Detection>, Vec<(usize, Embedding)>), Self::Err> {
        let dets = self.detect(gray);
        let embs = self.embed(gray, &dets);
        Ok((dets, embs))
    }
}

/// Clamp a detection bbox to the image and widen to a square so the LBPH
/// resize isotropically samples the face region.
fn clamp_bbox(img: &GrayImage, d: &Detection) -> (usize, usize, usize, usize) {
    let (w, h) = (img.width(), img.height());
    let x0 = d.x.min(w);
    let y0 = d.y.min(h);
    let x1 = (d.x + d.w).min(w);
    let y1 = (d.y + d.h).min(h);
    (x0, y0, x1, y1)
}

/// Cheap crop: copy pixel rows + columns. `GrayImage` doesn't expose a view
/// type, so a fresh allocation is the cleanest path. At short-drama face
/// sizes (<= 200px) this is well under a megabyte per crop.
fn crop_gray(img: &GrayImage, x0: usize, y0: usize, x1: usize, y1: usize) -> GrayImage {
    let w = x1 - x0;
    let h = y1 - y0;
    let mut out = GrayImage::new(w, h);
    for (oy, iy) in (y0..y1).enumerate() {
        let src = img.row(iy);
        let dst = out.row_mut(oy);
        dst[..w].copy_from_slice(&src[x0..x0 + w]);
    }
    out
}

/// Re-open the source once, seek to each track's representative frame and
/// crop the bbox. Zero-dep: pulls every frame in order, stops as soon as the
/// last `last_frame` is past.
fn render_track_crops(
    src: &mut dyn rsface::source::FrameSource,
    ident: &VideoIdentification,
    out_dir: &Path,
    video_stem: &str,
) {
    if ident.tracks.is_empty() {
        return;
    }
    let last_needed = ident.tracks.iter().map(|t| t.last_frame).max().unwrap_or(0);
    let mut by_frame: std::collections::HashMap<u64, Vec<&Track>> =
        std::collections::HashMap::new();
    for t in &ident.tracks {
        by_frame.entry(t.last_frame).or_default().push(t);
    }

    while let Ok(Some(frame)) = src.next_frame() {
        if let Some(tracks) = by_frame.remove(&frame.index) {
            let gray = frame.gray.clone();
            for t in tracks {
                let (x0, y0, x1, y1) = clamp_bbox(&gray, &t.best_bbox);
                if x1 <= x0 || y1 <= y0 {
                    continue;
                }
                let crop = crop_gray(&gray, x0, y0, x1, y1);
                let mut buf: Vec<u8> = Vec::new();
                if rsface::image::png::write_png_gray(&mut buf, &crop).is_err() {
                    continue;
                }
                let name = format!(
                    "{stem}_track{tid:03}_frame{f:05}_cluster{c}.png",
                    stem = video_stem,
                    tid = t.track_id,
                    f = frame.index,
                    c = cluster_for_track(ident, t.track_id),
                );
                let _ = std::fs::write(out_dir.join(name), buf);
            }
        }
        if frame.index >= last_needed {
            break;
        }
    }
}

fn cluster_for_track(ident: &VideoIdentification, track_id: u32) -> u32 {
    for id in &ident.identities {
        if id.tracks.iter().any(|t| t.track_id == track_id) {
            return id.cluster_id;
        }
    }
    0
}

fn render_global_manifest(
    videos: &[PathBuf],
    identities: &[rsface::video_id::Identity],
    per_video: &[VideoIdentification],
) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(2048);
    let _ = writeln!(s, "{{");
    let _ = writeln!(s, "  \"videos\": [");
    for (i, v) in videos.iter().enumerate() {
        let comma = if i + 1 == videos.len() { "" } else { "," };
        let _ = writeln!(s, "    {:?}{}", v.display().to_string(), comma);
    }
    let _ = writeln!(s, "  ],");
    let _ = writeln!(s, "  \"per_video_summary\": [");
    for (i, run) in per_video.iter().enumerate() {
        let comma = if i + 1 == per_video.len() { "" } else { "," };
        let _ = writeln!(
            s,
            "    {{ \"video\": {:?}, \"frames\": {}, \"faces\": {}, \"local_identities\": {} }}{}",
            run.video.display().to_string(),
            run.frames_processed,
            run.faces_embedded,
            run.identities.len(),
            comma
        );
    }
    let _ = writeln!(s, "  ],");
    let _ = writeln!(s, "  \"global_identities\": [");
    for (i, id) in identities.iter().enumerate() {
        let comma = if i + 1 == identities.len() { "" } else { "," };
        let _ = writeln!(s, "    {{");
        let _ = writeln!(s, "      \"cluster_id\": {},", id.cluster_id);
        let _ = writeln!(s, "      \"appearances\": [");
        for (j, t) in id.tracks.iter().enumerate() {
            let tcomma = if j + 1 == id.tracks.len() { "" } else { "," };
            let _ = writeln!(
                s,
                "        {{ \"video\": {:?}, \"track_id\": {}, \
                 \"first_ts_ms\": {}, \"last_ts_ms\": {}, \"best_score\": {:.4} }}{}",
                t.video.display().to_string(),
                t.track_id,
                t.first_ts_ms,
                t.last_ts_ms,
                t.best_score,
                tcomma
            );
        }
        let _ = writeln!(s, "      ]");
        let _ = writeln!(s, "    }}{comma}", comma = comma);
    }
    let _ = writeln!(s, "  ]");
    let _ = writeln!(s, "}}");
    s
}

#[derive(Debug)]
struct ParsedArgs {
    videos: Vec<PathBuf>,
    out_dir: PathBuf,
    algo: Algo,
    iou: f32,
    age: u32,
    threshold: f32,
}

fn parse_args(args: &[String]) -> Result<ParsedArgs, String> {
    let mut videos = Vec::new();
    let mut out_dir: Option<PathBuf> = None;
    let mut algo = Algo::Haar;
    let mut iou: Option<f32> = None;
    let mut age: Option<u32> = None;
    let mut threshold: Option<f32> = None;

    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        match a.as_str() {
            "--out" => {
                out_dir = Some(PathBuf::from(args.get(i + 1).ok_or("--out needs a value")?));
                i += 2;
            }
            "--algo" => {
                let v = args.get(i + 1).ok_or("--algo needs a value")?;
                algo = match v.as_str() {
                    "haar" => Algo::Haar,
                    "luminance" => Algo::Luminance,
                    other => return Err(format!("unknown --algo {other}")),
                };
                i += 2;
            }
            "--iou" => {
                iou = Some(
                    args.get(i + 1)
                        .ok_or("--iou needs a value")?
                        .parse()
                        .map_err(|e| format!("--iou: {e}"))?,
                );
                i += 2;
            }
            "--age" => {
                age = Some(
                    args.get(i + 1)
                        .ok_or("--age needs a value")?
                        .parse()
                        .map_err(|e| format!("--age: {e}"))?,
                );
                i += 2;
            }
            "--threshold" => {
                threshold = Some(
                    args.get(i + 1)
                        .ok_or("--threshold needs a value")?
                        .parse()
                        .map_err(|e| format!("--threshold: {e}"))?,
                );
                i += 2;
            }
            "--help" | "-h" => return Err("help".into()),
            other if other.starts_with("--") => return Err(format!("unknown flag {other}")),
            _ => {
                videos.push(PathBuf::from(a));
                i += 1;
            }
        }
    }
    if videos.is_empty() {
        return Err("no input videos given".into());
    }
    Ok(ParsedArgs {
        videos,
        out_dir: out_dir.ok_or("--out is required")?,
        algo,
        iou: iou.unwrap_or(0.3),
        age: age.unwrap_or(8),
        threshold: threshold.unwrap_or(0.36),
    })
}
