//! Batch directory processor.
//!
//! Walks a directory of standalone images (jpg/png/ppm/pgm), runs the same
//! detector on each one independently, and emits:
//!
//! - `<out>/detections/<input_basename>.png` — annotated PNG per input image
//!   (boxes drawn at the same coordinates the manifest reports). Skipped when
//!   `--only-with-face` is set and no faces were found.
//! - `<out>/batch_manifest.json` — single combined manifest with per-image
//!   summary stats (input path, output path, width, height, detection count,
//!   detect time, error string if any).
//!
//! The per-image output is named after the **input** basename rather than a
//! monotonic `frame_NNNNN.png` like the multi-frame pipeline produces — the
//! point of this driver is "scan a folder of photos, find which ones have
//! faces". That name parity lets a user run `rs-face --batch-dir ./photos`
//! once and instantly locate each annotated render.
//!
//! Scope deliberately excludes:
//! - Video / HLS / RTSP / `test://N` sources — the existing `Pipeline::run`
//!   driver handles those.
//! - Multi-algo ensemble — each batch invocation runs one detector. Pair the
//!   call with `--only-with-face` for fast triage of large photo dumps.
//! - Parallel workers — the canonical CLI passes `--threads N` to the
//!   multi-threaded pipeline; for image sequences where each frame is a
//!   multi-megapixel PNG, single-threaded iteration keeps memory pressure
//!   predictable. The function still leaves room for future per-image
//!   parallelism without changing the public surface.
//!
//! Zero-dep by construction (only `std` + the public `rsface` API).
//!
//! ## CLI wiring
//!
//! `src/main.rs` exposes `--batch-dir <DIR>`. When set, it supersedes the
//! positional `INPUT` for source resolution — the directory is scanned once,
//! every image is processed, and the pipeline driver is bypassed.

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use crate::detector::{Detector, DetectorConfig};
use crate::face::Detection;
use crate::haar::Cascade;
use crate::image::{png, GrayImage, RgbImage};

/// Configuration for a single batch run. Mirrors the CLI knobs that affect
/// per-image detection quality (vs. the pipeline-only knobs like queue depth).
#[derive(Clone, Debug)]
pub struct BatchConfig {
    pub min_size: usize,
    pub max_size: usize,
    pub scale_factor: f32,
    pub window_stride: usize,
    pub nms_iou_threshold: f32,
    pub min_neighbors: i32,
    pub min_score: f32,
    /// Drop empty annotated PNGs from `<out>/detections/` (the manifest still
    /// records them so a user can see "this photo had no face").
    pub only_with_face: bool,
    /// Optional cascade stage-bias override (forwarded to `Cascade::stage_bias`
    /// by the caller before the detector is built).
    pub stage_bias: Option<f32>,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            min_size: 24,
            max_size: 1024,
            scale_factor: 1.2,
            window_stride: 4,
            nms_iou_threshold: 0.3,
            min_neighbors: 3,
            min_score: 0.0,
            only_with_face: false,
            stage_bias: None,
        }
    }
}

impl BatchConfig {
    fn into_detector_config(self) -> DetectorConfig {
        DetectorConfig {
            min_size: self.min_size,
            max_size: self.max_size,
            scale_factor: self.scale_factor,
            window_stride: self.window_stride,
            nms_iou_threshold: self.nms_iou_threshold,
            min_neighbors: self.min_neighbors,
            min_score: self.min_score,
            // Batch mode defaults to conservative CPU-only behaviour;
            // the GPU pre-filter offers little on isolated photos and would
            // turn each invocation into a dlopen/JIT wait.
            use_gpu: false,
            ..DetectorConfig::default()
        }
    }
}

/// Per-image outcome in a batch run. Used both internally (for the manifest
/// writer) and externally (callers that want to react to per-image errors
/// without re-parsing the JSON).
#[derive(Clone, Debug)]
pub struct BatchImageResult {
    pub input: PathBuf,
    pub output: Option<PathBuf>,
    pub width: u32,
    pub height: u32,
    pub detections: Vec<Detection>,
    pub detect_ms: f64,
    pub error: Option<String>,
}

/// Aggregate counters for a batch run.
#[derive(Clone, Debug, Default)]
pub struct BatchStats {
    pub images_processed: u64,
    pub images_with_face: u64,
    pub total_detections: u64,
    pub elapsed_ms: u64,
}

/// Returns the absolute path to the annotated-output subdirectory.
///
/// Layout:
/// ```text
/// <output_dir>/
///   detections/<basename>.png   (per input image)
///   batch_manifest.json         (single combined)
/// ```
pub fn detections_dir(output_dir: &Path) -> PathBuf {
    output_dir.join("detections")
}

pub fn manifest_path(output_dir: &Path) -> PathBuf {
    output_dir.join("batch_manifest.json")
}

/// Scan `input_dir` non-recursively for images rs-face can decode. The
/// extension filter matches the source-image decoder: `.png`, `.jpg`,
/// `.jpeg`, `.ppm`, `.pgm`. Files are sorted by path so the manifest order
/// is deterministic and matches the conventional
/// `IMG_0001.png, IMG_0002.png, …` lexicographic order.
///
/// Returns an empty `Vec` (not an error) when the directory exists but
/// contains no recognised images — a user with a misnamed extension set
/// still wants a clear "0 images found" rather than an `io::Error`.
pub fn discover_images(input_dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    if !input_dir.is_dir() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("not a directory: {}", input_dir.display()),
        ));
    }
    let mut files: Vec<PathBuf> = fs::read_dir(input_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .filter(|p| {
            let name = p
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            name.ends_with(".png")
                || name.ends_with(".jpg")
                || name.ends_with(".jpeg")
                || name.ends_with(".ppm")
                || name.ends_with(".pgm")
        })
        .collect();
    files.sort();
    Ok(files)
}

/// Detect every image in `input_dir` independently and write a batch manifest
/// plus per-image annotated PNGs under `output_dir`. The cascade is consumed
/// once and reused across all images — Viola-Jones evaluation is the hot
/// loop, not setup, so cloning a 25-stage cascade per image would be wasteful.
///
/// On a per-image error (e.g. truncated JPEG, decode failure) the image is
/// recorded in the manifest with `"error"` set and detection skipped. The
/// run as a whole does not abort.
pub fn run_batch_dir(
    mut cascade: Cascade,
    input_dir: &Path,
    output_dir: &Path,
    cfg: &BatchConfig,
) -> std::io::Result<(BatchStats, Vec<BatchImageResult>)> {
    fs::create_dir_all(output_dir)?;
    fs::create_dir_all(detections_dir(output_dir))?;

    let images = discover_images(input_dir)?;
    if let Some(bias) = cfg.stage_bias {
        cascade.stage_bias = bias;
    }

    // Build the detector once and reuse — its per-call setup (integral image
    // allocation + EvalCache prime) is the bulk of the per-frame cost on small
    // images; cloning the Cascade is cheap relative to that.
    let det_cfg = cfg.clone().into_detector_config();
    let detector = Arc::new(Detector::new(cascade, det_cfg));

    let run_started = Instant::now();
    let mut results: Vec<BatchImageResult> = Vec::with_capacity(images.len());
    let mut stats = BatchStats::default();

    for input in &images {
        let r = process_one(&detector, input, output_dir, cfg);
        stats.images_processed += 1;
        if r.error.is_none() {
            if !r.detections.is_empty() {
                stats.images_with_face += 1;
            }
            stats.total_detections += r.detections.len() as u64;
        }
        results.push(r);
    }
    stats.elapsed_ms = run_started.elapsed().as_millis() as u64;

    write_batch_manifest(&manifest_path(output_dir), &stats, &results)?;
    Ok((stats, results))
}

/// Process one image: decode → detect → optionally write annotated PNG.
/// All errors are captured into `BatchImageResult.error`; the function itself
/// never returns Err for per-image failures. I/O errors on the output
/// directory (rare — we just created it) propagate so the caller can surface
/// them loudly.
fn process_one(
    detector: &Detector,
    input: &Path,
    output_dir: &Path,
    cfg: &BatchConfig,
) -> BatchImageResult {
    let t0 = Instant::now();
    let decoded = match decode_any(input) {
        Ok(d) => d,
        Err(e) => {
            return BatchImageResult {
                input: input.to_path_buf(),
                output: None,
                width: 0,
                height: 0,
                detections: Vec::new(),
                detect_ms: t0.elapsed().as_secs_f64() * 1000.0,
                error: Some(format!("decode: {e}")),
            };
        }
    };
    let (gray, rgb) = decoded;
    let dets = detector.detect(&gray);
    let detect_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let width = gray.width() as u32;
    let height = gray.height() as u32;
    let skip_empty = cfg.only_with_face && dets.is_empty();
    let written = if !skip_empty {
        match write_annotated_for(input, output_dir, &dets, &gray, rgb.as_ref()) {
            Ok(p) => Some(p),
            // Per-image write failure is non-fatal: the manifest still records
            // the detection result so downstream consumers can retry.
            Err(e) => {
                return BatchImageResult {
                    input: input.to_path_buf(),
                    output: None,
                    width,
                    height,
                    detections: dets,
                    detect_ms,
                    error: Some(format!("write: {e}")),
                };
            }
        }
    } else {
        None
    };

    BatchImageResult {
        input: input.to_path_buf(),
        output: written,
        width,
        height,
        detections: dets,
        detect_ms,
        error: None,
    }
}

/// Decode an image file by magic-byte sniff. PNG / PGM / PPM are routed to
/// the in-tree zero-dep decoders; JPEG falls back to writing a temp PGM via
/// ffmpeg if it's on PATH, otherwise we surface a clear "unsupported format"
/// error rather than silently skipping the file.
///
/// The same magic-byte precedence as `source::read_image_file`: PNG signature
/// → PGM (`P5`) → PPM (`P6`). We don't try to be clever with JPEG detection
/// — `.jpg`/`.jpeg` extension goes straight to the ffmpeg fallback.
fn decode_any(path: &Path) -> std::io::Result<(GrayImage, Option<RgbImage>)> {
    let f = File::open(path)?;
    let mut head = [0u8; 8];
    {
        let mut r = BufReader::new(f);
        r.read_exact(&mut head)?;
    }
    // PNG
    if head.starts_with(b"\x89PNG\r\n\x1a\n") {
        let f = File::open(path)?;
        let mut r = BufReader::new(f);
        let gray = png::decode_to_gray(&mut r)?;
        let f2 = File::open(path)?;
        let mut r2 = BufReader::new(f2);
        let rgb = png::decode_to_rgb(&mut r2).ok();
        return Ok((gray, rgb));
    }
    // PPM (P6) — three channels, convert to gray ourselves for the cascade.
    if head.starts_with(b"P6") {
        let f = File::open(path)?;
        let mut r = BufReader::new(f);
        let rgb = crate::image::codec::read_ppm(&mut r)?;
        let gray = rgb.to_gray();
        return Ok((gray, Some(rgb)));
    }
    // PGM (P5)
    if head.starts_with(b"P5") {
        let f = File::open(path)?;
        let mut r = BufReader::new(f);
        let gray = crate::image::codec::read_pgm(&mut r)?;
        return Ok((gray, None));
    }
    // JPEG/WebP/anything else: hand off to ffmpeg if available. The CLI
    // already depends on ffmpeg for video; routing image-decode through it
    // keeps the zero-dep default build viable for image-only users while
    // letting the batch dir handle real-world photo dumps without surprise.
    if which("ffmpeg").is_some() {
        return decode_via_ffmpeg(path);
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "unsupported image format (PNG/PGM/PPM only; install ffmpeg for JPEG/WebP)",
    ))
}

fn which(cmd: &str) -> Option<String> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        if dir.join(cmd).is_file() {
            return Some(cmd.to_string());
        }
        #[cfg(windows)]
        {
            if dir.join(format!("{cmd}.exe")).is_file() {
                return Some(format!("{cmd}.exe"));
            }
        }
    }
    None
}

/// ffmpeg → PGM helper for JPEG / WebP / HEIC in `--batch-dir`. Returns the
/// decoded gray + the RGB reconstructed from the same PGM stream (PGM is
/// single-channel so RGB is reconstructed by replication; the annotated
/// writer doesn't care because we draw boxes on the gray-derived canvas).
fn decode_via_ffmpeg(path: &Path) -> std::io::Result<(GrayImage, Option<RgbImage>)> {
    use std::process::Command;
    let mut child = Command::new("ffmpeg")
        .args(["-loglevel", "error", "-i"])
        .arg(path)
        .args(["-pix_fmt", "gray", "-f", "image2", "-"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| std::io::Error::other(format!("ffmpeg spawn: {e}")))?;
    let mut out = child
        .stdout
        .take()
        .ok_or_else(|| std::io::Error::other("ffmpeg stdout missing"))?;
    let mut buf = Vec::new();
    use std::io::Read;
    out.read_to_end(&mut buf)?;
    let status = child.wait()?;
    if !status.success() {
        return Err(std::io::Error::other(format!(
            "ffmpeg failed (status {status})"
        )));
    }
    let mut cur = std::io::Cursor::new(buf);
    let gray = crate::image::codec::read_pgm(&mut cur)?;
    Ok((gray, None))
}

/// Write a single annotated PNG for one image. Box colour matches the
/// pipeline's `output::write_annotated_png` (red `(255, 64, 64)`).
///
/// The output filename is `<input_basename>.png`, NOT `frame_NNNNN.png`,
/// so users can map "which photo produced which detection" by filename.
fn write_annotated_for(
    input: &Path,
    output_dir: &Path,
    dets: &[Detection],
    gray: &GrayImage,
    rgb: Option<&RgbImage>,
) -> std::io::Result<PathBuf> {
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("image");
    let fname = format!("{stem}.png");
    let out_path = detections_dir(output_dir).join(&fname);

    // Prefer the original RGB if we decoded one (PNG, PPM); fall back to
    // gray replication so PGM-only inputs still produce a colour-annotated
    // PNG instead of a flat grayscale image.
    let canvas: RgbImage = match rgb.cloned() {
        Some(rgb) => rgb,
        None => gray_to_rgb(gray),
    };
    let mut canvas = canvas;
    for d in dets {
        canvas.draw_rect(d.x, d.y, d.w, d.h, (255, 64, 64));
    }
    let f = File::create(&out_path)?;
    let mut w = BufWriter::new(f);
    png::write_png_rgb(&mut w, &canvas)?;
    w.flush()?;
    Ok(out_path)
}

fn gray_to_rgb(gray: &GrayImage) -> RgbImage {
    let (w, h) = (gray.width(), gray.height());
    let mut rgb = RgbImage::new(w, h);
    for y in 0..h {
        let row = rgb.row_mut(y);
        let gray_row = gray.row(y);
        for (x, &v) in gray_row.iter().enumerate() {
            row[x * 3] = v;
            row[x * 3 + 1] = v;
            row[x * 3 + 2] = v;
        }
    }
    rgb
}

/// Hand-rolled JSON manifest writer. Same zero-dep, allocation-light style as
/// `output::write_manifest`; the only structural difference is that we use
/// `input` / `output` paths instead of `image_file` and we include per-image
/// `error` strings.
fn write_batch_manifest(
    path: &Path,
    stats: &BatchStats,
    results: &[BatchImageResult],
) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut f = BufWriter::new(File::create(path)?);
    writeln!(f, "{{")?;
    writeln!(f, "  \"version\": \"rs-face-0.2-batch-v1\",")?;
    writeln!(f, "  \"stats\": {{")?;
    writeln!(f, "    \"images_processed\": {},", stats.images_processed)?;
    writeln!(f, "    \"images_with_face\": {},", stats.images_with_face)?;
    writeln!(f, "    \"total_detections\": {},", stats.total_detections)?;
    writeln!(f, "    \"elapsed_ms\": {},", stats.elapsed_ms)?;
    let avg = if stats.images_processed > 0 {
        stats.elapsed_ms as f64 / stats.images_processed as f64
    } else {
        0.0
    };
    writeln!(f, "    \"avg_detect_ms_per_image\": {avg:.3}")?;
    writeln!(f, "  }},")?;
    writeln!(f, "  \"images\": [")?;
    for (i, r) in results.iter().enumerate() {
        write!(f, "    {{")?;
        write!(
            f,
            "\"input\": \"{}\", ",
            json_escape(&r.input.display().to_string())
        )?;
        write!(
            f,
            "\"output\": \"{}\", ",
            json_escape(
                &r.output
                    .as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_default()
            )
        )?;
        write!(f, "\"width\": {}, \"height\": {}, ", r.width, r.height)?;
        write!(f, "\"detect_ms\": {:.3}, ", r.detect_ms)?;
        if let Some(e) = &r.error {
            write!(f, "\"error\": \"{}\", ", json_escape(e))?;
        }
        write!(f, "\"detections\": [")?;
        for (j, d) in r.detections.iter().enumerate() {
            if j > 0 {
                write!(f, ", ")?;
            }
            write!(
                f,
                "{{\"x\": {}, \"y\": {}, \"w\": {}, \"h\": {}, \"conf\": {:.4}}}",
                d.x, d.y, d.w, d.h, d.score
            )?;
        }
        write!(f, "]")?;
        write!(f, "}}")?;
        if i + 1 < results.len() {
            writeln!(f, ",")?;
        } else {
            writeln!(f)?;
        }
    }
    writeln!(f, "  ]")?;
    writeln!(f, "}}")?;
    f.flush()?;
    Ok(())
}

/// Minimal JSON string escape for `\"` and `\\` only — filenames and error
/// messages don't typically contain control chars, and we're not parsing
/// this back, just emitting. Stays alloc-light vs. pulling in serde_json.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Lightweight tempdir substitute to avoid pulling in a dev-dep just for
    /// tests. Returns a unique path under $TMPDIR/rsface-batch-test-<n>.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(prefix: &str) -> std::io::Result<Self> {
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            let pid = std::process::id();
            let path = std::env::temp_dir().join(format!("{prefix}-{pid}-{n}"));
            std::fs::create_dir_all(&path)?;
            Ok(Self(path))
        }
        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn empty_dir() -> std::io::Result<TempDir> {
        TempDir::new("rsface-batch-test")
    }

    #[test]
    fn discover_images_returns_empty_for_empty_dir() {
        let d = empty_dir().unwrap();
        let found = discover_images(d.path()).unwrap();
        assert!(found.is_empty());
    }

    #[test]
    fn discover_images_errors_on_missing_dir() {
        let p = std::path::PathBuf::from("/nonexistent/path/rsface-discover-test");
        let err = discover_images(&p).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    }

    #[test]
    fn discover_images_filters_to_known_extensions() {
        let d = empty_dir().unwrap();
        std::fs::write(d.path().join("a.png"), b"").unwrap();
        std::fs::write(d.path().join("b.jpg"), b"").unwrap();
        std::fs::write(d.path().join("c.ppm"), b"").unwrap();
        std::fs::write(d.path().join("d.txt"), b"").unwrap();
        std::fs::write(d.path().join("README"), b"").unwrap();
        let found = discover_images(d.path()).unwrap();
        let names: Vec<String> = found
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec!["a.png", "b.jpg", "c.ppm"]);
    }

    #[test]
    fn discover_images_is_sorted_lexicographically() {
        let d = empty_dir().unwrap();
        for name in ["z.png", "a.png", "m.png"] {
            std::fs::write(d.path().join(name), b"").unwrap();
        }
        let found = discover_images(d.path()).unwrap();
        let names: Vec<String> = found
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec!["a.png", "m.png", "z.png"]);
    }

    #[test]
    fn json_escape_handles_quotes_and_backslash() {
        assert_eq!(json_escape("plain"), "plain");
        assert_eq!(json_escape("a\"b"), "a\\\"b");
        assert_eq!(json_escape("a\\b"), "a\\\\b");
        assert_eq!(json_escape("line\nfeed"), "line\\nfeed");
    }
}
