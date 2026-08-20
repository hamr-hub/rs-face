//! Output writers — annotated PNG frames + JSON manifest.

use crate::detector::Detection;
use crate::image::{png, RgbImage};
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

/// Detected-face record for the manifest.
#[derive(Clone, Debug)]
pub struct DetectionRecord {
    pub frame_index: u64,
    pub timestamp_ms: u64,
    pub image_file: String,
    pub width: usize,
    pub height: usize,
    pub detections: Vec<Detection>,
}

/// Write an RGB image with detection boxes drawn, as PNG, to the given path.
/// Returns the file name (basename) on success.
pub fn write_annotated_png(
    dir: &Path,
    rec: &DetectionRecord,
    rgb: &RgbImage,
) -> std::io::Result<String> {
    fs::create_dir_all(dir)?;
    let mut canvas = rgb.clone();
    for d in &rec.detections {
        canvas.draw_rect(d.x, d.y, d.w, d.h, (255, 64, 64));
    }
    let fname = format!("frame_{:06}.png", rec.frame_index);
    let path = dir.join(&fname);
    let mut f = BufWriter::new(File::create(&path)?);
    png::write_png_rgb(&mut f, &canvas)?;
    Ok(fname)
}

/// Hand-rolled JSON writer — zero deps. Writes a manifest describing every detection.
pub fn write_manifest(
    path: &Path,
    records: &[DetectionRecord],
    stats: &PipelineSummary,
) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut f = BufWriter::new(File::create(path)?);
    writeln!(f, "{{")?;
    writeln!(f, "  \"version\": \"rs-face-0.1\",")?;
    writeln!(f, "  \"stats\": {{")?;
    writeln!(f, "    \"frames_processed\": {},", stats.frames_processed)?;
    writeln!(f, "    \"frames_with_face\": {},", stats.frames_with_face)?;
    writeln!(f, "    \"total_detections\": {},", stats.total_detections)?;
    writeln!(f, "    \"elapsed_ms\": {},", stats.elapsed_ms)?;
    writeln!(f, "    \"fps\": {:.2}", stats.fps())?;
    writeln!(f, "  }},")?;
    writeln!(f, "  \"frames\": [")?;
    for (i, r) in records.iter().enumerate() {
        write!(f, "    {{")?;
        write!(f, "\"frame_index\": {}, \"timestamp_ms\": {}, \"image\": \"{}\", \"width\": {}, \"height\": {}, \"detections\": [",
               r.frame_index, r.timestamp_ms, json_escape(&r.image_file), r.width, r.height)?;
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
        if i + 1 < records.len() {
            writeln!(f, "}},")?;
        } else {
            writeln!(f, "}}")?;
        }
    }
    writeln!(f, "  ]")?;
    writeln!(f, "}}")?;
    Ok(())
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
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

/// Lightweight buffered writer wrapper.
struct BufWriter {
    inner: File,
}
impl BufWriter {
    fn new(f: File) -> Self {
        Self { inner: f }
    }
}
impl Write for BufWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.inner.write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

#[derive(Default, Debug, Clone)]
pub struct PipelineSummary {
    pub frames_processed: u64,
    pub frames_with_face: u64,
    pub total_detections: u64,
    pub elapsed_ms: u64,
}

impl PipelineSummary {
    pub fn fps(&self) -> f32 {
        if self.elapsed_ms == 0 {
            return 0.0;
        }
        self.frames_processed as f32 * 1000.0 / self.elapsed_ms as f32
    }
}
