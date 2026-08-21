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
    /// Wall-clock detection time for this frame in milliseconds (pyramid +
    /// integrals + cascade + NMS). Populated by the pipeline; defaults to
    /// `0.0` when the caller doesn't measure.
    pub detect_ms: f64,
}

impl Default for DetectionRecord {
    fn default() -> Self {
        Self {
            frame_index: 0,
            timestamp_ms: 0,
            image_file: String::new(),
            width: 0,
            height: 0,
            detections: Vec::new(),
            detect_ms: 0.0,
        }
    }
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
    writeln!(f, "    \"fps\": {:.2},", stats.fps())?;
    writeln!(f, "    \"detect_ms_avg\": {:.3}", stats.detect_ms_avg())?;
    writeln!(f, "  }},")?;
    writeln!(f, "  \"frames\": [")?;
    for (i, r) in records.iter().enumerate() {
        write!(f, "    {{")?;
        write!(
            f,
            "\"frame_index\": {}, \"timestamp_ms\": {}, \"image\": \"{}\", \"width\": {}, \"height\": {}, \"detect_ms\": {:.3}, \"detections\": [",
            r.frame_index,
            r.timestamp_ms,
            json_escape(&r.image_file),
            r.width,
            r.height,
            r.detect_ms
        )?;
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
    /// Mean per-frame detection time in ms (sum of per-frame `detect_ms`
    /// over `frames_processed`). Kept as `f64` sum + count to avoid storing
    /// per-frame history; 0.0 when not measured.
    pub detect_ms_total: f64,
}

impl PipelineSummary {
    pub fn fps(&self) -> f32 {
        if self.elapsed_ms == 0 {
            return 0.0;
        }
        self.frames_processed as f32 * 1000.0 / self.elapsed_ms as f32
    }

    /// Mean per-frame detection time in milliseconds (`detect_ms_total /
    /// frames_processed`; 0.0 when no frames were processed).
    pub fn detect_ms_avg(&self) -> f64 {
        if self.frames_processed == 0 {
            return 0.0;
        }
        self.detect_ms_total / self.frames_processed as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_record(detect_ms: f64) -> DetectionRecord {
        DetectionRecord {
            frame_index: 3,
            timestamp_ms: 120,
            image_file: "frame_000003.png".to_string(),
            width: 64,
            height: 48,
            detections: vec![
                Detection {
                    x: 1,
                    y: 2,
                    w: 10,
                    h: 12,
                    score: 0.9876,
                },
                Detection {
                    x: 30,
                    y: 4,
                    w: 8,
                    h: 8,
                    score: 0.5,
                },
            ],
            detect_ms,
        }
    }

    #[test]
    fn manifest_contains_confidence_and_timing() {
        let dir = std::env::temp_dir().join("rsface_output_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("manifest.json");
        let records = vec![sample_record(7.25), sample_record(2.5)];
        let summary = PipelineSummary {
            frames_processed: 2,
            frames_with_face: 2,
            total_detections: 4,
            elapsed_ms: 100,
            detect_ms_total: 9.75,
        };
        write_manifest(&path, &records, &summary).unwrap();
        let body = std::fs::read_to_string(&path).unwrap();
        // Per-detection confidence.
        assert!(
            body.contains("\"conf\": 0.9876"),
            "manifest missing confidence: {body}"
        );
        // Per-frame detection time.
        assert!(
            body.contains("\"detect_ms\": 7.250"),
            "manifest missing per-frame detect_ms: {body}"
        );
        // Aggregate average.
        assert!(
            body.contains("\"detect_ms_avg\": 4.875"),
            "manifest missing detect_ms_avg: {body}"
        );
        // Sanity: JSON round-brackets balance.
        assert_eq!(body.matches('{').count(), body.matches('}').count());
        assert_eq!(body.matches('[').count(), body.matches(']').count());
    }

    #[test]
    fn detect_ms_avg_handles_zero_frames() {
        let s = PipelineSummary::default();
        assert_eq!(s.detect_ms_avg(), 0.0);
        let s = PipelineSummary {
            frames_processed: 4,
            detect_ms_total: 10.0,
            ..PipelineSummary::default()
        };
        assert_eq!(s.detect_ms_avg(), 2.5);
    }

    #[test]
    fn detection_record_default_is_zeroed() {
        let r = DetectionRecord::default();
        assert_eq!(r.frame_index, 0);
        assert!(r.detections.is_empty());
        assert_eq!(r.detect_ms, 0.0);
        assert!(r.image_file.is_empty());
    }
}
