//! Frame source abstraction.
//!
//! A `FrameSource` yields grayscale frames one at a time. Sources handle:
//! - local file sequences (`img_0001.png`, `img_0002.png`, ...)
//! - HTTP progressive streams (multipart MJPEG, image sequence over HTTP)
//! - `ffmpeg` shell-out for MP4 / arbitrary containers (only when `ffmpeg` is on PATH)
//! - synthetic test streams
//!
//! Designed to plug into the multi-threaded pipeline without extra deps.

use crate::image::GrayImage;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::sync::Arc;

pub mod image_seq;
pub mod http;
pub mod ffmpeg_pipe;
pub mod synthetic;

pub use image_seq::ImageSequenceSource;
pub use http::HttpImageSource;
pub use ffmpeg_pipe::FfmpegPipeSource;
pub use synthetic::SyntheticSource;

/// A single decoded video frame plus metadata.
#[derive(Clone, Debug)]
pub struct Frame {
    /// Monotonic frame index (0-based) from the source.
    pub index: u64,
    /// Presentation timestamp in milliseconds (best-effort; 0 if unknown).
    pub timestamp_ms: u64,
    /// Decoded grayscale pixels.
    pub gray: Arc<GrayImage>,
    /// Optional original RGB pixels for output rendering.
    pub rgb: Option<Arc<crate::image::RgbImage>>,
}

/// Trait for objects that can produce frames sequentially.
pub trait FrameSource: Send {
    fn next_frame(&mut self) -> std::io::Result<Option<Frame>>;
    fn total_hint(&self) -> Option<u64> { None }
}

/// Build the most appropriate `FrameSource` for a URL/path string.
///
/// Accepted forms:
/// - `file:///abs/dir`           → image sequence (frame_NNNNN.png|jpg|ppm)
/// - `/abs/dir` or `./dir`       → image sequence
/// - `http://host/path/...png`   → single PNG over HTTP
/// - `http://host/path/...mjpeg` → multipart MJPEG
/// - `http://host/stream`        → MJPEG if `?mjpeg=1`, else try PNG sequence
/// - `https://...mp4` / `*.mp4`  → ffmpeg pipe (if `ffmpeg` on PATH)
/// - `rtsp://...`                → ffmpeg pipe (if `ffmpeg` on PATH)
/// - `test://grid`               → synthetic grid pattern
pub fn open(source: &str) -> std::io::Result<Box<dyn FrameSource>> {
    if source.starts_with("test://") {
        return Ok(Box::new(SyntheticSource::new(source)));
    }
    if source.starts_with("http://") || source.starts_with("https://") {
        let lower = source.to_ascii_lowercase();
        if lower.ends_with(".png") || lower.ends_with(".ppm") {
            return Ok(Box::new(HttpImageSource::new_single(source)));
        }
        if lower.contains(".mjpeg") || lower.contains("multipart") {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "MJPEG decoding requires a JPEG decoder (not bundled to keep zero-dep); convert server-side to PNG sequence or single PNG",
            ));
        }
        // If URL looks like a video, try ffmpeg pipe.
        if lower.ends_with(".mp4") || lower.ends_with(".mov") || lower.ends_with(".avi")
            || lower.ends_with(".mkv") || lower.ends_with(".webm")
            || lower.contains("/upload-") || lower.contains("video")
        {
            if which("ffmpeg").is_some() {
                return Ok(Box::new(FfmpegPipeSource::new(source, 30)?));
            }
        }
        // Default: treat as base URL for image sequence at /frame_NNNNN.png.
        let base = if source.ends_with('/') { source.to_string() } else { format!("{}/", source) };
        return Ok(Box::new(HttpImageSource::new_sequence(&base, 0)));
    }
    if source.starts_with("rtsp://") || source.to_ascii_lowercase().ends_with(".mp4")
        || source.to_ascii_lowercase().ends_with(".mov") || source.to_ascii_lowercase().ends_with(".avi")
        || source.to_ascii_lowercase().ends_with(".mkv") || source.to_ascii_lowercase().ends_with(".webm")
    {
        if which("ffmpeg").is_some() {
            return Ok(Box::new(FfmpegPipeSource::new(source, 30)?));
        } else {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "video decoding requires `ffmpeg` on PATH; install or use image-sequence input",
            ));
        }
    }
    // Local path → image sequence (dir or file pattern).
    Ok(Box::new(ImageSequenceSource::new(source)?))
}

fn which(cmd: &str) -> Option<String> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(cmd);
        if candidate.is_file() { return Some(cmd.to_string()); }
        #[cfg(windows)]
        {
            let candidate_exe = dir.join(format!("{}.exe", cmd));
            if candidate_exe.is_file() { return Some(format!("{}.exe", cmd)); }
        }
    }
    None
}

pub(crate) fn read_image_file(path: &Path) -> std::io::Result<(GrayImage, Option<crate::image::RgbImage>)> {
    let mut f = BufReader::new(File::open(path)?);
    let mut head = [0u8; 8];
    f.read_exact(&mut head)?;
    f.seek_relative(-(head.len() as i64))?;
    if &head[..8] == b"\x89PNG\r\n\x1a\n" {
        let rgb = crate::image::png::decode_to_rgb(&mut f).ok().map(Arc::new);
        f.seek_relative(-((head.len() as i64) + 0))?; // rewind for gray decode
        let mut f2 = BufReader::new(File::open(path)?);
        let gray = crate::image::png::decode_to_gray(&mut f2)?;
        return Ok((gray, rgb.map(|a| (*a).clone())));
    }
    // PPM/PGM
    let mut f = BufReader::new(File::open(path)?);
    let mut peek = [0u8; 2];
    f.read_exact(&mut peek)?;
    f.seek_relative(-2)?;
    if &peek == b"P5" {
        let gray = crate::image::codec::read_pgm(&mut f)?;
        return Ok((gray, None));
    }
    if &peek == b"P6" {
        let rgb = crate::image::codec::read_ppm(&mut f)?;
        let gray = rgb.to_gray();
        return Ok((gray, Some(rgb)));
    }
    Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "unsupported image format"))
}
