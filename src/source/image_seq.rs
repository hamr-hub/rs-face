//! Image-sequence source: reads `frame_NNNNN.png` (or `.jpg`, `.ppm`) from a directory
//! or file pattern.

use super::{read_image_file, Frame, FrameSource};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub struct ImageSequenceSource {
    files: Vec<PathBuf>,
    pos: usize,
    frame_index: u64,
    fps: u32, // for timestamp calculation
}

impl ImageSequenceSource {
    pub fn new(path: &str) -> io::Result<Self> {
        let p = Path::new(path);
        let (dir, ext_filter) = if p.is_dir() {
            (p.to_path_buf(), None)
        } else {
            // Treat as file pattern prefix (e.g. "./frames/frame_").
            let parent = p.parent().unwrap_or_else(|| Path::new("."));
            let prefix = p.file_name().and_then(|s| s.to_str()).unwrap_or("").to_string();
            (parent.to_path_buf(), Some(prefix))
        };

        let mut files: Vec<PathBuf> = fs::read_dir(&dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_file())
            .filter(|p| {
                let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
                let ext_ok = name.ends_with(".png") || name.ends_with(".jpg")
                    || name.ends_with(".jpeg") || name.ends_with(".ppm") || name.ends_with(".pgm");
                if !ext_ok { return false; }
                if let Some(pref) = &ext_filter {
                    if !pref.is_empty() && !name.starts_with(pref.as_str()) { return false; }
                }
                true
            })
            .collect();
        files.sort();

        Ok(Self { files, pos: 0, frame_index: 0, fps: 30 })
    }

    pub fn with_fps(mut self, fps: u32) -> Self { self.fps = fps; self }
}

impl FrameSource for ImageSequenceSource {
    fn next_frame(&mut self) -> io::Result<Option<Frame>> {
        if self.pos >= self.files.len() { return Ok(None); }
        let path = self.files[self.pos].clone();
        self.pos += 1;
        let (gray, rgb) = read_image_file(&path)?;
        let ts = (self.frame_index * 1000 / self.fps as u64) as u64;
        let idx = self.frame_index;
        self.frame_index += 1;
        Ok(Some(Frame { index: idx, timestamp_ms: ts, gray: Arc::new(gray), rgb: rgb.map(Arc::new) }))
    }

    fn total_hint(&self) -> Option<u64> { Some(self.files.len() as u64) }
}
