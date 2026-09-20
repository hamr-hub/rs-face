//! Synthetic test source — generates a grayscale gradient grid for development.

use super::{Frame, FrameSource};
use crate::image::GrayImage;
use std::io;
use std::sync::Arc;

pub struct SyntheticSource {
    frames: u64,
    pos: u64,
    width: usize,
    height: usize,
}

/// Default frame count when the spec gives no parseable number.
const DEFAULT_FRAMES: u64 = 60;

impl SyntheticSource {
    /// Parse a synthetic-source spec.
    ///
    /// Accepted forms (all render the same animated sine-grid):
    /// - `test://N`                 → `N` frames, e.g. `test://10`
    /// - `test://frames=N`          → `N` frames
    /// - `test://grid?frames=N`     → named grid with `N` frames
    /// - `test://grid` / `test://` → [`DEFAULT_FRAMES`] frames
    ///
    /// An unparsable count (e.g. `test://lots`) also falls back to the
    /// default rather than failing — the source is a development fixture.
    pub fn new(spec: &str) -> Self {
        let body = spec.strip_prefix("test://").unwrap_or("");
        let (path, query) = body.split_once('?').map_or((body, ""), |(p, q)| (p, q));
        let query_frames = query
            .split('&')
            .find_map(|kv| kv.strip_prefix("frames="))
            .and_then(|n| n.parse().ok());
        let path_frames = path
            .strip_prefix("frames=")
            .and_then(|n| n.parse().ok())
            .or_else(|| path.parse::<u64>().ok());
        let frames = query_frames.or(path_frames).unwrap_or(DEFAULT_FRAMES);
        Self {
            frames,
            pos: 0,
            width: 320,
            height: 240,
        }
    }
}

impl FrameSource for SyntheticSource {
    fn next_frame(&mut self) -> io::Result<Option<Frame>> {
        if self.pos >= self.frames {
            return Ok(None);
        }
        let mut img = GrayImage::new(self.width, self.height);
        let phase = (self.pos as f32) * 0.1;
        for y in 0..self.height {
            for x in 0..self.width {
                let v = ((x + y) as f32 * 0.02 + phase).sin() * 127.0 + 128.0;
                img[(x, y)] = v.clamp(0.0, 255.0) as u8;
            }
        }
        let idx = self.pos;
        self.pos += 1;
        Ok(Some(Frame {
            index: idx,
            timestamp_ms: idx * 33,
            gray: Arc::new(img),
            rgb: None,
        }))
    }

    fn total_hint(&self) -> Option<u64> {
        Some(self.frames)
    }
}

#[cfg(test)]
mod tests {
    use super::SyntheticSource;

    #[test]
    fn parses_bare_frame_count() {
        // The form the CLI help promises: test://N means N frames.
        assert_eq!(SyntheticSource::new("test://10").frames, 10);
        assert_eq!(SyntheticSource::new("test://1").frames, 1);
    }

    #[test]
    fn parses_named_grid_forms() {
        assert_eq!(SyntheticSource::new("test://frames=7").frames, 7);
        assert_eq!(SyntheticSource::new("test://grid?frames=12").frames, 12);
    }

    #[test]
    fn falls_back_to_default_frames() {
        assert_eq!(SyntheticSource::new("test://grid").frames, 60);
        assert_eq!(SyntheticSource::new("test://").frames, 60);
        assert_eq!(SyntheticSource::new("test://lots").frames, 60);
    }
}
