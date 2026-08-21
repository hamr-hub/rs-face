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

impl SyntheticSource {
    pub fn new(spec: &str) -> Self {
        let frames: u64 = spec
            .strip_prefix("test://")
            .unwrap_or("grid")
            .split('?')
            .next()
            .and_then(|s| s.split('=').nth(1).and_then(|n| n.parse().ok()))
            .unwrap_or(60);
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
