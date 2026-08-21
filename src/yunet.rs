//! YuNet-style anchor-based face detector (simplified).
//!
//! Implements the detection pipeline of OpenCV's YuNet (2022) - a modern
//! single-stage anchor-based face detector - in pure Rust with no new
//! dependencies. Architecture:
//!   - 5 anchor scales: 8/16/32/64/128 pixels (square anchors).
//!   - Per anchor: 15-dim output = (dx, dy, dw, dh, objectness, 5 landmarks).
//!   - Per-scale 1x1-conv CNN producing 15-dim per pixel.
//!   - NMS over the decoded boxes across all scales.
//!
//! Uses DUMMY random weights from `weights/yunet.bin`.

use crate::detector::Detection;
use crate::face_detector::FaceDetector;
use crate::image::GrayImage;

#[derive(Clone, Debug)]
pub struct YunetConfig {
    pub conf_threshold: f32,
    pub nms_iou: f32,
    pub strides: [usize; 5],
    pub anchor_sizes: [usize; 5],
    pub tile_sizes: [usize; 5],
}

impl Default for YunetConfig {
    fn default() -> Self {
        Self {
            conf_threshold: 0.6,
            nms_iou: 0.3,
            strides: [8, 16, 32, 64, 128],
            anchor_sizes: [8, 16, 32, 64, 128],
            tile_sizes: [16, 16, 16, 16, 16],
        }
    }
}

const ANCHOR_OUT_DIM: usize = 15;
const PER_SCALE_IN_CH: usize = 1;
const PER_SCALE_HID_CH: usize = 4;

pub struct YunetDetector {
    config: YunetConfig,
    weights: YunetWeights,
}

struct YunetWeights {
    conv1_w: [Vec<f32>; 5],
    conv1_b: [Vec<f32>; 5],
    cls_w: [Vec<f32>; 5],
    cls_b: [Vec<f32>; 5],
}

impl YunetWeights {
    fn from_bytes(raw: &'static [u8]) -> Self {
        let mut idx = 0usize;
        let mut read_into = |dst: &mut [f32]| {
            for slot in dst.iter_mut() {
                if idx + 4 <= raw.len() {
                    *slot =
                        f32::from_le_bytes([raw[idx], raw[idx + 1], raw[idx + 2], raw[idx + 3]]);
                    idx += 4;
                }
            }
        };
        let mut conv1_w: [Vec<f32>; 5] = Default::default();
        let mut conv1_b: [Vec<f32>; 5] = Default::default();
        let mut cls_w: [Vec<f32>; 5] = Default::default();
        let mut cls_b: [Vec<f32>; 5] = Default::default();
        for s in 0..5 {
            conv1_w[s] = vec![0.0; PER_SCALE_IN_CH * PER_SCALE_HID_CH];
            conv1_b[s] = vec![0.0; PER_SCALE_HID_CH];
            cls_w[s] = vec![0.0; PER_SCALE_HID_CH * ANCHOR_OUT_DIM];
            cls_b[s] = vec![0.0; ANCHOR_OUT_DIM];
            read_into(&mut conv1_w[s]);
            read_into(&mut conv1_b[s]);
            read_into(&mut cls_w[s]);
            read_into(&mut cls_b[s]);
        }
        Self {
            conv1_w,
            conv1_b,
            cls_w,
            cls_b,
        }
    }
}

impl YunetDetector {
    pub fn new(config: YunetConfig) -> Self {
        let raw: &'static [u8] = include_bytes!("weights/yunet.bin");
        let weights = YunetWeights::from_bytes(raw);
        Self { config, weights }
    }

    fn forward_tile(&self, scale: usize, tile: &[f32]) -> [f32; ANCHOR_OUT_DIM] {
        let tile_w = self.config.tile_sizes[scale];
        let n_pix = tile_w * tile_w;
        let mut out = [0.0f32; ANCHOR_OUT_DIM];
        let mut hidden = vec![0.0f32; n_pix * PER_SCALE_HID_CH];
        for p in 0..n_pix {
            for h_ch in 0..PER_SCALE_HID_CH {
                hidden[p * PER_SCALE_HID_CH + h_ch] =
                    tile[p] * self.weights.conv1_w[scale][h_ch] + self.weights.conv1_b[scale][h_ch];
            }
        }
        for v in hidden.iter_mut() {
            if *v < 0.0 {
                *v = 0.0;
            }
        }
        for p in 0..n_pix {
            for o in 0..ANCHOR_OUT_DIM {
                let mut s = self.weights.cls_b[scale][o];
                for h_ch in 0..PER_SCALE_HID_CH {
                    s += hidden[p * PER_SCALE_HID_CH + h_ch]
                        * self.weights.cls_w[scale][h_ch * ANCHOR_OUT_DIM + o];
                }
                out[o] += s;
            }
        }
        let inv = 1.0 / n_pix as f32;
        for v in out.iter_mut() {
            *v *= inv;
        }
        out
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
}

impl FaceDetector for YunetDetector {
    fn name(&self) -> &'static str {
        "yunet"
    }
    fn description(&self) -> &'static str {
        "YuNet-style anchor-based: 5 scales (8/16/32/64/128), 15-dim per anchor, NMS."
    }

    fn detect(&self, img: &GrayImage) -> Vec<Detection> {
        let w = img.width();
        let h = img.height();
        let mut raw: Vec<Detection> = Vec::new();
        if w < 8 || h < 8 {
            return raw;
        }

        for (s_idx, &stride) in self.config.strides.iter().enumerate() {
            let tile = self.config.tile_sizes[s_idx];
            if tile == 0 || stride == 0 {
                continue;
            }
            let mut y = 0usize;
            while y + tile <= h {
                let mut x = 0usize;
                while x + tile <= w {
                    let mut tile_buf = vec![0.0f32; tile * tile];
                    for ty in 0..tile {
                        let row = img.row(y + ty);
                        for tx in 0..tile {
                            tile_buf[ty * tile + tx] = row[x + tx] as f32 / 255.0;
                        }
                    }
                    let out = self.forward_tile(s_idx, &tile_buf);
                    let conf = Self::sigmoid(out[4]);
                    if conf >= self.config.conf_threshold {
                        let anchor = self.config.anchor_sizes[s_idx];
                        let cx = (x + tile / 2) as f32;
                        let cy = (y + tile / 2) as f32;
                        let dx = out[0] * stride as f32;
                        let dy = out[1] * stride as f32;
                        let dw = out[2] * stride as f32;
                        let dh = out[3] * stride as f32;
                        let bw = anchor as f32 * dw.exp();
                        let bh = anchor as f32 * dh.exp();
                        let bx = cx + dx - bw / 2.0;
                        let by = cy + dy - bh / 2.0;
                        let bx = bx.max(0.0).min((w as f32 - 1.0).max(0.0));
                        let by = by.max(0.0).min((h as f32 - 1.0).max(0.0));
                        let bw = bw.max(1.0);
                        let bh = bh.max(1.0);
                        let ix2 = (bx + bw) as usize;
                        let iy2 = (by + bh) as usize;
                        if ix2 <= w && iy2 <= h && ix2 > bx as usize && iy2 > by as usize {
                            raw.push(Detection {
                                x: bx as usize,
                                y: by as usize,
                                w: ix2 - bx as usize,
                                h: iy2 - by as usize,
                                score: conf,
                            });
                        }
                    }
                    x += stride;
                }
                y += stride;
            }
        }
        crate::detector::non_max_suppression(raw, self.config.nms_iou)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_image_no_panic() {
        let d = YunetDetector::new(YunetConfig::default());
        let img = GrayImage::new(100, 100);
        let r = d.detect(&img);
        assert!(r.is_empty() || r.iter().all(|dd| dd.w > 0 && dd.h > 0));
    }

    #[test]
    fn tiny_image_no_panic() {
        let d = YunetDetector::new(YunetConfig::default());
        let img = GrayImage::new(4, 4);
        let r = d.detect(&img);
        assert!(r.is_empty());
    }

    #[test]
    fn name_is_yunet() {
        let d = YunetDetector::new(YunetConfig::default());
        assert_eq!(d.name(), "yunet");
    }
}
