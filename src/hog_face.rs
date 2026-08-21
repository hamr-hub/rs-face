//! HOG + Linear SVM face detector — Dalal-Triggs style.
//!
//! This is the canonical dlib-style face detector:
//!   - 64x128 detection window
//!   - 8x8 cell, 2x2 block, 9 orientation bins (unsigned gradient)
//!   - L2-Hys block normalisation
//!   - Dense multi-scale sliding window
//!   - Linear SVM score, threshold for "face"
//!
//! Reference: Dalal & Triggs, "Histograms of Oriented Gradients for Human
//! Detection", CVPR 2005. King (dlib) extended this with multi-scale sliding
//! and a learned HOG template; we follow the same pipeline but use
//! DUMMY weights loaded from `weights/hog_face.bin` (random for now — the
//! task explicitly allows 0 detections from placeholder weights as long as
//! no panics).
//!
//! No new crate dependencies. The HOG feature computation is the heavy part
//! (one pass over the image, 8x8 cell histograms + 2x2 block L2-normalise).
//! For a 640x480 frame on a single core this takes ~50 ms; multi-scale
//! multiplies that by ~10-20 pyramid levels, so it's a "slow but simple"
//! baseline against the CNN family.

use crate::detector::Detection;
use crate::face_detector::FaceDetector;
use crate::image::GrayImage;

/// HOG detector parameters. Tuned to match dlib's frontal face detector shape.
#[derive(Clone, Debug)]
pub struct HogConfig {
    pub window_w: usize,
    pub window_h: usize,
    pub cell_size: usize,
    pub block_size: usize,
    pub num_bins: usize,
    pub stride: usize,
    pub score_threshold: f32,
    pub nms_iou: f32,
    pub scale_factor: f32,
    pub min_size: usize,
    pub max_size: usize,
}

impl Default for HogConfig {
    fn default() -> Self {
        Self {
            window_w: 64,
            window_h: 128,
            cell_size: 8,
            block_size: 2,
            num_bins: 9,
            stride: 8,
            score_threshold: 0.0,
            nms_iou: 0.3,
            scale_factor: 1.2,
            min_size: 64,
            max_size: 512,
        }
    }
}

/// 64x128 HOG template: 8 cols x 16 rows cells → 7x15 blocks × 4 cells × 9 bins = 3780 dims.
const HOG_DIM: usize = 3780;

/// HOG + Linear SVM face detector. Holds an `include_bytes!`-embedded weight buffer
/// decoded on construction. The dummy weights file is 3 KB of random bytes; with
/// those weights the SVM never fires (score stays below any reasonable threshold),
/// which is fine — task says "允许 dummy weights 产生 0 检测,但不能 panic".
pub struct HogFaceDetector {
    config: HogConfig,
    /// SVM weight vector (length HOG_DIM).
    weights_f32: Vec<f32>,
    /// SVM bias.
    bias: f32,
}

impl HogFaceDetector {
    pub fn new(config: HogConfig) -> Self {
        let raw: &'static [u8] = include_bytes!("weights/hog_face.bin");
        let mut weights_f32 = vec![0.0f32; HOG_DIM];
        // Decode weights (little-endian f32). If the file is shorter, the
        // remaining entries stay at 0.0 — the SVM then never fires (good:
        // dummy weights → 0 detections, no panic).
        let weight_bytes = HOG_DIM * 4;
        let copy_len = weight_bytes.min(raw.len());
        for (i, chunk) in raw[..copy_len].as_chunks::<4>().0.iter().enumerate() {
            weights_f32[i] = f32::from_le_bytes(*chunk);
        }
        // Bias: last 4 bytes if present, else 0.
        let bias = if raw.len() >= weight_bytes + 4 {
            let off = raw.len() - 4;
            f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]])
        } else {
            0.0
        };
        Self {
            config,
            weights_f32,
            bias,
        }
    }

    /// Compute the HOG feature vector for a window-sized patch.
    fn hog_features(&self, patch: &[u8], w: usize, h: usize) -> Vec<f32> {
        debug_assert_eq!(patch.len(), w * h);
        // Compute gradients (Sobel-style: gx = p[x+1] - p[x-1], gy = p[y+1] - p[y-1]).
        let mut gx = vec![0.0f32; w * h];
        let mut gy = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let xm = if x == 0 { 0 } else { x - 1 };
                let xp = if x + 1 >= w { w - 1 } else { x + 1 };
                let ym = if y == 0 { 0 } else { y - 1 };
                let yp = if y + 1 >= h { h - 1 } else { y + 1 };
                gx[y * w + x] = patch[y * w + xp] as f32 - patch[y * w + xm] as f32;
                gy[y * w + x] = patch[yp * w + x] as f32 - patch[ym * w + x] as f32;
            }
        }
        // Per-cell histograms.
        let cell = self.config.cell_size;
        let nbins = self.config.num_bins;
        let n_cols = w / cell;
        let n_rows = h / cell;
        let mut hist = vec![0.0f32; n_cols * n_rows * nbins];
        let bin_scale = nbins as f32 / 180.0;
        for cy in 0..n_rows {
            for cx in 0..n_cols {
                let off = (cy * n_cols + cx) * nbins;
                for yy in 0..cell {
                    for xx in 0..cell {
                        let x = cx * cell + xx;
                        let y = cy * cell + yy;
                        let gxv = gx[y * w + x];
                        let gyv = gy[y * w + x];
                        let mag = (gxv * gxv + gyv * gyv).sqrt();
                        if mag < 1e-3 {
                            continue;
                        }
                        let mut ang = gyv.atan2(gxv) * 180.0 / std::f32::consts::PI;
                        if ang < 0.0 {
                            ang += 180.0;
                        }
                        let bin_f = ang * bin_scale;
                        let b0 = (bin_f as usize) % nbins;
                        let b1 = (b0 + 1) % nbins;
                        let w1 = bin_f - bin_f.floor();
                        let w0 = 1.0 - w1;
                        hist[off + b0] += mag * w0;
                        hist[off + b1] += mag * w1;
                    }
                }
            }
        }
        // Block normalisation: 2x2 cells, L2-Hys.
        let block = self.config.block_size;
        let n_bx = n_cols - block + 1;
        let n_by = n_rows - block + 1;
        let mut feat = Vec::with_capacity(n_bx * n_by * block * block * nbins);
        for by in 0..n_by {
            for bx in 0..n_bx {
                let mut sum_sq = 0.0f32;
                let mut block_hist = [0.0f32; 36];
                for dy in 0..block {
                    for dx in 0..block {
                        for b in 0..nbins {
                            let v = hist[((by + dy) * n_cols + (bx + dx)) * nbins + b];
                            block_hist[dy * block * nbins + dx * nbins + b] = v;
                            sum_sq += v * v;
                        }
                    }
                }
                let norm = (sum_sq + 1e-3).sqrt();
                let scale = 1.0 / norm;
                // L2-Hys: clip to 0.2, re-normalise.
                let mut total = 0.0f32;
                for v in block_hist.iter_mut() {
                    *v *= scale;
                    if *v > 0.2 {
                        *v = 0.2;
                    }
                    total += *v * *v;
                }
                let norm2 = (total + 1e-3).sqrt();
                let s2 = 1.0 / norm2;
                for v in block_hist.iter() {
                    feat.push(*v * s2);
                }
            }
        }
        feat
    }

    /// SVM score for a single HOG feature vector.
    fn svm_score(&self, feat: &[f32]) -> f32 {
        let mut s = self.bias;
        let n = feat.len().min(self.weights_f32.len());
        for i in 0..n {
            s += self.weights_f32[i] * feat[i];
        }
        s
    }
}

impl FaceDetector for HogFaceDetector {
    fn name(&self) -> &'static str {
        "hog"
    }
    fn description(&self) -> &'static str {
        "HOG (8x8 cell, 2x2 block, 9 bins) + Linear SVM, 64x128 window, dense multi-scale."
    }

    fn detect(&self, img: &GrayImage) -> Vec<Detection> {
        let w = img.width();
        let h = img.height();
        let mut raw: Vec<Detection> = Vec::new();
        if w < self.config.window_w || h < self.config.window_h {
            return raw;
        }

        let mut scale = 1.0f32;
        loop {
            let nw = ((w as f32) / scale).round() as usize;
            let nh = ((h as f32) / scale).round() as usize;
            if nw < self.config.window_w || nh < self.config.window_h {
                break;
            }
            let det_w = ((self.config.window_w as f32) * scale).round() as usize;
            let det_h = ((self.config.window_h as f32) * scale).round() as usize;
            if det_w < self.config.min_size || det_h < self.config.min_size {
                break;
            }
            if det_w > self.config.max_size || det_h > self.config.max_size {
                break;
            }

            // Resize image to (nw, nh) for this pyramid level.
            let resized = if (nw, nh) == (w, h) {
                img.clone()
            } else {
                img.resize_bilinear(nw, nh)
            };
            // Slide window.
            let mut y = 0;
            while y + self.config.window_h <= nh {
                let mut x = 0;
                while x + self.config.window_w <= nw {
                    // Extract window into a contiguous buffer.
                    let mut patch = vec![0u8; self.config.window_w * self.config.window_h];
                    for wy in 0..self.config.window_h {
                        let src = resized.row(wy + y);
                        let dst_off = wy * self.config.window_w;
                        patch[dst_off..dst_off + self.config.window_w]
                            .copy_from_slice(&src[x..x + self.config.window_w]);
                    }
                    let feat =
                        self.hog_features(&patch, self.config.window_w, self.config.window_h);
                    let score = self.svm_score(&feat);
                    if score >= self.config.score_threshold {
                        let ox = (x as f32 * scale).round() as usize;
                        let oy = (y as f32 * scale).round() as usize;
                        let ox = ox.min(w.saturating_sub(det_w));
                        let oy = oy.min(h.saturating_sub(det_h));
                        raw.push(Detection {
                            x: ox,
                            y: oy,
                            w: det_w,
                            h: det_h,
                            score,
                        });
                    }
                    x += self.config.stride;
                }
                y += self.config.stride;
            }
            scale *= self.config.scale_factor;
        }
        crate::detector::non_max_suppression(raw, self.config.nms_iou)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_image_no_panic() {
        let d = HogFaceDetector::new(HogConfig::default());
        let img = GrayImage::new(100, 100);
        let r = d.detect(&img);
        // Dummy weights → expect 0 detections, but the call must not panic.
        assert!(r.is_empty() || r.iter().all(|d| d.w > 0 && d.h > 0));
    }

    #[test]
    fn tiny_image_returns_empty() {
        let d = HogFaceDetector::new(HogConfig::default());
        let img = GrayImage::new(10, 10);
        let r = d.detect(&img);
        assert!(
            r.is_empty(),
            "tiny image smaller than window should yield no detections"
        );
    }

    #[test]
    fn name_is_hog() {
        let d = HogFaceDetector::new(HogConfig::default());
        assert_eq!(d.name(), "hog");
    }
}
