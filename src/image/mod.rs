//! Image types and zero-dependency codecs.
//!
//! Supported:
//! - 8-bit grayscale (`GrayImage`) and RGB (`RgbImage`).
//! - PPM/PGM (P5/P6) encode + decode — trivial text/binary format, ideal for tests.
//! - PNG encode — uses stored (uncompressed) DEFLATE blocks; produces valid PNG
//!   but slightly larger than `zlib`-compressed output. Acceptable for our use case.
//! - PNG decode — supports filter types 0..=4 (None, Sub, Up, Average, Paeth).

pub mod codec;
pub mod png;

use std::ops::{Index, IndexMut};

/// 8-bit grayscale image, row-major.
#[derive(Clone, Debug)]
pub struct GrayImage {
    data: Vec<u8>,
    width: usize,
    height: usize,
}

impl GrayImage {
    pub fn new(width: usize, height: usize) -> Self {
        Self { data: vec![0; width * height], width, height }
    }

    pub fn from_vec(data: Vec<u8>, width: usize, height: usize) -> Self {
        assert_eq!(data.len(), width * height, "data size mismatch");
        Self { data, width, height }
    }

    #[inline] pub fn width(&self) -> usize { self.width }
    #[inline] pub fn height(&self) -> usize { self.height }
    #[inline] pub fn as_slice(&self) -> &[u8] { &self.data }
    #[inline] pub fn as_mut_slice(&mut self) -> &mut [u8] { &mut self.data }
    #[inline] pub fn row(&self, y: usize) -> &[u8] { &self.data[y * self.width..(y + 1) * self.width] }
    #[inline] pub fn row_mut(&mut self, y: usize) -> &mut [u8] { &mut self.data[y * self.width..(y + 1) * self.width] }

    /// Mean luminance — used for adaptive thresholding / lighting normalization.
    pub fn mean(&self) -> f32 {
        if self.data.is_empty() { return 0.0; }
        let s: u64 = self.data.iter().map(|&v| v as u64).sum();
        s as f32 / self.data.len() as f32
    }

    /// Standard deviation of luminance.
    pub fn stddev(&self) -> f32 {
        let m = self.mean();
        let v: f64 = self.data.iter().map(|&p| {
            let d = p as f64 - m as f64;
            d * d
        }).sum();
        ((v / self.data.len() as f64) as f32).sqrt()
    }

    /// Histogram-based contrast stretch to `[0, 255]` (ignoring saturated 1% tails).
    pub fn normalize_contrast(&mut self) {
        if self.data.is_empty() { return; }
        let mut hist = [0u32; 256];
        for &p in &self.data { hist[p as usize] += 1; }
        let total = self.data.len() as u32;
        let low_cut = total / 100;
        let high_cut = total - total / 100;
        let mut acc = 0u32;
        let mut lo = 0u8;
        let mut hi = 255u8;
        for (i, &c) in hist.iter().enumerate() {
            acc += c;
            if acc >= low_cut { lo = i as u8; break; }
        }
        acc = 0;
        for (i, &c) in hist.iter().enumerate().rev() {
            acc += c;
            if acc >= high_cut { hi = i as u8; break; }
        }
        if hi <= lo { return; }
        let range = (hi - lo) as f32;
        for p in self.data.iter_mut() {
            let v = *p as i32 - lo as i32;
            let stretched = (v as f32 / range * 255.0).clamp(0.0, 255.0) as u8;
            *p = stretched;
        }
    }

    /// Histogram equalization — `cv::equalizeHist` reference.
    ///
    /// Maps each pixel `p` to `round(cdf[p] * 255 / total)` where `cdf` is
    /// the cumulative distribution of the image histogram. This is the
    /// preprocessing OpenCV's `detectMultiScale` callers are expected to
    /// apply before running the cascade — the trained thresholds and
    /// variance normalisation factors assume input in the equalized
    /// luminance range. Without it the cascade's stage-0 weak features
    /// (which are calibrated to "bright forehead over dark border"-type
    /// responses on equalized data) consistently reject real faces on
    /// raw photographs.
    ///
    /// Cost: one histogram pass + one CDF pass over 256 entries + one
    /// pixel remap = O(W·H) for typical face window sizes.
    pub fn equalize_hist_inplace(&mut self) {
        if self.data.is_empty() { return; }
        let mut hist = [0u32; 256];
        for &p in &self.data { hist[p as usize] += 1; }
        // Cumulative distribution, then scale into 0..=255.
        let mut cdf = [0u32; 256];
        let mut acc: u32 = 0;
        for i in 0..256 {
            acc += hist[i];
            cdf[i] = acc;
        }
        // The OpenCV equalizeHist formula: `round(cdf[p] * 255 / total)`,
        // but only using the part of the cdf after the first non-zero
        // entry (`cdf_min`). This avoids a global brightness shift on
        // mostly-dark images and matches `cv::equalizeHist` byte-for-byte.
        let mut cdf_min: u32 = 0;
        for i in 0..256 {
            if cdf[i] != 0 { cdf_min = cdf[i]; break; }
        }
        let total = self.data.len() as u32;
        let denom = total - cdf_min;
        if denom == 0 { return; }
        // Build a 256-entry LUT so the per-pixel remap is a single load.
        let mut lut = [0u8; 256];
        for i in 0..256 {
            let num = cdf[i].saturating_sub(cdf_min) as u64;
            let v = (num * 255 + (denom as u64 / 2)) / (denom as u64);
            lut[i] = (v as u32).min(255) as u8;
        }
        for p in self.data.iter_mut() { *p = lut[*p as usize]; }
    }

    /// Resize with bilinear interpolation.
    pub fn resize_bilinear(&self, new_w: usize, new_h: usize) -> GrayImage {
        if new_w == 0 || new_h == 0 || self.width == 0 || self.height == 0 {
            return GrayImage::new(new_w.max(1), new_h.max(1));
        }
        let mut out = GrayImage::new(new_w, new_h);
        let sx = self.width as f32 / new_w as f32;
        let sy = self.height as f32 / new_h as f32;
        for y in 0..new_h {
            let fy = (y as f32 + 0.5) * sy - 0.5;
            let y0 = fy.floor().max(0.0) as usize;
            let y1 = (y0 + 1).min(self.height - 1);
            let dy = (fy - y0 as f32).clamp(0.0, 1.0);
            for x in 0..new_w {
                let fx = (x as f32 + 0.5) * sx - 0.5;
                let x0 = fx.floor().max(0.0) as usize;
                let x1 = (x0 + 1).min(self.width - 1);
                let dx = (fx - x0 as f32).clamp(0.0, 1.0);
                let p00 = self.data[y0 * self.width + x0] as f32;
                let p01 = self.data[y0 * self.width + x1] as f32;
                let p10 = self.data[y1 * self.width + x0] as f32;
                let p11 = self.data[y1 * self.width + x1] as f32;
                let top = p00 * (1.0 - dx) + p01 * dx;
                let bot = p10 * (1.0 - dx) + p11 * dx;
                let v = top * (1.0 - dy) + bot * dy;
                out.data[y * new_w + x] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
        out
    }

    /// Resize with area-averaging (matches OpenCV's `INTER_AREA` for downscaling).
    /// Each output pixel is the average of the input pixels it covers.
    /// This is significantly more accurate than bilinear for >2× downscaling and
    /// matches OpenCV's default behavior in `cv::resize` with downscaling.
    pub fn resize_area(&self, new_w: usize, new_h: usize) -> GrayImage {
        if new_w == 0 || new_h == 0 || self.width == 0 || self.height == 0 {
            return GrayImage::new(new_w.max(1), new_h.max(1));
        }
        let mut out = GrayImage::new(new_w, new_h);
        let sx = self.width as f32 / new_w as f32;
        let sy = self.height as f32 / new_h as f32;
        for y in 0..new_h {
            let fy_start = y as f32 * sy;
            let fy_end = (y + 1) as f32 * sy;
            let y0 = fy_start.floor() as usize;
            let y1 = (fy_end.ceil() as usize).min(self.height);
            for x in 0..new_w {
                let fx_start = x as f32 * sx;
                let fx_end = (x + 1) as f32 * sx;
                let x0 = fx_start.floor() as usize;
                let x1 = (fx_end.ceil() as usize).min(self.width);
                // Compute weighted sum using exact sub-pixel coverage.
                let mut sum = 0.0f64;
                let mut area = 0.0f64;
                for yy in y0..y1 {
                    let ycov = (yy + 1) as f32 - fy_start.max(yy as f32);
                    let ycov = ycov.min(fy_end - yy as f32).max(0.0);
                    if ycov <= 0.0 { continue; }
                    for xx in x0..x1 {
                        let xcov = (xx + 1) as f32 - fx_start.max(xx as f32);
                        let xcov = xcov.min(fx_end - xx as f32).max(0.0);
                        if xcov <= 0.0 { continue; }
                        let w = (xcov * ycov) as f64;
                        sum += w * self.data[yy * self.width + xx] as f64;
                        area += w;
                    }
                }
                out.data[y * new_w + x] = if area > 0.0 { (sum / area).round().clamp(0.0, 255.0) as u8 } else { 0 };
            }
        }
        out
    }

    /// Downscale by an integer factor using box averaging (memory- and time-friendly).
    pub fn downscale(&self, factor: usize) -> GrayImage {
        if factor <= 1 { return self.clone(); }
        let nw = self.width / factor;
        let nh = self.height / factor;
        let mut out = GrayImage::new(nw, nh);
        let f = factor as u32;
        for y in 0..nh {
            for x in 0..nw {
                let mut s = 0u32;
                for j in 0..factor {
                    for i in 0..factor {
                        s += self.data[(y * factor + j) * self.width + (x * factor + i)] as u32;
                    }
                }
                out.data[y * nw + x] = (s / (f * f)) as u8;
            }
        }
        out
    }
}

impl Index<(usize, usize)> for GrayImage {
    type Output = u8;
    #[inline]
    fn index(&self, (x, y): (usize, usize)) -> &u8 {
        &self.data[y * self.width + x]
    }
}

impl IndexMut<(usize, usize)> for GrayImage {
    #[inline]
    fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut u8 {
        &mut self.data[y * self.width + x]
    }
}

/// 8-bit RGB image, row-major, 3 bytes per pixel.
#[derive(Clone, Debug)]
pub struct RgbImage {
    data: Vec<u8>,
    width: usize,
    height: usize,
}

impl RgbImage {
    pub fn new(width: usize, height: usize) -> Self {
        Self { data: vec![0; width * height * 3], width, height }
    }

    #[inline] pub fn width(&self) -> usize { self.width }
    #[inline] pub fn height(&self) -> usize { self.height }
    #[inline] pub fn as_slice(&self) -> &[u8] { &self.data }
    #[inline] pub fn as_mut_slice(&mut self) -> &mut [u8] { &mut self.data }
    #[inline] pub fn row(&self, y: usize) -> &[u8] { &self.data[y * self.width * 3..(y + 1) * self.width * 3] }
    #[inline] pub fn row_mut(&mut self, y: usize) -> &mut [u8] { &mut self.data[y * self.width * 3..(y + 1) * self.width * 3] }

    /// RGB → grayscale using BT.601 luma weights.
    pub fn to_gray(&self) -> GrayImage {
        let mut g = GrayImage::new(self.width, self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                let i = (y * self.width + x) * 3;
                let r = self.data[i] as u32;
                let gg = self.data[i + 1] as u32;
                let b = self.data[i + 2] as u32;
                // BT.601: 0.299 R + 0.587 G + 0.114 B, fixed-point.
                let v = (r * 77 + gg * 150 + b * 29) >> 8;
                g.data[y * self.width + x] = v as u8;
            }
        }
        g
    }

    /// Draw a 1-pixel-thick rectangle outline in the given RGB color.
    pub fn draw_rect(&mut self, x: usize, y: usize, w: usize, h: usize, color: (u8, u8, u8)) {
        let (cr, cg, cb) = color;
        for i in 0..w {
            let xx = x + i;
            if xx >= self.width { break; }
            if y < self.height {
                let idx = (y * self.width + xx) * 3;
                self.data[idx] = cr; self.data[idx + 1] = cg; self.data[idx + 2] = cb;
            }
            let yy = y + h;
            if yy < self.height {
                let idx = (yy * self.width + xx) * 3;
                self.data[idx] = cr; self.data[idx + 1] = cg; self.data[idx + 2] = cb;
            }
        }
        for j in 0..h {
            let yy = y + j;
            if yy >= self.height { break; }
            if x < self.width {
                let idx = (yy * self.width + x) * 3;
                self.data[idx] = cr; self.data[idx + 1] = cg; self.data[idx + 2] = cb;
            }
            let xx = x + w;
            if xx < self.width {
                let idx = (yy * self.width + xx) * 3;
                self.data[idx] = cr; self.data[idx + 1] = cg; self.data[idx + 2] = cb;
            }
        }
    }
}
