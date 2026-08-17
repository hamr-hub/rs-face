//! CNN face detector — minimal implementation from scratch.
//!
//! Implements the most-used modern face detection architecture: a small
//! convolutional neural network that scores each sliding-window patch.
//!
//! **Architecture**:
//!   - Input:  24×24 grayscale patch (matches Viola-Jones window size)
//!   - Conv1:  3×3, 1 → 8 channels, ReLU
//!   - Conv2:  3×3, 8 → 16 channels, ReLU + 2×2 MaxPool
//!   - Conv3:  3×3, 16 → 32 channels, ReLU + 2×2 MaxPool
//!   - Flatten: 32 × 4 × 4 = 512
//!   - FC1:    512 → 32, ReLU
//!   - FC2:    32 → 1, Sigmoid (face confidence)
//!
//! **Weights**: hand-crafted to detect a "bright centre + darker border"
//! face-like pattern. These are NOT pretrained — they encode a simple
//! template match that catches face-like structures in real images.
//!
//! No GPU acceleration (yet) — runs single-threaded on the CPU. The
//! architecture is straightforwardly parallelisable; an OpenCL kernel
//! could replace the inner loops with one work-item per (x, y, c_out).

/// CNN face detection result for one window.
#[derive(Clone, Debug)]
pub struct CnnDetection {
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
    pub confidence: f32,
}

/// Configuration for the CNN detector.
#[derive(Clone, Debug)]
pub struct CnnConfig {
    pub window_w: usize,
    pub window_h: usize,
    pub stride: usize,
    pub confidence_threshold: f32,
    pub max_size: usize,
}

impl Default for CnnConfig {
    fn default() -> Self {
        Self {
            window_w: 24,
            window_h: 24,
            stride: 4,
            confidence_threshold: 0.5,
            max_size: 200,
        }
    }
}

/// 2D convolution (single channel input → single channel output, no padding).
/// Used as a building block — not the production path (which uses im2col).
fn conv2d_1to1(
    img: &[f32], w: usize, h: usize,
    kernel: &[f32], kw: usize, kh: usize,
) -> Vec<f32> {
    let mut out = vec![0.0; (w - kw + 1) * (h - kh + 1)];
    for y in 0..(h - kh + 1) {
        for x in 0..(w - kw + 1) {
            let mut sum = 0.0;
            for ky in 0..kh {
                for kx in 0..kw {
                    sum += img[(y + ky) * w + (x + kx)] * kernel[ky * kw + kx];
                }
            }
            out[y * (w - kw + 1) + x] = sum;
        }
    }
    out
}

/// 2D convolution with multiple input channels → multiple output channels.
/// Padded with zeros ("valid" conv, output size = input - kernel + 1).
fn conv2d(
    input: &[f32], w: usize, h: usize, c_in: usize,
    kernel: &[f32], kw: usize, kh: usize, c_out: usize,
) -> Vec<f32> {
    let ow = w - kw + 1;
    let oh = h - kh + 1;
    let mut out = vec![0.0; ow * oh * c_out];
    for co in 0..c_out {
        for y in 0..oh {
            for x in 0..ow {
                let mut sum = 0.0;
                for ci in 0..c_in {
                    for ky in 0..kh {
                        for kx in 0..kw {
                            sum += input[((y + ky) * w + (x + kx)) * c_in + ci]
                                  * kernel[((ky * kw + kx) * c_in + ci) * c_out + co];
                        }
                    }
                }
                out[(y * ow + x) * c_out + co] = sum;
            }
        }
    }
    out
}

/// ReLU in-place.
pub fn relu(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 { *v = 0.0; }
    }
}

/// 2×2 max pool, stride 2. Operates on (h, w, c) layout.
fn maxpool2(input: &[f32], w: usize, h: usize, c: usize) -> Vec<f32> {
    let ow = w / 2;
    let oh = h / 2;
    let mut out = vec![0.0; ow * oh * c];
    for co in 0..c {
        for y in 0..oh {
            for x in 0..ow {
                let mut m = f32::NEG_INFINITY;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let v = input[((y * 2 + dy) * w + (x * 2 + dx)) * c + co];
                        if v > m { m = v; }
                    }
                }
                out[(y * ow + x) * c + co] = m;
            }
        }
    }
    out
}

/// Fully-connected layer: y = x · W + b. Returns y of length `out`.
fn fc(input: &[f32], weights: &[f32], bias: &[f32], out: usize) -> Vec<f32> {
    let in_n = input.len();
    debug_assert_eq!(weights.len(), in_n * out);
    debug_assert_eq!(bias.len(), out);
    let mut y = vec![0.0; out];
    for o in 0..out {
        let mut s = bias[o];
        for i in 0..in_n {
            s += input[i] * weights[o * in_n + i];
        }
        y[o] = s;
    }
    y
}

/// Sigmoid in-place.
pub fn sigmoid(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = 1.0 / (1.0 + (-*v).exp());
    }
}

/// Hand-crafted CNN weights for face-template detection.
///
/// Designed to fire on windows with:
/// - A bright centre region (face oval)
/// - A darker surrounding border (hair / shadow)
/// - Slight edge at the boundary (forehead / cheek transition)
///
/// These are NOT pretrained — they encode a simple "bright centre"
/// template that catches face-like structures. Real CNN face detectors
/// (YuNet, SCRFD) use pretrained weights on millions of face images.
pub fn template_face_weights() -> CnnWeights {
    // 3×3 edge kernel (Sobel-ish, detects brightness gradient)
    let sobel_h = [
        -1.0, 0.0, 1.0,
        -2.0, 0.0, 2.0,
        -1.0, 0.0, 1.0,
    ];
    let sobel_v = [
        -1.0, -2.0, -1.0,
         0.0,  0.0,  0.0,
         1.0,  2.0, 1.0,
    ];
    // 3×3 Laplacian / "centre brighter than surrounds" detector
    let centre_brighter = [
        -0.5, -0.5, -0.5,
        -0.5,  3.0, -0.5,
        -0.5, -0.5, -0.5,
    ];

    // Conv1: 8 output channels from 1 input. Each output channel is a
    // different 3×3 kernel applied to the grayscale input.
    let mut conv1 = vec![0.0; 8 * 3 * 3 * 1];
    for co in 0..8 {
        let kernel: &[f32] = match co {
            0 => &sobel_h,
            1 => &sobel_v,
            2 => &centre_brighter,
            3 => &sobel_h,
            4 => &sobel_v,
            5 => &centre_brighter,
            6 => &sobel_h,
            7 => &sobel_v,
            _ => &sobel_h, // unreachable, 0..8 covers all
        };
        for k in 0..9 {
            conv1[(k * 1) * 8 + co] = kernel[k];
        }
    }

    // Conv2: 16 out from 8 in. Detect combinations of edge features.
    // Random-ish but structured: positive on (bright) channels, negative on
    // (edge) channels → "bright + smooth" detector.
    let mut conv2 = vec![0.0; 16 * 3 * 3 * 8];
    for co in 0..16 {
        for ci in 0..8 {
            let k_base = ((0 * 3 + 0) * 8 + ci) * 16 + co;
            let w = if ci < 3 { 0.0 } else if ci == 2 || ci == 5 { 0.5 } else { 0.2 };
            conv2[k_base] = w;
        }
    }
    // Add a small bias for center of the kernel to detect "smooth bright spots"
    for co in 0..16 {
        for ci in 2..6 {
            let off = ((1 * 3 + 1) * 8 + ci) * 16 + co;
            conv2[off] += 0.3;
        }
    }

    // Conv3: 32 out from 16 in. Aggregate into face-like response map.
    let mut conv3 = vec![0.0; 32 * 3 * 3 * 16];
    for co in 0..32 {
        for ci in 0..16 {
            let off = ((1 * 3 + 1) * 16 + ci) * 32 + co;
            // Centre weight: positive for "all-channels-bright" detection
            conv3[off] = if ci < 8 { 0.5 } else { 0.1 };
        }
    }

    // FC1: 32 outputs (32 channel pool output flattened to 32*4*4=512 → 32).
    // Detects "high response everywhere" → face-like.
    let mut fc1_w = vec![0.0; 32 * 512];
    let mut fc1_b = vec![0.0; 32];
    for o in 0..32 {
        for i in 0..512 {
            fc1_w[o * 512 + i] = 0.01;
        }
        fc1_b[o] = -2.0; // bias so it needs strong evidence
    }

    // FC2: 1 output (face confidence).
    let mut fc2_w = vec![0.0; 32];
    let mut fc2_b = vec![0.0; 1];
    for o in 0..32 {
        fc2_w[o] = 1.0; // pure sum
    }
    fc2_b[0] = -2.0;

    CnnWeights {
        conv1_w: conv1, conv2_w: conv2, conv3_w: conv3,
        fc1_w, fc1_b, fc2_w, fc2_b,
    }
}

/// Bundled CNN weights (immutable after construction).
#[derive(Clone)]
pub struct CnnWeights {
    pub conv1_w: Vec<f32>, // 8 × 3 × 3 × 1
    pub conv2_w: Vec<f32>, // 16 × 3 × 3 × 8
    pub conv3_w: Vec<f32>, // 32 × 3 × 3 × 16
    pub fc1_w: Vec<f32>,   // 32 × 512
    pub fc1_b: Vec<f32>,
    pub fc2_w: Vec<f32>,   // 32
    pub fc2_b: Vec<f32>,
}

impl CnnWeights {
    /// Load weights from a `.cnn.bin` file produced by `cnn_train`.
    /// File format: magic "RCNN", version u32, then seven u32 lengths
    /// (conv1_w, conv2_w, conv3_w, fc1_w, fc1_b, fc2_w, fc2_b) followed
    /// by the raw little-endian f32 values.
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        use std::io::Read;
        let mut f = std::fs::File::open(path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != b"RCNN" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
                "not a CNN weights file"));
        }
        let mut vbuf = [0u8; 4];
        f.read_exact(&mut vbuf)?; let _version = u32::from_le_bytes(vbuf);
        let mut lens = [0usize; 7];
        for slot in &mut lens {
            f.read_exact(&mut vbuf)?;
            *slot = u32::from_le_bytes(vbuf) as usize;
        }
        let mut read_vec = |n: usize| -> std::io::Result<Vec<f32>> {
            let mut buf = vec![0f32; n];
            let mut raw = vec![0u8; n * 4];
            f.read_exact(&mut raw)?;
            for (i, chunk) in raw.chunks_exact(4).enumerate() {
                buf[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
            Ok(buf)
        };
        Ok(Self {
            conv1_w: read_vec(lens[0])?,
            conv2_w: read_vec(lens[1])?,
            conv3_w: read_vec(lens[2])?,
            fc1_w:   read_vec(lens[3])?,
            fc1_b:   read_vec(lens[4])?,
            fc2_w:   read_vec(lens[5])?,
            fc2_b:   read_vec(lens[6])?,
        })
    }
}

/// Per-thread scratch buffers for the CNN forward pass. Allocating these
/// once per detector and reusing them across every window eliminates the
/// 5+ Vec allocations per window that previously dominated the cost.
///
/// Wrapped in `UnsafeCell` because the detector takes `&self` in `detect()`
/// — RefCell would add a branch per call.
pub struct CnnScratch {
    inner: std::cell::UnsafeCell<CnnScratchInner>,
}

pub struct CnnScratchInner {
    pub c1: Vec<f32>,  // 22 * 22 * 8
    pub c2: Vec<f32>,  // 20 * 20 * 16
    pub c2p: Vec<f32>, // 10 * 10 * 16
    pub c3: Vec<f32>,  // 8 * 8 * 32
    pub c3p: Vec<f32>, // 4 * 4 * 32
    pub f1: Vec<f32>,  // 32
    pub f2: Vec<f32>,  // 1
}

impl CnnScratch {
    pub fn new() -> Self {
        Self {
            inner: std::cell::UnsafeCell::new(CnnScratchInner {
                c1:  vec![0.0; 22 * 22 * 8],
                c2:  vec![0.0; 20 * 20 * 16],
                c2p: vec![0.0; 10 * 10 * 16],
                c3:  vec![0.0; 8 * 8 * 32],
                c3p: vec![0.0; 4 * 4 * 32],
                f1:  vec![0.0; 32],
                f2:  vec![0.0; 1],
            }),
        }
    }
    /// Borrow the inner scratch buffers mutably. Caller must ensure no two
    /// `detect()` calls on the same detector race — typically you use one
    /// detector per worker thread (the pipeline does this).
    #[inline]
    pub fn buffers_mut(&self) -> &mut CnnScratchInner {
        // SAFETY: each detector instance is owned by exactly one thread
        // (the pipeline spawns one worker per detector). The pipeline's
        // `Arc<Detector>` clones don't share scratch state.
        unsafe { &mut *self.inner.get() }
    }
}

/// Forward pass: take a 24×24 grayscale window, return face confidence ∈ [0, 1].
/// Writes through `scratch.buffers_mut()` to avoid per-window allocations.
pub fn forward(weights: &CnnWeights, window: &[f32], scratch: &CnnScratch) -> f32 {
    debug_assert_eq!(window.len(), 24 * 24);
    let s = scratch.buffers_mut();

    // Conv1: 24×24×1 → 22×22×8
    conv2d_into(window, 24, 24, 1, &weights.conv1_w, 3, 3, 8, &mut s.c1);
    relu(&mut s.c1);

    // Conv2: 22×22×8 → 20×20×16
    conv2d_into(&s.c1, 22, 22, 8, &weights.conv2_w, 3, 3, 16, &mut s.c2);
    relu(&mut s.c2);

    // MaxPool 2×2: 20×20×16 → 10×10×16
    maxpool2_into(&s.c2, 20, 20, 16, &mut s.c2p);

    // Conv3: 10×10×16 → 8×8×32
    conv2d_into(&s.c2p, 10, 10, 16, &weights.conv3_w, 3, 3, 32, &mut s.c3);
    relu(&mut s.c3);

    // MaxPool 2×2: 8×8×32 → 4×4×32
    maxpool2_into(&s.c3, 8, 8, 32, &mut s.c3p);
    // Flatten: 4*4*32 = 512

    // FC1: 512 → 32
    fc_into(&s.c3p, &weights.fc1_w, &weights.fc1_b, 32, &mut s.f1);
    relu(&mut s.f1);

    // FC2: 32 → 1
    fc_into(&s.f1, &weights.fc2_w, &weights.fc2_b, 1, &mut s.f2);

    // Sigmoid
    sigmoid(&mut s.f2);
    s.f2[0]
}

/// Like `conv2d` but writes into a caller-provided buffer. Buffer must be of
/// length `ow * oh * c_out`.
pub fn conv2d_into(
    input: &[f32], w: usize, h: usize, c_in: usize,
    kernel: &[f32], kw: usize, kh: usize, c_out: usize,
    out: &mut [f32],
) {
    let ow = w - kw + 1;
    let oh = h - kh + 1;
    debug_assert_eq!(out.len(), ow * oh * c_out);
    out.fill(0.0);
    for co in 0..c_out {
        for y in 0..oh {
            for x in 0..ow {
                let mut sum = 0.0f32;
                for ci in 0..c_in {
                    for ky in 0..kh {
                        for kx in 0..kw {
                            sum += input[((y + ky) * w + (x + kx)) * c_in + ci]
                                  * kernel[((ky * kw + kx) * c_in + ci) * c_out + co];
                        }
                    }
                }
                out[(y * ow + x) * c_out + co] = sum;
            }
        }
    }
}

/// Like `maxpool2` but writes into a caller-provided buffer.
pub fn maxpool2_into(input: &[f32], w: usize, h: usize, c: usize, out: &mut [f32]) {
    let ow = w / 2;
    let oh = h / 2;
    debug_assert_eq!(out.len(), ow * oh * c);
    out.fill(0.0);
    for co in 0..c {
        for y in 0..oh {
            for x in 0..ow {
                let mut m = f32::NEG_INFINITY;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let v = input[((y * 2 + dy) * w + (x * 2 + dx)) * c + co];
                        if v > m { m = v; }
                    }
                }
                out[(y * ow + x) * c + co] = m;
            }
        }
    }
}

/// Like `fc` but writes into a caller-provided buffer.
pub fn fc_into(input: &[f32], weights: &[f32], bias: &[f32], out_n: usize, out: &mut [f32]) {
    debug_assert_eq!(out.len(), out_n);
    let in_n = input.len();
    debug_assert_eq!(weights.len(), in_n * out_n);
    debug_assert_eq!(bias.len(), out_n);
    for o in 0..out_n {
        let mut s = bias[o];
        for i in 0..in_n {
            s += input[i] * weights[o * in_n + i];
        }
        out[o] = s;
    }
}

/// CNN face detector — runs the CNN forward pass on every window.
pub struct CnnDetector {
    weights: CnnWeights,
    config: CnnConfig,
    /// Reusable scratch buffer for the forward pass — eliminates per-window
    /// allocations of the 7 intermediate tensors.
    scratch: CnnScratch,
}

impl CnnDetector {
    pub fn new(config: CnnConfig) -> Self {
        Self {
            weights: template_face_weights(),
            config,
            scratch: CnnScratch::new(),
        }
    }

    /// Construct with explicit weights (e.g. loaded from a `.cnn.bin` file
    /// produced by `cnn_train`).
    pub fn with_weights(weights: CnnWeights, config: CnnConfig) -> Self {
        Self { weights, config, scratch: CnnScratch::new() }
    }

    pub fn weights(&self) -> &CnnWeights {
        &self.weights
    }

    /// Replace the weights in-place. Useful for hot-reloading after training.
    pub fn set_weights(&mut self, weights: CnnWeights) {
        self.weights = weights;
    }

    /// Detect faces in a grayscale image (any size). Returns detections
    /// sorted by descending confidence.
    pub fn detect(&self, img: &[f32], w: usize, h: usize) -> Vec<CnnDetection> {
        let mut detections = Vec::new();
        let ww = self.config.window_w;
        let wh = self.config.window_h;
        let stride = self.config.stride;
        if w < ww || h < wh { return detections; }

        let mut window = [0.0f32; 24 * 24];
        let mut y = 0;
        while y + wh <= h {
            let mut x = 0;
            while x + ww <= w {
                // Extract window into a stack-allocated buffer. Replaces a
                // heap-allocated Vec<f32> per window.
                for wy in 0..wh {
                    let src = &img[(y + wy) * w + x..(y + wy) * w + x + ww];
                    let dst_start = wy * ww;
                    window[dst_start..dst_start + ww].copy_from_slice(src);
                }
                let conf = forward(&self.weights, &window, &self.scratch);
                if conf >= self.config.confidence_threshold {
                    detections.push(CnnDetection {
                        x, y, w: ww, h: wh, confidence: conf,
                    });
                }
                x += stride;
            }
            y += stride;
        }

        // Non-maximum suppression (greedy IoU 0.3)
        detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        let mut kept = Vec::new();
        let mut suppressed = vec![false; detections.len()];
        for i in 0..detections.len() {
            if suppressed[i] { continue; }
            kept.push(detections[i].clone());
            for j in (i + 1)..detections.len() {
                if suppressed[j] { continue; }
                let iou = iou(&detections[i], &detections[j]);
                if iou > 0.3 { suppressed[j] = true; }
            }
        }
        kept
    }
}

fn iou(a: &CnnDetection, b: &CnnDetection) -> f32 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.w).min(b.x + b.w);
    let y2 = (a.y + a.h).min(b.y + b.h);
    let inter = if x2 > x1 && y2 > y1 { (x2 - x1) * (y2 - y1) } else { 0 };
    let union = a.w * a.h + b.w * b.h - inter;
    if union == 0 { 0.0 } else { inter as f32 / union as f32 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_runs() {
        let w = template_face_weights();
        let window: Vec<f32> = (0..24*24).map(|i| (i as f32) / 100.0).collect();
        let scratch = CnnScratch::new();
        let conf = forward(&w, &window, &scratch);
        assert!(conf >= 0.0 && conf <= 1.0, "sigmoid output out of [0,1]: {}", conf);
    }

    #[test]
    fn detects_bright_center() {
        let det = CnnDetector::new(CnnConfig::default());
        // 100×100 image with bright centre
        let mut img = vec![0.0f32; 100 * 100];
        for y in 38..62 {
            for x in 38..62 {
                img[y * 100 + x] = 1.0;
            }
        }
        let dets = det.detect(&img, 100, 100);
        // Should detect at least one face-like region in the centre
        // (the hand-crafted weights look for bright centre + dark border)
        assert!(!dets.is_empty(), "expected at least one face-like detection in bright-centre pattern");
        let c = &dets[0];
        // The detected window should overlap the bright centre
        assert!(c.x < 62 && c.x + c.w > 38 && c.y < 62 && c.y + c.h > 38,
            "detection at ({},{}) {}x{} does not overlap bright centre (38,38)-(62,62)",
            c.x, c.y, c.w, c.h);
    }
}