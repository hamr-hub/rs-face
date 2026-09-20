//! # EmbedNet — a zero-dependency, trainable embedding network.
//!
//! This module is the answer to "can the crate produce deep-learning face
//! embeddings without ONNX Runtime, `tract`, or any other runtime
//! dependency?". Yes: a small convolutional network, its im2col forward
//! pass, its full backprop, and an Adam trainer are all implemented in
//! pure `std` here, behind the **default** (zero-dep) build.
//!
//! ## Pipeline
//!
//! ```text
//! GrayImage crop
//!   └─▶ resize 56×56 (INTER_AREA) + per-image standardisation   ([`EmbedNet::preprocess`])
//!         └─▶ Conv 3×3 1→24  ReLU       56 → 54
//!               └─▶ MaxPool 2×2         54 → 27
//!                     └─▶ Conv 3×3 24→48 ReLU  → pool  27 → 25 → 12
//!                           └─▶ Conv 3×3 48→64 ReLU  → pool  12 → 10 → 5
//!                                 └─▶ Conv 3×3 64→96 ReLU         5 → 3
//!                                       └─▶ Flatten 864 → FC 128
//!                                             └─▶ L2-normalised [`Embedding`]
//!                                                   └─▶ cosine [`Gallery`]
//! ```
//!
//! The resulting 128-d embedding plugs into the *same* zero-dep
//! [`crate::embedding::Gallery`] matcher as the ONNX ArcFace path —
//! [`EmbedNetRecognizer`] additionally adapts it to the grey-crop
//! [`crate::recognizer::FaceRecognizer`] trait shared by LBPH /
//! eigenfaces / Fisherfaces.
//!
//! ## Training
//!
//! Networks are trained with the classic **contrastive pair loss**
//! (Hadsell–Chopra–LeCun 2006) on unit-normalised embeddings:
//!
//! ```text
//! L = ½ y·D² + ½ (1−y)·max(0, m−D)²,   D = ‖ê_a − ê_b‖₂
//! ```
//!
//! so same-identity pairs (`y = 1`) collapse onto each other and
//! different-identity pairs (`y = 0`) are pushed apart to at least the
//! margin. See [`ContrastiveTrainer`]. The `embednet_train` binary trains
//! on a folder-of-folders (`root/<label>/*.{pgm,ppm,png}`) and writes the
//! `.rsen` weight file consumed by [`EmbedNet::load`].
//!
//! ## Honesty about accuracy
//!
//! What is verified here: the forward numerics, the **analytic gradients
//! against finite differences** (see the `gradcheck_*` tests), and that
//! training separates a synthetic multi-identity toy task. What is *not*
//! claimed: production face accuracy from random initial weights. A
//! metric network is only as good as its labelled data; the bundled
//! maturity level matches the CNN detector — genuinely trainable, weights
//! you train yourself. The ONNX ArcFace path remains the industrial
//! option; this module keeps the single-static-binary, zero-dep story for
//! users who own a labelled gallery and cannot ship a C++ runtime.
//!
//! ## Persistence
//!
//! `.rsen` is a tiny little-endian container: magic `RSEN`, u32 version,
//! then the ten weight tensors (`4× conv W`, `4× conv b`, FC W, FC b) as
//! u32-length-prefixed f32 runs. Shape mismatch on load is an error, so a
//! file from a different architecture is refused rather than reshaped.

use crate::embedding::{Embedding, Gallery, MatchConfig, MatchOutcome};
use crate::image::GrayImage;
use crate::recognizer::{FaceRecognizer, IncrementalRecognizer, Recognition};
use std::fs;
use std::io::{Read, Write};
use std::path::Path;

// ---- Architecture constants (single source of truth for save/load) --------

/// Network input edge length: crops are resized to `INPUT × INPUT`.
pub const INPUT: usize = 56;
/// Conv output channels for the four stages.
pub const CONV_CHANNELS: [usize; 4] = [24, 48, 64, 96];
/// Conv kernel edge (all four stages use 3×3 valid convolution).
pub const KERNEL: usize = 3;
/// Embedding dimensionality produced by the FC head.
pub const EMBED_DIM: usize = 128;

// Stage edge sizes: 56 → conv 54 → pool 27 → conv 25 → pool 12 →
// conv 10 → pool 5 → conv 3 (no final pool; 3×3×96 = 864 is flattened).
const STAGE_IN: [usize; 4] = [56, 27, 12, 5];
const STAGE_OUT: [usize; 4] = [54, 25, 10, 3];
const POOL_OUT: [usize; 3] = [27, 12, 5];
const FLAT_DIM: usize = 3 * 3 * CONV_CHANNELS[3]; // 864

const RSEN_MAGIC: &[u8; 4] = b"RSEN";
const RSEN_VERSION: u32 = 1;
const TENSOR_COUNT: usize = 10;

/// Default contrastive margin on the unit sphere (random 128-d vectors
/// start at D ≈ √2, so 1.0 is immediately learnable).
pub const DEFAULT_MARGIN: f32 = 1.0;
/// Adam learning rate used by the trainer binary.
pub const DEFAULT_LR: f32 = 2e-3;

// ---- Tiny deterministic RNG (mirrors cnn_train's xorshift) ----------------

/// Deterministic xorshift64* PRNG so trained weights are reproducible
/// from a seed and tests never depend on platform RNG support.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform f32 in `[0, 1)`.
    #[inline]
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    #[inline]
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.uniform()
    }

    /// Standard normal via Box–Muller.
    pub fn gaussian(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-7);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

// ---- Network weights -------------------------------------------------------

/// All trainable parameters of the EmbedNet architecture.
///
/// Conv weights are stored `[cout][k·k·cin]` (GEMM-friendly against the
/// im2col matrix); biases per output channel. The FC head is
/// `[EMBED_DIM][FLAT_DIM]`.
#[derive(Clone)]
pub struct EmbedNet {
    conv_w: [Vec<f32>; 4],
    conv_b: [Vec<f32>; 4],
    fc_w: Vec<f32>,
    fc_b: Vec<f32>,
}

fn conv_weight_len(stage: usize) -> usize {
    let cin = if stage == 0 {
        1
    } else {
        CONV_CHANNELS[stage - 1]
    };
    KERNEL * KERNEL * cin * CONV_CHANNELS[stage]
}

impl EmbedNet {
    /// Construct a He-initialised network (ReLU variant: `N(0, 2/fan_in)`).
    pub fn new(seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let mut conv_w = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut conv_b = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for s in 0..4 {
            let fan_in = KERNEL * KERNEL * if s == 0 { 1 } else { CONV_CHANNELS[s - 1] };
            let scale = (2.0 / fan_in as f32).sqrt();
            conv_w[s] = (0..conv_weight_len(s))
                .map(|_| rng.gaussian() * scale)
                .collect();
            conv_b[s] = vec![0.0; CONV_CHANNELS[s]];
        }
        let fc_scale = (2.0 / FLAT_DIM as f32).sqrt();
        let fc_w = (0..EMBED_DIM * FLAT_DIM)
            .map(|_| rng.gaussian() * fc_scale)
            .collect();
        let fc_b = vec![0.0; EMBED_DIM];
        Self {
            conv_w,
            conv_b,
            fc_w,
            fc_b,
        }
    }

    /// Embedding dimensionality (always [`EMBED_DIM`], exposed for parity
    /// with the ONNX recogniser's `dim()`).
    pub fn dim(&self) -> usize {
        EMBED_DIM
    }

    /// Resize a crop to the network input and standardise it per image.
    ///
    /// Per-image zero-mean/unit-stdev normalisation (rather than fixed
    /// `[0,1]` scaling) is what makes the descriptor robust to the
    /// lighting changes a zero-landmark detector hands us: contrast
    /// becomes a scale factor and brightness an offset, both divided out.
    pub fn preprocess(crop: &GrayImage) -> Vec<f32> {
        let resized = if crop.width() == INPUT && crop.height() == INPUT {
            crop.clone()
        } else {
            crop.resize_area(INPUT, INPUT)
        };
        let n = (INPUT * INPUT) as f32;
        let mut mean = 0.0f32;
        for &v in resized.as_slice() {
            mean += v as f32 / 255.0;
        }
        mean /= n;
        let mut var = 0.0f32;
        for &v in resized.as_slice() {
            let d = v as f32 / 255.0 - mean;
            var += d * d;
        }
        let inv_std = 1.0 / ((var / n).sqrt() + 1e-6);
        resized
            .as_slice()
            .iter()
            .map(|&v| (v as f32 / 255.0 - mean) * inv_std)
            .collect()
    }

    /// Raw (pre-normalisation) 128-d head output for a standardised patch.
    pub fn forward_raw(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), INPUT * INPUT);
        let mut cache = ForwardCache::new();
        self.forward_cached(x, &mut cache);
        cache.fc_out.clone()
    }

    /// Embed a grey crop: preprocess, forward, L2-normalise.
    ///
    /// Returns `None` when the head collapses to a zero/non-finite vector
    /// (same contract as [`Embedding::from_raw`]), so a degenerate crop
    /// can never enter a gallery as a vector that matches everything.
    pub fn embed(&self, crop: &GrayImage) -> Option<Embedding> {
        let x = Self::preprocess(crop);
        Embedding::from_raw(&self.forward_raw(&x))
    }

    /// Forward pass filling a reusable cache; used by both inference
    /// (`cache.fc_out` is the raw embedding) and training.
    fn forward_cached(&self, x: &[f32], c: &mut ForwardCache) {
        c.reset();
        let mut cur = x.to_vec();
        let mut h = INPUT;
        let mut w = INPUT;
        let mut cin = 1usize;
        for s in 0..4 {
            let oh = STAGE_OUT[s];
            let n = oh * oh;
            let kdim = KERNEL * KERNEL * cin;
            c.col[s].resize(kdim * n, 0.0);
            im2col(&cur, h, w, cin, KERNEL, &mut c.col[s]);
            c.pre_relu[s].resize(CONV_CHANNELS[s] * n, 0.0);
            gemm_conv_forward(
                &self.conv_w[s],
                &self.conv_b[s],
                &c.col[s],
                &mut c.pre_relu[s],
                CONV_CHANNELS[s],
                kdim,
                n,
            );
            // Post-ReLU HWC map is rebuilt on demand during backprop from
            // `pre_relu`, so the forward inference path just allocates the
            // next input here.
            let mut activated = vec![0.0f32; n * CONV_CHANNELS[s]];
            for n_i in 0..n {
                for co in 0..CONV_CHANNELS[s] {
                    let z = c.pre_relu[s][co * n + n_i];
                    activated[n_i * CONV_CHANNELS[s] + co] = z.max(0.0);
                }
            }
            if s < 3 {
                let p = POOL_OUT[s];
                cur = vec![0.0f32; p * p * CONV_CHANNELS[s]];
                maxpool_forward(&activated, oh, CONV_CHANNELS[s], &mut cur);
                h = p;
                w = p;
            } else {
                cur = activated;
            }
            cin = CONV_CHANNELS[s];
        }
        // cur = flattened 3×3×96 post-ReLU map in HWC order == row-major.
        debug_assert_eq!(cur.len(), FLAT_DIM);
        c.fc_out.clear();
        c.fc_out.resize(EMBED_DIM, 0.0);
        for j in 0..EMBED_DIM {
            let mut s = self.fc_b[j];
            let row = &self.fc_w[j * FLAT_DIM..(j + 1) * FLAT_DIM];
            for i in 0..FLAT_DIM {
                s += row[i] * cur[i];
            }
            c.fc_out[j] = s;
        }
        c.fc_in = cur;
    }

    /// Save weights to a `.rsen` file (little-endian; see module docs).
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let tmp = path.with_extension("rsen.tmp");
        {
            let mut f = fs::File::create(&tmp)?;
            f.write_all(RSEN_MAGIC)?;
            f.write_all(&RSEN_VERSION.to_le_bytes())?;
            f.write_all(&(TENSOR_COUNT as u32).to_le_bytes())?;
            for s in 0..4 {
                write_tensor(&mut f, &self.conv_w[s])?;
            }
            for s in 0..4 {
                write_tensor(&mut f, &self.conv_b[s])?;
            }
            write_tensor(&mut f, &self.fc_w)?;
            write_tensor(&mut f, &self.fc_b)?;
            f.flush()?;
        }
        fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Load weights from a `.rsen` file. A version/shape mismatch is an
    /// [`std::io::ErrorKind::InvalidData`] rather than a panic.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let mut f = fs::File::open(path)?;
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != RSEN_MAGIC {
            return Err(invalid("not an RSEN weights file"));
        }
        let version = read_u32(&mut f)?;
        if version != RSEN_VERSION {
            return Err(invalid(&format!(
                "unsupported RSEN version {version}, expected {RSEN_VERSION}"
            )));
        }
        let count = read_u32(&mut f)? as usize;
        if count != TENSOR_COUNT {
            return Err(invalid(&format!(
                "RSEN tensor count {count} does not match architecture ({TENSOR_COUNT})"
            )));
        }
        let mut conv_w = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut conv_b = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for s in 0..4 {
            conv_w[s] = read_tensor(&mut f, conv_weight_len(s))?;
        }
        for s in 0..4 {
            conv_b[s] = read_tensor(&mut f, CONV_CHANNELS[s])?;
        }
        let fc_w = read_tensor(&mut f, EMBED_DIM * FLAT_DIM)?;
        let fc_b = read_tensor(&mut f, EMBED_DIM)?;
        Ok(Self {
            conv_w,
            conv_b,
            fc_w,
            fc_b,
        })
    }
}

fn invalid(msg: &str) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg)
}

fn read_u32(r: &mut impl Read) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn write_tensor(w: &mut impl Write, data: &[f32]) -> std::io::Result<()> {
    w.write_all(&(data.len() as u32).to_le_bytes())?;
    for v in data {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

fn read_tensor(r: &mut impl Read, expected_len: usize) -> std::io::Result<Vec<f32>> {
    let len = read_u32(r)? as usize;
    if len != expected_len {
        return Err(invalid(&format!(
            "tensor length {len} does not match architecture ({expected_len})"
        )));
    }
    let mut out = vec![0.0f32; len];
    let mut raw = vec![0u8; len * 4];
    r.read_exact(&mut raw)?;
    for (i, chunk) in raw.as_chunks::<4>().0.iter().enumerate() {
        out[i] = f32::from_le_bytes(*chunk);
    }
    Ok(out)
}

// ---- im2col / GEMM primitives ----------------------------------------------

/// Build the im2col patch matrix for a HWC `(h,w,cin)` map.
///
/// `col` is laid out `[k·k·cin][oh·ow]`: each column is one valid-output
/// position's gathered neighbourhood. Convolution then becomes a single
/// GEMM, and transposed scatter-add (`col2im`) is its exact backward.
fn im2col(x: &[f32], h: usize, w: usize, cin: usize, k: usize, col: &mut [f32]) {
    let oh = h - k + 1;
    let ow = w - k + 1;
    let kdim = k * k * cin;
    debug_assert_eq!(col.len(), kdim * oh * ow);
    for n in 0..oh * ow {
        let oy = n / ow;
        let ox = n % ow;
        for ky in 0..k {
            for kx in 0..k {
                let row = (ky * k + kx) * cin;
                let src_base = ((oy + ky) * w + (ox + kx)) * cin;
                for ci in 0..cin {
                    col[(row + ci) * (oh * ow) + n] = x[src_base + ci];
                }
            }
        }
    }
}

/// Scatter-add transpose of [`im2col`].
fn col2im(col: &[f32], h: usize, w: usize, cin: usize, k: usize, dx: &mut [f32]) {
    let oh = h - k + 1;
    let ow = w - k + 1;
    for n in 0..oh * ow {
        let oy = n / ow;
        let ox = n % ow;
        for ky in 0..k {
            for kx in 0..k {
                let row = (ky * k + kx) * cin;
                let dst_base = ((oy + ky) * w + (ox + kx)) * cin;
                for ci in 0..cin {
                    dx[dst_base + ci] += col[(row + ci) * (oh * ow) + n];
                }
            }
        }
    }
}

/// `Z = W · Col + b`: `[cout,kdim] · [kdim,n] → [cout,n]`.
fn gemm_conv_forward(
    w: &[f32],
    b: &[f32],
    col: &[f32],
    z: &mut [f32],
    cout: usize,
    kdim: usize,
    n: usize,
) {
    debug_assert_eq!(w.len(), cout * kdim);
    debug_assert_eq!(z.len(), cout * n);
    for co in 0..cout {
        let wrow = &w[co * kdim..(co + 1) * kdim];
        let zrow = &mut z[co * n..(co + 1) * n];
        for j in 0..n {
            zrow[j] = b[co];
        }
        for kk in 0..kdim {
            let wk = wrow[kk];
            if wk == 0.0 {
                continue;
            }
            let crow = &col[kk * n..(kk + 1) * n];
            for j in 0..n {
                zrow[j] += wk * crow[j];
            }
        }
    }
}

/// 2×2 stride-2 max pool over a HWC map.
fn maxpool_forward(x: &[f32], edge: usize, c: usize, out: &mut [f32]) {
    let p = edge / 2;
    debug_assert_eq!(out.len(), p * p * c);
    for y in 0..p {
        for x_i in 0..p {
            for co in 0..c {
                let mut m = f32::NEG_INFINITY;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let v = x[((y * 2 + dy) * edge + (x_i * 2 + dx)) * c + co];
                        m = m.max(v);
                    }
                }
                out[(y * p + x_i) * c + co] = m;
            }
        }
    }
}

/// Backward of 2×2 max pool: route each pooled gradient to the argmax
/// input. The pre-pool map is rebuilt from the stored pre-ReLU tensor
/// (`max(0, z)`), so no separate activation cache is needed.
fn maxpool_backward(pre_relu: &[f32], pre_edge: usize, c: usize, dp: &[f32], dx: &mut [f32]) {
    let p = pre_edge / 2;
    let n = pre_edge * pre_edge;
    debug_assert_eq!(pre_relu.len(), c * n);
    dx.fill(0.0);
    for y in 0..p {
        for x_i in 0..p {
            for co in 0..c {
                let mut best = f32::NEG_INFINITY;
                let mut arg = 0usize;
                for dy in 0..2 {
                    for dx_i in 0..2 {
                        let yy = y * 2 + dy;
                        let xx = x_i * 2 + dx_i;
                        let z = pre_relu[co * n + yy * pre_edge + xx].max(0.0);
                        if z > best {
                            best = z;
                            arg = (yy * pre_edge + xx) * c + co;
                        }
                    }
                }
                dx[arg] += dp[(y * p + x_i) * c + co];
            }
        }
    }
}

/// Per-image forward activations retained for backprop.
struct ForwardCache {
    /// im2col matrices per stage: `[k·k·cin][out_positions]`.
    col: [Vec<f32>; 4],
    /// Pre-ReLU conv outputs per stage, channel-major: `[cout][positions]`.
    pre_relu: [Vec<f32>; 4],
    /// Flattened final post-ReLU map fed to the FC head.
    fc_in: Vec<f32>,
    /// Raw (unnormalised) embedding.
    fc_out: Vec<f32>,
}

impl ForwardCache {
    fn new() -> Self {
        Self {
            col: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            pre_relu: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            fc_in: Vec::new(),
            fc_out: Vec::new(),
        }
    }

    fn reset(&mut self) {
        self.fc_in.clear();
        self.fc_out.clear();
    }
}

// ---- Gradients + Adam ------------------------------------------------------

/// Gradient accumulators with the same layout as [`EmbedNet`].
#[derive(Clone)]
struct Grads {
    conv_w: [Vec<f32>; 4],
    conv_b: [Vec<f32>; 4],
    fc_w: Vec<f32>,
    fc_b: Vec<f32>,
}

impl Grads {
    fn zero_like(net: &EmbedNet) -> Self {
        Self {
            conv_w: [
                vec![0.0; net.conv_w[0].len()],
                vec![0.0; net.conv_w[1].len()],
                vec![0.0; net.conv_w[2].len()],
                vec![0.0; net.conv_w[3].len()],
            ],
            conv_b: [
                vec![0.0; net.conv_b[0].len()],
                vec![0.0; net.conv_b[1].len()],
                vec![0.0; net.conv_b[2].len()],
                vec![0.0; net.conv_b[3].len()],
            ],
            fc_w: vec![0.0; net.fc_w.len()],
            fc_b: vec![0.0; net.fc_b.len()],
        }
    }

    fn zero(&mut self) {
        for s in 0..4 {
            self.conv_w[s].fill(0.0);
            self.conv_b[s].fill(0.0);
        }
        self.fc_w.fill(0.0);
        self.fc_b.fill(0.0);
    }
}

/// Adam in the exact spirit of `cnn_train`'s optimizer, with the same
/// bias-correction and a ±5 per-element gradient clip for chain stability.
struct Adam {
    m: Vec<f32>,
    v: Vec<f32>,
    t: u32,
}

impl Adam {
    fn new(n: usize) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
        }
    }

    fn step(&mut self, params: &mut [f32], grads: &[f32], lr: f32) {
        const BETA1: f32 = 0.9;
        const BETA2: f32 = 0.999;
        const EPS: f32 = 1e-8;
        const CLIP: f32 = 5.0;
        self.t += 1;
        let bc1 = 1.0 - BETA1.powi(self.t as i32);
        let bc2 = 1.0 - BETA2.powi(self.t as i32);
        for i in 0..params.len() {
            let g = grads[i].clamp(-CLIP, CLIP);
            self.m[i] = BETA1 * self.m[i] + (1.0 - BETA1) * g;
            self.v[i] = BETA2 * self.v[i] + (1.0 - BETA2) * g * g;
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            params[i] -= lr * m_hat / (v_hat.sqrt() + EPS);
        }
    }
}

// ---- Contrastive trainer ---------------------------------------------------

/// Trains an [`EmbedNet`] on same/different identity pairs.
///
/// One [`ContrastiveTrainer::train_step`] runs both branches of a pair,
/// accumulates the shared-weight gradients from the contrastive loss, and
/// applies one Adam update. The trainer owns the network; finish with
/// [`Self::into_net`] (or grab it at any point with [`Self::net`]).
pub struct ContrastiveTrainer {
    net: EmbedNet,
    grads: Grads,
    /// One Adam state per parameter tensor, in update order.
    conv_w_adam: [Adam; 4],
    conv_b_adam: [Adam; 4],
    fc_w_adam: Adam,
    fc_b_adam: Adam,
    cache_a: ForwardCache,
    cache_b: ForwardCache,
    /// Learning rate.
    pub lr: f32,
    /// Contrastive margin `m`.
    pub margin: f32,
}

impl ContrastiveTrainer {
    pub fn new(seed: u64) -> Self {
        Self::with_hyperparams(seed, DEFAULT_LR, DEFAULT_MARGIN)
    }

    pub fn with_hyperparams(seed: u64, lr: f32, margin: f32) -> Self {
        let net = EmbedNet::new(seed);
        let grads = Grads::zero_like(&net);
        Self {
            conv_w_adam: [
                Adam::new(net.conv_w[0].len()),
                Adam::new(net.conv_w[1].len()),
                Adam::new(net.conv_w[2].len()),
                Adam::new(net.conv_w[3].len()),
            ],
            conv_b_adam: [
                Adam::new(net.conv_b[0].len()),
                Adam::new(net.conv_b[1].len()),
                Adam::new(net.conv_b[2].len()),
                Adam::new(net.conv_b[3].len()),
            ],
            fc_w_adam: Adam::new(net.fc_w.len()),
            fc_b_adam: Adam::new(net.fc_b.len()),
            net,
            grads,
            cache_a: ForwardCache::new(),
            cache_b: ForwardCache::new(),
            lr,
            margin,
        }
    }

    pub fn net(&self) -> &EmbedNet {
        &self.net
    }

    /// Current network by value (cloned); keeps the trainer usable for
    /// periodic validation snapshots.
    pub fn snapshot(&self) -> EmbedNet {
        self.net.clone()
    }

    pub fn into_net(self) -> EmbedNet {
        self.net
    }

    /// One forward/backward/update on a labelled pair. Both patches are
    /// standardised 56×56 f32 buffers (see [`EmbedNet::preprocess`]);
    /// `same` selects the attracting vs repelling term. Returns the
    /// scalar loss for logging.
    pub fn train_step(&mut self, a: &[f32], b: &[f32], same: bool) -> f32 {
        debug_assert_eq!(a.len(), INPUT * INPUT);
        debug_assert_eq!(b.len(), INPUT * INPUT);
        self.net.forward_cached(a, &mut self.cache_a);
        self.net.forward_cached(b, &mut self.cache_b);
        self.grads.zero();

        let loss = contrastive_backward(
            &self.net,
            &self.cache_a,
            &self.cache_b,
            same,
            self.margin,
            &mut self.grads,
        );

        for s in 0..4 {
            self.conv_w_adam[s].step(&mut self.net.conv_w[s], &self.grads.conv_w[s], self.lr);
            self.conv_b_adam[s].step(&mut self.net.conv_b[s], &self.grads.conv_b[s], self.lr);
        }
        self.fc_w_adam
            .step(&mut self.net.fc_w, &self.grads.fc_w, self.lr);
        self.fc_b_adam
            .step(&mut self.net.fc_b, &self.grads.fc_b, self.lr);
        loss
    }
}

/// Backprop both branches of a contrastive pair into one shared
/// [`Grads`]. Returns the loss value.
fn contrastive_backward(
    net: &EmbedNet,
    ca: &ForwardCache,
    cb: &ForwardCache,
    same: bool,
    margin: f32,
    grads: &mut Grads,
) -> f32 {
    // L2-normalise each branch's raw head output.
    let (ea, na) = normalize(&ca.fc_out);
    let (eb, nb) = normalize(&cb.fc_out);

    let mut diff = vec![0.0f32; EMBED_DIM];
    let mut d2 = 0.0f32;
    for j in 0..EMBED_DIM {
        diff[j] = ea[j] - eb[j];
        d2 += diff[j] * diff[j];
    }
    let d = d2.sqrt();

    let y = if same { 1.0 } else { 0.0 };
    // Hinge term only acts while D < margin.
    let hinge = (margin - d).max(0.0);
    let loss = 0.5 * y * d2 + 0.5 * (1.0 - y) * hinge * hinge;

    // dL/dD. Both squares carry a 0.5 whose factor cancels against the
    // chain-rule 2: d(½D²)/dD = D and d(½(m−D)²)/dD = D−m.
    let dd = if same {
        d
    } else if d < margin {
        d - margin
    } else {
        0.0
    };

    // dL/dê_a (and the negated term goes to ê_b). ∂D/∂ê_a = diff/D with
    // no extra 0.5 — the loss's 0.5 already cancelled above. Guard D≈0:
    // coincident embeddings contribute no direction.
    let mut dea = vec![0.0f32; EMBED_DIM];
    let mut deb = vec![0.0f32; EMBED_DIM];
    if d > 1e-6 {
        let scale = dd / d;
        for j in 0..EMBED_DIM {
            dea[j] = scale * diff[j];
            deb[j] = -scale * diff[j];
        }
    }

    // Through L2 normalisation: d_raw = (d_hat − (d_hat·ê)·ê) / ‖raw‖.
    let mut draw_a = vec![0.0f32; EMBED_DIM];
    let mut draw_b = vec![0.0f32; EMBED_DIM];
    denormalize_grad(&ea, na, &dea, &mut draw_a);
    denormalize_grad(&eb, nb, &deb, &mut draw_b);

    // Both branches share weights — gradients accumulate.
    backward_branch(net, ca, &draw_a, grads);
    backward_branch(net, cb, &draw_b, grads);
    loss
}

/// Normalise to unit length; returns `(normalised, raw_norm)`.
fn normalize(raw: &[f32]) -> (Vec<f32>, f32) {
    let norm = raw.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm < 1e-8 {
        return (raw.to_vec(), norm);
    }
    (raw.iter().map(|v| v / norm).collect(), norm)
}

/// Apply the transpose of L2 normalisation to a gradient.
fn denormalize_grad(unit: &[f32], norm: f32, d_unit: &[f32], d_raw: &mut [f32]) {
    if norm < 1e-8 {
        for v in d_raw.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    let dot: f32 = d_unit.iter().zip(unit).map(|(g, u)| g * u).sum();
    for j in 0..EMBED_DIM {
        d_raw[j] = (d_unit[j] - dot * unit[j]) / norm;
    }
}

/// Backprop one image branch: FC → (conv → pool)×3 → conv1.
fn backward_branch(net: &EmbedNet, c: &ForwardCache, d_emb_raw: &[f32], g: &mut Grads) {
    // ---- FC head: e = W·a4 + b --------------------------------------
    for j in 0..EMBED_DIM {
        let de = d_emb_raw[j];
        g.fc_b[j] += de;
        let wrow = j * FLAT_DIM;
        for i in 0..FLAT_DIM {
            g.fc_w[wrow + i] += de * c.fc_in[i];
        }
    }
    let mut da4 = vec![0.0f32; FLAT_DIM];
    for j in 0..EMBED_DIM {
        let de = d_emb_raw[j];
        let wrow = &net.fc_w[j * FLAT_DIM..(j + 1) * FLAT_DIM];
        for i in 0..FLAT_DIM {
            da4[i] += wrow[i] * de;
        }
    }
    // ReLU mask of the stage-4 conv output (fc_in is post-ReLU HWC;
    // pre_relu[3] is channel-major of the same 3×3 map).
    let n4 = STAGE_OUT[3] * STAGE_OUT[3];
    for n_i in 0..n4 {
        for co in 0..CONV_CHANNELS[3] {
            if c.pre_relu[3][co * n4 + n_i] <= 0.0 {
                da4[n_i * CONV_CHANNELS[3] + co] = 0.0;
            }
        }
    }

    let mut grad_hwc = da4;
    // Stages 3 → 0. Stages 0..2 sit behind a maxpool.
    for s in (0..4).rev() {
        let cin = if s == 0 { 1 } else { CONV_CHANNELS[s - 1] };
        let cout = CONV_CHANNELS[s];
        let oh = STAGE_OUT[s];
        let n = oh * oh;
        let kdim = KERNEL * KERNEL * cin;

        // Gradient w.r.t this stage's conv output map (post-ReLU HWC).
        let da = &grad_hwc;

        // 1) ReLU + gather into channel-major dz.
        let mut dz = vec![0.0f32; cout * n];
        for n_i in 0..n {
            for co in 0..cout {
                if c.pre_relu[s][co * n + n_i] > 0.0 {
                    dz[co * n + n_i] = da[n_i * cout + co];
                }
            }
        }

        // 2) db, dW via the im2col patches.
        for co in 0..cout {
            let mut bs = 0.0f32;
            let zrow = &dz[co * n..(co + 1) * n];
            for v in zrow {
                bs += v;
            }
            g.conv_b[s][co] += bs;
            let gw = &mut g.conv_w[s][co * kdim..(co + 1) * kdim];
            for kk in 0..kdim {
                let crow = &c.col[s][kk * n..(kk + 1) * n];
                let mut acc = 0.0f32;
                for j in 0..n {
                    acc += zrow[j] * crow[j];
                }
                gw[kk] += acc;
            }
        }

        // 3) dcol = Wᵀ · dz.
        let mut dcol = vec![0.0f32; kdim * n];
        for kk in 0..kdim {
            let drow = &mut dcol[kk * n..(kk + 1) * n];
            for co in 0..cout {
                let wk = net.conv_w[s][co * kdim + kk];
                let zrow = &dz[co * n..(co + 1) * n];
                for j in 0..n {
                    drow[j] += wk * zrow[j];
                }
            }
        }

        // 4) col2im → gradient of this stage's HWC input.
        let in_edge = STAGE_IN[s];
        let mut dconv_in = vec![0.0f32; in_edge * in_edge * cin];
        col2im(&dcol, in_edge, in_edge, cin, KERNEL, &mut dconv_in);

        // 5) Through the preceding maxpool (stages 0..2): scatter the
        //    conv-input gradient onto pool-input positions, which becomes
        //    the next outer stage's output gradient.
        if s > 0 {
            let pool_in_edge = STAGE_OUT[s - 1]; // conv output edge of stage s-1
            let mut dpool_in = vec![0.0f32; pool_in_edge * pool_in_edge * cin];
            maxpool_backward(
                &c.pre_relu[s - 1],
                pool_in_edge,
                cin,
                &dconv_in,
                &mut dpool_in,
            );
            grad_hwc = dpool_in;
        }
    }
}

// ---- FaceRecognizer adapter ------------------------------------------------

/// Accept policy for [`EmbedNetRecognizer`], expressed in **Euclidean
/// embedding distance** (smaller is better; 0 = identical, 2 = antipodal).
#[derive(Clone, Debug)]
pub struct EmbedNetConfig {
    /// Accept only probes closer than this (mapped to a cosine threshold
    /// internally). `1.0` ≈ cosine 0.5.
    pub distance_threshold: f32,
    /// Require the winner to beat the runner-up by this distance gap.
    pub min_distance_margin: f32,
}

impl Default for EmbedNetConfig {
    /// An intentionally neutral, **uncalibrated** policy. A freshly
    /// initialised network has no identity semantics — retune on your own
    /// trained model (the `embednet_train` binary reports pair-accuracy
    /// sweeps for exactly this purpose).
    fn default() -> Self {
        Self {
            distance_threshold: 1.0,
            min_distance_margin: 0.0,
        }
    }
}

/// [`EmbedNet`] plus an incremental embedding gallery, exposed through the
/// same [`FaceRecognizer`] trait as the classical recognisers.
///
/// Enrolment is incremental (only gallery vectors are appended — the
/// network itself is trained offline via [`ContrastiveTrainer`] /
/// `embednet_train`).
pub struct EmbedNetRecognizer {
    net: EmbedNet,
    gallery: Gallery,
    /// Distance form of the configured accept threshold (kept for
    /// diagnostics; the gallery stores the cosine form).
    distance_threshold: f32,
}

impl EmbedNetRecognizer {
    pub fn new(net: EmbedNet) -> Self {
        Self::with_config(net, EmbedNetConfig::default())
    }

    pub fn with_config(net: EmbedNet, config: EmbedNetConfig) -> Self {
        // Convert Euclidean distance on unit vectors to cosine:
        // D² = 2 − 2·cos ⇒ cos = 1 − D²/2. The runner-up margin converts
        // the same way: a distance gap ΔD corresponds to a cosine gap of
        // (2·D1 + ΔD)·ΔD/2, which the gallery applies in similarity space.
        let cos_threshold =
            (1.0 - config.distance_threshold * config.distance_threshold / 2.0).clamp(-1.0, 1.0);
        // Translate the distance margin at the operating point cos ≈
        // cos_threshold: Δcos ≈ D·ΔD for small gaps.
        let cos_margin = (config.distance_threshold * config.min_distance_margin).max(0.0);
        Self {
            net,
            gallery: Gallery::new(
                MatchConfig::default()
                    .with_threshold(cos_threshold)
                    .with_min_margin(cos_margin),
            ),
            distance_threshold: config.distance_threshold,
        }
    }

    /// Configured Euclidean accept threshold.
    pub fn distance_threshold(&self) -> f32 {
        self.distance_threshold
    }

    /// Embed a crop directly (bypasses the gallery; useful for callers that
    /// manage embeddings themselves).
    pub fn embed(&self, crop: &GrayImage) -> Option<Embedding> {
        self.net.embed(crop)
    }

    /// Enrol a crop; returns `false` when the embedding collapsed (blank
    /// crop) and nothing was enrolled.
    pub fn enroll_crop(&mut self, label: impl Into<String>, crop: &GrayImage) -> bool {
        match self.net.embed(crop) {
            Some(e) => {
                self.gallery.enroll(label, e);
                true
            }
            None => false,
        }
    }

    /// Remove an identity; `true` when it was present.
    pub fn remove_label(&mut self, label: &str) -> bool {
        self.gallery.remove(label)
    }

    /// Every identity with its embedding distance, ascending.
    pub fn rank_crop_embed(&self, crop: &GrayImage) -> Vec<(String, f32)> {
        let Some(e) = self.net.embed(crop) else {
            return Vec::new();
        };
        self.gallery
            .rank(&e)
            .into_iter()
            .map(|(l, cos)| (l, 2.0 - 2.0 * cos))
            .collect()
    }

    fn outcome_to_recognition(outcome: MatchOutcome) -> Recognition {
        match outcome {
            MatchOutcome::Match {
                label,
                similarity,
                margin,
            } => Recognition::Match {
                label,
                distance: 2.0 - 2.0 * similarity,
                // Distance gap: d2 − d1 = 2·(s1 − s2).
                margin: 2.0 * margin,
            },
            MatchOutcome::BelowThreshold { best } => Recognition::BelowThreshold {
                best: best.map(|(l, cos)| (l, 2.0 - 2.0 * cos)),
            },
            MatchOutcome::Ambiguous {
                first,
                second,
                margin,
            } => Recognition::Ambiguous {
                first,
                second,
                margin: 2.0 * margin,
            },
            MatchOutcome::NoCandidates => Recognition::NoCandidates,
        }
    }
}

impl FaceRecognizer for EmbedNetRecognizer {
    fn name(&self) -> &'static str {
        "embednet"
    }

    fn identify_crop(&self, crop: &GrayImage) -> Recognition {
        let Some(e) = self.net.embed(crop) else {
            return Recognition::NoCandidates;
        };
        Self::outcome_to_recognition(self.gallery.identify(&e))
    }

    fn rank_crop(&self, crop: &GrayImage) -> Vec<(String, f32)> {
        self.rank_crop_embed(crop)
    }

    fn verify(&self, label: &str, crop: &GrayImage) -> Option<f32> {
        let e = self.net.embed(crop)?;
        let id = self
            .gallery
            .identities()
            .iter()
            .find(|i| i.label == label)?;
        id.embeddings
            .iter()
            .filter_map(|g| e.cosine(g).map(|c| 2.0 - 2.0 * c))
            .fold(None, |acc: Option<f32>, d| {
                Some(acc.map_or(d, |a| a.min(d)))
            })
    }

    fn len(&self) -> usize {
        self.gallery.len()
    }

    fn crop_count(&self) -> usize {
        self.gallery.embedding_count()
    }
}

impl IncrementalRecognizer for EmbedNetRecognizer {
    fn enroll(&mut self, label: String, crop: &GrayImage) {
        // The trait contract is best-effort append; a collapsed embedding
        // is silently ignored (callers needing the signal use
        // [`EmbedNetRecognizer::enroll_crop`]).
        self.enroll_crop(label, crop);
    }

    fn remove(&mut self, label: &str) -> bool {
        self.remove_label(label)
    }
}

// ---- Synthetic identity toy data (tests + examples) ------------------------

/// A deterministic synthetic "face identity": a low-frequency intensity
/// field (plane waves + elliptical blobs) that stands in for a person's
/// stable appearance. Jittered *samples* of one identity mimic
/// pose/lighting/registration noise.
///
/// This is **not** a claim that real faces are sinusoids — it is a
/// controllable metric-learning benchmark: if contrastive training cannot
/// pull jittered copies of one field together and push distinct fields
/// apart, nothing about the trainer works. It is used by the unit tests
/// and `examples/recognise_embednet.rs`.
pub mod toy {
    use super::Rng;

    const WAVES: usize = 4;
    const BLOBS: usize = 3;

    /// One synthetic identity's prototype field.
    #[derive(Clone)]
    pub struct ToyIdentity {
        wave_angle: [f32; WAVES],
        wave_freq: [f32; WAVES],
        wave_phase: [f32; WAVES],
        wave_amp: [f32; WAVES],
        blob_cx: [f32; BLOBS],
        blob_cy: [f32; BLOBS],
        blob_rx: [f32; BLOBS],
        blob_ry: [f32; BLOBS],
        blob_amp: [f32; BLOBS],
    }

    /// Generate identity `seed` (different seed ⇒ different field).
    pub fn identity(seed: u64) -> ToyIdentity {
        let mut rng = Rng::new(seed.wrapping_add(0x9E37_79B9_7F4A_7C15));
        let mut t = ToyIdentity {
            wave_angle: [0.0; WAVES],
            wave_freq: [0.0; WAVES],
            wave_phase: [0.0; WAVES],
            wave_amp: [0.0; WAVES],
            blob_cx: [0.0; BLOBS],
            blob_cy: [0.0; BLOBS],
            blob_rx: [0.0; BLOBS],
            blob_ry: [0.0; BLOBS],
            blob_amp: [0.0; BLOBS],
        };
        for i in 0..WAVES {
            t.wave_angle[i] = rng.range(0.0, 2.0 * std::f32::consts::PI);
            t.wave_freq[i] = rng.range(0.05, 0.22);
            t.wave_phase[i] = rng.range(0.0, 2.0 * std::f32::consts::PI);
            t.wave_amp[i] = rng.range(0.04, 0.14);
        }
        for i in 0..BLOBS {
            t.blob_cx[i] = rng.range(14.0, 42.0);
            t.blob_cy[i] = rng.range(14.0, 42.0);
            t.blob_rx[i] = rng.range(4.0, 11.0);
            t.blob_ry[i] = rng.range(5.0, 13.0);
            t.blob_amp[i] = rng.range(-0.22, 0.22);
        }
        t
    }

    impl ToyIdentity {
        /// Field value at continuous coordinates (used so translation
        /// jitter can be sub-pixel).
        fn field(&self, x: f32, y: f32) -> f32 {
            let cx = super::INPUT as f32 / 2.0;
            let cy = cx;
            let mut v = 0.5f32;
            for i in 0..WAVES {
                let dx = x - cx;
                let dy = y - cy;
                let p = dx * self.wave_angle[i].cos() + dy * self.wave_angle[i].sin();
                v += self.wave_amp[i] * (p * self.wave_freq[i] + self.wave_phase[i]).cos();
            }
            for i in 0..BLOBS {
                let nx = (x - self.blob_cx[i]) / self.blob_rx[i];
                let ny = (y - self.blob_cy[i]) / self.blob_ry[i];
                v += self.blob_amp[i] * (-(nx * nx + ny * ny)).exp();
            }
            v
        }

        /// One jittered 56×56 sample: sub-pixel translation, brightness
        /// offset, contrast scale, pixel noise.
        pub fn sample(&self, rng: &mut Rng) -> Vec<f32> {
            let shift_x = rng.range(-2.0, 2.0);
            let shift_y = rng.range(-2.0, 2.0);
            let brightness = rng.range(-0.08, 0.08);
            let contrast = rng.range(0.85, 1.15);
            let noise_sigma = 0.02;
            let edge = super::INPUT;
            let mut out = vec![0.0f32; edge * edge];
            for y in 0..edge {
                for x in 0..edge {
                    let v = self.field(x as f32 - shift_x, y as f32 - shift_y);
                    let v = (v - 0.5) * contrast + 0.5 + brightness + rng.gaussian() * noise_sigma;
                    out[y * edge + x] = v.clamp(0.0, 1.0);
                }
            }
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::toy::*;
    use super::*;

    /// Pair distance between two standardised patches under a network.
    fn pair_distance(net: &EmbedNet, a: &[f32], b: &[f32]) -> f32 {
        let ea = Embedding::from_raw(&net.forward_raw(a)).unwrap();
        let eb = Embedding::from_raw(&net.forward_raw(b)).unwrap();
        ea.sq_euclidean(&eb).unwrap().sqrt()
    }

    #[test]
    fn embedding_is_unit_length_and_deterministic() {
        let net = EmbedNet::new(1);
        let id = identity(7);
        let mut rng = Rng::new(100);
        let patch = id.sample(&mut rng);
        let e1 = net.forward_raw(&patch);
        let e2 = net.forward_raw(&patch);
        assert_eq!(e1, e2);
        // embed() L2-normalises. A structured toy crop is used because a
        // perfectly uniform crop collapses to zero activations (and that
        // collapse is correctly rejected by Embedding::from_raw).
        let mut g = GrayImage::new(INPUT, INPUT);
        for (i, v) in patch.iter().enumerate() {
            g.as_mut_slice()[i] = (v * 255.0).clamp(0.0, 255.0) as u8;
        }
        let emb = net.embed(&g).expect("structured crop must embed");
        let n: f32 = emb.as_slice().iter().map(|v| v * v).sum();
        assert!((n - 1.0).abs() < 1e-5, "unit norm = {n}");
    }

    #[test]
    fn blank_crop_collapses_to_none() {
        // All-uniform input standardises to zero → zero head output → the
        // type-system guard keeps a zero vector out of every gallery.
        let net = EmbedNet::new(1);
        assert!(net.embed(&GrayImage::new(INPUT, INPUT)).is_none());
    }

    #[test]
    fn save_load_roundtrips_bit_exact() {
        let net = EmbedNet::new(42);
        let dir = std::env::temp_dir();
        let path = dir.join(format!("rsface_embednet_test_{}.rsen", std::process::id()));
        net.save(&path).unwrap();
        let loaded = EmbedNet::load(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        for s in 0..4 {
            assert_eq!(net.conv_w[s], loaded.conv_w[s]);
            assert_eq!(net.conv_b[s], loaded.conv_b[s]);
        }
        assert_eq!(net.fc_w, loaded.fc_w);
        assert_eq!(net.fc_b, loaded.fc_b);
    }

    #[test]
    fn load_rejects_garbage_and_wrong_shape() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("rsface_embednet_bad_{}.rsen", std::process::id()));
        std::fs::write(&path, b"not a model file at all").unwrap();
        assert!(EmbedNet::load(&path).is_err());
        std::fs::remove_file(&path).unwrap();
    }

    /// Analytic vs central-difference gradients for one weight in each
    /// tensor type. This is the proof the whole backward pass is correct.
    #[test]
    fn gradcheck_matches_finite_differences() {
        let mut trainer = ContrastiveTrainer::with_hyperparams(123, 1e-3, DEFAULT_MARGIN);
        let mut rng = Rng::new(999);
        let a: Vec<f32> = (0..INPUT * INPUT).map(|_| rng.range(-1.5, 1.5)).collect();
        let b: Vec<f32> = (0..INPUT * INPUT).map(|_| rng.range(-1.5, 1.5)).collect();

        // Run one step to populate analytic grads (discard its Adam update
        // effects by computing analytic grads from a fresh copy instead).
        let net0 = trainer.snapshot();
        let mut ca = ForwardCache::new();
        let mut cb = ForwardCache::new();
        net0.forward_cached(&a, &mut ca);
        net0.forward_cached(&b, &mut cb);
        let mut grads = Grads::zero_like(&net0);
        let loss0 = contrastive_backward(&net0, &ca, &cb, false, DEFAULT_MARGIN, &mut grads);
        assert!(loss0.is_finite());

        // (tensor selector, index) spots across every layer.
        let eps = 1e-2f32;
        let mut check = |name: &str,
                         get: &dyn Fn(&EmbedNet) -> f32,
                         set: &dyn Fn(&mut EmbedNet, f32),
                         analytic: f32| {
            let mut np = net0.clone();
            set(&mut np, get(&net0) + eps);
            let lp = scalar_loss(&np, &a, &b, false);
            let mut nm = net0.clone();
            set(&mut nm, get(&net0) - eps);
            let lm = scalar_loss(&nm, &a, &b, false);
            let numeric = (lp - lm) / (2.0 * eps);
            let tol = 2e-2 * numeric.abs().max(analytic.abs()) + 2e-3;
            assert!(
                (numeric - analytic).abs() <= tol,
                "{name}: analytic={analytic:.6} numeric={numeric:.6} diff={:.6} tol={tol:.6}",
                (numeric - analytic).abs()
            );
        };

        // conv1 weight + bias
        let idx = 3 * 9 + 4;
        check(
            "conv1_w",
            &|n| n.conv_w[0][idx],
            &|n, v| n.conv_w[0][idx] = v,
            grads.conv_w[0][idx],
        );
        check(
            "conv1_b",
            &|n| n.conv_b[0][7],
            &|n, v| n.conv_b[0][7] = v,
            grads.conv_b[0][7],
        );
        // conv2
        let idx2 = 11 * 216 + 40;
        check(
            "conv2_w",
            &|n| n.conv_w[1][idx2],
            &|n, v| n.conv_w[1][idx2] = v,
            grads.conv_w[1][idx2],
        );
        // conv3 + conv4
        let idx3 = 30 * 432 + 7;
        check(
            "conv3_w",
            &|n| n.conv_w[2][idx3],
            &|n, v| n.conv_w[2][idx3] = v,
            grads.conv_w[2][idx3],
        );
        let idx4 = 2 * 576 + 500;
        check(
            "conv4_w",
            &|n| n.conv_w[3][idx4],
            &|n, v| n.conv_w[3][idx4] = v,
            grads.conv_w[3][idx4],
        );
        // FC weight + bias
        let fi = 50 * FLAT_DIM + 100;
        check(
            "fc_w",
            &|n| n.fc_w[fi],
            &|n, v| n.fc_w[fi] = v,
            grads.fc_w[fi],
        );
        check(
            "fc_b",
            &|n| n.fc_b[64],
            &|n, v| n.fc_b[64] = v,
            grads.fc_b[64],
        );

        // Also gradcheck a SAME pair (attracting branch).
        let mut grads_same = Grads::zero_like(&net0);
        let mut cas = ForwardCache::new();
        let mut cbs = ForwardCache::new();
        net0.forward_cached(&a, &mut cas);
        net0.forward_cached(&b, &mut cbs);
        contrastive_backward(&net0, &cas, &cbs, true, DEFAULT_MARGIN, &mut grads_same);
        let idx = 3 * 9 + 4;
        let analytic = grads_same.conv_w[0][idx];
        let mut np = net0.clone();
        np.conv_w[0][idx] += eps;
        let lp = scalar_loss(&np, &a, &b, true);
        let mut nm = net0.clone();
        nm.conv_w[0][idx] -= eps;
        let lm = scalar_loss(&nm, &a, &b, true);
        let numeric = (lp - lm) / (2.0 * eps);
        let tol = 2e-2 * numeric.abs().max(analytic.abs()) + 2e-3;
        assert!(
            (numeric - analytic).abs() <= tol,
            "same-pair conv1_w: analytic={analytic:.6} numeric={numeric:.6}"
        );
        let _ = trainer; // constructed but unused beyond this test
    }

    /// Recompute only the scalar loss for finite differencing.
    fn scalar_loss(net: &EmbedNet, a: &[f32], b: &[f32], same: bool) -> f32 {
        let mut ca = ForwardCache::new();
        let mut cb = ForwardCache::new();
        net.forward_cached(a, &mut ca);
        net.forward_cached(b, &mut cb);
        let (ea, _) = normalize(&ca.fc_out);
        let (eb, _) = normalize(&cb.fc_out);
        let d2: f32 = ea.iter().zip(eb.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        let d = d2.sqrt();
        let y = if same { 1.0 } else { 0.0 };
        let hinge = (DEFAULT_MARGIN - d).max(0.0);
        0.5 * y * d2 + 0.5 * (1.0 - y) * hinge * hinge
    }

    #[test]
    fn contrastive_training_separates_toy_identities() {
        const IDENTITIES: u64 = 5;
        const STEPS: usize = 400;
        let ids: Vec<ToyIdentity> = (0..IDENTITIES).map(identity).collect();
        let mut trainer = ContrastiveTrainer::new(2026);
        let mut rng = Rng::new(777);

        for step in 0..STEPS {
            let i = (step as u64) % IDENTITIES;
            let same = step % 2 == 0;
            let a = ids[i as usize].sample(&mut rng);
            let b = if same {
                ids[i as usize].sample(&mut rng)
            } else {
                let mut j = (rng.next_u64() % IDENTITIES) as usize;
                if j == i as usize {
                    j = (j + 1) % IDENTITIES as usize;
                }
                ids[j].sample(&mut rng)
            };
            trainer.train_step(&a, &b, same);
        }
        let net = trainer.into_net();

        // Held-out evaluation on freshly jittered samples.
        let mut eval_rng = Rng::new(0xABCD);
        let samples: Vec<Vec<f32>> = ids.iter().map(|id| id.sample(&mut eval_rng)).collect();
        let threshold = DEFAULT_MARGIN * 0.8;
        let (mut correct, mut total) = (0u32, 0u32);
        let (mut same_sum, mut same_n) = (0.0f32, 0u32);
        let (mut diff_sum, mut diff_n) = (0.0f32, 0u32);
        for i in 0..IDENTITIES as usize {
            for j in 0..IDENTITIES as usize {
                let d = pair_distance(&net, &samples[i], &samples[j]);
                let same = i == j;
                let predicted_same = d < threshold;
                if predicted_same == same {
                    correct += 1;
                }
                total += 1;
                if same {
                    same_sum += d;
                    same_n += 1;
                } else {
                    diff_sum += d;
                    diff_n += 1;
                }
            }
        }
        let acc = correct as f32 / total as f32;
        let mean_same = same_sum / same_n as f32;
        let mean_diff = diff_sum / diff_n as f32;
        assert!(
            acc >= 0.9,
            "toy pair accuracy {acc:.3} < 0.90 (same D={mean_same:.3}, diff D={mean_diff:.3})"
        );
        assert!(
            mean_diff - mean_same > 0.2,
            "weak separation: same={mean_same:.3} diff={mean_diff:.3}"
        );
    }

    #[test]
    fn recognizer_adapter_enrols_identifies_and_removes() {
        let net = EmbedNet::new(5);
        let mut rec = EmbedNetRecognizer::new(net);
        assert_eq!(rec.name(), "embednet");
        assert!(matches!(
            rec.identify_crop(&GrayImage::new(INPUT, INPUT)),
            Recognition::NoCandidates
        ));

        let id = identity(3);
        let mut rng = Rng::new(11);
        let to_crop = |patch: &[f32]| -> GrayImage {
            let mut g = GrayImage::new(INPUT, INPUT);
            for (i, v) in patch.iter().enumerate() {
                g.as_mut_slice()[i] = (v * 255.0).clamp(0.0, 255.0) as u8;
            }
            g
        };
        for _ in 0..3 {
            rec.enroll("p3".to_string(), &to_crop(&id.sample(&mut rng)));
        }
        assert_eq!(rec.len(), 1);
        assert_eq!(rec.crop_count(), 3);
        assert!(rec.verify("p3", &to_crop(&id.sample(&mut rng))).is_some());
        assert!(rec
            .verify("ghost", &to_crop(&id.sample(&mut rng)))
            .is_none());
        assert!(!rec.rank_crop(&to_crop(&id.sample(&mut rng))).is_empty());
        assert!(rec.remove("p3"));
        assert!(rec.is_empty());
        assert!(!rec.remove("p3"));
    }
}
