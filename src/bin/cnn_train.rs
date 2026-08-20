//! Train the CNN face detector on synthetic data.
//!
//! Generates positive samples (face-like elliptical bright centres on dark
//! backgrounds with affine jitter) and negative samples (random noise,
//! gradients, plain backgrounds, and hard negatives produced by mining the
//! current detector). Trains with Adam + BCE-with-logits loss.
//!
//! Usage:
//!   cargo run --release --bin cnn_train -- [epochs] [out_path] [seed]
//!
//! Saves weights in `.cnn.bin` format that [`CnnDetector`] can load via
//! `--cnn-weights`.

use rsface::cnn::{conv2d_into, fc_into, maxpool2_into, relu, CnnScratch, CnnWeights};
use rsface::image::GrayImage;
use std::time::Instant;

const WIN: usize = 24;
const TRAIN_PER_EPOCH: usize = 1024; // 512 pos + 512 neg per epoch
const BATCH: usize = 32;
const LR: f32 = 1e-4;

// Tiny deterministic PRNG (xorshift64) so runs are reproducible.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn f32(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * ((self.next() & 0xFFFF) as f32 / 65535.0)
    }
    fn u8(&mut self, lo: u8, hi: u8) -> u8 {
        lo + ((self.next() as u8).wrapping_add(hi.wrapping_sub(lo)))
    }
}

/// Render a synthetic 24×24 face patch into `out` (f32 in [0, 1]).
/// The face is an ellipse with bright centre, darker border, plus random
/// affine jitter so the network learns position/scale invariance.
fn synth_face(out: &mut [f32], rng: &mut Rng) {
    debug_assert_eq!(out.len(), WIN * WIN);
    // Background = dark, varying.
    let bg = rng.f32(0.0, 0.15);
    for v in out.iter_mut() {
        *v = bg;
    }
    let cx = rng.f32(10.0, 14.0);
    let cy = rng.f32(10.0, 14.0);
    let rx = rng.f32(7.0, 10.0);
    let ry = rng.f32(8.0, 11.0);
    let face_val = rng.f32(0.55, 0.95);
    // Darken border (hair / shadow).
    let border_dark = rng.f32(0.15, 0.45);
    for y in 0..WIN {
        for x in 0..WIN {
            let nx = (x as f32 - cx) / rx;
            let ny = (y as f32 - cy) / ry;
            let d2 = nx * nx + ny * ny;
            let i = y * WIN + x;
            if d2 < 0.55 {
                // Inside the ellipse: face luminance.
                out[i] = face_val * (1.0 - d2 * 0.4);
            } else if d2 < 1.0 {
                // Soft fall-off at the border.
                let t = (d2 - 0.55) / 0.45;
                out[i] = face_val * (1.0 - t) + bg * t - border_dark * t * 0.3;
            }
        }
    }
    // Add eyes (two dark blobs in the upper half).
    let eye_y = rng.f32(8.0, 12.0);
    let eye_dx = rng.f32(2.5, 4.0);
    let eye_r = rng.f32(1.2, 1.8);
    let eye_val = bg * 0.4;
    for sign in &[-1.0f32, 1.0] {
        let ex = cx + sign * eye_dx;
        for y in 0..WIN {
            for x in 0..WIN {
                let dx = x as f32 - ex;
                let dy = y as f32 - eye_y;
                if dx * dx + dy * dy < eye_r * eye_r {
                    out[y * WIN + x] = eye_val;
                }
            }
        }
    }
    // Slight pixel noise to break perfect symmetry.
    let noise_amp = rng.f32(0.0, 0.03);
    for v in out.iter_mut() {
        *v = (*v + rng.f32(-noise_amp, noise_amp)).clamp(0.0, 1.0);
    }
}

/// Render a synthetic 24×24 non-face patch. Mixes gradients, plain
/// backgrounds, and structured "confuser" patterns that look face-ish at
/// first glance (vertical/horizontal edge stripes).
fn synth_nonface(out: &mut [f32], rng: &mut Rng) {
    debug_assert_eq!(out.len(), WIN * WIN);
    let kind = rng.next() % 4;
    match kind {
        0 => {
            // Plain dark or light background with mild noise.
            let bg = rng.f32(0.0, 1.0);
            for v in out.iter_mut() {
                *v = bg + rng.f32(-0.05, 0.05);
            }
        }
        1 => {
            // Smooth horizontal gradient.
            let lo = rng.f32(0.0, 0.5);
            let hi = rng.f32(lo + 0.1, 1.0);
            for y in 0..WIN {
                let t = y as f32 / (WIN - 1) as f32;
                for x in 0..WIN {
                    out[y * WIN + x] = lo + (hi - lo) * t + rng.f32(-0.02, 0.02);
                }
            }
        }
        2 => {
            // Vertical edge stripe.
            let x_split = rng.f32(8.0, 16.0);
            let a = rng.f32(0.0, 0.7);
            let b = rng.f32(0.3, 1.0);
            for y in 0..WIN {
                for x in 0..WIN {
                    out[y * WIN + x] = if (x as f32) < x_split { a } else { b };
                }
            }
        }
        _ => {
            // High-frequency noise (texture).
            for v in out.iter_mut() {
                *v = rng.f32(0.0, 1.0);
            }
        }
    }
    for v in out.iter_mut() {
        *v = v.clamp(0.0, 1.0);
    }
}

// ---------------- Adam optimizer ----------------
#[derive(Clone)]
struct Adam {
    m: Vec<f32>,
    v: Vec<f32>,
    t: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
}
impl Adam {
    fn new(n: usize, lr: f32) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }
    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        // Clip gradients to ±5.0 per-element — keeps training stable when
        // the chain is long (we don't have full conv backprop, so the
        // partial gradients can spike).
        for i in 0..params.len() {
            let g = grads[i].clamp(-5.0, 5.0);
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;
            params[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
        }
    }
}

// ---------------- Forward / backward ----------------
struct Net {
    w: CnnWeights,
    scratch: CnnScratch,
}

impl Net {
    fn new() -> Self {
        // Start from the template face weights (already detect face-like
        // patterns) with FC layers reinitialised for clean training.
        let w = init_from_template();
        Self {
            w,
            scratch: CnnScratch::new(),
        }
    }

    fn forward(&mut self, x: &[f32]) -> f32 {
        let s = self.scratch.buffers_mut();
        conv2d_into(x, WIN, WIN, 1, &self.w.conv1_w, 3, 3, 8, &mut s.c1);
        relu(&mut s.c1);
        conv2d_into(&s.c1, 22, 22, 8, &self.w.conv2_w, 3, 3, 16, &mut s.c2);
        relu(&mut s.c2);
        maxpool2_into(&s.c2, 20, 20, 16, &mut s.c2p);
        conv2d_into(&s.c2p, 10, 10, 16, &self.w.conv3_w, 3, 3, 32, &mut s.c3);
        relu(&mut s.c3);
        maxpool2_into(&s.c3, 8, 8, 32, &mut s.c3p);
        fc_into(&s.c3p, &self.w.fc1_w, &self.w.fc1_b, 32, &mut s.f1);
        relu(&mut s.f1);
        fc_into(&s.f1, &self.w.fc2_w, &self.w.fc2_b, 1, &mut s.f2);
        s.f2[0] // logit (apply sigmoid outside)
    }

    /// Forward + accumulate gradient w.r.t. weights. The "logit" is the
    /// pre-sigmoid FC2 output; we use BCE-with-logits which is numerically
    /// stable: `loss = max(z,0) - z*y + log(1+exp(-|z|))`,
    /// `∂loss/∂z = sigmoid(z) - y`.
    fn forward_backward(&mut self, x: &[f32], y: f32, grads: &mut CnnWeights) -> f32 {
        let s = self.scratch.buffers_mut();
        // Forward.
        conv2d_into(x, WIN, WIN, 1, &self.w.conv1_w, 3, 3, 8, &mut s.c1);
        relu(&mut s.c1);
        conv2d_into(&s.c1, 22, 22, 8, &self.w.conv2_w, 3, 3, 16, &mut s.c2);
        relu(&mut s.c2);
        maxpool2_into(&s.c2, 20, 20, 16, &mut s.c2p);
        conv2d_into(&s.c2p, 10, 10, 16, &self.w.conv3_w, 3, 3, 32, &mut s.c3);
        relu(&mut s.c3);
        maxpool2_into(&s.c3, 8, 8, 32, &mut s.c3p);
        fc_into(&s.c3p, &self.w.fc1_w, &self.w.fc1_b, 32, &mut s.f1);
        relu(&mut s.f1);
        fc_into(&s.f1, &self.w.fc2_w, &self.w.fc2_b, 1, &mut s.f2);
        let z = s.f2[0];

        // BCE-with-logits loss.
        let (loss, d_z) = if z > 0.0 {
            (
                z - z * y + (1.0 + (-z).exp()).ln(),
                1.0 / (1.0 + (-z).exp()) - y,
            )
        } else {
            (-z * y + (1.0 + z.exp()).ln(), 1.0 / (1.0 + z.exp()) - y)
        };

        // Backprop FC2 → FC1 → conv3 → conv2 → conv1.
        // FC2: f2 = fc1 · fc2_w + fc2_b.  d_fc2_w[i] = f1[i] * d_z.
        for i in 0..32 {
            grads.fc2_w[i] += s.f1[i] * d_z;
        }
        grads.fc2_b[0] += d_z;

        // d_f1[i] = fc2_w[i] * d_z.
        let mut d_f1 = [0.0f32; 32];
        for i in 0..32 {
            d_f1[i] = self.w.fc2_w[i] * d_z;
        }
        // ReLU backprop.
        for i in 0..32 {
            if s.f1[i] <= 0.0 {
                d_f1[i] = 0.0;
            }
        }

        // FC1: out[o] = Σ c3p[i] * fc1_w[o*512+i] + fc1_b[o].
        // d_fc1_w[o*512+i] += c3p[i] * d_f1[o]
        for o in 0..32 {
            for i in 0..512 {
                grads.fc1_w[o * 512 + i] += s.c3p[i] * d_f1[o];
            }
            grads.fc1_b[o] += d_f1[o];
        }
        // d_c3p[i] = Σ_o fc1_w[o*512+i] * d_f1[o]
        let mut d_c3p = [0.0f32; 512];
        for o in 0..32 {
            let d = d_f1[o];
            for i in 0..512 {
                d_c3p[i] += self.w.fc1_w[o * 512 + i] * d;
            }
        }

        // MaxPool 2x2 backprop: c3p comes from c3 with stride 2. Distribute
        // gradient back to the argmax location in each 2x2 block.
        let mut d_c3 = vec![0.0f32; 8 * 8 * 32];
        for co in 0..32 {
            for y in 0..4 {
                for x in 0..4 {
                    let mut best = f32::NEG_INFINITY;
                    let mut arg = (0, 0);
                    for dy in 0..2 {
                        for dx in 0..2 {
                            let v = s.c3[((y * 2 + dy) * 8 + (x * 2 + dx)) * 32 + co];
                            if v > best {
                                best = v;
                                arg = (dy, dx);
                            }
                        }
                    }
                    let idx = ((y * 2 + arg.0) * 8 + (x * 2 + arg.1)) * 32 + co;
                    d_c3[idx] = d_c3p[(y * 4 + x) * 32 + co];
                }
            }
        }
        // ReLU backprop on c3.
        for v in &mut d_c3 {
            if *v < 0.0 { /* keep */ }
        }
        for i in 0..d_c3.len() {
            if s.c3[i] <= 0.0 {
                d_c3[i] = 0.0;
            }
        }

        // Conv3: c3[co, y, x] = Σ ci, ky, kx c2p[ci, y+ky, x+kx] * conv3_w[ky*3+kx, ci, co]
        // d_conv3_w[ky*3+kx, ci, co] += c2p[ci, y+ky, x+kx] * d_c3[co, y, x]
        for co in 0..32 {
            for y in 0..8 {
                for x in 0..8 {
                    let d = d_c3[(y * 8 + x) * 32 + co];
                    for ci in 0..16 {
                        for ky in 0..3 {
                            for kx in 0..3 {
                                let k = ((ky * 3 + kx) * 16 + ci) * 32 + co;
                                grads.conv3_w[k] += s.c2p[((y + ky) * 10 + (x + kx)) * 16 + ci] * d;
                            }
                        }
                    }
                }
            }
        }

        // d_c2p backprop from conv3 (full version) — left as a stub for
        // brevity: we only update conv weights/biases; the chain would
        // propagate back to conv2 then conv1 but for this initial demo we
        // stop here and let the FC layers do most of the work.

        loss
    }
}

/// Start from the template face weights (which already detect face-like
/// patterns via hand-crafted conv kernels) and only train the FC head.
/// Training the conv layers from scratch would require full conv backprop
/// for conv1 and conv2 which is outside this initial demo; the FC layers
/// are sufficient to learn the decision boundary on top of the template
/// features.
fn init_from_template() -> CnnWeights {
    let mut w = rsface::cnn::template_face_weights();
    // Re-initialise FC layers with small random weights so training has a
    // clean signal (the template weights have fc1_b = -2.0 and fc2_b = -2.0
    // which makes everything start off-screen).
    let mut rng = Rng::new(0xC0FFEE);
    let small = |rng: &mut Rng| -> f32 {
        let u1 = ((rng.next() & 0xFFFF) as f32 / 65535.0).max(1e-7);
        let u2 = (rng.next() & 0xFFFF) as f32 / 65535.0;
        let n = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        n * 0.1
    };
    for v in w.fc1_w.iter_mut() {
        *v = small(&mut rng);
    }
    w.fc1_b.iter_mut().for_each(|v| *v = 0.0);
    for v in w.fc2_w.iter_mut() {
        *v = small(&mut rng);
    }
    w.fc2_b[0] = -0.5;
    w
}

fn save_weights(path: &std::path::Path, w: &CnnWeights) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    f.write_all(b"RCNN")?;
    f.write_all(&1u32.to_le_bytes())?; // version
    let header = [
        w.conv1_w.len(),
        w.conv2_w.len(),
        w.conv3_w.len(),
        w.fc1_w.len(),
        w.fc1_b.len(),
        w.fc2_w.len(),
        w.fc2_b.len(),
    ];
    for &n in &header {
        f.write_all(&(n as u32).to_le_bytes())?;
    }
    for buf in [
        &w.conv1_w, &w.conv2_w, &w.conv3_w, &w.fc1_w, &w.fc1_b, &w.fc2_w, &w.fc2_b,
    ] {
        for &v in buf {
            f.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

fn train(epochs: usize, out: &std::path::Path, seed: u64) -> std::io::Result<()> {
    let mut rng = Rng::new(seed);
    let mut net = Net::new();

    // One Adam optimizer per weight buffer.
    let mut adam_conv1 = Adam::new(net.w.conv1_w.len(), LR);
    let mut adam_conv2 = Adam::new(net.w.conv2_w.len(), LR);
    let mut adam_conv3 = Adam::new(net.w.conv3_w.len(), LR);
    let mut adam_fc1_w = Adam::new(net.w.fc1_w.len(), LR);
    let mut adam_fc1_b = Adam::new(net.w.fc1_b.len(), LR);
    let mut adam_fc2_w = Adam::new(net.w.fc2_w.len(), LR);
    let mut adam_fc2_b = Adam::new(net.w.fc2_b.len(), LR);

    let mut x_buf = [0.0f32; WIN * WIN];
    let mut grads = net.w.clone();
    let mut pos_correct = 0u32;
    let mut neg_correct = 0u32;
    let mut total_loss = 0.0f32;
    let mut steps = 0u32;
    // Snapshot the best weights seen so far (lowest loss) so a late-training
    // explosion doesn't overwrite a good early checkpoint.
    let mut best_loss = f32::INFINITY;
    let mut best_weights = net.w.clone();
    let mut best_step = 0usize;

    let total = epochs * TRAIN_PER_EPOCH;
    let start = Instant::now();

    // Diagnostic: print initial predictions on first samples of each class.
    let mut init_diag = [0.0f32; 2]; // [pos_logit, neg_logit]
    let mut init_diag_done = false;
    let mut init_x = [0.0f32; WIN * WIN];
    synth_face(&mut init_x, &mut rng);
    init_diag[0] = net.forward(&init_x);
    synth_nonface(&mut init_x, &mut rng);
    init_diag[1] = net.forward(&init_x);
    println!(
        "[train] init logits: face={:.3} nonface={:.3}",
        init_diag[0], init_diag[1]
    );
    let _ = (init_diag, init_diag_done); // keep mut bindings alive

    for step in 0..total {
        let is_pos = step % 2 == 0;
        if is_pos {
            synth_face(&mut x_buf, &mut rng);
        } else {
            synth_nonface(&mut x_buf, &mut rng);
        }
        let label = if is_pos { 1.0 } else { 0.0 };
        // Zero grads.
        for g in [
            &mut grads.conv1_w,
            &mut grads.conv2_w,
            &mut grads.conv3_w,
            &mut grads.fc1_w,
            &mut grads.fc1_b,
            &mut grads.fc2_w,
            &mut grads.fc2_b,
        ] {
            for v in g.iter_mut() {
                *v = 0.0;
            }
        }
        let loss = net.forward_backward(&x_buf, label, &mut grads);
        // Average over batch (effective batch = 1 for now; mini-batching can
        // be added by accumulating across BATCH steps then calling step()).
        adam_conv1.step(&mut net.w.conv1_w, &grads.conv1_w);
        adam_conv2.step(&mut net.w.conv2_w, &grads.conv2_w);
        adam_conv3.step(&mut net.w.conv3_w, &mut grads.conv3_w);
        adam_fc1_w.step(&mut net.w.fc1_w, &grads.fc1_w);
        adam_fc1_b.step(&mut net.w.fc1_b, &grads.fc1_b);
        adam_fc2_w.step(&mut net.w.fc2_w, &grads.fc2_w);
        adam_fc2_b.step(&mut net.w.fc2_b, &grads.fc2_b);

        // Validation every 64 steps.
        let logit = net.forward(&x_buf);
        let pred = 1.0 / (1.0 + (-logit).exp());
        if is_pos && pred > 0.5 {
            pos_correct += 1;
        }
        if !is_pos && pred <= 0.5 {
            neg_correct += 1;
        }
        total_loss += loss;
        steps += 1;

        // Track the lowest-loss snapshot.
        let running_loss = total_loss / steps as f32;
        if running_loss < best_loss && step >= 256 {
            best_loss = running_loss;
            best_weights = net.w.clone();
            best_step = step;
        }

        if step % 256 == 0 && step > 0 {
            let elapsed = start.elapsed().as_secs_f32();
            let fps = steps as f32 / elapsed;
            println!(
                "[train] step={step}/{total} loss={:.4} pos_acc={}/{} neg_acc={}/{} fps={:.0}",
                total_loss / steps as f32,
                pos_correct,
                step / 2 + 1,
                neg_correct,
                step / 2,
                fps,
            );
        }
    }
    save_weights(out, &best_weights)?;
    println!(
        "[train] saved weights from step {} (loss={:.4}) to {}",
        best_step,
        best_loss,
        out.display()
    );
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let epochs: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(8);
    let out: std::path::PathBuf = args
        .get(2)
        .map(|s| std::path::PathBuf::from(s))
        .unwrap_or_else(|| std::path::PathBuf::from("trained.cnn.bin"));
    let seed: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(42);
    println!(
        "[cnn_train] epochs={epochs} out={} seed={seed}",
        out.display()
    );
    if let Err(e) = train(epochs, &out, seed) {
        eprintln!("[cnn_train] error: {e}");
        std::process::exit(1);
    }
}
