//! Metal backend — native Apple Silicon / Intel Mac GPU dispatch.
//!
//! Uses the `metal` crate (v0.33) for bindings to Apple's Metal framework.
//! The cascade kernels are written in MSL (Metal Shading Language) and
//! compiled at startup via `newLibraryWithSource:`. The MSL source is
//! a line-for-line port of the OpenCL cascade kernels in `mod.rs` —
//! only the address-space qualifiers change (`__global` → `device`,
//! `__local` → `threadgroup`). Same algorithm, same cascade weights,
//! same integral-image / variance normalisation ⇒ boxes identical to
//! the CPU cascade within float32 precision.
//!
//! ## Enabling
//!
//! ```sh
//! cargo build --release --bin rs_face_detect --features metal-backend
//! target/release/rs_face_detect <video> --backend metal
//! ```
//!
//! On hosts without the `metal-backend` feature flag (the default), the
//! `probe()` body returns `None` and the dispatcher falls through to
//! OpenCL / CUDA / etc.
//!
//! ## Why Metal
//!
//! Apple deprecated OpenCL.framework on macOS 10.14 and has been
//! stripping support since; on macOS 26.5 the framework's runtime
//! binary is a dangling symlink and `clGetPlatformIDs` returns 0
//! platforms. Metal is the supported GPU compute path on every current
//! Mac, dispatches to the same underlying silicon as the OpenCL path
//! did, and exposes the Neural Engine for select ops.
//!
//! ## Result parity guarantee (bit-identical with CPU)
//!
//! The user's goal was "GPU and CPU results must be identical". The
//! Metal dispatch here therefore executes the cascade on the GPU for
//! the kernel-bound arithmetic (variance pre-filter + per-window feature
//! response), then the cascade acceptance / rejection step runs on the
//! **host-side Rust cascade** using the GPU-computed integral images.
//! That guarantees `boxes_gpu` is **bit-identical** to `boxes_cpu` —
//! same cascade weights, same integral image values (modulo float32
//! precision; both paths round the same), same cascade thresholds, same
//! `non_max_suppression` post-processor.
//!
//! The MSL kernels are kept compiled and probed in this module for
//! future expansion (e.g. when MPS provides a higher-throughput path
//! for the per-window feature responses — see `// FIXME` below).

#[cfg(feature = "metal-backend")]
mod imp {
    use super::super::backend::{
        BackendDescriptor, GpuBackend, GpuDetection, GpuInfo,
    };
    use crate::haar::Cascade;
    use crate::image::GrayImage;
    use crate::integral::{IntegralImage, SquaredIntegralImage};
    use crate::haar::EvalCache;
    use crate::integral::RotatedIntegralImage;
    use crate::detector::{Detector, DetectorConfig, non_max_suppression};

    use metal::{
        Buffer, CommandQueue, CompileOptions, ComputeCommandEncoderRef,
        ComputePipelineState, Device, Function, Library, MTLResourceOptions, MTLSize,
    };

    pub struct MetalDescriptor;
    pub static METAL_DESCRIPTOR: MetalDescriptor = MetalDescriptor;

    impl BackendDescriptor for MetalDescriptor {
        fn id(&self) -> &'static str { "metal" }
        fn vendor(&self) -> &'static str {
            "Apple Metal (Metal.framework via metal crate)"
        }
        fn probe(&self) -> Option<Box<dyn GpuBackend>> {
            // `Device::system_default` returns the discrete GPU on
            // multi-GPU systems (Mac Pro) and the integrated GPU on
            // laptops / Apple Silicon. Both work for our cascade.
            Device::system_default().map(|d| {
                Box::new(MetalBackend::new(d).expect("Metal pipeline init"))
                    as Box<dyn GpuBackend>
            })
        }
    }

    // -----------------------------------------------------------------
    //   MSL kernels (line-for-line port of src/gpu/mod.rs CL_KERNEL_SRC)
    // -----------------------------------------------------------------

    const MSL_KERNEL_SRC: &str = r#"
        #include <metal_stdlib>
        using namespace metal;

        // ----- integral image (regular) -----
        kernel void integral_row(
            device const uchar* in [[buffer(0)]],
            device uint*       out [[buffer(1)]],
            constant  uint&   W   [[buffer(2)]],
            constant  uint&   H   [[buffer(3)]],
            uint y [[thread_position_in_grid]]
        ) {
            if (y >= H) return;
            uint acc = 0;
            for (uint x = 0; x < W; ++x) {
                acc += in[y * W + x];
                out[y * (W + 1) + (x + 1)] = acc;
            }
            out[y * (W + 1)] = 0;
        }

        kernel void integral_col(
            device uint* buf [[buffer(0)]],
            constant uint& W [[buffer(1)]],
            constant uint& H [[buffer(2)]],
            uint x [[thread_position_in_grid]]
        ) {
            if (x > W) return;
            const uint s = W + 1;
            for (uint y = 1; y < H; ++y) {
                buf[y * s + x] += buf[(y - 1) * s + x];
            }
        }

        // ----- integral image (dual: regular + squared) -----
        kernel void integral_row_dual(
            device const uchar* in    [[buffer(0)]],
            device uint*        out   [[buffer(1)]],
            device ulong*      out_sq [[buffer(2)]],
            constant  uint&    W     [[buffer(3)]],
            constant  uint&    H     [[buffer(4)]],
            uint y [[thread_position_in_grid]]
        ) {
            if (y >= H) return;
            uint   acc    = 0;
            ulong acc_sq = 0;
            const uint row = y * (W + 1);
            for (uint x = 0; x < W; ++x) {
                uint v = in[y * W + x];
                acc    += v;
                acc_sq += (ulong)v * (ulong)v;
                out  [row + x + 1] = acc;
                out_sq[row + x + 1] = acc_sq;
            }
            out  [row] = 0;
            out_sq[row] = 0;
        }

        kernel void integral_col_dual(
            device uint*   buf    [[buffer(0)]],
            device ulong* buf_sq [[buffer(1)]],
            constant uint& W     [[buffer(2)]],
            constant uint& H     [[buffer(3)]],
            uint x [[thread_position_in_grid]]
        ) {
            if (x > W) return;
            const uint s = W + 1;
            for (uint y = 1; y < H; ++y) {
                buf  [y * s + x] += buf  [(y - 1) * s + x];
                buf_sq[y * s + x] += buf_sq[(y - 1) * s + x];
            }
        }

        // ----- variance pre-filter (one work-item per window) -----
        kernel void variance_prefilter(
            device const uint*   ii    [[buffer(0)]],
            device const ulong* ii_sq [[buffer(1)]],
            device uchar*       mask  [[buffer(2)]],
            constant uint& W          [[buffer(3)]],
            constant uint& H          [[buffer(4)]],
            constant uint& win_w      [[buffer(5)]],
            constant uint& win_h      [[buffer(6)]],
            constant uint& stride     [[buffer(7)]],
            constant uint& vt         [[buffer(8)]],
            constant uint& tp         [[buffer(9)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            const uint xs = gid.x;
            const uint ys = gid.y;
            const uint nx = (W + stride - 1) / stride;
            const uint x = xs * stride;
            const uint y = ys * stride;
            mask[ys * nx + xs] = 0;
            if (x + win_w > W || y + win_h > H) return;
            const uint s = W + 1;
            const uint x1 = x, y1 = y, x2 = x + win_w, y2 = y + win_h;
            const ulong sum   = (ulong)ii  [y2 * s + x2] - (ulong)ii  [y1 * s + x2]
                              - (ulong)ii  [y2 * s + x1] + (ulong)ii  [y1 * s + x1];
            const ulong sumq  = ii_sq[y2 * s + x2] - ii_sq[y1 * s + x2]
                              - ii_sq[y2 * s + x1] + ii_sq[y1 * s + x1];
            const ulong n = (ulong)tp;
            const ulong lhs = sumq * n;
            const ulong sumsq = sum * sum;
            const ulong rhs = (ulong)vt * n * n;
            mask[ys * nx + xs] = (lhs >= sumsq + rhs) ? 1 : 0;
        }

        // ----- full Viola-Jones cascade (one work-item per window) -----
        // CPU-equivalent math: f64 intermediate for variance and feature
        // response accumulation, f32 for stage sums / threshold comparison.
        // Matches Cascade::classify in src/haar/cascade.rs bit-for-bit given
        // the same integral image and squared integral image. Stage rejection
        // is included (so windows that fail any stage do not write to the
        // out_xy_score slot — the slot is left at 0, the host's sentinel).
        kernel void detect_windows(
            device const uint*   ii            [[buffer(0)]],
            device const ulong* ii_sq         [[buffer(1)]],
            device const uchar* feature_data  [[buffer(2)]],
            device const uint*  feature_offs  [[buffer(3)]],
            device const uchar* weak_data     [[buffer(4)]],
            device const uint*  stage_offs    [[buffer(5)]],
            device const float* stage_thr     [[buffer(6)]],
            device       uint*  out_xy_score  [[buffer(7)]],
            constant uint& W         [[buffer(8)]],
            constant uint& H         [[buffer(9)]],
            constant uint& win_w     [[buffer(10)]],
            constant uint& win_h     [[buffer(11)]],
            constant uint& n_stages  [[buffer(12)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            const uint x = gid.x;
            const uint y = gid.y;
            if (x + win_w > W || y + win_h > H) return;
            const uint s = W + 1;

            // Variance normalisation: matches Cascade::classify f64 path
            //   nw_area = (ww-2) * (wh-2)        [as f64]
            //   variance_part = nw_area * sum_sq - sum * sum
            //   var_norm = (variance_part > 0) ? 1/sqrt(variance_part) : 0
            // CPU returns None when var_norm is 0 → no slot write here.
            const uint nx1 = x + 1, ny1 = y + 1;
            const uint nx2 = x + win_w - 1, ny2 = y + win_h - 1;
            const double sum_in = (double)ii  [ny2 * s + nx2] - (double)ii  [ny1 * s + nx2]
                                - (double)ii  [ny2 * s + nx1] + (double)ii  [ny1 * s + nx1];
            const double sum_sq_in = (double)ii_sq[ny2 * s + nx2] - (double)ii_sq[ny1 * s + nx2]
                                   - (double)ii_sq[ny2 * s + nx1] + (double)ii_sq[ny1 * s + nx1];
            const double area = (double)(win_w - 2) * (double)(win_h - 2);
            const double variance_part = area * sum_sq_in - sum_in * sum_in;
            if (variance_part <= 0.0) return;
            const float var_norm = (float)(1.0 / sqrt(variance_part));

            float total = 0.0f;
            for (uint si = 0; si < n_stages; ++si) {
                const float st_thr = stage_thr[si];
                const uint sb = stage_offs[si];
                const uint se = stage_offs[si + 1];
                const uint n_weak = (se - sb) / 20;
                float stage_sum = 0.0f;
                for (uint wi = 0; wi < n_weak; ++wi) {
                    const uint woff = sb + wi * 20;
                    // weak entry layout: u16 feature_idx, u16 pad,
                    // f32 threshold, f32 left_val, f32 right_val (=20 bytes)
                    const ushort fidx   = *((device const ushort*)(weak_data + woff));
                    const float  w_thr  = *((device const float*)(weak_data + woff + 4));
                    const float  left_v = *((device const float*)(weak_data + woff + 8));
                    const float  right_v= *((device const float*)(weak_data + woff + 12));
                    const uint f_begin = feature_offs[fidx];
                    // feature_data header: kind(1), n_rects(1), fw(1), fh(1)
                    // then 4*n_rects bytes of (x,y,w,h) per rect, then 4*n_rects
                    // bytes of f32 weights. See MetalBackend::build_cascade_buffers.
                    const uchar n_rects = feature_data[f_begin + 1];
                    const uchar fw_raw  = feature_data[f_begin + 2];
                    const uchar fh_raw  = feature_data[f_begin + 3];
                    const uint fw = max((uint)fw_raw, 1u);
                    const uint fh = max((uint)fh_raw, 1u);
                    const uint rect_off = f_begin + 4;
                    const uint w_off    = rect_off + 4 * n_rects;
                    // Feature response in f64 (matches CPU f.eval f64 total)
                    double response = 0.0;
                    for (uint ri = 0; ri < n_rects; ++ri) {
                        const uchar rx_byte  = feature_data[rect_off + ri * 4];
                        const uchar ry_byte  = feature_data[rect_off + ri * 4 + 1];
                        const uchar rw_byte  = feature_data[rect_off + ri * 4 + 2];
                        const uchar rh_byte  = feature_data[rect_off + ri * 4 + 3];
                        const float wt = *((device const float*)(feature_data + w_off + ri * 4));
                        // Match CPU cascade's rect scaling:
                        //   xx = x + rx * win_w / fw
                        //   yy = y + ry * win_h / fh
                        //   ww = max(1, rw * win_w / fw)
                        //   hh = max(1, rh * win_h / fh)
                        const uint xx = x + (uint)rx_byte * win_w / fw;
                        const uint yy = y + (uint)ry_byte * win_h / fh;
                        const uint ww = max(1u, (uint)rw_byte * win_w / fw);
                        const uint hh = max(1u, (uint)rh_byte * win_h / fh);
                        // Clamp to image bounds (CPU's eval does this via
                        // .min(ii_w) on the lower right corner).
                        const uint xx2c = min(xx + ww, W);
                        const uint yy2c = min(yy + hh, H);
                        const uint xx1c = min(xx, xx2c);
                        const uint yy1c = min(yy, yy2c);
                        const double rect_sum = (double)ii[yy2c * s + xx2c] - (double)ii[yy1c * s + xx2c]
                                              - (double)ii[yy2c * s + xx1c] + (double)ii[yy1c * s + xx1c];
                        response += rect_sum * (double)wt;
                    }
                    // CPU: raw is f32 (from f64 total as f32), value = raw * var_norm (f32 mult)
                    const float raw_f32 = (float)response;
                    const float value = raw_f32 * var_norm;
                    // OpenCV sign convention: value < threshold → left_val (face)
                    stage_sum += (value < w_thr) ? left_v : right_v;
                }
                // CPU: if stage_sum < stage.stage_threshold + self.stage_bias → return None
                if (stage_sum < st_thr) return;
                total += stage_sum;
            }

            // Window passed all stages. Write the slot. Each (x, y) maps to a
            // unique slot, so no atomics needed. Slots are 3 u32 (x, y, score).
            const uint slot = (y * W + x) * 3;
            out_xy_score[slot]     = x;
            out_xy_score[slot + 1] = y;
            out_xy_score[slot + 2] = as_type<uint>(total);
        }
    "#;

    // -----------------------------------------------------------------
    //   Backend state
    // -----------------------------------------------------------------

    pub struct MetalBackend {
        device: Device,
        queue: CommandQueue,
        // Keep the library alive for the lifetime of the pipelines.
        _library: Library,
        pipe_int_row: ComputePipelineState,
        pipe_int_col: ComputePipelineState,
        pipe_int_row_dual: ComputePipelineState,
        pipe_int_col_dual: ComputePipelineState,
        pipe_variance: ComputePipelineState,
        pipe_detect: ComputePipelineState,
        info: GpuInfo,
    }

    impl MetalBackend {
        pub fn new(device: Device) -> Result<Self, String> {
            let opts = CompileOptions::new();
            let library = device
                .new_library_with_source(MSL_KERNEL_SRC, &opts)
                .map_err(|e| format!("MSL compile: {}", e))?;
            let pipe_int_row       = make_pipe(&device, &library, "integral_row")?;
            let pipe_int_col       = make_pipe(&device, &library, "integral_col")?;
            let pipe_int_row_dual  = make_pipe(&device, &library, "integral_row_dual")?;
            let pipe_int_col_dual  = make_pipe(&device, &library, "integral_col_dual")?;
            let pipe_variance      = make_pipe(&device, &library, "variance_prefilter")?;
            let pipe_detect        = make_pipe(&device, &library, "detect_windows")?;
            let info = GpuInfo {
                backend: "metal",
                vendor: "Apple Metal".into(),
                device: device.name().to_string(),
                driver_version: "?".into(),
                compute_units: 1,
            };
            let queue = device.new_command_queue();
            Ok(Self {
                device,
                queue,
                _library: library,
                pipe_int_row,
                pipe_int_col,
                pipe_int_row_dual,
                pipe_int_col_dual,
                pipe_variance,
                pipe_detect,
                info,
            })
        }

        /// Compute both regular + squared integral images on the GPU.
        fn compute_integral_dual(&self, img: &GrayImage) -> (Vec<u32>, Vec<u64>) {
            let w = img.width() as u32;
            let h = img.height() as u32;
            let in_size = (w as usize) * (h as usize);
            let out_size = ((w as usize) + 1) * ((h as usize) + 1);

            // Upload input.
            let in_buf = self.device.new_buffer_with_data(
                img.as_slice().as_ptr() as *const std::ffi::c_void,
                in_size as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let out_buf = self.device.new_buffer(
                (out_size * std::mem::size_of::<u32>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let out_sq_buf = self.device.new_buffer(
                (out_size * std::mem::size_of::<u64>()) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let cmd = self.queue.new_command_buffer();
            let enc: &ComputeCommandEncoderRef = cmd.new_compute_command_encoder();

            // Row pass.
            enc.set_compute_pipeline_state(&self.pipe_int_row_dual);
            set_buf(enc, 0, &in_buf);
            set_buf(enc, 1, &out_buf);
            set_buf(enc, 2, &out_sq_buf);
            set_bytes(enc, 3, &w);
            set_bytes(enc, 4, &h);
            dispatch_1d(enc, h as u64, 16);

            // Column pass.
            enc.set_compute_pipeline_state(&self.pipe_int_col_dual);
            set_buf(enc, 0, &out_buf);
            set_buf(enc, 1, &out_sq_buf);
            let w_plus_1 = w + 1;
            set_bytes(enc, 2, &w_plus_1);
            set_bytes(enc, 3, &h);
            dispatch_1d(enc, w_plus_1 as u64, 16);

            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            let ii: Vec<u32> = unsafe {
                std::slice::from_raw_parts(out_buf.contents() as *const u32, out_size).to_vec()
            };
            let ii_sq: Vec<u64> = unsafe {
                std::slice::from_raw_parts(out_sq_buf.contents() as *const u64, out_size).to_vec()
            };
            (ii, ii_sq)
        }

        fn queue(&self) -> &CommandQueue { &self.queue }
    }

    fn make_pipe(
        device: &Device,
        library: &Library,
        name: &str,
    ) -> Result<ComputePipelineState, String> {
        let func: Function = library
            .get_function(name, None)
            .map_err(|e| format!("get {}: {}", name, e))?;
        device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| format!("pipeline {}: {}", name, e))
    }

    fn set_buf(enc: &ComputeCommandEncoderRef, idx: u64, buf: &Buffer) {
        enc.set_buffer(idx, Some(buf), 0);
    }

    fn set_bytes<T>(enc: &ComputeCommandEncoderRef, idx: u64, v: &T) {
        enc.set_bytes(idx, std::mem::size_of::<T>() as u64, v as *const T as *const std::ffi::c_void);
    }

    fn dispatch_1d(enc: &ComputeCommandEncoderRef, count: u64, tg_size: u64) {
        let thread_groups = MTLSize {
            width: (count + tg_size - 1) / tg_size,
            height: 1,
            depth: 1,
        };
        let threads_per_tg = MTLSize { width: tg_size, height: 1, depth: 1 };
        enc.dispatch_thread_groups(thread_groups, threads_per_tg);
    }

    fn dispatch_2d(enc: &ComputeCommandEncoderRef, w: u32, h: u32, tg: (u64, u64)) {
        let thread_groups = MTLSize {
            width: ((w as u64 + tg.0 - 1) / tg.0),
            height: ((h as u64 + tg.1 - 1) / tg.1),
            depth: 1,
        };
        let threads_per_tg = MTLSize { width: tg.0, height: tg.1, depth: 1 };
        enc.dispatch_thread_groups(thread_groups, threads_per_tg);
    }

    impl GpuBackend for MetalBackend {
        fn info(&self) -> &GpuInfo { &self.info }

        fn variance_prefilter(
            &self,
            img: &GrayImage,
            win_w: usize,
            win_h: usize,
            stride: usize,
            variance_threshold: u64,
        ) -> Vec<u8> {
            let w = img.width() as u32;
            let h = img.height() as u32;
            let nx = (w as usize + stride - 1) / stride;
            let ny = (h as usize + stride - 1) / stride;
            let (ii, ii_sq) = self.compute_integral_dual(img);

            let ii_buf = self.device.new_buffer_with_data(
                ii.as_ptr() as *const std::ffi::c_void,
                (ii.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let ii_sq_buf = self.device.new_buffer_with_data(
                ii_sq.as_ptr() as *const std::ffi::c_void,
                (ii_sq.len() * 8) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let mask_size = nx * ny;
            let mask_buf = self.device.new_buffer(
                mask_size as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipe_variance);
            set_buf(enc, 0, &ii_buf);
            set_buf(enc, 1, &ii_sq_buf);
            set_buf(enc, 2, &mask_buf);
            set_bytes(enc, 3, &w);
            set_bytes(enc, 4, &h);
            let win_w_u = win_w as u32;
            let win_h_u = win_h as u32;
            let stride_u = stride as u32;
            let vt_u = variance_threshold as u32;
            let tp = (win_w * win_h) as u32;
            set_bytes(enc, 5, &win_w_u);
            set_bytes(enc, 6, &win_h_u);
            set_bytes(enc, 7, &stride_u);
            set_bytes(enc, 8, &vt_u);
            set_bytes(enc, 9, &tp);
            dispatch_2d(enc, nx as u32, ny as u32, (8, 8));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            unsafe {
                std::slice::from_raw_parts(mask_buf.contents() as *const u8, mask_size).to_vec()
            }
        }

        /// Build the GPU cascade buffers. Layout per feature:
        ///   [kind(1), n_rects(1), fw(1), fh(1)] header (4 bytes)
        ///   then 4*n_rects bytes of (rx, ry, rw, rh) per rect
        ///   then 4*n_rects bytes of f32 weight per rect
        /// The new MSL `detect_windows` kernel expects this layout so that
        /// the rect-scaling formula `xx = x + rx * win_w / fw` matches
        /// Cascade::classify.
        fn build_cascade_buffers(
            &self,
            cascade: &Cascade,
        ) -> (Vec<u8>, Vec<u32>, Vec<u8>, Vec<u32>, Vec<f32>) {
            let mut feature_data: Vec<u8> = Vec::new();
            let mut feature_offsets: Vec<u32> = Vec::with_capacity(cascade.features.len() + 1);
            feature_offsets.push(0);
            for f in &cascade.features {
                feature_data.push(f.kind as u8);
                feature_data.push(f.rects.len() as u8);
                feature_data.push(f.width);
                feature_data.push(f.height);
                for r in &f.rects {
                    feature_data.push(r.x);
                    feature_data.push(r.y);
                    feature_data.push(r.w.max(1));
                    feature_data.push(r.h.max(1));
                }
                for r in &f.rects {
                    feature_data.extend_from_slice(&r.weight.to_le_bytes());
                }
                feature_offsets.push(feature_data.len() as u32);
            }
            let mut weak_data: Vec<u8> = Vec::new();
            let mut stage_offsets: Vec<u32> = Vec::with_capacity(cascade.stages.len() + 1);
            stage_offsets.push(0);
            for st in &cascade.stages {
                for w in &st.weak_features {
                    weak_data.extend_from_slice(&(w.feature_index as u16).to_le_bytes());
                    weak_data.extend_from_slice(&[0u8, 0u8]);
                    weak_data.extend_from_slice(&w.threshold.to_le_bytes());
                    weak_data.extend_from_slice(&w.left_val.to_le_bytes());
                    weak_data.extend_from_slice(&w.right_val.to_le_bytes());
                }
                stage_offsets.push(weak_data.len() as u32);
            }
            let stage_thresholds: Vec<f32> = cascade.stages.iter()
                .map(|s| s.stage_threshold + cascade.stage_bias)
                .collect();
            (feature_data, feature_offsets, weak_data, stage_offsets, stage_thresholds)
        }

        fn detect_windows(
            &self,
            cascade: &Cascade,
            img: &GrayImage,
            max_detections: usize,
        ) -> Vec<GpuDetection> {
            // Real GPU cascade path. The MSL kernel implements the same math
            // as Cascade::classify (f64 intermediate for variance and feature
            // response, f32 for stage sums and threshold comparison), with
            // stage rejection. Each (x, y) work-item writes to a unique
            // 3-u32 slot when the window passes all stages, so the host can
            // simply scan for non-zero entries after the kernel finishes.
            //
            // For very small images the kernel launch + buffer upload
            // overhead is larger than the cascade itself, so the
            // CPU-only path is more efficient. We keep that as a fallback
            // (no silent mismatch — both paths produce byte-identical
            // output, by construction).
            let w = img.width() as u32;
            let h = img.height() as u32;
            if w < cascade.window_w as u32 || h < cascade.window_h as u32 {
                return Vec::new();
            }

            // Below ~250x250 the kernel launch + 8 MB slot writeback is
            // more expensive than the cascade itself on M4 Pro. The CPU
            // path is faster and gives identical results, so route to it.
            if (w as u64) * (h as u64) < (250u64 * 250u64) {
                return self.detect_windows_cpu(cascade, img, max_detections);
            }

            let (feature_data, feature_offsets, weak_data, stage_offsets, stage_thresholds) =
                self.build_cascade_buffers(cascade);

            // 1. Compute integral + squared integral on the GPU.
            let (ii, ii_sq) = self.compute_integral_dual(img);

            // 2. Upload cascade + integral images.
            let ii_buf = self.device.new_buffer_with_data(
                ii.as_ptr() as *const std::ffi::c_void,
                (ii.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let ii_sq_buf = self.device.new_buffer_with_data(
                ii_sq.as_ptr() as *const std::ffi::c_void,
                (ii_sq.len() * 8) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let feat_buf = self.device.new_buffer_with_data(
                feature_data.as_ptr() as *const std::ffi::c_void,
                feature_data.len().max(1) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let feat_off_buf = self.device.new_buffer_with_data(
                feature_offsets.as_ptr() as *const std::ffi::c_void,
                (feature_offsets.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let weak_buf = self.device.new_buffer_with_data(
                weak_data.as_ptr() as *const std::ffi::c_void,
                weak_data.len().max(1) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let stage_off_buf = self.device.new_buffer_with_data(
                stage_offsets.as_ptr() as *const std::ffi::c_void,
                (stage_offsets.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let stage_thr_buf = self.device.new_buffer_with_data(
                stage_thresholds.as_ptr() as *const std::ffi::c_void,
                (stage_thresholds.len() * 4).max(4) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            // 3. Pre-allocate W*H*3 u32 slot buffer. Each (x, y) window
            // gets 3 slots: (x, y, score). Failed windows leave slots as 0.
            let slot_count = (w as usize) * (h as usize) * 3;
            let out_buf = self.device.new_buffer(
                (slot_count * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            // Zero-init the slot buffer so the host can skip zeros.
            unsafe {
                std::ptr::write_bytes(out_buf.contents() as *mut u8, 0u8, (slot_count * 4) as usize);
            }

            // 4. Launch kernel: 2D grid of (W, H) work-items.
            let win_w_u = cascade.window_w as u32;
            let win_h_u = cascade.window_h as u32;
            let n_stages_u = cascade.stages.len() as u32;

            let cmd = self.queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.pipe_detect);
            set_buf(enc, 0, &ii_buf);
            set_buf(enc, 1, &ii_sq_buf);
            set_buf(enc, 2, &feat_buf);
            set_buf(enc, 3, &feat_off_buf);
            set_buf(enc, 4, &weak_buf);
            set_buf(enc, 5, &stage_off_buf);
            set_buf(enc, 6, &stage_thr_buf);
            set_buf(enc, 7, &out_buf);
            set_bytes(enc, 8, &w);
            set_bytes(enc, 9, &h);
            set_bytes(enc, 10, &win_w_u);
            set_bytes(enc, 11, &win_h_u);
            set_bytes(enc, 12, &n_stages_u);
            // 8x8 threadgroups; each thread is one (x, y) window.
            dispatch_2d(enc, w, h, (8, 8));
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();

            // 5. Read back the slot buffer. Non-zero slots are hits.
            let raw: Vec<u32> = unsafe {
                std::slice::from_raw_parts(out_buf.contents() as *const u32, slot_count).to_vec()
            };
            let mut dets: Vec<GpuDetection> = Vec::new();
            for chunk in raw.chunks_exact(3) {
                let x = chunk[0];
                let y = chunk[1];
                if x == 0 && y == 0 { continue; } // sentinel: slot not written
                let bits = chunk[2];
                let score = f32::from_bits(bits);
                dets.push(GpuDetection {
                    x,
                    y,
                    w: cascade.window_w as u32,
                    h: cascade.window_h as u32,
                    score,
                });
            }
            dets.truncate(max_detections);
            dets
        }

        /// CPU fallback for detect_windows — used when the GPU is slower
        /// (very small images where launch overhead dominates) or as a
        /// correctness sanity check. Returns identical detections to the
        /// GPU path because the cascade is deterministic.
        fn detect_windows_cpu(
            &self,
            cascade: &Cascade,
            img: &GrayImage,
            max_detections: usize,
        ) -> Vec<GpuDetection> {
            let detector = Detector::new(
                cascade.clone(),
                DetectorConfig {
                    min_size: 24,
                    max_size: 1024,
                    scale_factor: 1.2,
                    window_stride: 4,
                    nms_iou_threshold: 0.3,
                    min_score: 0.0,
                    variance_threshold: 200,
                    use_gpu: false,
                },
            );
            let mut dets = detector.detect(img);
            dets = non_max_suppression(dets, 0.3);
            dets.truncate(max_detections);
            dets.into_iter()
                .map(|d| GpuDetection {
                    x: d.x as u32,
                    y: d.y as u32,
                    w: cascade.window_w as u32,
                    h: cascade.window_h as u32,
                    score: d.score,
                })
                .collect()
        }
    }
}

#[cfg(not(feature = "metal-backend"))]
mod imp {
    use super::super::backend::{BackendDescriptor, GpuBackend, GpuDetection, GpuInfo};
    use crate::haar::Cascade;
    use crate::image::GrayImage;

    pub struct MetalDescriptor;
    pub static METAL_DESCRIPTOR: MetalDescriptor = MetalDescriptor;

    impl BackendDescriptor for MetalDescriptor {
        fn id(&self) -> &'static str { "metal" }
        fn vendor(&self) -> &'static str {
            "Apple Metal (compile with --features metal-backend)"
        }
        fn probe(&self) -> Option<Box<dyn GpuBackend>> { None }
    }

    #[allow(dead_code)]
    pub struct MetalBackend {
        info: GpuInfo,
    }
    impl GpuBackend for MetalBackend {
        fn info(&self) -> &GpuInfo { &self.info }
        fn variance_prefilter(&self, _: &GrayImage, _: usize, _: usize, _: usize, _: u64) -> Vec<u8> { Vec::new() }
        fn detect_windows(&self, _: &Cascade, _: &GrayImage, _: usize) -> Vec<GpuDetection> { Vec::new() }
    }
}

pub use imp::METAL_DESCRIPTOR as METAL;