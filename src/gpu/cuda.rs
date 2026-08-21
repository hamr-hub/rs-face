//! CUDA backend (NVIDIA GPUs).
//!
//! Status: real implementation gated behind the ``cuda-backend`` Cargo
//! feature. The default build (`cargo check` / `cargo build` with no
//! flags) compiles a stub that returns ``None`` from ``probe()`` so the
//! dispatcher falls through to OpenCL / Metal — keeping the library
//! zero-dep on hosts without an NVIDIA driver.
//!
//! Enabling on Linux / Windows hosts with an NVIDIA GPU + CUDA toolkit
//! -------------------------------------------------------------------
//!
//! ```sh
//! cargo build --release --features cuda-backend
//! target/release/rs_face_detect <video> --backend cuda
//! ```
//!
//! Runtime requirements:
//!   * NVIDIA driver installed and loaded (``nvidia-smi`` reports the card).
//!   * CUDA toolkit (>= 12.x) so ``libcuda`` + NVRTC are available on
//!     ``LD_LIBRARY_PATH`` (Linux) / ``PATH`` (Windows).
//!   * You also need to pick a CUDA version feature for ``cudarc``
//!     itself, e.g. ``--features "cuda-backend,cudarc/cuda-12060"``.
//!     See ``cudarc``'s ``build.rs`` for the list of supported versions.
//!
//! Implementation
//! --------------
//!
//! Kernels are written in CUDA C and compiled **just-in-time** at startup
//! via NVRTC (``cudarc::nvrtc::compile_ptx``). The source is a
//! line-for-line port of the OpenCL cascade kernels in
//! ``mod.rs::CL_KERNEL_SRC``; only the address-space qualifiers and
//! index intrinsics change:
//!
//! ```text
//!   __kernel void name(...)       → __global__ void name(...)
//!   __global       (address space) → drop qualifier (CUDA implicit global ptr)
//!   __local                      → __shared__
//!   get_global_id(d)             → blockIdx.*dim * blockDim.*dim + threadIdx.*dim
//!   atom_inc(out)                → atomicAdd(out, 1u) (returns the old value)
//!   as_uint(x)                   → __float_as_uint(x)
//! ```
//!
//! The same algorithm, the same cascade weights, and the same integral
//! image / variance normalisation ⇒ boxes identical to the CPU cascade
//! within float32 precision.
//!
//! Hardware parity
//! ---------------
//!
//! ``tools/compare_cpu_gpu.py`` verifies box equality against the
//! reference OpenCV detector on real video frames. The CUDA dispatch
//! should match the OpenCL dispatch box-for-box because both run the
//! same kernel source (modulo the OpenCL↔CUDA translation above).
//!
//! Tested on
//! ---------
//!
//! The author has no NVIDIA GPU on this Mac; the code compiles when
//! ``--features cuda-backend`` is enabled (against ``cudarc`` 0.12 with
//! CUDA toolkit 12.x) but has NOT been run on real silicon yet. See
//! the "Test plan once on NVIDIA hardware" comment block at the
//! bottom of the `imp` module for the manual verification checklist.

#[cfg(feature = "cuda-backend")]
mod imp {
    use super::super::backend::{BackendDescriptor, GpuBackend, GpuDetection, GpuInfo};
    use crate::haar::Cascade;
    use crate::image::GrayImage;

    use cudarc::driver::{
        CudaDevice, CudaFunction, CudaSlice, DeviceRepr, LaunchAsync, LaunchConfig,
    };
    use cudarc::nvrtc::compile_ptx;
    use std::sync::Arc;

    pub struct CudaDescriptor;
    pub static CUDA_DESCRIPTOR: CudaDescriptor = CudaDescriptor;

    impl BackendDescriptor for CudaDescriptor {
        fn id(&self) -> &'static str {
            "cuda"
        }
        fn vendor(&self) -> &'static str {
            "NVIDIA CUDA (cudarc driver API + NVRTC)"
        }
        fn probe(&self) -> Option<Box<dyn GpuBackend>> {
            // `CudaDevice::new(0)` opens device 0 via libcuda. On hosts
            // without an NVIDIA driver, cudarc 0.12 panics inside
            // `result::init()` (it can't `dlopen("libcuda")`); we
            // `catch_unwind` so the dispatcher can still gracefully
            // fall through to OpenCL.
            let dev = match std::panic::catch_unwind(|| CudaDevice::new(0)) {
                Ok(Ok(d)) => d,
                Ok(Err(e)) => {
                    eprintln!("[rs-face] CUDA backend disabled: {}", e);
                    return None;
                }
                Err(_) => {
                    // cudarc panicked because libcuda is missing — the
                    // common case on macOS / CI containers without
                    // NVIDIA hardware. Quietly skip.
                    return None;
                }
            };
            let backend = match CudaBackend::new(dev) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("[rs-face] CUDA backend init failed: {}", e);
                    return None;
                }
            };
            Some(Box::new(backend) as Box<dyn GpuBackend>)
        }
    }

    // -----------------------------------------------------------------
    //   CUDA kernel source
    // -----------------------------------------------------------------
    //
    // Line-for-line port of `mod.rs::CL_KERNEL_SRC` (OpenCL C) to CUDA C.
    // Every address-space qualifier is stripped because CUDA's pointer
    // space is implicit on global pointers; `__local` becomes `__shared__`
    // (unused in these kernels); the index intrinsic is the standard
    // `blockIdx * blockDim + threadIdx` pattern; the OpenCL
    // `atom_inc(out)` → `atomicAdd(out, 1u)` (returns the previous
    // value, which is exactly what `atom_inc` returns in OpenCL).

    const CUDA_KERNEL_SRC: &str = r#"
        // ===== integral image (regular) =====
        __global__ void integral_row(
            const unsigned char* in,
            unsigned int*       out32,
            unsigned int  width,
            unsigned int  height
        ) {
            const unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;
            if (y >= height) return;
            unsigned int acc = 0;
            for (unsigned int x = 0; x < width; ++x) {
                acc += in[y * width + x];
                out32[y * (width + 1) + (x + 1)] = acc;
            }
            out32[y * (width + 1)] = 0;
        }

        __global__ void integral_col(
            unsigned int* buf,
            unsigned int  width,
            unsigned int  height
        ) {
            const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
            if (x > width) return;
            const unsigned int s = width + 1;
            for (unsigned int y = 1; y < height; ++y) {
                buf[y * s + x] += buf[(y - 1) * s + x];
            }
        }

        // ===== integral image (dual: regular + squared) =====
        __global__ void integral_row_dual(
            const unsigned char* in,
            unsigned int*        out,
            unsigned long long*  out_sq,
            unsigned int  width,
            unsigned int  height
        ) {
            const unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;
            if (y >= height) return;
            unsigned int        acc    = 0;
            unsigned long long  acc_sq = 0;
            const unsigned int  row    = y * (width + 1);
            for (unsigned int x = 0; x < width; ++x) {
                unsigned int v = in[y * width + x];
                acc    += v;
                acc_sq += (unsigned long long)v * (unsigned long long)v;
                out  [row + x + 1] = acc;
                out_sq[row + x + 1] = acc_sq;
            }
            out  [row] = 0;
            out_sq[row] = 0;
        }

        __global__ void integral_col_dual(
            unsigned int*        buf,
            unsigned long long*  buf_sq,
            unsigned int  width,
            unsigned int  height
        ) {
            const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
            if (x > width) return;
            const unsigned int s = width + 1;
            for (unsigned int y = 1; y < height; ++y) {
                buf  [y * s + x] += buf  [(y - 1) * s + x];
                buf_sq[y * s + x] += buf_sq[(y - 1) * s + x];
            }
        }

        // ===== variance pre-filter (one work-item per window) =====
        __global__ void variance_prefilter(
            const unsigned int*       ii,
            const unsigned long long* ii_sq,
            unsigned char*            mask,
            unsigned int  W,  unsigned int  H,
            unsigned int  win_w, unsigned int  win_h,
            unsigned int  stride,
            unsigned int  variance_threshold,
            unsigned int  total_pixels
        ) {
            const unsigned int xs = blockIdx.x * blockDim.x + threadIdx.x;
            const unsigned int ys = blockIdx.y * blockDim.y + threadIdx.y;
            const unsigned int nx = (W + stride - 1) / stride;
            const unsigned int x  = xs * stride;
            const unsigned int y  = ys * stride;
            mask[ys * nx + xs] = 0;
            if (x + win_w > W || y + win_h > H) return;
            const unsigned int s = W + 1;
            const unsigned int x1 = x, y1 = y, x2 = x + win_w, y2 = y + win_h;
            const unsigned long long sum   = (unsigned long long)ii  [y2 * s + x2] - (unsigned long long)ii  [y1 * s + x2]
                                          - (unsigned long long)ii  [y2 * s + x1] + (unsigned long long)ii  [y1 * s + x1];
            const unsigned long long sumq  = ii_sq[y2 * s + x2] - ii_sq[y1 * s + x2]
                                          - ii_sq[y2 * s + x1] + ii_sq[y1 * s + x1];
            const unsigned long long n     = (unsigned long long)total_pixels;
            const unsigned long long lhs   = sumq * n;
            const unsigned long long sumsq = sum * sum;
            const unsigned long long rhs   = (unsigned long long)variance_threshold * n * n;
            mask[ys * nx + xs] = (lhs >= sumsq + rhs) ? 1u : 0u;
        }

        // ===== full Viola-Jones cascade (one work-item per window) =====
        __global__ void detect_windows(
            const unsigned int*       ii,
            const unsigned long long* ii_sq,
            const unsigned char*      feature_data,
            const unsigned int*       feature_offsets,
            const unsigned char*      weak_data,
            const unsigned int*       stage_offsets,
            const float*              stage_thresholds,
            unsigned int*             out_count,
            unsigned int*             out_xy_score,
            unsigned int  W,  unsigned int  H,
            unsigned int  win_w, unsigned int  win_h,
            unsigned int  n_stages,
            unsigned int  max_detections
        ) {
            const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
            const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x + win_w > W || y + win_h > H) return;

            const unsigned int s = W + 1;

            // Variance normalisation over the inner (1,1,win_w-2,win_h-2) rect.
            const unsigned int nx1 = x + 1, ny1 = y + 1;
            const unsigned int nx2 = x + win_w - 1, ny2 = y + win_h - 1;
            const unsigned long long sum_in   = (unsigned long long)ii  [ny2 * s + nx2] - (unsigned long long)ii  [ny1 * s + nx2]
                                              - (unsigned long long)ii  [ny2 * s + nx1] + (unsigned long long)ii  [ny1 * s + nx1];
            const unsigned long long sum_sq_in = ii_sq[ny2 * s + nx2] - ii_sq[ny1 * s + nx2]
                                              - ii_sq[ny2 * s + nx1] + ii_sq[ny1 * s + nx1];
            const float area = (float)((win_w - 2) * (win_h - 2));
            const float variance_part = area * (float)sum_sq_in - (float)sum_in * (float)sum_in;
            if (variance_part <= 0.0f) return;
            const float var_norm = 1.0f / sqrtf(variance_part);

            // Cascade evaluation. Each stage: compute stage_sum, reject if < threshold.
            float total = 0.0f;
            for (unsigned int si = 0; si < n_stages; ++si) {
                const float stage_thr = stage_thresholds[si];
                const unsigned int s_begin = stage_offsets[si];
                const unsigned int s_end   = stage_offsets[si + 1];
                // Each weak feature is 20 bytes: u16 feature_idx, u16 pad,
                // f32 threshold, f32 left_val, f32 right_val.
                const unsigned int n_weak = (s_end - s_begin) / 20;
                float stage_sum = 0.0f;
                for (unsigned int wi = 0; wi < n_weak; ++wi) {
                    const unsigned int  woff  = s_begin + wi * 20;
                    const unsigned short fidx  = *reinterpret_cast<const unsigned short*>(weak_data + woff);
                    const float         w_thr = *reinterpret_cast<const float*>(weak_data + woff + 4);
                    const float         left_v= *reinterpret_cast<const float*>(weak_data + woff + 8);
                    const float         right_v=*reinterpret_cast<const float*>(weak_data + woff + 12);

                    const unsigned int f_begin = feature_offsets[fidx];
                    const unsigned int f_end   = feature_offsets[fidx + 1];
                    const unsigned char n_rects = feature_data[f_begin + 1];
                    const unsigned int rect_off = f_begin + 2;
                    const unsigned int w_off    = rect_off + 4 * n_rects;

                    float response = 0.0f;
                    for (unsigned int ri = 0; ri < n_rects; ++ri) {
                        const unsigned char rx = feature_data[rect_off + ri * 4];
                        const unsigned char ry = feature_data[rect_off + ri * 4 + 1];
                        const unsigned char rw = feature_data[rect_off + ri * 4 + 2];
                        const unsigned char rh = feature_data[rect_off + ri * 4 + 3];
                        const float wt = *reinterpret_cast<const float*>(feature_data + w_off + ri * 4);
                        const unsigned int xx1 = x + rx, yy1 = y + ry;
                        const unsigned int xx2 = xx1 + rw, yy2 = yy1 + rh;
                        const unsigned long long rect_sum = (unsigned long long)ii[yy2 * s + xx2]
                                                          - (unsigned long long)ii[yy1 * s + xx2]
                                                          - (unsigned long long)ii[yy2 * s + xx1]
                                                          + (unsigned long long)ii[yy1 * s + xx1];
                        response += wt * (float)rect_sum;
                    }
                    const float value = response * var_norm;
                    // OpenCV convention: value < threshold -> left_val (face), else right_val (non-face).
                    stage_sum += (value < w_thr) ? left_v : right_v;
                }
                if (stage_sum < stage_thr) return;
                total += stage_sum;
            }

            // Atomic append to output. `atomicAdd` returns the OLD value,
            // matching OpenCL's `atom_inc` semantics — so `idx` is the
            // slot this thread was just assigned.
            const unsigned int idx = atomicAdd(out_count, 1u);
            if (idx < max_detections) {
                const unsigned int base = idx * 3;
                out_xy_score[base]     = x;
                out_xy_score[base + 1] = y;
                out_xy_score[base + 2] = __float_as_uint(total);
            }
        }
    "#;

    // -----------------------------------------------------------------
    //   Backend state
    // -----------------------------------------------------------------

    pub struct CudaBackend {
        // `CudaDevice::new` returns `Arc<CudaDevice>` directly; we keep
        // the Arc so multiple detect threads can clone() it if needed.
        device: Arc<CudaDevice>,
        // Function handles for each kernel. ``CudaDevice`` retains a
        // reference to the loaded PTX module internally (keyed by
        // module name), so dropping these function handles does not
        // unload the module — they are cheap to clone if needed.
        func_int_row: CudaFunction,
        func_int_col: CudaFunction,
        func_int_row_dual: CudaFunction,
        func_int_col_dual: CudaFunction,
        func_variance: CudaFunction,
        func_detect: CudaFunction,
        info: GpuInfo,
    }

    impl CudaBackend {
        pub fn new(device: Arc<CudaDevice>) -> Result<Self, String> {
            // Compile the kernel source via NVRTC and load the resulting
            // PTX into the device. Compilation happens once per backend
            // instance; subsequent calls reuse the compiled module
            // (cached inside ``CudaDevice`` keyed by module name).
            let ptx =
                compile_ptx(CUDA_KERNEL_SRC).map_err(|e| format!("NVRTC compile failed: {}", e))?;
            device
                .load_ptx(
                    ptx,
                    "rsface_cuda",
                    &[
                        "integral_row",
                        "integral_col",
                        "integral_row_dual",
                        "integral_col_dual",
                        "variance_prefilter",
                        "detect_windows",
                    ],
                )
                .map_err(|e| format!("load_ptx failed: {}", e))?;

            // `get_func` returns Option in cudarc 0.12 — convert it to a
            // Result so we can use `?` consistently.
            let func_int_row = device
                .get_func("rsface_cuda", "integral_row")
                .ok_or_else(|| "get integral_row: not found".to_string())?;
            let func_int_col = device
                .get_func("rsface_cuda", "integral_col")
                .ok_or_else(|| "get integral_col: not found".to_string())?;
            let func_int_row_dual = device
                .get_func("rsface_cuda", "integral_row_dual")
                .ok_or_else(|| "get integral_row_dual: not found".to_string())?;
            let func_int_col_dual = device
                .get_func("rsface_cuda", "integral_col_dual")
                .ok_or_else(|| "get integral_col_dual: not found".to_string())?;
            let func_variance = device
                .get_func("rsface_cuda", "variance_prefilter")
                .ok_or_else(|| "get variance_prefilter: not found".to_string())?;
            let func_detect = device
                .get_func("rsface_cuda", "detect_windows")
                .ok_or_else(|| "get detect_windows: not found".to_string())?;

            let info = GpuInfo {
                backend: "cuda",
                vendor: "NVIDIA CUDA".into(),
                device: device.name().unwrap_or_else(|_| "NVIDIA".into()),
                driver_version: "?".into(),
                compute_units: 1,
            };

            Ok(Self {
                device,
                func_int_row,
                func_int_col,
                func_int_row_dual,
                func_int_col_dual,
                func_variance,
                func_detect,
                info,
            })
        }

        /// Compute both regular + squared integral images on the GPU.
        /// Returns CPU-side ``Vec``s the caller can use for downstream
        /// kernels or for the host-side cascade re-eval path.
        fn compute_integral_dual(&self, img: &GrayImage) -> (Vec<u32>, Vec<u64>) {
            let w = img.width() as u32;
            let h = img.height() as u32;
            let out_size = ((w as usize) + 1) * ((h as usize) + 1);

            let d_in: CudaSlice<u8> = self
                .device
                .htod_copy(img.as_slice().to_vec())
                .expect("upload input image");
            let d_out: CudaSlice<u32> = self.device.alloc_zeros(out_size).expect("alloc ii buffer");
            let d_out_sq: CudaSlice<u64> = self
                .device
                .alloc_zeros(out_size)
                .expect("alloc ii_sq buffer");

            // Row pass: 1D grid, one thread per row.
            let cfg_row = LaunchConfig {
                grid_dim: ((h + 255) / 256, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            // cudarc's LaunchAsync requires scalars to be passed BY VALUE
            // (DeviceRepr is impl'd for primitives, not `&u32`), and
            // slices by reference (D deref -> DevicePtr<T>). Also,
            // `launch` consumes the receiver, so we `clone()` the
            // `CudaFunction` handle each call (cheap: it's just an Arc).
            unsafe {
                self.func_int_row_dual
                    .clone()
                    .launch(cfg_row, (&d_in, &d_out, &d_out_sq, w, h))
                    .expect("launch integral_row_dual");
            }

            // Column pass: 1D grid, one thread per column (W+1 entries).
            let w_plus_1 = w + 1;
            let cfg_col = LaunchConfig {
                grid_dim: ((w_plus_1 + 255) / 256, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                self.func_int_col_dual
                    .clone()
                    .launch(cfg_col, (&d_out, &d_out_sq, w_plus_1, h))
                    .expect("launch integral_col_dual");
            }

            let ii = self.device.dtoh_sync_copy(&d_out).expect("download ii");
            let ii_sq = self
                .device
                .dtoh_sync_copy(&d_out_sq)
                .expect("download ii_sq");
            (ii, ii_sq)
        }
    }

    // -----------------------------------------------------------------
    //   Trait implementation
    // -----------------------------------------------------------------

    impl GpuBackend for CudaBackend {
        fn info(&self) -> &GpuInfo {
            &self.info
        }

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
            let mask_size = nx * ny;

            let (ii, ii_sq) = self.compute_integral_dual(img);

            let d_ii: CudaSlice<u32> = self.device.htod_copy(ii).expect("upload ii");
            let d_ii_sq: CudaSlice<u64> = self.device.htod_copy(ii_sq).expect("upload ii_sq");
            let d_mask: CudaSlice<u8> = self.device.alloc_zeros(mask_size).expect("alloc mask");

            // 2D launch — 16x16 thread blocks. The kernel early-returns
            // for OOB windows, so we round the grid up rather than
            // tightening it.
            let cfg = LaunchConfig {
                grid_dim: (((nx as u32) + 15) / 16, ((ny as u32) + 15) / 16, 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };
            let win_w_u = win_w as u32;
            let win_h_u = win_h as u32;
            let stride_u = stride as u32;
            let vt_u = variance_threshold as u32;
            let tp = (win_w * win_h) as u32;
            unsafe {
                self.func_variance
                    .clone()
                    .launch(
                        cfg,
                        (
                            &d_ii, &d_ii_sq, &d_mask, w, h, win_w_u, win_h_u, stride_u, vt_u, tp,
                        ),
                    )
                    .expect("launch variance_prefilter");
            }
            self.device.dtoh_sync_copy(&d_mask).expect("download mask")
        }

        fn detect_windows(
            &self,
            cascade: &Cascade,
            img: &GrayImage,
            max_detections: usize,
        ) -> Vec<GpuDetection> {
            // ---- Serialise cascade into the GPU buffer layout ----
            // Mirrors `mod.rs::opencl::Context::detect_windows` exactly
            // so the CUDA kernel sees the same byte stream as the
            // OpenCL one — bit-identical cascade evaluation.
            let mut feature_data: Vec<u8> = Vec::new();
            let mut feature_offsets: Vec<u32> = Vec::with_capacity(cascade.features.len() + 1);
            feature_offsets.push(0);
            for f in &cascade.features {
                feature_data.push(f.kind as u8);
                feature_data.push(f.rects.len() as u8);
                for r in &f.rects {
                    feature_data.push(r.x);
                    feature_data.push(r.y);
                    feature_data.push(r.w.max(1));
                    feature_data.push(r.h.max(1));
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
            let stage_thresholds: Vec<f32> =
                cascade.stages.iter().map(|s| s.stage_threshold).collect();

            // ---- Compute integral images ----
            let (ii, ii_sq) = self.compute_integral_dual(img);
            let w = img.width() as u32;
            let h = img.height() as u32;

            // ---- Upload everything ----
            let d_ii: CudaSlice<u32> = self.device.htod_copy(ii).expect("upload ii");
            let d_ii_sq: CudaSlice<u64> = self.device.htod_copy(ii_sq).expect("upload ii_sq");
            let d_feat: CudaSlice<u8> = self
                .device
                .htod_copy(feature_data)
                .expect("upload feature_data");
            let d_feat_off: CudaSlice<u32> = self
                .device
                .htod_copy(feature_offsets)
                .expect("upload feature_offsets");
            let d_weak: CudaSlice<u8> = self.device.htod_copy(weak_data).expect("upload weak_data");
            let d_stage_off: CudaSlice<u32> = self
                .device
                .htod_copy(stage_offsets)
                .expect("upload stage_offsets");
            let d_stage_thr: CudaSlice<f32> = self
                .device
                .htod_copy(stage_thresholds)
                .expect("upload stage_thresholds");
            let d_count: CudaSlice<u32> = self.device.alloc_zeros(1).expect("alloc count");
            let d_out: CudaSlice<u32> = self
                .device
                .alloc_zeros(max_detections * 3)
                .expect("alloc out_xy_score");

            // ---- Launch ----
            let cfg = LaunchConfig {
                grid_dim: ((w + 15) / 16, (h + 15) / 16, 1),
                block_dim: (16, 16, 1),
                shared_mem_bytes: 0,
            };
            let win_w_arg = cascade.window_w as u32;
            let win_h_arg = cascade.window_h as u32;
            let n_stages_arg = cascade.stages.len() as u32;
            let max_det_arg = max_detections as u32;
            // cudarc's tuple-based `launch` is only impl'd for tuples
            // up to 12 elements; detect_windows has 15 args (9 buffers
            // + 6 scalars) so we drop down to the raw-pointer form.
            // `as_kernel_param` (from the `DeviceRepr` trait) converts
            // each arg into a `*mut c_void` that CUDA accepts in its
            // variadic kernel-arg ABI. The order MUST match the kernel
            // signature in `CUDA_KERNEL_SRC` exactly.
            let mut args: [*mut std::ffi::c_void; 15] = [
                (&d_ii).as_kernel_param(),
                (&d_ii_sq).as_kernel_param(),
                (&d_feat).as_kernel_param(),
                (&d_feat_off).as_kernel_param(),
                (&d_weak).as_kernel_param(),
                (&d_stage_off).as_kernel_param(),
                (&d_stage_thr).as_kernel_param(),
                (&d_count).as_kernel_param(),
                (&d_out).as_kernel_param(),
                (&w).as_kernel_param(),
                (&h).as_kernel_param(),
                (&win_w_arg).as_kernel_param(),
                (&win_h_arg).as_kernel_param(),
                (&n_stages_arg).as_kernel_param(),
                (&max_det_arg).as_kernel_param(),
            ];
            unsafe {
                self.func_detect
                    .clone()
                    .launch(cfg, args.as_mut_slice())
                    .expect("launch detect_windows");
            }

            // ---- Read back ----
            let count: Vec<u32> = self
                .device
                .dtoh_sync_copy(&d_count)
                .expect("download count");
            let actual = (count[0] as usize).min(max_detections);
            let xy: Vec<u32> = if actual > 0 {
                self.device
                    .dtoh_sync_copy(&d_out)
                    .expect("download out_xy_score")
            } else {
                Vec::new()
            };

            let mut out = Vec::with_capacity(actual);
            for chunk in xy.chunks(3) {
                if chunk.len() < 3 {
                    break;
                }
                let x = chunk[0];
                let y = chunk[1];
                let bits = chunk[2];
                let score = f32::from_bits(bits);
                out.push(GpuDetection {
                    x,
                    y,
                    w: 0,
                    h: 0,
                    score,
                });
            }
            out
        }
    }

    // -----------------------------------------------------------------
    //   Test plan once on NVIDIA hardware
    // -----------------------------------------------------------------
    //
    // The author could not exercise the kernels in-tree (no NVIDIA GPU
    // on hand). Before merging, the following checks should be run on
    // a Linux/Windows box with an NVIDIA card + CUDA toolkit 12.x:
    //
    // 1. Smoke test — does probe() succeed and report the right device?
    //
    //      cargo run --release --features cuda-backend -- --backend cuda \
    //          --dry-run
    //
    //    Expect: "cuda on <GPU name> (1 CU, ?)" in the backend line.
    //
    // 2. Integral image parity — `compute_integral_dual` must produce the
    //    same (ii, ii_sq) buffers as the CPU integral image builder in
    //    `src/integral.rs` for a known input. Diff byte-by-byte.
    //
    // 3. Variance pre-filter parity — run `variance_prefilter` on a
    //    small synthetic image and compare its output to
    //    `crate::detector::variance_prefilter_cpu` (or the OpenCL one
    //    on a dual-GPU machine). The two must agree cell-by-cell.
    //
    // 4. Cascade box equality — feed a labelled video to:
    //
    //      tools/compare_cpu_gpu.py --backend cuda
    //
    //    and verify the CUDA boxes match the CPU (and OpenCL) boxes
    //    within a 1-pixel tolerance, on a frame-by-frame basis.
    //
    // 5. Sanity: `--backend auto` should prefer CUDA when the NVIDIA
    //    driver is loaded (it's listed before OpenCL in BACKENDS).
    //
    // 6. Negative test — with the NVIDIA driver unloaded, `probe()`
    //    must return `None` and the dispatcher must fall through to
    //    OpenCL without panicking.
}

// ---------------------------------------------------------------------
//   Stub backend — compiled when ``cuda-backend`` is NOT enabled.
// ---------------------------------------------------------------------
//
// Keeps the linker symbols (`CUDA`, `CudaDescriptor`) defined so the
// `BACKENDS` array in `backend.rs` compiles regardless of feature flag,
// and `probe()` returns `None` so the dispatcher skips CUDA.

#[cfg(not(feature = "cuda-backend"))]
mod imp {
    use super::super::backend::{BackendDescriptor, GpuBackend, GpuDetection, GpuInfo};
    use crate::haar::Cascade;
    use crate::image::GrayImage;

    pub struct CudaDescriptor;
    pub static CUDA_DESCRIPTOR: CudaDescriptor = CudaDescriptor;

    impl BackendDescriptor for CudaDescriptor {
        fn id(&self) -> &'static str {
            "cuda"
        }
        fn vendor(&self) -> &'static str {
            "NVIDIA CUDA (compile with --features cuda-backend)"
        }
        fn probe(&self) -> Option<Box<dyn GpuBackend>> {
            None
        }
    }

    #[allow(dead_code)]
    pub struct CudaBackend {
        info: GpuInfo,
    }
    impl GpuBackend for CudaBackend {
        fn info(&self) -> &GpuInfo {
            &self.info
        }
        fn variance_prefilter(
            &self,
            img: &GrayImage,
            _w: usize,
            _h: usize,
            s: usize,
            _t: u64,
        ) -> Vec<u8> {
            let nx = (img.width() + s - 1) / s;
            let ny = (img.height() + s - 1) / s;
            vec![1u8; nx * ny]
        }
        fn detect_windows(&self, _: &Cascade, _: &GrayImage, _: usize) -> Vec<GpuDetection> {
            Vec::new()
        }
    }
}

pub use imp::CUDA_DESCRIPTOR as CUDA;
