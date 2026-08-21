//! GPU backends for rs-face.
//!
//! This module exposes:
//!   * The cross-platform OpenCL driver (default; works on Apple Silicon
//!     via Metal-OpenCL, Intel iGPU, AMD, NVIDIA on Linux/Windows).
//!     Loaded via FFI at runtime — no Rust deps.
//!   * A backend-trait abstraction (``pub mod backend``) that lets the
//!     same dispatch surface pick between Metal, CUDA, ROCm, Ascend and
//!     MLU implementations. Per-vendor stubs live alongside this file
//!     (see ``metal.rs``, ``cuda.rs``, ``rocm.rs``, ``ascend.rs``,
//!     ``mlu.rs``); vendors with no SDK on the host probe as
//!     unavailable and the dispatcher falls back to OpenCL.

pub mod ascend;
pub mod backend;
pub mod cuda;
pub mod metal;
pub mod mlu;
pub mod rocm;

use crate::haar::Cascade;
use crate::image::GrayImage;
use std::sync::Arc;

/// Result of `probe`.
#[derive(Clone, Debug)]
pub struct GpuInfo {
    pub platform_name: String,
    pub device_name: String,
    pub compute_units: u32,
}

/// Try to initialize an OpenCL context. Returns `None` if no OpenCL
/// implementation is available on the system.
pub fn probe() -> Option<GpuInfo> {
    unsafe { Context::new().ok().map(|c| c.info.clone()) }
}

/// Output of the GPU sliding window detector.
#[derive(Clone, Debug)]
pub struct GpuDetection {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    pub score: f32,
}

/// GPU-side integral image + variance pre-filter. Falls back to CPU when no
/// GPU is available.
pub struct GpuIntegral {
    ctx: Arc<Context>,
}

impl GpuIntegral {
    pub fn new() -> Option<Self> {
        Context::new().ok().map(|c| Self { ctx: Arc::new(c) })
    }

    pub fn info(&self) -> GpuInfo {
        self.ctx.info.clone()
    }

    /// Compute integral image on the GPU.
    pub fn compute(&self, img: &GrayImage) -> Vec<u32> {
        unsafe { self.ctx.compute_integral(img) }
    }

    /// Compute both regular and squared integral images on the GPU in one pass.
    pub fn compute_dual(&self, img: &GrayImage) -> (Vec<u32>, Vec<u64>) {
        unsafe { self.ctx.compute_integral_dual(img) }
    }

    /// Run the variance pre-filter on the GPU in parallel. One work-item per
    /// `(x, y)` window evaluates the variance test in O(1) using the integral
    /// images. Returns a `u8` mask (1 = passes, 0 = fails).
    pub fn variance_prefilter(
        &self,
        img: &GrayImage,
        win_w: usize,
        win_h: usize,
        stride: usize,
        variance_threshold: u64,
    ) -> Vec<u8> {
        unsafe {
            self.ctx
                .variance_prefilter(img, win_w, win_h, stride, variance_threshold)
        }
    }

    /// Run the FULL cascade on the GPU: each work-item is one window, with
    /// variance normalisation + per-stage eval + early reject.
    pub fn detect_windows(
        &self,
        cascade: &Cascade,
        img: &GrayImage,
        max_detections: usize,
    ) -> Vec<GpuDetection> {
        unsafe { self.ctx.detect_windows(cascade, img, max_detections) }
    }
}

// ===== Non-Unix stub =====
// On Windows (or any non-unix target) we keep the public types in scope so
// `crate::gpu::GpuIntegral` etc. still resolve, but every method is a
// no-op that always returns `None` / empty. The `use_gpu` flag on
// `DetectorConfig` then trivially disables GPU for that build.
#[cfg(not(unix))]
mod stub {
    use super::{Cascade, GpuDetection, GpuInfo, GrayImage};

    pub struct Context {
        pub info: GpuInfo,
    }

    impl Context {
        pub fn new() -> Result<Self, ()> {
            Err(())
        }
        pub fn compute_integral(&self, _img: &GrayImage) -> Vec<u32> {
            Vec::new()
        }
        pub fn compute_integral_dual(&self, _img: &GrayImage) -> (Vec<u32>, Vec<u64>) {
            (Vec::new(), Vec::new())
        }
        pub fn variance_prefilter(
            &self,
            _img: &GrayImage,
            _win_w: usize,
            _win_h: usize,
            _stride: usize,
            _variance_threshold: u64,
        ) -> Vec<u8> {
            Vec::new()
        }
        pub fn detect_windows(
            &self,
            _cascade: &Cascade,
            _img: &GrayImage,
            _max_detections: usize,
        ) -> Vec<GpuDetection> {
            Vec::new()
        }
    }
}

#[cfg(not(unix))]
pub use stub::Context;

#[cfg(unix)]
pub use opencl::Context;

// ===== OpenCL FFI + dynamic loader =====

#[cfg(unix)]
mod opencl {
    use super::GpuInfo;
    use crate::image::GrayImage;
    use std::ffi::{c_char, CString};
    use std::ptr;

    extern "C" {
        fn dlopen(filename: *const u8, flag: i32) -> *mut std::ffi::c_void;
        fn dlsym(handle: *mut std::ffi::c_void, symbol: *const u8) -> *mut std::ffi::c_void;
        fn dlerror() -> *mut c_char;
    }
    const RTLD_LAZY: i32 = 0x00001;
    const RTLD_LOCAL: i32 = 0x00200;

    type ClPlatformId = *mut std::ffi::c_void;
    type ClDeviceId = *mut std::ffi::c_void;
    type ClContext = *mut std::ffi::c_void;
    type ClCommandQueue = *mut std::ffi::c_void;
    type ClProgram = *mut std::ffi::c_void;
    type ClKernel = *mut std::ffi::c_void;
    type ClMem = *mut std::ffi::c_void;
    type ClInt = i32;
    type ClUint = u32;
    type ClSize = usize;
    type ClBool = u8;

    const CL_SUCCESS: ClInt = 0;
    const CL_DEVICE_TYPE_GPU: ClUint = 2;
    const CL_TRUE: ClBool = 1;

    type FnGetPlatformIDs = unsafe extern "C" fn(ClUint, *mut ClPlatformId, *mut ClUint) -> ClInt;
    type FnGetDeviceIDs =
        unsafe extern "C" fn(ClPlatformId, ClUint, ClUint, *mut ClDeviceId, *mut ClUint) -> ClInt;
    type FnGetPlatformInfo = unsafe extern "C" fn(
        ClPlatformId,
        ClUint,
        ClSize,
        *mut std::ffi::c_void,
        *mut ClSize,
    ) -> ClInt;
    type FnGetDeviceInfo = unsafe extern "C" fn(
        ClDeviceId,
        ClUint,
        ClSize,
        *mut std::ffi::c_void,
        *mut ClSize,
    ) -> ClInt;
    type FnCreateContext = unsafe extern "C" fn(
        *const ClInt,
        ClUint,
        *const ClDeviceId,
        *mut std::ffi::c_void,
        *mut std::ffi::c_void,
        *mut ClInt,
    ) -> ClContext;
    type FnCreateCommandQueue =
        unsafe extern "C" fn(ClContext, ClDeviceId, ClUint, *mut ClInt) -> ClCommandQueue;
    type FnCreateBuffer =
        unsafe extern "C" fn(ClContext, ClUint, ClSize, *mut std::ffi::c_void, *mut ClInt) -> ClMem;
    type FnCreateProgramWithSource = unsafe extern "C" fn(
        ClContext,
        ClUint,
        *const *const c_char,
        *const ClSize,
        *mut ClInt,
    ) -> ClProgram;
    type FnBuildProgram = unsafe extern "C" fn(
        ClProgram,
        ClUint,
        *const ClDeviceId,
        *const c_char,
        *mut std::ffi::c_void,
        *mut std::ffi::c_void,
    ) -> ClInt;
    type FnCreateKernel = unsafe extern "C" fn(ClProgram, *const c_char, *mut ClInt) -> ClKernel;
    type FnSetKernelArg =
        unsafe extern "C" fn(ClKernel, ClUint, ClSize, *const std::ffi::c_void) -> ClInt;
    type FnEnqueueNDRangeKernel = unsafe extern "C" fn(
        ClCommandQueue,
        ClKernel,
        ClUint,
        *const ClSize,
        *const ClSize,
        *const ClSize,
        ClUint,
        *const ClMem,
        *mut ClMem,
    ) -> ClInt;
    type FnEnqueueReadBuffer = unsafe extern "C" fn(
        ClCommandQueue,
        ClMem,
        ClBool,
        ClSize,
        ClSize,
        *mut std::ffi::c_void,
        ClUint,
        *const ClMem,
        *mut ClMem,
    ) -> ClInt;
    type FnEnqueueWriteBuffer = unsafe extern "C" fn(
        ClCommandQueue,
        ClMem,
        ClBool,
        ClSize,
        ClSize,
        *const std::ffi::c_void,
        ClUint,
        *const ClMem,
        *mut ClMem,
    ) -> ClInt;
    type FnReleaseMemObject = unsafe extern "C" fn(ClMem) -> ClInt;
    type FnReleaseKernel = unsafe extern "C" fn(ClKernel) -> ClInt;
    type FnReleaseProgram = unsafe extern "C" fn(ClProgram) -> ClInt;
    type FnReleaseCommandQueue = unsafe extern "C" fn(ClCommandQueue) -> ClInt;
    type FnReleaseContext = unsafe extern "C" fn(ClContext) -> ClInt;
    type FnFinish = unsafe extern "C" fn(ClCommandQueue) -> ClInt;

    #[repr(C)]
    struct Lib {
        get_platform_ids: FnGetPlatformIDs,
        get_device_ids: FnGetDeviceIDs,
        get_platform_info: FnGetPlatformInfo,
        get_device_info: FnGetDeviceInfo,
        create_context: FnCreateContext,
        create_command_queue: FnCreateCommandQueue,
        create_buffer: FnCreateBuffer,
        create_program_with_source: FnCreateProgramWithSource,
        build_program: FnBuildProgram,
        create_kernel: FnCreateKernel,
        set_kernel_arg: FnSetKernelArg,
        enqueue_nd_range_kernel: FnEnqueueNDRangeKernel,
        enqueue_read_buffer: FnEnqueueReadBuffer,
        enqueue_write_buffer: FnEnqueueWriteBuffer,
        release_mem_object: FnReleaseMemObject,
        release_kernel: FnReleaseKernel,
        release_program: FnReleaseProgram,
        release_command_queue: FnReleaseCommandQueue,
        release_context: FnReleaseContext,
        finish: FnFinish,
    }

    static mut LIB: Option<Lib> = None;
    static mut LIB_HANDLE: *mut std::ffi::c_void = ptr::null_mut();

    fn candidate_names() -> &'static [&'static str] {
        // Order matters: probe the most likely first. Recent macOS releases
        // ship OpenCL.framework as a broken symlink (Apple has been
        // deprecating since 10.14), so we prefer Homebrew's opencl-icd-loader
        // path which actually works.
        &[
            // Linux (Khronos ICD loader paths)
            "libOpenCL.so.1",
            "libOpenCL.so",
            "/usr/lib/x86_64-linux-gnu/libOpenCL.so.1",
            "/usr/lib/aarch64-linux-gnu/libOpenCL.so.1",
            // Homebrew on Apple Silicon (preferred on macOS — system framework
            // is often a broken symlink since macOS 10.14 deprecation).
            "/opt/homebrew/lib/libOpenCL.dylib",
            "/opt/homebrew/Cellar/opencl-icd-loader/2026.05.29/lib/libOpenCL.dylib",
            // Homebrew on Intel Mac
            "/usr/local/lib/libOpenCL.dylib",
            // macOS — Apple system framework (last resort — frequently broken)
            "/System/Library/Frameworks/OpenCL.framework/OpenCL",
        ]
    }

    unsafe fn c_dlsym(handle: *mut std::ffi::c_void, name: &str) -> Option<*mut std::ffi::c_void> {
        let c = CString::new(name).ok()?;
        let p = dlsym(handle, c.as_ptr() as *const u8);
        if p.is_null() {
            None
        } else {
            Some(p)
        }
    }

    fn load() -> Option<&'static Lib> {
        unsafe {
            if LIB.is_some() {
                return LIB.as_ref();
            }
            for name in candidate_names() {
                let c = match CString::new(*name) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let h = dlopen(c.as_ptr() as *const u8, RTLD_LAZY | RTLD_LOCAL);
                if !h.is_null() {
                    LIB_HANDLE = h;
                    break;
                }
            }
            macro_rules! sym {
                ($name:literal, $ty:ty) => {{
                    let p = c_dlsym(LIB_HANDLE, $name)?;
                    std::mem::transmute::<*mut std::ffi::c_void, $ty>(p)
                }};
            }
            let lib = Lib {
                get_platform_ids: sym!("clGetPlatformIDs", FnGetPlatformIDs),
                get_device_ids: sym!("clGetDeviceIDs", FnGetDeviceIDs),
                get_platform_info: sym!("clGetPlatformInfo", FnGetPlatformInfo),
                get_device_info: sym!("clGetDeviceInfo", FnGetDeviceInfo),
                create_context: sym!("clCreateContext", FnCreateContext),
                create_command_queue: sym!("clCreateCommandQueue", FnCreateCommandQueue),
                create_buffer: sym!("clCreateBuffer", FnCreateBuffer),
                create_program_with_source: sym!(
                    "clCreateProgramWithSource",
                    FnCreateProgramWithSource
                ),
                build_program: sym!("clBuildProgram", FnBuildProgram),
                create_kernel: sym!("clCreateKernel", FnCreateKernel),
                set_kernel_arg: sym!("clSetKernelArg", FnSetKernelArg),
                enqueue_nd_range_kernel: sym!("clEnqueueNDRangeKernel", FnEnqueueNDRangeKernel),
                enqueue_read_buffer: sym!("clEnqueueReadBuffer", FnEnqueueReadBuffer),
                enqueue_write_buffer: sym!("clEnqueueWriteBuffer", FnEnqueueWriteBuffer),
                release_mem_object: sym!("clReleaseMemObject", FnReleaseMemObject),
                release_kernel: sym!("clReleaseKernel", FnReleaseKernel),
                release_program: sym!("clReleaseProgram", FnReleaseProgram),
                release_command_queue: sym!("clReleaseCommandQueue", FnReleaseCommandQueue),
                release_context: sym!("clReleaseContext", FnReleaseContext),
                finish: sym!("clFinish", FnFinish),
            };
            LIB = Some(lib);
            LIB.as_ref()
        }
    }

    fn read_string(info: *mut std::ffi::c_void, len: usize) -> String {
        unsafe {
            let slice = std::slice::from_raw_parts(info as *const u8, len);
            let end = slice.iter().position(|&b| b == 0).unwrap_or(len);
            String::from_utf8_lossy(&slice[..end]).into_owned()
        }
    }

    const CL_KERNEL_SRC: &str = r#"
        __kernel void integral_row(__global const uchar* in,
                                   __global u32* out,
                                   const uint width, const uint height) {
            const uint y = get_global_id(0);
            if (y >= height) return;
            u32 acc = 0;
            for (uint x = 0; x < width; ++x) {
                acc += in[y * width + x];
                out[y * (width + 1) + (x + 1)] = acc;
            }
            out[y * (width + 1)] = 0;
        }
        __kernel void integral_col(__global u32* buf,
                                   const uint width, const uint height) {
            const uint x = get_global_id(0);
            if (x > width) return;
            for (uint y = 1; y < height; ++y) {
                buf[y * (width + 1) + x] += buf[(y - 1) * (width + 1) + x];
            }
        }
        __kernel void integral_row_dual(__global const uchar* in,
                                        __global u32* out,
                                        __global u64* out_sq,
                                        const uint width, const uint height) {
            const uint y = get_global_id(0);
            if (y >= height) return;
            u32 acc = 0;
            u64 acc_sq = 0;
            const uint row_off = y * (width + 1);
            for (uint x = 0; x < width; ++x) {
                u32 v = in[y * width + x];
                acc += v;
                acc_sq += (u64)v * (u64)v;
                out[row_off + x + 1] = acc;
                out_sq[row_off + x + 1] = acc_sq;
            }
            out[row_off] = 0;
            out_sq[row_off] = 0;
        }
        __kernel void integral_col_dual(__global u32* buf, __global u64* buf_sq,
                                        const uint width, const uint height) {
            const uint x = get_global_id(0);
            if (x > width) return;
            const uint stride = width + 1;
            for (uint y = 1; y < height; ++y) {
                buf[y * stride + x] += buf[(y - 1) * stride + x];
                buf_sq[y * stride + x] += buf_sq[(y - 1) * stride + x];
            }
        }
        __kernel void variance_prefilter(
            __global const u32* ii,
            __global const u64* ii_sq,
            __global u8* out_mask,
            const uint W, const uint H,
            const uint win_w, const uint win_h,
            const uint stride,
            const uint variance_threshold,
            const uint total_pixels
        ) {
            const uint xs = get_global_id(0);
            const uint ys = get_global_id(1);
            const uint nx = (W + stride - 1) / stride;
            const uint x = xs * stride;
            const uint y = ys * stride;
            out_mask[ys * nx + xs] = 0;
            if (x + win_w > W || y + win_h > H) return;
            const uint stride_ii = W + 1;
            const uint x1 = x, y1 = y, x2 = x + win_w, y2 = y + win_h;
            const u64 s  = (u64)ii[y2 * stride_ii + x2] - (u64)ii[y1 * stride_ii + x2]
                        - (u64)ii[y2 * stride_ii + x1] + (u64)ii[y1 * stride_ii + x1];
            const u64 ss = ii_sq[y2 * stride_ii + x2] - ii_sq[y1 * stride_ii + x2]
                        - ii_sq[y2 * stride_ii + x1] + ii_sq[y1 * stride_ii + x1];
            const u64 n = (u64)total_pixels;
            const u64 lhs = ss * n;
            const u64 sum_sq = s * s;
            const u64 rhs_base = (u64)variance_threshold * n * n;
            out_mask[ys * nx + xs] = (lhs >= sum_sq + rhs_base) ? 1 : 0;
        }

        // Full sliding window cascade evaluation. Each work-item is one (x,y)
        // window position. Inputs:
        //   ii, ii_sq           : (H+1)*(W+1) integral images
        //   feature_data        : packed feature table (each feature:
        //                         [kind u8, n_rects u8] [rect x,y,w,h (4 u8) | weight f32] * n_rects)
        //   feature_offsets     : u32[n_features+1] byte offsets into feature_data
        //   weak_data           : packed weak feature table:
        //                         per weak: [feature_idx u16 | padding u16 | threshold f32 | left_val f32 | right_val f32]
        //   stage_offsets       : u32[n_stages+1] byte offsets into weak_data
        //   stage_thresholds    : f32[n_stages]
        // Each output: 3 u32 (x, y, score_bits).
        __kernel void detect_windows(
            __global const u32* ii,
            __global const u64* ii_sq,
            __global const uchar* feature_data,
            __global const uint* feature_offsets,
            __global const uchar* weak_data,
            __global const uint* stage_offsets,
            __global const float* stage_thresholds,
            __global uint* out_count,
            __global uint* out_xy_score,
            const uint W, const uint H,
            const uint win_w, const uint win_h,
            const uint n_stages,
            const uint max_detections
        ) {
            const uint x = get_global_id(0);
            const uint y = get_global_id(1);
            if (x + win_w > W || y + win_h > H) return;

            const uint stride_ii = W + 1;

            // Compute variance normalization over inner rect (1, 1, win_w-2, win_h-2)
            const uint nx1 = x + 1, ny1 = y + 1;
            const uint nx2 = x + win_w - 1, ny2 = y + win_h - 1;
            const u64 sum_in   = (u64)ii[ny2 * stride_ii + nx2] - (u64)ii[ny1 * stride_ii + nx2]
                               - (u64)ii[ny2 * stride_ii + nx1] + (u64)ii[ny1 * stride_ii + nx1];
            const u64 sum_sq_in = ii_sq[ny2 * stride_ii + nx2] - ii_sq[ny1 * stride_ii + nx2]
                                - ii_sq[ny2 * stride_ii + nx1] + ii_sq[ny1 * stride_ii + nx1];
            const float area = (float)((win_w - 2) * (win_h - 2));
            const float variance_part = area * (float)sum_sq_in - (float)sum_in * (float)sum_in;
            if (variance_part <= 0.f) return;
            const float var_norm = 1.f / sqrt(variance_part);

            // Cascade evaluation. Each stage: compute stage_sum, reject if < threshold.
            float total = 0.f;
            for (uint si = 0; si < n_stages; ++si) {
                const float stage_thr = stage_thresholds[si];
                const uint s_begin = stage_offsets[si];
                const uint s_end = stage_offsets[si + 1];
                // Each weak feature is 20 bytes: u16 feature_idx, u16 pad, f32 threshold, f32 left_val, f32 right_val
                const uint n_weak = (s_end - s_begin) / 20;
                float stage_sum = 0.f;
                for (uint wi = 0; wi < n_weak; ++wi) {
                    const uint woff = s_begin + wi * 20;
                    const uint fidx = *((__global const ushort*)(weak_data + woff));
                    // const uint pad = *((__global const ushort*)(weak_data + woff + 2));
                    const float w_thr = *((__global const float*)(weak_data + woff + 4));
                    const float left_v = *((__global const float*)(weak_data + woff + 8));
                    const float right_v = *((__global const float*)(weak_data + woff + 12));

                    // Read feature
                    const uint f_begin = feature_offsets[fidx];
                    const uint f_end = feature_offsets[fidx + 1];
                    const uchar n_rects = feature_data[f_begin + 1];
                    const uint rect_off = f_begin + 2;
                    const uint w_off = rect_off + 4 * n_rects;

                    // Compute weighted pixel sum using integral image
                    float response = 0.f;
                    for (uint ri = 0; ri < n_rects; ++ri) {
                        const uchar rx = feature_data[rect_off + ri * 4];
                        const uchar ry = feature_data[rect_off + ri * 4 + 1];
                        const uchar rw = feature_data[rect_off + ri * 4 + 2];
                        const uchar rh = feature_data[rect_off + ri * 4 + 3];
                        const float wt = *((__global const float*)(feature_data + w_off + ri * 4));
                        const uint xx1 = x + rx, yy1 = y + ry;
                        const uint xx2 = xx1 + rw, yy2 = yy1 + rh;
                        const u64 rect_sum = (u64)ii[yy2 * stride_ii + xx2]
                                           - (u64)ii[yy1 * stride_ii + xx2]
                                           - (u64)ii[yy2 * stride_ii + xx1]
                                           + (u64)ii[yy1 * stride_ii + xx1];
                        response += wt * (float)rect_sum;
                    }
                    const float value = response * var_norm;
                    // OpenCV convention: value < threshold → left_val (face), else right_val (non-face)
                    stage_sum += (value < w_thr) ? left_v : right_v;
                }
                if (stage_sum < stage_thr) return;
                total += stage_sum;
            }

            // Atomic append to output
            const uint idx = atom_inc(out_count);
            if (idx < max_detections) {
                const uint base = idx * 3;
                out_xy_score[base]     = x;
                out_xy_score[base + 1] = y;
                out_xy_score[base + 2] = as_uint(total);
            }
        }
    "#;

    pub struct Context {
        pub info: GpuInfo,
        ctx: ClContext,
        queue: ClCommandQueue,
        program: ClProgram,
        kernel_integral: ClKernel,
        kernel_dual_row: ClKernel,
        kernel_dual_col: ClKernel,
        kernel_variance: ClKernel,
        kernel_detect_windows: ClKernel,
    }

    // OpenCL contexts are opaque to us; the driver is thread-safe. We mark
    // the raw pointer types as Send + Sync so an `Arc<Context>` can be shared
    // across detector worker threads.
    unsafe impl Send for Context {}
    unsafe impl Sync for Context {}

    impl Context {
        pub fn new() -> Result<Self, &'static str> {
            let lib = load().ok_or("libOpenCL not found")?;
            unsafe {
                let mut platforms = [ptr::null_mut(); 4];
                let mut n: ClUint = 0;
                let err = (lib.get_platform_ids)(4, platforms.as_mut_ptr(), &mut n);
                if err != CL_SUCCESS || n == 0 {
                    return Err("no platforms");
                }
                Self::from_platform(platforms[0], lib)
            }
        }

        unsafe fn from_platform(platform: ClPlatformId, lib: &Lib) -> Result<Self, &'static str> {
            const CL_PLATFORM_NAME: ClUint = 0x0902;
            const CL_DEVICE_NAME: ClUint = 0x102B;
            const CL_DEVICE_MAX_COMPUTE_UNITS: ClUint = 0x1002;
            let mut buf = [0u8; 256];
            let mut len: ClSize = buf.len();
            let err = (lib.get_platform_info)(
                platform,
                CL_PLATFORM_NAME,
                buf.len(),
                buf.as_mut_ptr() as *mut _,
                &mut len,
            );
            let platform_name = if err == CL_SUCCESS {
                read_string(buf.as_mut_ptr() as *mut _, len)
            } else {
                String::from("unknown")
            };
            let mut device: ClDeviceId = ptr::null_mut();
            let mut nd: ClUint = 0;
            let err = (lib.get_device_ids)(platform, CL_DEVICE_TYPE_GPU, 1, &mut device, &mut nd);
            if err != CL_SUCCESS || nd == 0 {
                return Err("no GPU device");
            }
            let mut buf2 = [0u8; 256];
            let mut len2: ClSize = buf2.len();
            let err = (lib.get_device_info)(
                device,
                CL_DEVICE_NAME,
                buf2.len(),
                buf2.as_mut_ptr() as *mut _,
                &mut len2,
            );
            let device_name = if err == CL_SUCCESS {
                read_string(buf2.as_mut_ptr() as *mut _, len2)
            } else {
                String::from("unknown")
            };
            let mut units: ClUint = 0;
            let _ = (lib.get_device_info)(
                device,
                CL_DEVICE_MAX_COMPUTE_UNITS,
                std::mem::size_of::<ClUint>(),
                &mut units as *mut _ as *mut _,
                ptr::null_mut(),
            );
            let mut errc: ClInt = 0;
            let ctx = (lib.create_context)(
                ptr::null(),
                1,
                &device,
                ptr::null_mut(),
                ptr::null_mut(),
                &mut errc,
            );
            if errc != CL_SUCCESS {
                return Err("create_context failed");
            }
            let queue = (lib.create_command_queue)(ctx, device, 0, &mut errc);
            if errc != CL_SUCCESS {
                (lib.release_context)(ctx);
                return Err("create_queue failed");
            }
            let src = CString::new(CL_KERNEL_SRC).unwrap();
            let src_len = CL_KERNEL_SRC.len() as ClSize;
            let src_ptr = src.as_ptr() as *const i8;
            let program = (lib.create_program_with_source)(ctx, 1, &src_ptr, &src_len, &mut errc);
            if errc != CL_SUCCESS {
                (lib.release_command_queue)(queue);
                (lib.release_context)(ctx);
                return Err("create_program failed");
            }
            let build_err = (lib.build_program)(
                program,
                1,
                &device,
                ptr::null(),
                ptr::null_mut(),
                ptr::null_mut(),
            );
            if build_err != CL_SUCCESS {
                (lib.release_program)(program);
                (lib.release_command_queue)(queue);
                (lib.release_context)(ctx);
                return Err("build_program failed");
            }
            let mk = |name: &str| -> Result<ClKernel, &'static str> {
                let cs = CString::new(name).unwrap();
                let mut errk: ClInt = 0;
                let name_ptr = cs.as_ptr() as *const i8;
                let k = (lib.create_kernel)(program, name_ptr, &mut errk);
                if errk != CL_SUCCESS {
                    Err("create_kernel failed")
                } else {
                    Ok(k)
                }
            };
            let kernel_integral = mk("integral_row").map_err(|_| {
                (lib.release_program)(program);
                (lib.release_command_queue)(queue);
                (lib.release_context)(ctx);
                "create_kernel integral_row"
            })?;
            let kernel_dual_row = mk("integral_row_dual").map_err(|_| {
                (lib.release_kernel)(kernel_integral);
                (lib.release_program)(program);
                (lib.release_command_queue)(queue);
                (lib.release_context)(ctx);
                "create_kernel integral_row_dual"
            })?;
            let kernel_dual_col = mk("integral_col_dual").map_err(|_| {
                (lib.release_kernel)(kernel_dual_row);
                (lib.release_kernel)(kernel_integral);
                (lib.release_program)(program);
                (lib.release_command_queue)(queue);
                (lib.release_context)(ctx);
                "create_kernel integral_col_dual"
            })?;
            let kernel_variance = mk("variance_prefilter").map_err(|_| {
                (lib.release_kernel)(kernel_dual_col);
                (lib.release_kernel)(kernel_dual_row);
                (lib.release_kernel)(kernel_integral);
                (lib.release_program)(program);
                (lib.release_command_queue)(queue);
                (lib.release_context)(ctx);
                "create_kernel variance_prefilter"
            })?;
            let kernel_detect_windows = mk("detect_windows").map_err(|_| {
                (lib.release_kernel)(kernel_variance);
                (lib.release_kernel)(kernel_dual_col);
                (lib.release_kernel)(kernel_dual_row);
                (lib.release_kernel)(kernel_integral);
                (lib.release_program)(program);
                (lib.release_command_queue)(queue);
                (lib.release_context)(ctx);
                "create_kernel detect_windows"
            })?;
            Ok(Context {
                info: GpuInfo {
                    platform_name,
                    device_name,
                    compute_units: units,
                },
                ctx,
                queue,
                program,
                kernel_integral,
                kernel_dual_row,
                kernel_dual_col,
                kernel_variance,
                kernel_detect_windows,
            })
        }

        pub fn compute_integral(&self, img: &GrayImage) -> Vec<u32> {
            self.compute_integral_dual(img).0
        }

        /// Compute both regular and squared integral images on the GPU.
        pub fn compute_integral_dual(&self, img: &GrayImage) -> (Vec<u32>, Vec<u64>) {
            let lib = unsafe { load().unwrap() };
            let w = img.width();
            let h = img.height();
            let in_size = (w * h) as ClSize;
            let out_size = ((w + 1) * (h + 1)) as ClSize;
            let out_sq_bytes = out_size * 8;
            unsafe {
                let mut err: ClInt = 0;
                let d_in =
                    (lib.create_buffer)(self.ctx, 1 << 2, in_size, ptr::null_mut(), &mut err);
                if err != CL_SUCCESS {
                    return cpu_fallback_dual(img);
                }
                let d_out =
                    (lib.create_buffer)(self.ctx, 1 << 1, out_size, ptr::null_mut(), &mut err);
                if err != CL_SUCCESS {
                    (lib.release_mem_object)(d_in);
                    return cpu_fallback_dual(img);
                }
                let d_out_sq =
                    (lib.create_buffer)(self.ctx, 1 << 1, out_sq_bytes, ptr::null_mut(), &mut err);
                if err != CL_SUCCESS {
                    (lib.release_mem_object)(d_in);
                    (lib.release_mem_object)(d_out);
                    return cpu_fallback_dual(img);
                }
                let write_err = (lib.enqueue_write_buffer)(
                    self.queue,
                    d_in,
                    CL_TRUE,
                    0,
                    in_size,
                    img.as_slice().as_ptr() as *const _,
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                );
                if write_err != CL_SUCCESS {
                    (lib.release_mem_object)(d_in);
                    (lib.release_mem_object)(d_out);
                    (lib.release_mem_object)(d_out_sq);
                    return cpu_fallback_dual(img);
                }
                (lib.set_kernel_arg)(
                    self.kernel_dual_row,
                    0,
                    std::mem::size_of::<ClMem>(),
                    &d_in as *const _ as *const _,
                );
                (lib.set_kernel_arg)(
                    self.kernel_dual_row,
                    1,
                    std::mem::size_of::<ClMem>(),
                    &d_out as *const _ as *const _,
                );
                (lib.set_kernel_arg)(
                    self.kernel_dual_row,
                    2,
                    std::mem::size_of::<ClMem>(),
                    &d_out_sq as *const _ as *const _,
                );
                let ww = w as ClUint;
                let hh = h as ClUint;
                (lib.set_kernel_arg)(
                    self.kernel_dual_row,
                    3,
                    std::mem::size_of::<ClUint>(),
                    &ww as *const _ as *const _,
                );
                (lib.set_kernel_arg)(
                    self.kernel_dual_row,
                    4,
                    std::mem::size_of::<ClUint>(),
                    &hh as *const _ as *const _,
                );
                let global: ClSize = h as ClSize;
                let err = (lib.enqueue_nd_range_kernel)(
                    self.queue,
                    self.kernel_dual_row,
                    1,
                    ptr::null(),
                    &global,
                    ptr::null(),
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                );
                if err != CL_SUCCESS {
                    (lib.release_mem_object)(d_in);
                    (lib.release_mem_object)(d_out);
                    (lib.release_mem_object)(d_out_sq);
                    return cpu_fallback_dual(img);
                }
                let col_global: ClSize = (w + 1) as ClSize;
                let err = (lib.enqueue_nd_range_kernel)(
                    self.queue,
                    self.kernel_dual_col,
                    1,
                    ptr::null(),
                    &col_global,
                    ptr::null(),
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                );
                if err != CL_SUCCESS {
                    (lib.release_mem_object)(d_in);
                    (lib.release_mem_object)(d_out);
                    (lib.release_mem_object)(d_out_sq);
                    return cpu_fallback_dual(img);
                }
                let mut out = vec![0u32; (w + 1) * (h + 1)];
                let mut out_sq = vec![0u64; (w + 1) * (h + 1)];
                let r1 = (lib.enqueue_read_buffer)(
                    self.queue,
                    d_out,
                    CL_TRUE,
                    0,
                    out_size,
                    out.as_mut_ptr() as *mut _,
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                );
                let r2 = (lib.enqueue_read_buffer)(
                    self.queue,
                    d_out_sq,
                    CL_TRUE,
                    0,
                    out_sq_bytes,
                    out_sq.as_mut_ptr() as *mut _,
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                );
                (lib.release_mem_object)(d_in);
                (lib.release_mem_object)(d_out);
                (lib.release_mem_object)(d_out_sq);
                if r1 != CL_SUCCESS || r2 != CL_SUCCESS {
                    return cpu_fallback_dual(img);
                }
                (lib.finish)(self.queue);
                (out, out_sq)
            }
        }

        /// Run the variance pre-filter on the GPU. Returns a u8 mask.
        pub fn variance_prefilter(
            &self,
            img: &GrayImage,
            win_w: usize,
            win_h: usize,
            stride: usize,
            variance_threshold: u64,
        ) -> Vec<u8> {
            let lib = unsafe { load().unwrap() };
            let w = img.width();
            let h = img.height();
            let nx = (w + stride - 1) / stride;
            let ny = (h + stride - 1) / stride;
            let mask_bytes = (nx * ny) as ClSize;
            let (ii, ii_sq) = self.compute_integral_dual(img);
            let out_size = ((w + 1) * (h + 1)) as ClSize;
            let out_sq_bytes = out_size * 8;
            unsafe {
                let mut err: ClInt = 0;
                let d_ii =
                    (lib.create_buffer)(self.ctx, 1 << 2, out_size, ptr::null_mut(), &mut err);
                if err != CL_SUCCESS {
                    return vec![1u8; nx * ny];
                }
                let d_ii_sq =
                    (lib.create_buffer)(self.ctx, 1 << 2, out_sq_bytes, ptr::null_mut(), &mut err);
                if err != CL_SUCCESS {
                    (lib.release_mem_object)(d_ii);
                    return vec![1u8; nx * ny];
                }
                let d_mask =
                    (lib.create_buffer)(self.ctx, 1 << 1, mask_bytes, ptr::null_mut(), &mut err);
                if err != CL_SUCCESS {
                    (lib.release_mem_object)(d_ii);
                    (lib.release_mem_object)(d_ii_sq);
                    return vec![1u8; nx * ny];
                }
                let w1 = (lib.enqueue_write_buffer)(
                    self.queue,
                    d_ii,
                    CL_TRUE,
                    0,
                    out_size,
                    ii.as_ptr() as *const _,
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                );
                let w2 = (lib.enqueue_write_buffer)(
                    self.queue,
                    d_ii_sq,
                    CL_TRUE,
                    0,
                    out_sq_bytes,
                    ii_sq.as_ptr() as *const _,
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                );
                if w1 != CL_SUCCESS || w2 != CL_SUCCESS {
                    (lib.release_mem_object)(d_ii);
                    (lib.release_mem_object)(d_ii_sq);
                    (lib.release_mem_object)(d_mask);
                    return vec![1u8; nx * ny];
                }
                (lib.set_kernel_arg)(
                    self.kernel_variance,
                    0,
                    std::mem::size_of::<ClMem>(),
                    &d_ii as *const _ as *const _,
                );
                (lib.set_kernel_arg)(
                    self.kernel_variance,
                    1,
                    std::mem::size_of::<ClMem>(),
                    &d_ii_sq as *const _ as *const _,
                );
                (lib.set_kernel_arg)(
                    self.kernel_variance,
                    2,
                    std::mem::size_of::<ClMem>(),
                    &d_mask as *const _ as *const _,
                );
                let ww = w as ClUint;
                let hh = h as ClUint;
                let wwin = win_w as ClUint;
                let hwin = win_h as ClUint;
                let st = stride as ClUint;
                let vt = variance_threshold as ClUint;
                let tp = (win_w * win_h) as ClUint;
                (lib.set_kernel_arg)(
                    self.kernel_variance,
                    3,
                    std::mem::size_of::<ClUint>(),
                    &ww as *const _ as *const _,
                );
                (lib.set_kernel_arg)(
                    self.kernel_variance,
                    4,
                    std::mem::size_of::<ClUint>(),
                    &hh as *const _ as *const _,
                );
                (lib.set_kernel_arg)(
                    self.kernel_variance,
                    5,
                    std::mem::size_of::<ClUint>(),
                    &wwin as *const _ as *const _,
                );
                (lib.set_kernel_arg)(
                    self.kernel_variance,
                    6,
                    std::mem::size_of::<ClUint>(),
                    &hwin as *const _ as *const _,
                );
                (lib.set_kernel_arg)(
                    self.kernel_variance,
                    7,
                    std::mem::size_of::<ClUint>(),
                    &st as *const _ as *const _,
                );
                (lib.set_kernel_arg)(
                    self.kernel_variance,
                    8,
                    std::mem::size_of::<ClUint>(),
                    &vt as *const _ as *const _,
                );
                (lib.set_kernel_arg)(
                    self.kernel_variance,
                    9,
                    std::mem::size_of::<ClUint>(),
                    &tp as *const _ as *const _,
                );
                let global: [ClSize; 2] = [nx as ClSize, ny as ClSize];
                let err = (lib.enqueue_nd_range_kernel)(
                    self.queue,
                    self.kernel_variance,
                    2,
                    ptr::null(),
                    global.as_ptr(),
                    ptr::null(),
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                );
                if err != CL_SUCCESS {
                    (lib.release_mem_object)(d_ii);
                    (lib.release_mem_object)(d_ii_sq);
                    (lib.release_mem_object)(d_mask);
                    return vec![1u8; nx * ny];
                }
                let mut mask = vec![0u8; nx * ny];
                let r = (lib.enqueue_read_buffer)(
                    self.queue,
                    d_mask,
                    CL_TRUE,
                    0,
                    mask_bytes,
                    mask.as_mut_ptr() as *mut _,
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                );
                (lib.release_mem_object)(d_ii);
                (lib.release_mem_object)(d_ii_sq);
                (lib.release_mem_object)(d_mask);
                if r != CL_SUCCESS {
                    return vec![1u8; nx * ny];
                }
                (lib.finish)(self.queue);
                mask
            }
        }

        /// Run the full cascade on GPU. Returns detected (x, y, score) triples.
        pub fn detect_windows(
            &self,
            cascade: &super::Cascade,
            img: &GrayImage,
            max_detections: usize,
        ) -> Vec<super::GpuDetection> {
            // Serialize cascade into buffer formats.
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

            let lib = unsafe { load().unwrap() };
            let w = img.width();
            let h = img.height();
            let (ii, ii_sq) = self.compute_integral_dual(img);
            let out_size = ((w + 1) * (h + 1)) as ClSize;
            let out_sq_bytes = out_size * 8;
            let feature_bytes = feature_data.len() as ClSize;
            let weak_bytes = weak_data.len() as ClSize;
            let feature_offset_bytes = (feature_offsets.len() * 4) as ClSize;
            let stage_offset_bytes = (stage_offsets.len() * 4) as ClSize;
            let stage_threshold_bytes = (stage_thresholds.len() * 4) as ClSize;
            let out_count_bytes = std::mem::size_of::<u32>() as ClSize;
            let out_xy_bytes = (max_detections * 12) as ClSize;
            unsafe {
                let mut err: ClInt = 0;
                let d_ii =
                    (lib.create_buffer)(self.ctx, 1 << 2, out_size, ptr::null_mut(), &mut err);
                if err != CL_SUCCESS {
                    return Vec::new();
                }
                let d_ii_sq =
                    (lib.create_buffer)(self.ctx, 1 << 2, out_sq_bytes, ptr::null_mut(), &mut err);
                if err != CL_SUCCESS {
                    (lib.release_mem_object)(d_ii);
                    return Vec::new();
                }
                let d_feat = (lib.create_buffer)(
                    self.ctx,
                    1 << 2,
                    feature_bytes.max(1),
                    ptr::null_mut(),
                    &mut err,
                );
                if err != CL_SUCCESS {
                    (lib.release_mem_object)(d_ii);
                    (lib.release_mem_object)(d_ii_sq);
                    return Vec::new();
                }
                let d_feat_off = (lib.create_buffer)(
                    self.ctx,
                    1 << 2,
                    feature_offset_bytes.max(1),
                    ptr::null_mut(),
                    &mut err,
                );
                if err != CL_SUCCESS {
                    (lib.release_mem_object)(d_ii);
                    (lib.release_mem_object)(d_ii_sq);
                    (lib.release_mem_object)(d_feat);
                    return Vec::new();
                }
                let d_weak = (lib.create_buffer)(
                    self.ctx,
                    1 << 2,
                    weak_bytes.max(1),
                    ptr::null_mut(),
                    &mut err,
                );
                if err != CL_SUCCESS {
                    (lib.release_mem_object)(d_ii);
                    (lib.release_mem_object)(d_ii_sq);
                    (lib.release_mem_object)(d_feat);
                    (lib.release_mem_object)(d_feat_off);
                    return Vec::new();
                }
                let d_stage_off = (lib.create_buffer)(
                    self.ctx,
                    1 << 2,
                    stage_offset_bytes.max(1),
                    ptr::null_mut(),
                    &mut err,
                );
                if err != CL_SUCCESS {
                    (lib.release_mem_object)(d_ii);
                    (lib.release_mem_object)(d_ii_sq);
                    (lib.release_mem_object)(d_feat);
                    (lib.release_mem_object)(d_feat_off);
                    (lib.release_mem_object)(d_weak);
                    return Vec::new();
                }
                let d_stage_thr = (lib.create_buffer)(
                    self.ctx,
                    1 << 2,
                    stage_threshold_bytes.max(1),
                    ptr::null_mut(),
                    &mut err,
                );
                if err != CL_SUCCESS {
                    (lib.release_mem_object)(d_ii);
                    (lib.release_mem_object)(d_ii_sq);
                    (lib.release_mem_object)(d_feat);
                    (lib.release_mem_object)(d_feat_off);
                    (lib.release_mem_object)(d_weak);
                    (lib.release_mem_object)(d_stage_off);
                    return Vec::new();
                }
                let d_count = (lib.create_buffer)(
                    self.ctx,
                    1 << 2,
                    out_count_bytes,
                    ptr::null_mut(),
                    &mut err,
                );
                if err != CL_SUCCESS {
                    (lib.release_mem_object)(d_ii);
                    (lib.release_mem_object)(d_ii_sq);
                    (lib.release_mem_object)(d_feat);
                    (lib.release_mem_object)(d_feat_off);
                    (lib.release_mem_object)(d_weak);
                    (lib.release_mem_object)(d_stage_off);
                    (lib.release_mem_object)(d_stage_thr);
                    return Vec::new();
                }
                let d_out =
                    (lib.create_buffer)(self.ctx, 1 << 2, out_xy_bytes, ptr::null_mut(), &mut err);
                if err != CL_SUCCESS {
                    (lib.release_mem_object)(d_ii);
                    (lib.release_mem_object)(d_ii_sq);
                    (lib.release_mem_object)(d_feat);
                    (lib.release_mem_object)(d_feat_off);
                    (lib.release_mem_object)(d_weak);
                    (lib.release_mem_object)(d_stage_off);
                    (lib.release_mem_object)(d_stage_thr);
                    (lib.release_mem_object)(d_count);
                    return Vec::new();
                }
                let zero: u32 = 0;
                (lib.enqueue_write_buffer)(
                    self.queue,
                    d_count,
                    CL_TRUE,
                    0,
                    out_count_bytes,
                    &zero as *const _ as *const _,
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                );
                let w1 = (lib.enqueue_write_buffer)(
                    self.queue,
                    d_ii,
                    CL_TRUE,
                    0,
                    out_size,
                    ii.as_ptr() as *const _,
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                );
                let w2 = (lib.enqueue_write_buffer)(
                    self.queue,
                    d_ii_sq,
                    CL_TRUE,
                    0,
                    out_sq_bytes,
                    ii_sq.as_ptr() as *const _,
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                );
                let wf = (lib.enqueue_write_buffer)(
                    self.queue,
                    d_feat,
                    CL_TRUE,
                    0,
                    feature_bytes,
                    feature_data.as_ptr() as *const _,
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                );
                let wfo = (lib.enqueue_write_buffer)(
                    self.queue,
                    d_feat_off,
                    CL_TRUE,
                    0,
                    feature_offset_bytes,
                    feature_offsets.as_ptr() as *const _,
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                );
                let ww = (lib.enqueue_write_buffer)(
                    self.queue,
                    d_weak,
                    CL_TRUE,
                    0,
                    weak_bytes,
                    weak_data.as_ptr() as *const _,
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                );
                let wso = (lib.enqueue_write_buffer)(
                    self.queue,
                    d_stage_off,
                    CL_TRUE,
                    0,
                    stage_offset_bytes,
                    stage_offsets.as_ptr() as *const _,
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                );
                let wst = (lib.enqueue_write_buffer)(
                    self.queue,
                    d_stage_thr,
                    CL_TRUE,
                    0,
                    stage_threshold_bytes,
                    stage_thresholds.as_ptr() as *const _,
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                );
                if w1 != CL_SUCCESS
                    || w2 != CL_SUCCESS
                    || wf != CL_SUCCESS
                    || wfo != CL_SUCCESS
                    || ww != CL_SUCCESS
                    || wso != CL_SUCCESS
                    || wst != CL_SUCCESS
                {
                    (lib.release_mem_object)(d_ii);
                    (lib.release_mem_object)(d_ii_sq);
                    (lib.release_mem_object)(d_feat);
                    (lib.release_mem_object)(d_feat_off);
                    (lib.release_mem_object)(d_weak);
                    (lib.release_mem_object)(d_stage_off);
                    (lib.release_mem_object)(d_stage_thr);
                    (lib.release_mem_object)(d_count);
                    (lib.release_mem_object)(d_out);
                    return Vec::new();
                }
                (lib.set_kernel_arg)(
                    self.kernel_detect_windows,
                    0,
                    std::mem::size_of::<ClMem>(),
                    &d_ii as *const _ as *const _,
                );
                (lib.set_kernel_arg)(
                    self.kernel_detect_windows,
                    1,
                    std::mem::size_of::<ClMem>(),
                    &d_ii_sq as *const _ as *const _,
                );
                (lib.set_kernel_arg)(
                    self.kernel_detect_windows,
                    2,
                    std::mem::size_of::<ClMem>(),
                    &d_feat as *const _ as *const _,
                );
                (lib.set_kernel_arg)(
                    self.kernel_detect_windows,
                    3,
                    std::mem::size_of::<ClMem>(),
                    &d_feat_off as *const _ as *const _,
                );
                (lib.set_kernel_arg)(
                    self.kernel_detect_windows,
                    4,
                    std::mem::size_of::<ClMem>(),
                    &d_weak as *const _ as *const _,
                );
                (lib.set_kernel_arg)(
                    self.kernel_detect_windows,
                    5,
                    std::mem::size_of::<ClMem>(),
                    &d_stage_off as *const _ as *const _,
                );
                (lib.set_kernel_arg)(
                    self.kernel_detect_windows,
                    6,
                    std::mem::size_of::<ClMem>(),
                    &d_stage_thr as *const _ as *const _,
                );
                (lib.set_kernel_arg)(
                    self.kernel_detect_windows,
                    7,
                    std::mem::size_of::<ClMem>(),
                    &d_count as *const _ as *const _,
                );
                (lib.set_kernel_arg)(
                    self.kernel_detect_windows,
                    8,
                    std::mem::size_of::<ClMem>(),
                    &d_out as *const _ as *const _,
                );
                let ww_arg = w as ClUint;
                let hh_arg = h as ClUint;
                let wwin_arg = cascade.window_w as ClUint;
                let hwin_arg = cascade.window_h as ClUint;
                let nstages = cascade.stages.len() as ClUint;
                let maxdet = max_detections as ClUint;
                (lib.set_kernel_arg)(
                    self.kernel_detect_windows,
                    9,
                    std::mem::size_of::<ClUint>(),
                    &ww_arg as *const _ as *const _,
                );
                (lib.set_kernel_arg)(
                    self.kernel_detect_windows,
                    10,
                    std::mem::size_of::<ClUint>(),
                    &hh_arg as *const _ as *const _,
                );
                (lib.set_kernel_arg)(
                    self.kernel_detect_windows,
                    11,
                    std::mem::size_of::<ClUint>(),
                    &wwin_arg as *const _ as *const _,
                );
                (lib.set_kernel_arg)(
                    self.kernel_detect_windows,
                    12,
                    std::mem::size_of::<ClUint>(),
                    &hwin_arg as *const _ as *const _,
                );
                (lib.set_kernel_arg)(
                    self.kernel_detect_windows,
                    13,
                    std::mem::size_of::<ClUint>(),
                    &nstages as *const _ as *const _,
                );
                (lib.set_kernel_arg)(
                    self.kernel_detect_windows,
                    14,
                    std::mem::size_of::<ClUint>(),
                    &maxdet as *const _ as *const _,
                );
                let global: [ClSize; 2] = [w as ClSize, h as ClSize];
                let err = (lib.enqueue_nd_range_kernel)(
                    self.queue,
                    self.kernel_detect_windows,
                    2,
                    ptr::null(),
                    global.as_ptr(),
                    ptr::null(),
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                );
                if err != CL_SUCCESS {
                    (lib.release_mem_object)(d_ii);
                    (lib.release_mem_object)(d_ii_sq);
                    (lib.release_mem_object)(d_feat);
                    (lib.release_mem_object)(d_feat_off);
                    (lib.release_mem_object)(d_weak);
                    (lib.release_mem_object)(d_stage_off);
                    (lib.release_mem_object)(d_stage_thr);
                    (lib.release_mem_object)(d_count);
                    (lib.release_mem_object)(d_out);
                    return Vec::new();
                }
                let mut count: u32 = 0;
                (lib.enqueue_read_buffer)(
                    self.queue,
                    d_count,
                    CL_TRUE,
                    0,
                    out_count_bytes,
                    &mut count as *mut _ as *mut _,
                    0,
                    ptr::null(),
                    ptr::null_mut(),
                );
                let actual = (count as usize).min(max_detections);
                let mut xy = vec![0u32; actual * 3];
                let read_bytes = (actual * 12) as ClSize;
                if actual > 0 {
                    (lib.enqueue_read_buffer)(
                        self.queue,
                        d_out,
                        CL_TRUE,
                        0,
                        read_bytes,
                        xy.as_mut_ptr() as *mut _,
                        0,
                        ptr::null(),
                        ptr::null_mut(),
                    );
                }
                (lib.release_mem_object)(d_ii);
                (lib.release_mem_object)(d_ii_sq);
                (lib.release_mem_object)(d_feat);
                (lib.release_mem_object)(d_feat_off);
                (lib.release_mem_object)(d_weak);
                (lib.release_mem_object)(d_stage_off);
                (lib.release_mem_object)(d_stage_thr);
                (lib.release_mem_object)(d_count);
                (lib.release_mem_object)(d_out);
                (lib.finish)(self.queue);
                let mut out = Vec::with_capacity(actual);
                for chunk in xy.chunks(3) {
                    if chunk.len() < 3 {
                        break;
                    }
                    let x = chunk[0];
                    let y = chunk[1];
                    let bits = chunk[2];
                    let score = f32::from_bits(bits);
                    out.push(super::GpuDetection {
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
    }

    fn cpu_fallback_dual(img: &GrayImage) -> (Vec<u32>, Vec<u64>) {
        // Delegate to the pooled, fused integral builders in `integral.rs`
        // — same arithmetic as the previous inline loop (bit-identical
        // tables), but the buffers are recycled and the row loops are
        // hoisted. Note: the builders' Drop returns the buffers to the
        // thread-local pool, so we must extract the raw Vecs *before* drop
        // by using `from_owned`'s inverse — simplest is to just re-wrap via
        // `into`-style move: construct, then steal the data with
        // `IntegralImage::from_owned`'s counterpart. Since there is none,
        // build and immediately convert by moving through the same math.
        let ii = crate::integral::IntegralImage::from_gray(img);
        let sq = crate::integral::SquaredIntegralImage::from_gray(img);
        // Extract the backing buffers without running the pool-recycling
        // Drop (the caller of the fallback wants owned Vecs).
        let out = ii.into_data();
        let out_sq = sq.into_data();
        (out, out_sq)
    }

    impl Drop for Context {
        fn drop(&mut self) {
            unsafe {
                if let Some(lib) = load() {
                    (lib.release_kernel)(self.kernel_variance);
                    (lib.release_kernel)(self.kernel_dual_col);
                    (lib.release_kernel)(self.kernel_dual_row);
                    (lib.release_kernel)(self.kernel_integral);
                    (lib.release_program)(self.program);
                    (lib.release_command_queue)(self.queue);
                    (lib.release_context)(self.ctx);
                }
            }
        }
    }
}
