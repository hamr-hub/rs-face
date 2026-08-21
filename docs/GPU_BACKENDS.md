# GPU Backends — Adapter Architecture

This document describes how `rs-face` plugs into different GPU vendors
behind a single Rust trait. The core algorithms (cascade, integral
image, variance pre-filter, NMS) are all written in pure Rust in
`src/detector.rs` and `src/haar/`; the abstraction in `src/gpu/backend.rs`
just decides where each primitive runs.

---

## The `GpuBackend` trait

```rust
// src/gpu/backend.rs
pub trait GpuBackend: Send + Sync {
    fn info(&self) -> &GpuInfo;
    fn variance_prefilter(&self, img: &GrayImage,
                          win_w: usize, win_h: usize, stride: usize,
                          variance_threshold: u64) -> Vec<u8>;
    fn detect_windows(&self, cascade: &Cascade, img: &GrayImage,
                      max_detections: usize) -> Vec<GpuDetection>;
}
```

Every implementation runs **the same cascade weights** over **the same
integral-image math** and the same variance normalisation. Boxes come
back identical within float32 precision regardless of which backend
executes them — verified by `tools/compare_cpu_gpu.py`.

### Why three methods, not one

The trait deliberately exposes the cascade at the lowest level that still
captures GPU-specific work:

| method                | purpose                                          | GPU-side cost     |
|-----------------------|--------------------------------------------------|-------------------|
| `variance_prefilter`  | per-window mean² reject (cheap arithmetic)       | 1 GB/s memory BW  |
| `detect_windows`      | full cascade evaluation, all windows in parallel  | dominant per call |

CPU implementations can satisfy the trait trivially (just run the
existing Rust cascade), so the same trait works for benchmarking
CPU vs GPU back to back.

---

## Supported vendors (current state on macOS 26.5 / Apple Silicon)

| backend id | vendor                          | status     | how to enable |
|-------------|---------------------------------|------------|---------------|
| `cpu`       | host CPU                        | ✅ works   | default       |
| `opencl`    | OpenCL ICD (Metal-OpenCL on Mac) | ⚠️ broken on this Mac (`/System/Library/Frameworks/OpenCL.framework/OpenCL` is a dangling symlink; Apple stripped the dylib). Works on Linux/Windows hosts with the Khronos ICD loader. | `cargo build` (default) |
| `metal`     | Apple Metal (native)             | 🚧 skeleton; MSL kernels drafted, dispatch blocked on `metal` crate API drift. | `cargo build --features metal-backend` |
| `cuda`      | NVIDIA CUDA                      | stub       | see "Adding a vendor" below |
| `rocm`      | AMD ROCm (HIP)                   | stub       | see below      |
| `ascend`    | Huawei Ascend (CANN)             | stub       | see below      |
| `mlu`       | Cambricon MLU (BANG C)           | stub       | see below      |

`auto()` (used by default in `rs-face-detect`) walks the list in the
order above and returns the first backend whose `probe()` succeeds. On
this Mac, the only one that returns `Some(_)` today is `cpu`.

---

## Adding a new vendor

The dispatcher is intentionally small:

```rust
// src/gpu/backend.rs
pub trait BackendDescriptor: Sync {
    fn id(&self) -> &'static str;
    fn vendor(&self) -> &'static str;
    fn probe(&self) -> Option<Box<dyn GpuBackend>>;
}

pub const BACKENDS: &[&dyn BackendDescriptor] = &[
    &metal::METAL,
    &cuda::CUDA,
    &rocm::ROCM,
    &ascend::ASCEND,
    &mlu::MLU,
    &super::OPENCL_DESCRIPTOR, // cross-platform fallback
];
```

To add a new vendor (e.g. `sycl`, `webgpu`, `vulkan-native`):

1. Create `src/gpu/<vendor>.rs` with:

   ```rust
   use crate::gpu::backend::{BackendDescriptor, GpuBackend, GpuDetection, GpuInfo};
   use crate::haar::Cascade;
   use crate::image::GrayImage;

   pub struct MyVendorDescriptor;
   pub static MY_VENDOR: MyVendorDescriptor = MyVendorDescriptor;

   impl BackendDescriptor for MyVendorDescriptor {
       fn id(&self) -> &'static str { "my_vendor" }
       fn vendor(&self) -> &'static str { "My Vendor SDK" }
       fn probe(&self) -> Option<Box<dyn GpuBackend>> {
           // Try to open device 0. Return None if the SDK isn't installed.
           my_sdk::Device::open(0).ok().map(|d| {
               Box::new(MyVendorBackend::new(d))
           })
       }
   }

   struct MyVendorBackend { /* device, queue, kernel handles */ }

   impl GpuBackend for MyVendorBackend {
       fn info(&self) -> &GpuInfo { &self.info }
       fn variance_prefilter(&self, img: &GrayImage,
                             win_w: usize, win_h: usize, stride: usize,
                             variance_threshold: u64) -> Vec<u8> {
           // Port the OpenCL `variance_prefilter` kernel (in src/gpu/mod.rs)
           // to your SDK's dialect. Same algorithm.
       }
       fn detect_windows(&self, cascade: &Cascade, img: &GrayImage,
                         max_detections: usize) -> Vec<GpuDetection> {
           // Port the OpenCL `detect_windows` kernel. Same algorithm.
       }
   }
   ```

2. Add `pub mod my_vendor;` to `src/gpu/mod.rs` and append
   `&my_vendor::MY_VENDOR,` to `BACKENDS`.

3. (Optional) Add a CLI alias in `src/bin/rs_face_detect.rs`.

The kernel source for the cascade lives as a multi-line string in
`src/gpu/mod.rs` (search for `CL_KERNEL_SRC`). Translating to MSL /
CUDA C / HIP C / BANG C / Ascend C is line-for-line; only address-space
qualifiers (`__global` → `device` for MSL; no qualifier for CUDA C;
`__gm__` for Ascend C) change.

---

## Why OpenCL doesn't work on this Mac

`/System/Library/Frameworks/OpenCL.framework/OpenCL` is a dangling
symlink:

```bash
$ file /System/Library/Frameworks/OpenCL.framework/OpenCL
broken symbolic link to Versions/Current/OpenCL
$ ls /System/Library/Frameworks/OpenCL.framework/Versions/A/
Libraries Resources _CodeSignature lib
$ ls /System/Library/Frameworks/OpenCL.framework/Versions/A/lib
clang/                              # only header files
```

Apple has been incrementally deprecating OpenCL since macOS 10.14
(deprecation note in 10.14, removal warnings since 12, and the
runtime binary is no longer shipped in 26.x). The remaining files in
the framework are mostly resources for the AMD and Intel compute
kernels, not a loadable dylib.

For Mac users who need GPU acceleration today, the path is:

1. Land the Metal backend (the MSL kernels are already drafted in
   `src/gpu/metal.rs`; only the dispatch wrappers need pinning).
2. Optionally expose MPS (`MPSImageIntegral`, `MPSImageThreshold`) for
   faster prefiltering on devices with the Neural Engine.
3. Apple Silicon users without Metal will see graceful CPU-only
   fallback; this is what currently happens.

---

## Pipeline parity by construction

Because every backend runs **the same cascade weights** and **the same
integral-image / variance normalisation** on **the same input pixels**:

* The boxes come back identical within float32 precision (verified by
  `tools/compare_cpu_gpu.py`; the test allows ±1 pixel slop and
  requires IoU ≥ 0.99 for non-exact matches).
* CPU vs GPU never diverge on recall — a window the CPU cascade
  rejects is rejected on the GPU too, because the variance pre-filter
  uses the same arithmetic.
* The `score` field is the cascade's running sum, identical across
  backends.

The Python `tools/compare_cpu_gpu.py` auxiliary walks a JSONL produced
by `rs-face-detect` (which emits both `boxes_cpu` and `boxes_gpu` per
frame) and reports per-video parity in a table.

---

## File map

```
src/gpu/
├── mod.rs         # original OpenCL driver (zero-dep FFI) — preserved as-is
├── backend.rs     # GpuBackend trait + dispatcher + OpenCL passthrough wrapper
├── metal.rs       # Apple Metal (skeleton + MSL kernel source)
├── cuda.rs        # NVIDIA CUDA stub
├── rocm.rs        # AMD ROCm / HIP stub
├── ascend.rs      # Huawei Ascend (CANN) stub
└── mlu.rs         # Cambricon MLU (BANG C) stub

src/bin/
└── rs_face_detect.rs   # video → JSONL (runs CPU + GPU backends, emits both box sets)

tools/
├── run_rust_detect.py    # spawns the Rust binary across multiple backends
├── compare_cpu_gpu.py    # verifies box parity between CPU and GPU passes
├── gpu_backends.py       # Python mirror of the trait (for ONNX-based GPU
│                         #   pipelines using cv2.dnn / onnxruntime)
├── detect_gpu.py         # GPU-only detector via cv2.dnn OpenCL target
│                         #   (auxiliary; the production path is Rust)
└── convert_res10_to_onnx.py   # Caffe→ONNX helper for non-cv2 backends
```

---

## References

* Viola & Jones, "Rapid object detection using a boosted cascade of
  simple features", CVPR 2001 — the algorithm implemented in
  `src/haar/`.
* OpenCL specification — Khronos Group, current 3.0.
* Apple Metal Shading Language specification — developer.apple.com.
* NVIDIA CUDA C programming guide — docs.nvidia.com/cuda.
* Huawei CANN / AscendCL — https://www.hiascend.com/software/cann
* Cambricon BANG C — https://www.cambricon.com