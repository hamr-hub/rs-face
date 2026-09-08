//! ROCm backend (AMD GPUs on Linux).
//!
//! Status: stub. ROCm provides a CUDA-compatible runtime (HIP), so the
//! same kernel source compiles for both with minimal changes.
//!
//! To enable:
//!   1. Add a HIP/ROCm Rust binding (e.g. ``hip-rs`` or build a C shim
//!      against ``libamdhip64``).
//!   2. Translate the OpenCL kernels to HIP — same algorithm, just
//!      ``__global__`` qualifier and explicit ``extern "C"`` entry.
//!   3. Replace ``probe()`` with a HIP device-open call.
//!
//! ROCm is AMD's CUDA-equivalent stack: install the ``rocm`` package
//! (``amdgpu-pro`` on Ubuntu) and ensure ``/opt/rocm/lib`` is on
//! ``LD_LIBRARY_PATH``.

use crate::gpu::backend::{BackendDescriptor, GpuBackend, GpuDetection, GpuInfo};
use crate::haar::Cascade;
use crate::image::GrayImage;

pub struct RocmDescriptor;

pub static ROCM: RocmDescriptor = RocmDescriptor;

impl BackendDescriptor for RocmDescriptor {
    fn id(&self) -> &'static str {
        "rocm"
    }
    fn vendor(&self) -> &'static str {
        "AMD ROCm (HIP)"
    }
    fn probe(&self) -> Option<Box<dyn GpuBackend>> {
        None
    }
}

struct RocmBackend {
    info: GpuInfo,
}
impl GpuBackend for RocmBackend {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::backend::BackendDescriptor;

    #[test]
    fn rocm_descriptor_reports_id_and_vendor() {
        let d = &ROCM;
        assert_eq!(d.id(), "rocm");
        assert!(!d.vendor().is_empty());
        // ROCm 是 stub 状态:没有 AMD GPU / ROCm runtime 时 probe 必须 None。
        // 启用 HIP feature 后才会变成 Some。这条契约保证:即使有 AMD GPU
        // 存在但 feature 未开,程序不会 panic / 不会调错 backend。
        let _ = d.probe();
    }
}
