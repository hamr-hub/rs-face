//! Huawei Ascend backend (CANN runtime).
//!
//! Status: stub. Ascend is Huawei's domestic accelerator line. The runtime
//! is CANN (Compute Architecture for Neural Networks), exposing a
//! "AscendCL" C API that mirrors the OpenCL / CUDA command-queue model.
//!
//! To enable:
//!   1. Install the CANN toolkit on the host
//!      (``Ascend-cann-toolkit_<ver>_linux-<arch>.run``).
//!   2. Link against ``libascendcl`` via a Rust binding crate or a small
//!      ``build.rs`` shim. The ``acl`` crate family is the canonical
//!      choice but not published on crates.io — most users vendor it.
//!   3. Translate the OpenCL kernels to Ascend's "Ascend C" dialect:
//!      same algorithm; replace ``__global`` with ``__gm__`` and use
//!      Ascend's pipe-style work-item indexing.
//!   4. Replace ``probe()`` with an ``aclInit`` + ``aclrtGetDeviceCount`` call.
//!
//! Until wired up, ``probe()`` returns ``None`` and the dispatcher skips
//! to the OpenCL path.

use crate::gpu::backend::{BackendDescriptor, GpuBackend, GpuDetection, GpuInfo};
use crate::haar::Cascade;
use crate::image::GrayImage;

pub struct AscendDescriptor;

pub static ASCEND: AscendDescriptor = AscendDescriptor;

impl BackendDescriptor for AscendDescriptor {
    fn id(&self) -> &'static str {
        "ascend"
    }
    fn vendor(&self) -> &'static str {
        "Huawei Ascend (CANN)"
    }
    fn probe(&self) -> Option<Box<dyn GpuBackend>> {
        None
    }
}

struct AscendBackend {
    info: GpuInfo,
}
impl GpuBackend for AscendBackend {
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
