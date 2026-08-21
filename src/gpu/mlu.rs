//! Cambricon MLU backend.
//!
//! Status: stub. Cambricon's MLU accelerators use the BANG C dialect and
//! the CNRT runtime; the latter exposes a queue / device model similar
//! enough to CUDA that the same source code can be ported with modest
//! effort.
//!
//! To enable:
//!   1. Install the Cambricon NeuWare SDK on the target host.
//!   2. Link against ``libcnrt`` via a Rust binding or a C shim.
//!   3. Translate the OpenCL kernels to BANG C — same algorithm, but
//!      work-item indexing uses CNRT's NRAM / SRAM memory hierarchy.
//!   4. Replace ``probe()`` with a ``cnrtInit`` + device-count call.
//!
//! Until wired up, ``probe()`` returns ``None`` and the dispatcher skips
//! to the OpenCL path.

use crate::gpu::backend::{BackendDescriptor, GpuBackend, GpuDetection, GpuInfo};
use crate::haar::Cascade;
use crate::image::GrayImage;

pub struct MluDescriptor;

pub static MLU: MluDescriptor = MluDescriptor;

impl BackendDescriptor for MluDescriptor {
    fn id(&self) -> &'static str { "mlu" }
    fn vendor(&self) -> &'static str { "Cambricon MLU (BANG C)" }
    fn probe(&self) -> Option<Box<dyn GpuBackend>> { None }
}

struct MluBackend { info: GpuInfo }
impl GpuBackend for MluBackend {
    fn info(&self) -> &GpuInfo { &self.info }
    fn variance_prefilter(&self, img: &GrayImage, _w: usize, _h: usize, s: usize, _t: u64) -> Vec<u8> {
        let nx = (img.width() + s - 1) / s;
        let ny = (img.height() + s - 1) / s;
        vec![1u8; nx * ny]
    }
    fn detect_windows(&self, _: &Cascade, _: &GrayImage, _: usize) -> Vec<GpuDetection> { Vec::new() }
}