//! ONNX Runtime backend via the `ort` crate.
//!
//! This is the throughput-oriented path: ONNX Runtime has the broadest operator coverage
//! of any available runtime and is the only route to GPU acceleration
//! (CoreML on Apple, CUDA/TensorRT on NVIDIA, DirectML on Windows, ROCm on AMD).
//!
//! # Execution-provider fallback is deliberate, and reported
//!
//! Registering an execution provider can fail at runtime for reasons entirely outside the
//! program's control: no CUDA driver, an incompatible CoreML version, a machine with no
//! discrete GPU. ONNX Runtime's default behaviour is to fall back to CPU silently, which
//! turns "my GPU deployment is 40x slower than expected" into an unfalsifiable mystery.
//!
//! So this module registers providers one at a time, records which one actually took
//! effect, and exposes it through [`crate::onnx::Inference::device`]. Falling back is
//! still the right behaviour — a working slow pipeline beats a crash — but it is
//! *observable*.

use super::{read_verified, Backend, Device, Inference, OnnxError, SessionConfig, Tensor};
use crate::models::ModelSpec;
use ort::session::{builder::GraphOptimizationLevel, Session};
use std::cell::RefCell;
use std::path::Path;

/// A loaded ONNX Runtime session.
///
/// The session sits behind a `RefCell` because ONNX Runtime's `run` requires `&mut`
/// (it mutates internal allocator arenas) while our [`Inference`] trait exposes `&self`,
/// so that a session can be shared behind an `Arc` for read-only *configuration* queries.
/// `RefCell` rather than `Mutex` is correct here precisely because [`Inference`] is `Send`
/// but deliberately **not** `Sync`: each pipeline worker owns its own session, so there is
/// no cross-thread contention to lock against, and paying for a mutex on every frame would
/// be pure overhead. The borrow is confined to the body of `run`, so it can never overlap.
pub struct OrtSession {
    session: RefCell<Session>,
    num_outputs: usize,
    device: Device,
}

impl OrtSession {
    /// Load a model and initialise a session.
    pub fn open(
        path: &Path,
        spec: Option<&ModelSpec>,
        cfg: &SessionConfig,
    ) -> Result<Self, OnnxError> {
        let bytes = if cfg.verify_integrity {
            read_verified(path, spec)?
        } else {
            if !path.exists() {
                return Err(OnnxError::ModelNotFound(path.to_path_buf()));
            }
            std::fs::read(path)?
        };

        let mut builder = Session::builder()
            .map_err(|e| OnnxError::Backend(format!("creating session builder: {e}")))?
            // Level3 enables all graph fusions. Safe for inference-only use and worth a
            // significant fraction of throughput on the SCRFD backbone.
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| OnnxError::Backend(format!("setting optimization level: {e}")))?;

        if let Some(n) = cfg.threads {
            builder = builder
                .with_intra_threads(n)
                .map_err(|e| OnnxError::Backend(format!("setting intra threads: {e}")))?;
        }

        let (mut builder, device) = Self::register_providers(builder, cfg.device)?;

        let session = builder
            .commit_from_memory(&bytes)
            .map_err(|e| OnnxError::Backend(format!("loading graph: {e}")))?;

        let num_outputs = session.outputs().len();

        Ok(Self {
            session: RefCell::new(session),
            num_outputs,
            device,
        })
    }

    /// Attempt to register the requested execution provider, reporting what stuck.
    ///
    /// Registration failure is downgraded to CPU rather than propagated: a user asking for
    /// `Device::Auto` on a CPU-only box wants their program to run.
    fn register_providers(
        builder: ort::session::builder::SessionBuilder,
        requested: Device,
    ) -> Result<(ort::session::builder::SessionBuilder, Device), OnnxError> {
        // Candidate list in preference order. `Auto` tries whatever this build supports.
        // CUDA leads TensorRT: on shared-memory devices (8 GB Jetson Orin Nano)
        // the TRT engine build for a 174 MB R50 model OOMs, while CUDA EP runs
        // on the same device with no upfront build. Operators who have enough
        // VRAM can pass `RSFACE_ARCFACE_EP=tensorrt` (env path) or ask
        // explicitly via the platform to flip the preference.
        let candidates: Vec<Device> = match requested {
            Device::Auto => vec![
                #[cfg(feature = "ort-cuda")]
                Device::Cuda(0),
                #[cfg(feature = "ort-tensorrt")]
                Device::TensorRt(0),
                #[cfg(feature = "ort-coreml")]
                Device::CoreMl,
                #[cfg(feature = "ort-directml")]
                Device::DirectMl,
            ],
            Device::Cpu => vec![],
            explicit => vec![explicit],
        };

        // ort 2.0 RC API: build an EP into an `ExecutionProviderDispatch`,
        // then `builder.with_execution_providers(...)` to register. The
        // builder is consumed and returned by `with_execution_providers`,
        // so we thread it through every candidate.
        //
        // Preference order: CUDA first (no engine build, lighter VRAM),
        // then TensorRT (heavier upfront cost, faster steady-state). On
        // 8 GB shared-memory Jetsons the TRT engine build for 174 MB
        // R50 models can OOM, so CUDA is the safer automatic pick.
        let mut builder = builder;
        let mut chosen = Device::Cpu;
        for cand in candidates {
            let ep = match Self::build_ep(cand) {
                Some(ep) => ep,
                None => continue, // device not handled by this build
            };
            // Apply to a clone: on success, swap; on failure, keep the
            // original builder and report.
            let attempt = builder.clone().with_execution_providers([ep]);
            match attempt {
                Ok(b) => {
                    builder = b;
                    chosen = cand;
                    break;
                }
                Err(e) => {
                    if !matches!(requested, Device::Auto) {
                        eprintln!(
                            "[rs-face] WARNING: execution provider '{}' was requested but \
                             could not be registered ({e}); falling back to CPU. \
                             Inference will be substantially slower than expected.",
                            cand.as_str()
                        );
                    }
                }
            }
        }
        Ok((builder, chosen))
    }

    /// Build the dispatch for one device. `None` means this build does not compile
    /// in support for the requested device at all (no candidate to try).
    /// `.build()` is infallible in ort 2.0 RC; failures are deferred to
    /// `with_execution_providers()` at registration time.
    #[allow(unused_variables)]
    fn build_ep(device: Device) -> Option<ort::ep::ExecutionProviderDispatch> {
        use ort::ep::ExecutionProvider;
        match device {
            Device::Cpu | Device::Auto => None,

            #[cfg(feature = "ort-coreml")]
            Device::CoreMl => Some(ort::execution_providers::CoreML::default().build()),

            #[cfg(feature = "ort-cuda")]
            Device::Cuda(idx) => {
                // Cap GPU memory at 1.5 GB so the system can keep its
                // 8 GB pool usable for OS + other workloads. R50 + SCRFD
                // activations fit comfortably; bigger models would need a
                // bigger cap and a host with more VRAM.
                const CUDA_MEM_LIMIT_BYTES: usize = 1536 * 1024 * 1024;
                Some(
                    ort::execution_providers::CUDA::default()
                        .with_device_id(idx as i32)
                        .with_memory_limit(CUDA_MEM_LIMIT_BYTES)
                        .build(),
                )
            }

            #[cfg(feature = "ort-tensorrt")]
            Device::TensorRt(idx) => Some(
                ort::execution_providers::TensorRT::default()
                    .with_device_id(idx as i32)
                    .build(),
            ),

            #[cfg(feature = "ort-directml")]
            Device::DirectMl => Some(ort::execution_providers::DirectML::default().build()),

            // Provider not compiled into this build.
            #[allow(unreachable_patterns)]
            _ => None,
        }
    }
}

impl Inference for OrtSession {
    fn run(&self, input: &[f32], shape: &[usize]) -> Result<Vec<Tensor>, OnnxError> {
        let expected: usize = shape.iter().product();
        if input.len() != expected {
            return Err(OnnxError::UnexpectedShape(format!(
                "input has {} elements but shape {:?} implies {}",
                input.len(),
                shape,
                expected
            )));
        }

        let dims: Vec<i64> = shape.iter().map(|&d| d as i64).collect();
        let value = ort::value::Tensor::from_array((dims, input.to_vec()))
            .map_err(|e| OnnxError::Backend(format!("building input tensor: {e}")))?;

        let mut session = self.session.borrow_mut();
        let outputs = session
            .run(ort::inputs![value])
            .map_err(|e| OnnxError::Backend(format!("running inference: {e}")))?;

        let mut result = Vec::with_capacity(outputs.len());
        for (_, v) in outputs.iter() {
            let (shape, data) = v
                .try_extract_tensor::<f32>()
                .map_err(|e| OnnxError::Backend(format!("extracting output: {e}")))?;
            result.push(Tensor::new(
                data.to_vec(),
                shape.iter().map(|&d| d as usize).collect(),
            ));
        }
        Ok(result)
    }

    fn num_outputs(&self) -> usize {
        self.num_outputs
    }

    fn backend(&self) -> Backend {
        Backend::Ort
    }

    fn device(&self) -> Device {
        self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opening_a_missing_model_is_a_clean_error() {
        let err = OrtSession::open(
            Path::new("/definitely/not/here.onnx"),
            None,
            &SessionConfig::default(),
        )
        .err()
        .expect("must fail");
        assert!(matches!(err, OnnxError::ModelNotFound(_)));
    }

    #[test]
    fn opening_a_non_onnx_file_fails_at_the_backend_not_by_panicking() {
        let path = std::env::temp_dir().join("rsface_ort_garbage.onnx");
        std::fs::write(&path, b"this is definitely not a protobuf graph").unwrap();
        let err = OrtSession::open(&path, None, &SessionConfig::default())
            .err()
            .expect("must fail");
        assert!(
            matches!(err, OnnxError::Backend(_)),
            "expected a backend error, got {err:?}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn cpu_device_registers_no_provider() {
        // Sanity: requesting CPU must not attempt any accelerator registration.
        let builder = Session::builder().unwrap();
        let (_, dev) = OrtSession::register_providers(builder, Device::Cpu).unwrap();
        assert_eq!(dev, Device::Cpu);
    }
}
