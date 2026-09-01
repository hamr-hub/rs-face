//! ONNX inference abstraction, and the concrete backends behind it.
//!
//! # Why an abstraction at all
//!
//! The accuracy-critical logic of this crate — letterboxing, anchor decoding, landmark
//! alignment, normalisation constants — lives in [`crate::scrfd`] and [`crate::arcface`]
//! as pure functions over `&[f32]`. This module supplies only the missing piece: a
//! forward pass. Keeping that boundary sharp means the parts most likely to harbour a
//! silent accuracy bug are testable without any runtime installed, and swapping runtimes
//! cannot change numerical behaviour.
//!
//! # Backends
//!
//! Two are offered because neither dominates for every deployment:
//!
//! | Backend | Feature | GPU | Deps | Use when |
//! |---|---|---|---|---|
//! | ONNX Runtime | `ort-backend` | CoreML / CUDA / TensorRT / DirectML | C++ runtime | Throughput matters |
//! | tract | `tract-backend` | none | pure Rust | Static binary, cross-compile, no C++ |
//!
//! With no feature enabled the crate still builds with zero dependencies and the classical
//! Haar path is unaffected — [`Backend::available`] simply reports an empty list, and
//! constructing a session returns [`OnnxError::NoBackend`] rather than failing to compile.

use crate::models::{verify_bytes, Integrity, ModelSpec};
use core::fmt;
use std::path::{Path, PathBuf};

/// A tensor returned from a forward pass: flat `f32` data plus its shape.
///
/// Flat rather than a typed n-dimensional array because every consumer in this crate
/// indexes it linearly anyway, and it avoids leaking a backend's array type into the
/// public API — which would make the two backends mutually incompatible.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }

    /// Total element count implied by the shape.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Whether `shape` actually describes `data`.
    ///
    /// Worth checking explicitly: a shape/data disagreement means the backend reported
    /// something we misunderstood, and downstream slicing would read the wrong values
    /// rather than fail.
    pub fn is_consistent(&self) -> bool {
        self.shape.is_empty() || self.numel() == self.data.len()
    }

    /// Trailing dimension, i.e. the per-prediction width of an SCRFD head
    /// (1 for scores, 4 for boxes, 10 for keypoints).
    pub fn last_dim(&self) -> usize {
        self.shape.last().copied().unwrap_or(0)
    }
}

/// Errors from loading or running a model.
#[derive(Debug)]
pub enum OnnxError {
    /// The crate was built without any inference backend feature.
    NoBackend,
    /// Model file missing on disk.
    ModelNotFound(PathBuf),
    /// Failed to read the model file.
    Io(std::io::Error),
    /// The file failed its pinned integrity check.
    Integrity(Integrity),
    /// The backend rejected the graph or the run.
    Backend(String),
    /// The graph's shape does not match what this crate expects.
    UnexpectedShape(String),
}

impl fmt::Display for OnnxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OnnxError::NoBackend => write!(
                f,
                "no ONNX backend compiled in; rebuild with --features ort-backend (GPU-capable, \
                 needs a C++ runtime) or --features tract-backend (pure Rust, CPU only)"
            ),
            OnnxError::ModelNotFound(p) => write!(
                f,
                "model file not found: {}; run tools/fetch_models.sh to download it",
                p.display()
            ),
            OnnxError::Io(e) => write!(f, "reading model: {e}"),
            OnnxError::Integrity(i) => write!(f, "model integrity check failed: {i}"),
            OnnxError::Backend(m) => write!(f, "inference backend error: {m}"),
            OnnxError::UnexpectedShape(m) => write!(f, "unexpected model shape: {m}"),
        }
    }
}

impl std::error::Error for OnnxError {}

impl From<std::io::Error> for OnnxError {
    fn from(e: std::io::Error) -> Self {
        OnnxError::Io(e)
    }
}

/// Which inference backend to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Microsoft ONNX Runtime via the `ort` crate.
    Ort,
    /// Pure-Rust `tract`.
    Tract,
}

impl Backend {
    /// Backends compiled into this binary, in preference order (fastest first).
    ///
    /// Returned as a `Vec` rather than a compile-time array so callers can report the
    /// build's real capability to users, which is the difference between a comprehensible
    /// "rebuild with --features ort-backend" and a baffling empty result set.
    pub fn available() -> Vec<Backend> {
        let mut v = Vec::new();
        #[cfg(feature = "ort-backend")]
        v.push(Backend::Ort);
        #[cfg(feature = "tract-backend")]
        v.push(Backend::Tract);
        v
    }

    /// The best available backend, or `None` in a zero-dep build.
    pub fn preferred() -> Option<Backend> {
        Self::available().into_iter().next()
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Backend::Ort => "ort",
            Backend::Tract => "tract",
        }
    }

    /// Whether this backend can offload to a GPU.
    pub fn supports_gpu(&self) -> bool {
        matches!(self, Backend::Ort)
    }
}

/// Hardware target for a session.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Device {
    /// Try the best available accelerator, silently falling back to CPU.
    #[default]
    Auto,
    Cpu,
    /// Apple CoreML (Neural Engine / GPU).
    CoreMl,
    /// NVIDIA CUDA, by device index.
    Cuda(usize),
    /// NVIDIA TensorRT, by device index.
    TensorRt(usize),
    /// Microsoft DirectML.
    DirectMl,
}

impl Device {
    pub fn as_str(&self) -> &'static str {
        match self {
            Device::Auto => "auto",
            Device::Cpu => "cpu",
            Device::CoreMl => "coreml",
            Device::Cuda(_) => "cuda",
            Device::TensorRt(_) => "tensorrt",
            Device::DirectMl => "directml",
        }
    }
}

/// Session construction options.
#[derive(Clone, Debug)]
pub struct SessionConfig {
    pub backend: Option<Backend>,
    pub device: Device,
    /// Intra-op threads. `None` lets the backend decide.
    ///
    /// Relevant because this crate already runs a multi-threaded frame pipeline: letting
    /// each of N detector workers spawn its own intra-op pool oversubscribes the machine
    /// and can be slower than single-threaded inference. Callers running the pipeline
    /// should set this to 1.
    pub threads: Option<usize>,
    /// Verify the pinned digest before loading. Should stay on outside benchmarks.
    pub verify_integrity: bool,
    /// Concrete NCHW input shape, when the caller knows it.
    ///
    /// Needed by the pure-Rust `tract` backend, whose optimiser cannot resolve a graph
    /// with dynamic axes and so must be told the shape at load time. Callers such as
    /// [`crate::scrfd_detector::ScrfdDetector`] always know their own input resolution, so
    /// they supply it here rather than forcing the backend to guess — a guess would
    /// surface as an opaque shape error from deep inside tract.
    ///
    /// `None` means "infer from the ModelSpec"; the `ort` backend ignores this entirely
    /// because ONNX Runtime resolves shapes at run time.
    pub input_shape: Option<Vec<usize>>,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            backend: None,
            device: Device::Auto,
            threads: None,
            verify_integrity: true,
            input_shape: None,
        }
    }
}

impl SessionConfig {
    pub fn with_backend(mut self, b: Backend) -> Self {
        self.backend = Some(b);
        self
    }
    pub fn with_device(mut self, d: Device) -> Self {
        self.device = d;
        self
    }
    pub fn with_threads(mut self, n: usize) -> Self {
        self.threads = Some(n);
        self
    }
    /// Declare the concrete NCHW input shape. See [`SessionConfig::input_shape`].
    pub fn with_input_shape(mut self, shape: &[usize]) -> Self {
        self.input_shape = Some(shape.to_vec());
        self
    }

    pub fn without_integrity_check(mut self) -> Self {
        self.verify_integrity = false;
        self
    }
}

/// A loaded model that can run a forward pass.
///
/// `Send` so a session can be moved onto a pipeline worker thread. Deliberately **not**
/// `Sync`: ONNX Runtime sessions carry mutable internal arenas, and sharing one across
/// threads is a data race the type system should forbid rather than something a comment
/// asks callers to avoid. Give each worker its own session.
pub trait Inference: Send {
    /// Run the model on a single NCHW `f32` input.
    fn run(&self, input: &[f32], shape: &[usize]) -> Result<Vec<Tensor>, OnnxError>;

    /// Number of output tensors the graph exposes.
    ///
    /// Used by [`crate::scrfd::probe_outputs`] to detect a keypoint head from the loaded
    /// graph rather than trusting a filename.
    fn num_outputs(&self) -> usize;

    fn backend(&self) -> Backend;

    /// The device the session actually initialised on, which may differ from the request
    /// when an accelerator was unavailable and CPU fallback occurred.
    fn device(&self) -> Device;
}

/// Read a model file, enforcing its pinned integrity check.
///
/// Separated from session construction so both backends share exactly one integrity
/// policy — a backend that forgot to verify would otherwise be a silent regression.
pub fn read_verified(path: &Path, spec: Option<&ModelSpec>) -> Result<Vec<u8>, OnnxError> {
    if !path.exists() {
        return Err(OnnxError::ModelNotFound(path.to_path_buf()));
    }
    let bytes = std::fs::read(path)?;
    if let Some(spec) = spec {
        let integrity = verify_bytes(spec, &bytes);
        if !integrity.is_usable() {
            return Err(OnnxError::Integrity(integrity));
        }
        if matches!(integrity, Integrity::Unpinned) {
            // Not fatal — a user may legitimately supply their own export — but it must
            // not pass unremarked, since "verified" is the whole point of the registry.
            eprintln!(
                "[rs-face] WARNING: {} has no pinned SHA-256; integrity NOT verified",
                spec.id
            );
        }
    }
    Ok(bytes)
}

/// Open a session for `path` using the configured (or best available) backend.
///
/// Returns [`OnnxError::NoBackend`] in a zero-dependency build. That is a runtime error
/// rather than a compile error on purpose: the default build must keep working for the
/// classical detectors, and the message tells the user exactly which feature to add.
pub fn open_session(
    path: &Path,
    spec: Option<&ModelSpec>,
    cfg: &SessionConfig,
) -> Result<Box<dyn Inference>, OnnxError> {
    let chosen = match cfg.backend {
        Some(b) => {
            if !Backend::available().contains(&b) {
                return Err(OnnxError::Backend(format!(
                    "backend '{}' requested but not compiled in; available: {:?}",
                    b.as_str(),
                    Backend::available()
                        .iter()
                        .map(|b| b.as_str())
                        .collect::<Vec<_>>()
                )));
            }
            b
        }
        None => Backend::preferred().ok_or(OnnxError::NoBackend)?,
    };

    match chosen {
        #[cfg(feature = "ort-backend")]
        Backend::Ort => ort_backend::OrtSession::open(path, spec, cfg)
            .map(|s| Box::new(s) as Box<dyn Inference>),
        #[cfg(feature = "tract-backend")]
        Backend::Tract => tract_backend::TractSession::open(path, spec, cfg)
            .map(|s| Box::new(s) as Box<dyn Inference>),
        #[allow(unreachable_patterns)]
        _ => Err(OnnxError::NoBackend),
    }
}

#[cfg(feature = "ort-backend")]
pub mod ort_backend;

#[cfg(feature = "tract-backend")]
pub mod tract_backend;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_consistency_detects_shape_data_mismatch() {
        assert!(Tensor::new(vec![0.0; 12], vec![1, 3, 4]).is_consistent());
        assert!(!Tensor::new(vec![0.0; 11], vec![1, 3, 4]).is_consistent());
    }

    #[test]
    fn tensor_numel_and_last_dim() {
        let t = Tensor::new(vec![0.0; 40], vec![1, 10, 4]);
        assert_eq!(t.numel(), 40);
        assert_eq!(t.last_dim(), 4, "SCRFD box head width");
        assert_eq!(Tensor::new(vec![], vec![]).last_dim(), 0);
    }

    #[test]
    fn backend_availability_matches_enabled_features() {
        let avail = Backend::available();
        #[cfg(feature = "ort-backend")]
        assert!(avail.contains(&Backend::Ort));
        #[cfg(not(feature = "ort-backend"))]
        assert!(!avail.contains(&Backend::Ort));
        #[cfg(feature = "tract-backend")]
        assert!(avail.contains(&Backend::Tract));
        #[cfg(not(feature = "tract-backend"))]
        assert!(!avail.contains(&Backend::Tract));
    }

    #[test]
    fn zero_dep_build_reports_no_backend_rather_than_failing_to_compile() {
        // The central promise of the feature gating: a default build still works.
        #[cfg(not(any(feature = "ort-backend", feature = "tract-backend")))]
        {
            assert!(Backend::preferred().is_none());
            // `Box<dyn Inference>` is not Debug, so unwrap_err() is unavailable here.
            let err = open_session(
                Path::new("/nonexistent.onnx"),
                None,
                &SessionConfig::default(),
            )
            .err()
            .expect("a zero-dep build must refuse to open a session");
            assert!(matches!(err, OnnxError::NoBackend));
        }
        #[cfg(any(feature = "ort-backend", feature = "tract-backend"))]
        assert!(Backend::preferred().is_some());
    }

    #[test]
    fn no_backend_error_names_the_features_to_enable() {
        // Users hit this first; the message must be actionable.
        let msg = OnnxError::NoBackend.to_string();
        assert!(msg.contains("ort-backend"), "got {msg}");
        assert!(msg.contains("tract-backend"), "got {msg}");
    }

    #[test]
    fn model_not_found_error_points_at_the_fetch_script() {
        let msg = OnnxError::ModelNotFound(PathBuf::from("/tmp/x.onnx")).to_string();
        assert!(msg.contains("/tmp/x.onnx"));
        assert!(msg.contains("fetch_models"), "got {msg}");
    }

    #[test]
    fn only_ort_claims_gpu_support() {
        assert!(Backend::Ort.supports_gpu());
        assert!(!Backend::Tract.supports_gpu(), "tract is CPU-only");
    }

    #[test]
    fn requesting_an_uncompiled_backend_is_a_clear_error() {
        // Pick a backend that is definitely not compiled in.
        let absent = if Backend::available().contains(&Backend::Ort) {
            Backend::Tract
        } else {
            Backend::Ort
        };
        if Backend::available().contains(&absent) {
            return; // both compiled in; nothing to assert
        }
        let cfg = SessionConfig::default().with_backend(absent);
        let err = open_session(Path::new("/nonexistent.onnx"), None, &cfg)
            .err()
            .expect("an uncompiled backend must be refused");
        match err {
            OnnxError::Backend(m) => assert!(m.contains("not compiled in"), "got {m}"),
            OnnxError::NoBackend => {}
            other => panic!("expected a backend error, got {other:?}"),
        }
    }

    #[test]
    fn read_verified_reports_missing_file() {
        let err = read_verified(Path::new("/definitely/not/here.onnx"), None).unwrap_err();
        assert!(matches!(err, OnnxError::ModelNotFound(_)));
    }

    #[test]
    fn read_verified_rejects_a_corrupt_file() {
        use crate::models::{License, ModelKind};
        let dir = std::env::temp_dir();
        let path = dir.join("rsface_test_corrupt.onnx");
        std::fs::write(&path, b"not the real weights").unwrap();

        let spec = ModelSpec {
            id: "test",
            kind: ModelKind::Detector,
            url: "",
            file_name: "test.onnx",
            // Digest of "abc", which this file is not.
            sha256: Some("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
            size_bytes: None,
            license: License::Permissive("MIT"),
            input_size: 640,
            accuracy: "",
        };
        let err = read_verified(&path, Some(&spec)).unwrap_err();
        assert!(
            matches!(err, OnnxError::Integrity(Integrity::Mismatch { .. })),
            "corrupt weights must be refused, got {err:?}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn read_verified_accepts_matching_digest() {
        use crate::models::{License, ModelKind};
        let path = std::env::temp_dir().join("rsface_test_good.onnx");
        std::fs::write(&path, b"abc").unwrap();
        let spec = ModelSpec {
            id: "test",
            kind: ModelKind::Detector,
            url: "",
            file_name: "test.onnx",
            sha256: Some("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
            size_bytes: Some(3),
            license: License::Permissive("MIT"),
            input_size: 640,
            accuracy: "",
        };
        assert_eq!(read_verified(&path, Some(&spec)).unwrap(), b"abc");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn session_config_builders_compose() {
        let c = SessionConfig::default()
            .with_device(Device::Cuda(1))
            .with_threads(1)
            .without_integrity_check();
        assert_eq!(c.device, Device::Cuda(1));
        assert_eq!(c.threads, Some(1));
        assert!(!c.verify_integrity);
    }

    #[test]
    fn integrity_check_defaults_to_on() {
        // A default that skipped verification would defeat the registry.
        assert!(SessionConfig::default().verify_integrity);
    }

    #[test]
    fn device_auto_is_the_default() {
        assert_eq!(SessionConfig::default().device, Device::Auto);
        assert_eq!(Device::default(), Device::Auto);
    }
}
