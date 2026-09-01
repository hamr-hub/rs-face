//! Pure-Rust inference backend via `tract`.
//!
//! This path exists to preserve something the crate genuinely had and would otherwise
//! lose: a build with **no C++ toolchain and no shared-library dependency**. That is not
//! ideology. It is what makes `cargo build --target aarch64-unknown-linux-musl` produce a
//! single static binary you can drop into a scratch container or an embedded rootfs, and
//! it removes an entire class of "works on my machine" deployment failure.
//!
//! The trade is real and should be stated plainly: tract is **CPU-only** and slower than
//! ONNX Runtime, and its operator coverage lags upstream ONNX.
//!
//! # Fixed input shapes
//!
//! tract's optimiser requires fully-resolved shapes, while SCRFD is commonly exported with
//! a dynamic batch/height/width. This module therefore *concretises* the input shape at
//! load time via `with_input_fact`, which is why [`TractSession::open_with_shape`] needs to
//! be told the shape up front. A dynamic-shape model that cannot be resolved fails here
//! with a clear message rather than at the first inference.

use super::{read_verified, Backend, Device, Inference, OnnxError, SessionConfig, Tensor};
use crate::models::ModelSpec;
use std::path::Path;
use tract_onnx::prelude::*;

/// Model type after optimisation: a runnable, shape-resolved plan.
type Plan = std::sync::Arc<TypedRunnableModel>;

/// A loaded tract session.
pub struct TractSession {
    plan: Plan,
    num_outputs: usize,
    input_shape: Vec<usize>,
}

impl TractSession {
    /// Load a model, inferring the input shape from the model spec.
    ///
    /// Falls back to a batch-1 NCHW square input of `spec.input_size`, which covers every
    /// model in this crate's registry.
    pub fn open(
        path: &Path,
        spec: Option<&ModelSpec>,
        cfg: &SessionConfig,
    ) -> Result<Self, OnnxError> {
        // Resolve the shape from the caller's explicit hint first, then the model spec.
        // Guessing would produce a confusing shape error from deep inside tract's
        // optimiser instead of a clear one here.
        if let Some(shape) = cfg.input_shape.clone() {
            return Self::open_with_shape(path, spec, cfg, &shape);
        }
        let size = spec.map(|s| s.input_size).ok_or_else(|| {
            OnnxError::UnexpectedShape(
                "the tract backend needs a known input size; set \
                 SessionConfig::with_input_shape, pass a ModelSpec, or use \
                 TractSession::open_with_shape"
                    .into(),
            )
        })?;
        Self::open_with_shape(path, spec, cfg, &[1, 3, size, size])
    }

    /// Load a model with an explicit input shape.
    pub fn open_with_shape(
        path: &Path,
        spec: Option<&ModelSpec>,
        cfg: &SessionConfig,
        shape: &[usize],
    ) -> Result<Self, OnnxError> {
        let bytes = if cfg.verify_integrity {
            read_verified(path, spec)?
        } else {
            if !path.exists() {
                return Err(OnnxError::ModelNotFound(path.to_path_buf()));
            }
            std::fs::read(path)?
        };

        let mut cursor = std::io::Cursor::new(&bytes);
        let model = tract_onnx::onnx()
            .model_for_read(&mut cursor)
            .map_err(|e| OnnxError::Backend(format!("parsing ONNX graph: {e}")))?;

        // Pin the input shape so the optimiser can resolve every intermediate fact.
        let model = model
            .with_input_fact(0, f32::fact(shape).into())
            .map_err(|e| {
                OnnxError::UnexpectedShape(format!(
                    "could not fix input shape to {shape:?}: {e}. The export may expect a \
                     different resolution, or may need onnxsim to remove dynamic axes."
                ))
            })?;

        let model = model
            .into_optimized()
            .map_err(|e| OnnxError::Backend(format!("optimising graph: {e}")))?;

        let num_outputs = model.outputs.len();

        let plan = model
            .into_runnable()
            .map_err(|e| OnnxError::Backend(format!("building runnable plan: {e}")))?;

        Ok(Self {
            plan,
            num_outputs,
            input_shape: shape.to_vec(),
        })
    }
}

impl Inference for TractSession {
    fn run(&self, input: &[f32], shape: &[usize]) -> Result<Vec<Tensor>, OnnxError> {
        if shape != self.input_shape.as_slice() {
            // tract plans are shape-specialised, so a mismatch cannot be absorbed.
            return Err(OnnxError::UnexpectedShape(format!(
                "this tract session was built for input shape {:?} but was given {:?}; \
                 open a new session for the new shape",
                self.input_shape, shape
            )));
        }
        let expected: usize = shape.iter().product();
        if input.len() != expected {
            return Err(OnnxError::UnexpectedShape(format!(
                "input has {} elements but shape {:?} implies {}",
                input.len(),
                shape,
                expected
            )));
        }

        let tensor = tract_ndarray::ArrayD::from_shape_vec(shape.to_vec(), input.to_vec())
            .map_err(|e| OnnxError::UnexpectedShape(format!("building input array: {e}")))?;

        let outputs = self
            .plan
            .run(tvec!(tensor.into_tensor().into()))
            .map_err(|e| OnnxError::Backend(format!("running inference: {e}")))?;

        let mut result = Vec::with_capacity(outputs.len());
        for out in outputs.iter() {
            let shape: Vec<usize> = out.shape().to_vec();
            // tract 0.23 moved the typed slice accessor off `Tensor` and onto
            // `TensorView`, so go through `view()`. This also type-checks the datum:
            // a model exporting f16 or i64 outputs errors here rather than
            // reinterpreting the bytes as f32.
            let view = (**out).view();
            let data = view
                .as_slice::<f32>()
                .map_err(|e| {
                    OnnxError::Backend(format!(
                        "reading output tensor as f32 (dtype {:?}): {e}",
                        out.datum_type()
                    ))
                })?
                .to_vec();
            result.push(Tensor::new(data, shape));
        }
        Ok(result)
    }

    fn num_outputs(&self) -> usize {
        self.num_outputs
    }

    fn backend(&self) -> Backend {
        Backend::Tract
    }

    fn device(&self) -> Device {
        // tract has no accelerator path; reporting anything else would be a lie that
        // shows up as an unexplained performance shortfall.
        Device::Cpu
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opening_a_missing_model_is_a_clean_error() {
        let err = TractSession::open_with_shape(
            Path::new("/definitely/not/here.onnx"),
            None,
            &SessionConfig::default(),
            &[1, 3, 640, 640],
        )
        .err()
        .expect("must fail");
        assert!(matches!(err, OnnxError::ModelNotFound(_)));
    }

    #[test]
    fn opening_a_non_onnx_file_fails_at_parse_not_by_panicking() {
        let path = std::env::temp_dir().join("rsface_tract_garbage.onnx");
        std::fs::write(&path, b"this is definitely not a protobuf graph").unwrap();
        let err = TractSession::open_with_shape(
            &path,
            None,
            &SessionConfig::default(),
            &[1, 3, 640, 640],
        )
        .err()
        .expect("must fail");
        assert!(
            matches!(err, OnnxError::Backend(_)),
            "expected a parse error, got {err:?}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn open_without_a_spec_explains_the_missing_shape() {
        // The failure must name the way out, since tract genuinely cannot guess.
        let err = TractSession::open(
            Path::new("/definitely/not/here.onnx"),
            None,
            &SessionConfig::default(),
        )
        .err()
        .expect("must fail");
        match err {
            OnnxError::UnexpectedShape(m) => {
                assert!(m.contains("open_with_shape"), "got {m}");
            }
            other => panic!("expected a shape error, got {other:?}"),
        }
    }
}
