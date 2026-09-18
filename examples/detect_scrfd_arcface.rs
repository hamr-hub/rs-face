//! Industrial-accuracy path survey (SCRFD + ArcFace via ONNX).
//!
//! The counterpart to `detect_haar` / `recognise_lbph` for users who need real
//! accuracy on unconstrained faces. This example walks the model registry and
//! prints every spec with its licence tier — it does *not* load the ONNX
//! runtime, so it compiles under the zero-dep default build.
//!
//! To actually run the SCRFD / ArcFace forward pass:
//!
//! ```bash
//! cargo build --features ort-backend      # recommended; needs libonnxruntime
//! # or
//! cargo build --features tract-backend    # pure Rust, CPU only
//!
//! tools/fetch_models.sh                    # downloads pinned ONNX weights
//!
//! # then plug into your code:
//! #   rsface::scrfd_detector::ScrfdDetector::from_spec(...)
//! #   rsface::arcface_recognizer::ArcFaceRecognizer::from_spec(...)
//! ```
//!
//! Run with:
//!   cargo run --release --example detect_scrfd_arcface

use rsface::models::{self, ModelKind, ModelSpec};

fn main() {
    println!("rs-face model registry — every spec this crate knows about:\n");
    println!(
        "  {:<26}  {:<10}  {:<8}  commercial-use  sha256-prefix",
        "id", "kind", "licence"
    );
    println!("  {:<26}  {:<10}  {:<8}  --------------  --------------", "", "", "");
    for spec in models::REGISTRY {
        println!(
            "  {:<26}  {:<10}  {:<8}  {:<14}  {}",
            spec.id,
            kind_label(&spec.kind),
            spec.license,
            if spec.is_commercial_use_allowed() { "yes" } else { "no (research only)" },
            short_sha(spec.sha256.unwrap_or("<unpinned>")),
        );
    }
    println!(
        "\nAlgorithm tags consumed by --algo and the RSFACE_ALGO env var \
         (gated behind the ort-backend or tract-backend feature):\n\
         \n  scrfd   Industrial-accuracy face detector (SCRFD-10G).\n  arcface 512-d L2-normalised face embeddings (w600k_r50 / w600k_mbf)."
    );
}

fn kind_label(k: &ModelKind) -> &'static str {
    match k {
        ModelKind::Detector => "detector",
        ModelKind::Recognizer => "recogniser",
    }
}

fn short_sha(sha: &str) -> &str {
    if sha.len() > 12 { &sha[..12] } else { sha }
}