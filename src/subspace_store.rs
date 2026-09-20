//! Zero-dependency binary persistence for the **trained** subspace recognisers
//! ([`crate::eigenface::EigenfaceRecognizer`] /
//! [`crate::fisherface::FisherfaceRecognizer`]).
//!
//! Unlike LBPH (incremental enrolment of descriptors), eigenfaces/fisherfaces are
//! train-once transforms of the whole gallery. This codec stores the *trained
//! model* — the pixel-space mean, the learned axes, and each crop's projected
//! coefficients — so a restart never needs the original crops or a retrain.
//!
//! Two near-identical versioned little-endian formats live here:
//!
//! * `RSEF` v1 — eigenfaces/PCA (adds the metric, energy cap and the derived
//!   zero-training-FAR threshold; each axis carries its Mahalanobis `inv_scale`).
//! * `RSLD` v1 — Fisherfaces/LDA (adds the component cap only).
//!
//! Every model scalar is stored as raw IEEE-754 bits: round-tripping is
//! bit-exact, so post-reload distances, rankings and identifications are
//! identical. All entries are bounds- and validity-checked before a recogniser
//! is constructed through a crate-internal `from_parts` constructor.

use std::fs;
use std::io;
use std::path::Path;

use crate::binio::{push_f32, push_u16, push_u32, BinError, Reader};
use crate::eigenface::{
    Component as EigenComponent, EigenMetric, EigenfaceConfig, EigenfaceRecognizer,
    Member as EigenMember,
};
use crate::fisherface::{FisherfaceConfig, FisherfaceRecognizer, Member as FisherMember};

const EIGEN_MAGIC: &[u8; 4] = b"RSEF";
const FISHER_MAGIC: &[u8; 4] = b"RSLD";
const FORMAT_VERSION: u16 = 1;

const MAX_FACE_SIZE: u32 = 1024;
const MAX_MEMBERS: u32 = 1_000_000;
const MAX_LABEL_BYTES: u32 = 4096;
/// Cap on mean + axes + coefficient `f32`s so a corrupt header cannot force a
/// pathological allocation (256 MiB of floats; real models are a few MB at most).
const MAX_MODEL_FLOATS: u64 = 67_108_864;

const METRIC_EUCLIDEAN: u8 = 0;
const METRIC_MAHALANOBIS: u8 = 1;

/// Anything that can go wrong loading a stored subspace model.
#[derive(Debug)]
pub enum SubspaceStoreError {
    /// Blob does not start with the expected `RSEF`/`RSLD` magic.
    BadMagic,
    /// Blob declares a format version this build cannot read.
    UnsupportedVersion(u16),
    /// Blob ended while a field was still expected.
    Truncated,
    /// A member label is not valid UTF-8 or contained an interior NUL.
    InvalidLabel,
    /// Counts/shape/scalars in the blob are inconsistent (see message).
    InvalidConfig(String),
    /// Bytes remain after the last declared coefficient was consumed.
    TrailingBytes(usize),
    /// Filesystem error while reading or writing the model file.
    Io(io::Error),
}

impl std::fmt::Display for SubspaceStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SubspaceStoreError::BadMagic => {
                write!(f, "not an rsface eigen/fisher model blob (bad magic)")
            }
            SubspaceStoreError::UnsupportedVersion(v) => {
                write!(f, "unsupported subspace model format version {v}")
            }
            SubspaceStoreError::Truncated => write!(f, "truncated subspace model blob"),
            SubspaceStoreError::InvalidLabel => write!(f, "invalid member label (not UTF-8)"),
            SubspaceStoreError::InvalidConfig(msg) => write!(f, "invalid model contents: {msg}"),
            SubspaceStoreError::TrailingBytes(n) => {
                write!(f, "{n} trailing byte(s) after the model data")
            }
            SubspaceStoreError::Io(e) => write!(f, "model file I/O error: {e}"),
        }
    }
}

impl std::error::Error for SubspaceStoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SubspaceStoreError::Io(e) => Some(e),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

/// Serialize a trained eigenfaces recogniser to the `RSEF` v1 format.
#[must_use]
pub fn encode_eigen(rec: &EigenfaceRecognizer) -> Vec<u8> {
    let cfg = rec.config();
    let mut out = Vec::new();
    out.extend_from_slice(EIGEN_MAGIC);
    write_common_header(
        &mut out,
        cfg.face_size,
        cfg.equalize,
        cfg.max_distance,
        cfg.min_margin,
    );
    out.push(match cfg.metric {
        EigenMetric::Euclidean => METRIC_EUCLIDEAN,
        EigenMetric::Mahalanobis => METRIC_MAHALANOBIS,
    });
    out.push(u8::from(cfg.auto_threshold));
    push_f32(&mut out, cfg.variance_kept);
    push_u32(&mut out, cfg.max_components as u32);
    match rec.suggested_threshold {
        Some(t) => {
            out.push(1);
            push_f32(&mut out, t);
        }
        None => {
            out.push(0);
            push_f32(&mut out, 0.0);
        }
    }
    write_model_body(&mut out, &rec.mean, rec.components.len(), |i| {
        AxisSpec::Eigen {
            axis: &rec.components[i].axis,
            inv_scale: rec.components[i].inv_scale,
        }
    });
    write_members(
        &mut out,
        rec.members.iter().map(|m| (&m.label[..], &m.coeffs[..])),
    );
    out
}

/// Serialize a trained Fisherfaces recogniser to the `RSLD` v1 format.
#[must_use]
pub fn encode_fisher(rec: &FisherfaceRecognizer) -> Vec<u8> {
    let cfg = rec.config();
    let mut out = Vec::new();
    out.extend_from_slice(FISHER_MAGIC);
    write_common_header(
        &mut out,
        cfg.face_size,
        cfg.equalize,
        cfg.max_distance,
        cfg.min_margin,
    );
    push_u32(&mut out, cfg.max_components as u32);
    write_model_body(&mut out, &rec.mean, rec.axes.len(), |i| {
        AxisSpec::Plain(&rec.axes[i])
    });
    write_members(
        &mut out,
        rec.members.iter().map(|m| (&m.label[..], &m.coeffs[..])),
    );
    out
}

fn write_common_header(
    out: &mut Vec<u8>,
    face_size: usize,
    equalize: bool,
    max_distance: f32,
    min_margin: f32,
) {
    push_u16(out, FORMAT_VERSION);
    push_u32(out, face_size as u32);
    out.push(u8::from(equalize));
    push_f32(out, max_distance);
    push_f32(out, min_margin);
}

enum AxisSpec<'a> {
    /// Fisher axis: `d` floats only.
    Plain(&'a [f32]),
    /// Eigen axis: Mahalanobis weight then `d` floats.
    Eigen { inv_scale: f32, axis: &'a [f32] },
}

fn write_model_body<'a>(
    out: &mut Vec<u8>,
    mean: &'a [f32],
    axis_count: usize,
    axis: impl Fn(usize) -> AxisSpec<'a>,
) {
    push_u32(out, axis_count as u32);
    for &v in mean {
        push_f32(out, v);
    }
    for i in 0..axis_count {
        match axis(i) {
            AxisSpec::Plain(values) => {
                for &v in values {
                    push_f32(out, v);
                }
            }
            AxisSpec::Eigen {
                inv_scale,
                axis: values,
            } => {
                push_f32(out, inv_scale);
                for &v in values {
                    push_f32(out, v);
                }
            }
        }
    }
}

fn write_members<'a>(out: &mut Vec<u8>, members: impl Iterator<Item = (&'a str, &'a [f32])>) {
    let members: Vec<(&str, &[f32])> = members.collect();
    push_u32(out, members.len() as u32);
    for (label, coeffs) in members {
        let bytes = label.as_bytes();
        push_u32(out, bytes.len() as u32);
        out.extend_from_slice(bytes);
        push_u32(out, coeffs.len() as u32);
        for &v in coeffs {
            push_f32(out, v);
        }
    }
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------

/// Parse and validate `RSEF` v1 output back into an eigenfaces recogniser.
///
/// # Errors
///
/// Any [`SubspaceStoreError`] variant describing a malformed blob.
pub fn decode_eigen(bytes: &[u8]) -> Result<EigenfaceRecognizer, SubspaceStoreError> {
    let mut r = Reader::new(bytes);
    expect_magic(&mut r, EIGEN_MAGIC)?;
    let version = tr(r.u16())?;
    if version != FORMAT_VERSION {
        return Err(SubspaceStoreError::UnsupportedVersion(version));
    }
    let (face_size, equalize, max_distance, min_margin) = read_common_header(&mut r)?;

    let metric = match tr(r.u8())? {
        METRIC_EUCLIDEAN => EigenMetric::Euclidean,
        METRIC_MAHALANOBIS => EigenMetric::Mahalanobis,
        other => {
            return Err(SubspaceStoreError::InvalidConfig(format!(
                "metric byte {other} is not 0 (Euclidean) or 1 (Mahalanobis)"
            )))
        }
    };
    let auto_threshold = read_flag(&mut r, "auto_threshold")?;
    let variance_kept = tr(r.f32())?;
    if !variance_kept.is_finite() || !(0.0..=1.0).contains(&variance_kept) || variance_kept == 0.0 {
        return Err(SubspaceStoreError::InvalidConfig(format!(
            "variance_kept {variance_kept} must be in (0, 1]"
        )));
    }
    let max_components = tr(r.u32())? as usize;
    let has_suggested = read_flag(&mut r, "suggested-threshold flag")?;
    let suggested_value = tr(r.f32())?;
    let suggested_threshold = if has_suggested {
        if !suggested_value.is_finite() || suggested_value < 0.0 {
            return Err(SubspaceStoreError::InvalidConfig(format!(
                "suggested_threshold {suggested_value} must be finite and >= 0"
            )));
        }
        Some(suggested_value)
    } else {
        None
    };

    let dim = model_dim(face_size)?;
    let (mean, k) = read_mean_and_axis_count(&mut r, dim)?;
    let mut components = Vec::with_capacity(k);
    for _ in 0..k {
        let inv_scale = tr(r.f32())?;
        if !inv_scale.is_finite() || inv_scale < 0.0 {
            return Err(SubspaceStoreError::InvalidConfig(format!(
                "axis inv_scale {inv_scale} must be finite and >= 0"
            )));
        }
        components.push(EigenComponent {
            inv_scale,
            axis: read_finite_vector(&mut r, dim, "eigenface axis")?,
        });
    }
    let stored = read_members(&mut r, k)?;
    if stored.len() < 2 {
        return Err(SubspaceStoreError::InvalidConfig(format!(
            "eigenfaces model needs >= 2 projected crops, found {}",
            stored.len()
        )));
    }
    check_exhausted(&r, bytes)?;
    let members = stored
        .into_iter()
        .map(|(label, coeffs)| EigenMember { label, coeffs })
        .collect();

    let config = EigenfaceConfig {
        face_size,
        variance_kept,
        max_components,
        max_distance,
        auto_threshold,
        min_margin,
        equalize,
        metric,
    };
    Ok(EigenfaceRecognizer::from_parts(
        config,
        mean,
        components,
        members,
        suggested_threshold,
    ))
}

/// Parse and validate `RSLD` v1 output back into a Fisherfaces recogniser.
///
/// # Errors
///
/// Any [`SubspaceStoreError`] variant describing a malformed blob.
pub fn decode_fisher(bytes: &[u8]) -> Result<FisherfaceRecognizer, SubspaceStoreError> {
    let mut r = Reader::new(bytes);
    expect_magic(&mut r, FISHER_MAGIC)?;
    let version = tr(r.u16())?;
    if version != FORMAT_VERSION {
        return Err(SubspaceStoreError::UnsupportedVersion(version));
    }
    let (face_size, equalize, max_distance, min_margin) = read_common_header(&mut r)?;
    let max_components = tr(r.u32())? as usize;

    let dim = model_dim(face_size)?;
    let (mean, k) = read_mean_and_axis_count(&mut r, dim)?;
    if k == 0 {
        return Err(SubspaceStoreError::InvalidConfig(
            "fisherfaces model has zero discriminant axes".into(),
        ));
    }
    let mut axes = Vec::with_capacity(k);
    for _ in 0..k {
        axes.push(read_finite_vector(&mut r, dim, "fisherface axis")?);
    }
    let stored = read_members(&mut r, k)?;
    let distinct_labels = stored
        .iter()
        .map(|(label, _)| label.as_str())
        .collect::<std::collections::HashSet<_>>()
        .len();
    if distinct_labels < 2 {
        return Err(SubspaceStoreError::InvalidConfig(format!(
            "fisherfaces model needs >= 2 identities, found {distinct_labels}"
        )));
    }
    check_exhausted(&r, bytes)?;
    let members = stored
        .into_iter()
        .map(|(label, coeffs)| FisherMember { label, coeffs })
        .collect();

    let config = FisherfaceConfig {
        face_size,
        max_components,
        max_distance,
        min_margin,
        equalize,
    };
    Ok(FisherfaceRecognizer::from_parts(
        config, mean, axes, members,
    ))
}

fn expect_magic(r: &mut Reader<'_>, expected: &[u8; 4]) -> Result<(), SubspaceStoreError> {
    let magic = tr(r.take(4))?;
    if magic == expected {
        Ok(())
    } else {
        Err(SubspaceStoreError::BadMagic)
    }
}

fn read_common_header(r: &mut Reader<'_>) -> Result<(usize, bool, f32, f32), SubspaceStoreError> {
    let face_size = tr(r.u32())?;
    if face_size == 0 || face_size > MAX_FACE_SIZE {
        return Err(SubspaceStoreError::InvalidConfig(format!(
            "face_size {face_size} not in 1..={MAX_FACE_SIZE}"
        )));
    }
    let equalize = read_flag(r, "equalize")?;
    let max_distance = tr(r.f32())?;
    require_nonneg_finite(max_distance, "max_distance")?;
    let min_margin = tr(r.f32())?;
    require_nonneg_finite(min_margin, "min_margin")?;
    Ok((face_size as usize, equalize, max_distance, min_margin))
}

fn read_flag(r: &mut Reader<'_>, name: &str) -> Result<bool, SubspaceStoreError> {
    match tr(r.u8())? {
        0 => Ok(false),
        1 => Ok(true),
        other => Err(SubspaceStoreError::InvalidConfig(format!(
            "{name} byte {other} is not 0 or 1"
        ))),
    }
}

fn require_nonneg_finite(v: f32, name: &str) -> Result<(), SubspaceStoreError> {
    if !v.is_finite() || v < 0.0 {
        Err(SubspaceStoreError::InvalidConfig(format!(
            "{name} {v} must be finite and >= 0"
        )))
    } else {
        Ok(())
    }
}

/// `face_size²`, with a sanity ceiling.
fn model_dim(face_size: usize) -> Result<usize, SubspaceStoreError> {
    face_size
        .checked_mul(face_size)
        .ok_or_else(|| SubspaceStoreError::InvalidConfig("face_size^2 overflows usize".into()))
}

fn read_mean_and_axis_count(
    r: &mut Reader<'_>,
    dim: usize,
) -> Result<(Vec<f32>, usize), SubspaceStoreError> {
    let k = tr(r.u32())? as u64;
    // Mean once plus k axes and (per member) k coefficients: bound the
    // header-driven float budget before allocating anything.
    let floats = (u64::try_from(dim).unwrap_or(u64::MAX))
        .checked_mul(k.saturating_add(1))
        .ok_or_else(|| capacity_err(dim, k))?;
    if floats > MAX_MODEL_FLOATS {
        return Err(capacity_err(dim, k));
    }
    let k = k as usize;
    let mean = read_finite_vector(r, dim, "model mean")?;
    Ok((mean, k))
}

fn capacity_err(dim: usize, k: u64) -> SubspaceStoreError {
    SubspaceStoreError::InvalidConfig(format!(
        "model size {dim} x {k} floats exceeds the {MAX_MODEL_FLOATS}-float sanity budget"
    ))
}

fn read_finite_vector(
    r: &mut Reader<'_>,
    n: usize,
    what: &str,
) -> Result<Vec<f32>, SubspaceStoreError> {
    let raw = tr(r.take(n.checked_mul(4).ok_or_else(|| {
        SubspaceStoreError::InvalidConfig(format!("{what} byte length overflows usize"))
    })?))?;
    let mut out = Vec::with_capacity(n);
    for chunk in raw.chunks_exact(4) {
        let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if !v.is_finite() {
            return Err(SubspaceStoreError::InvalidConfig(format!(
                "{what} contains NaN/infinite values"
            )));
        }
        out.push(v);
    }
    Ok(out)
}

/// Parse the shared member section (repeated crop: label + projected
/// coefficients), returned as neutral `(label, coeffs)` pairs that each
/// recogniser maps onto its own member type.
fn read_members(
    r: &mut Reader<'_>,
    k: usize,
) -> Result<Vec<(String, Vec<f32>)>, SubspaceStoreError> {
    let n = bounded_u32(r, MAX_MEMBERS)? as usize;
    if n.checked_mul(k)
        .map(|floats| (floats as u64) > MAX_MODEL_FLOATS)
        .unwrap_or(true)
    {
        return Err(SubspaceStoreError::InvalidConfig(format!(
            "{n} members x {k} coefficients exceed the {MAX_MODEL_FLOATS}-float budget"
        )));
    }
    let mut members = Vec::with_capacity(n);
    for _ in 0..n {
        let label = read_label(r)?;
        let n_coeffs = tr(r.u32())? as usize;
        if n_coeffs != k {
            return Err(SubspaceStoreError::InvalidConfig(format!(
                "member {label:?} has {n_coeffs} coefficients, model has {k} axes"
            )));
        }
        members.push((label, read_finite_vector(r, k, "member coefficients")?));
    }
    Ok(members)
}

fn bounded_u32(r: &mut Reader<'_>, max: u32) -> Result<u32, SubspaceStoreError> {
    let v = tr(r.u32())?;
    if v > max {
        Err(SubspaceStoreError::InvalidConfig(format!(
            "count {v} exceeds sanity limit {max}"
        )))
    } else {
        Ok(v)
    }
}

fn read_label(r: &mut Reader<'_>) -> Result<String, SubspaceStoreError> {
    let n = bounded_u32(r, MAX_LABEL_BYTES)? as usize;
    let raw = tr(r.take(n))?;
    if raw.contains(&0) {
        return Err(SubspaceStoreError::InvalidLabel);
    }
    match std::str::from_utf8(raw) {
        Ok(s) if !s.is_empty() => Ok(s.to_string()),
        Ok(_) => Err(SubspaceStoreError::InvalidConfig(
            "empty member label".into(),
        )),
        Err(_) => Err(SubspaceStoreError::InvalidLabel),
    }
}

fn check_exhausted(r: &Reader<'_>, bytes: &[u8]) -> Result<(), SubspaceStoreError> {
    if r.position() != bytes.len() {
        Err(SubspaceStoreError::TrailingBytes(
            bytes.len() - r.position(),
        ))
    } else {
        Ok(())
    }
}

fn tr<T>(r: Result<T, BinError>) -> Result<T, SubspaceStoreError> {
    r.map_err(|BinError::Truncated| SubspaceStoreError::Truncated)
}

/// Atomically write any model blob: sibling temp file then rename.
///
/// # Errors
///
/// [`SubspaceStoreError::Io`] for any filesystem failure.
pub fn save(bytes: &[u8], path: impl AsRef<Path>) -> Result<(), SubspaceStoreError> {
    let path = path.as_ref();
    let parent = path.parent().filter(|p| !p.as_os_str().is_empty());
    let tmp = match (parent, path.file_name()) {
        (Some(dir), Some(name)) => {
            let mut tmp_name = std::ffi::OsString::from(".");
            tmp_name.push(name);
            tmp_name.push(".tmp");
            dir.join(tmp_name)
        }
        _ => path.with_extension("tmp"),
    };
    fs::write(&tmp, bytes).map_err(SubspaceStoreError::Io)?;
    fs::rename(&tmp, path).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        SubspaceStoreError::Io(e)
    })?;
    Ok(())
}

/// Read a model file and decode it with `decoder`.
///
/// # Errors
///
/// [`SubspaceStoreError::Io`] for filesystem failures, or a decoder error.
pub fn load<T>(
    path: impl AsRef<Path>,
    decoder: impl FnOnce(&[u8]) -> Result<T, SubspaceStoreError>,
) -> Result<T, SubspaceStoreError> {
    let bytes = fs::read(path).map_err(SubspaceStoreError::Io)?;
    decoder(&bytes)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::GrayImage;

    fn crop(seed: u8) -> GrayImage {
        let mut img = GrayImage::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                img[(x, y)] = ((x as u16 + y as u16 + u16::from(seed)) % 256) as u8;
            }
        }
        img
    }

    fn trained_eigen() -> EigenfaceRecognizer {
        let a1 = crop(1);
        let a2 = crop(2);
        let a3 = crop(3);
        let b1 = crop(40);
        let b2 = crop(41);
        let b3 = crop(42);
        EigenfaceRecognizer::train(
            EigenfaceConfig::default(),
            vec![
                ("alice", &a1),
                ("alice", &a2),
                ("alice", &a3),
                ("bob", &b1),
                ("bob", &b2),
                ("bob", &b3),
            ],
        )
        .expect("train eigen")
    }

    fn trained_fisher() -> FisherfaceRecognizer {
        let a1 = crop(1);
        let a2 = crop(2);
        let a3 = crop(3);
        let b1 = crop(40);
        let b2 = crop(41);
        let b3 = crop(42);
        FisherfaceRecognizer::train(
            FisherfaceConfig::default(),
            vec![
                ("alice", &a1),
                ("alice", &a2),
                ("alice", &a3),
                ("bob", &b1),
                ("bob", &b2),
                ("bob", &b3),
            ],
        )
        .expect("train fisher")
    }

    #[test]
    fn eigen_roundtrip_preserves_model_and_rankings() {
        let rec = trained_eigen();
        let probe = crop(2);
        let before = rec.rank_crop(&probe);
        let bytes = rec.to_bytes();
        assert_eq!(&bytes[0..4], EIGEN_MAGIC);

        let back = EigenfaceRecognizer::from_bytes(&bytes).expect("decode eigen");
        assert_eq!(back.config(), rec.config());
        assert_eq!(back.len(), rec.len());
        assert_eq!(back.crop_count(), rec.crop_count());
        assert_eq!(back.component_count(), rec.component_count());
        assert_eq!(back.suggested_threshold(), rec.suggested_threshold());
        assert_eq!(back.rank_crop(&probe), before);
        assert_eq!(
            format!("{:?}", back.identify_crop(&probe)),
            format!("{:?}", rec.identify_crop(&probe))
        );
    }

    #[test]
    fn fisher_roundtrip_preserves_model_and_rankings() {
        let rec = trained_fisher();
        let probe = crop(41);
        let before = rec.rank_crop(&probe);
        let bytes = rec.to_bytes();
        assert_eq!(&bytes[0..4], FISHER_MAGIC);

        let back = FisherfaceRecognizer::from_bytes(&bytes).expect("decode fisher");
        assert_eq!(back.config(), rec.config());
        assert_eq!(back.len(), rec.len());
        assert_eq!(back.crop_count(), rec.crop_count());
        assert_eq!(back.component_count(), rec.component_count());
        assert_eq!(back.rank_crop(&probe), before);
    }

    #[test]
    fn cross_magic_is_rejected() {
        let eigen_bytes = trained_eigen().to_bytes();
        assert!(matches!(
            decode_fisher(&eigen_bytes),
            Err(SubspaceStoreError::BadMagic)
        ));
        let fisher_bytes = trained_fisher().to_bytes();
        assert!(matches!(
            decode_eigen(&fisher_bytes),
            Err(SubspaceStoreError::BadMagic)
        ));
    }

    #[test]
    fn files_roundtrip_through_disk_for_both_models() {
        let dir = std::env::temp_dir();
        let tag = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let ep = dir.join(format!("rsface_eigen_{}_{}.bin", std::process::id(), tag));
        let fp = dir.join(format!("rsface_fisher_{}_{}.bin", std::process::id(), tag));

        trained_eigen().save(&ep).expect("save eigen");
        trained_fisher().save(&fp).expect("save fisher");
        let e = EigenfaceRecognizer::load(&ep).expect("load eigen");
        let f = FisherfaceRecognizer::load(&fp).expect("load fisher");
        assert_eq!(e.crop_count(), 6);
        assert_eq!(f.crop_count(), 6);
        fs::remove_file(&ep).unwrap();
        fs::remove_file(&fp).unwrap();
    }

    #[test]
    fn rejects_truncation_trailing_bad_version_and_missing_file() {
        let bytes = trained_fisher().to_bytes();
        assert!(matches!(
            FisherfaceRecognizer::from_bytes(&bytes[..bytes.len() - 1]),
            Err(SubspaceStoreError::Truncated)
        ));
        assert!(matches!(
            FisherfaceRecognizer::from_bytes(&[]),
            Err(SubspaceStoreError::Truncated)
        ));
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(matches!(
            FisherfaceRecognizer::from_bytes(&trailing),
            Err(SubspaceStoreError::TrailingBytes(1))
        ));
        let mut future = bytes.clone();
        future[4..6].copy_from_slice(&42u16.to_le_bytes());
        assert!(matches!(
            FisherfaceRecognizer::from_bytes(&future),
            Err(SubspaceStoreError::UnsupportedVersion(42))
        ));
        let missing = std::env::temp_dir().join("rsface_subspace_missing_987654.bin");
        match FisherfaceRecognizer::load(&missing) {
            Err(SubspaceStoreError::Io(e)) => assert_eq!(e.kind(), io::ErrorKind::NotFound),
            other => panic!("expected Io(NotFound), got {other:?}"),
        }
    }

    #[test]
    fn rejects_shape_and_flag_corruption() {
        let bytes = trained_fisher().to_bytes();
        // Common header: magic4 + ver2 + face_size u32 at offset 6; claim 2048.
        let mut too_big = bytes.clone();
        too_big[6..10].copy_from_slice(&2048u32.to_le_bytes());
        assert!(matches!(
            FisherfaceRecognizer::from_bytes(&too_big),
            Err(SubspaceStoreError::InvalidConfig(_))
        ));
        // equalize byte at offset 10 must be 0/1.
        let mut bad_flag = bytes.clone();
        bad_flag[10] = 7;
        assert!(matches!(
            FisherfaceRecognizer::from_bytes(&bad_flag),
            Err(SubspaceStoreError::InvalidConfig(_))
        ));
        // Fisher header: magic4 ver2 size4 eq1 max4 min4 cap4 = 23 bytes, then k.
        let mut no_axes = bytes.clone();
        no_axes[23..27].copy_from_slice(&0u32.to_le_bytes());
        assert!(matches!(
            FisherfaceRecognizer::from_bytes(&no_axes),
            Err(SubspaceStoreError::InvalidConfig(_))
        ));

        // Eigen header after the 19 common bytes: metric u8 @19, auto u8 @20,
        // variance_kept f32 @21.
        let ebytes = trained_eigen().to_bytes();
        let mut bad_metric = ebytes.clone();
        bad_metric[19] = 9;
        assert!(matches!(
            EigenfaceRecognizer::from_bytes(&bad_metric),
            Err(SubspaceStoreError::InvalidConfig(_))
        ));
        let mut bad_variance = ebytes.clone();
        bad_variance[21..25].copy_from_slice(&0.0f32.to_le_bytes());
        assert!(matches!(
            EigenfaceRecognizer::from_bytes(&bad_variance),
            Err(SubspaceStoreError::InvalidConfig(_))
        ));
    }

    #[test]
    fn mahalanobis_config_roundtrips() {
        let a1 = crop(1);
        let a2 = crop(2);
        let b1 = crop(40);
        let b2 = crop(41);
        let rec = EigenfaceRecognizer::train(
            EigenfaceConfig::with_metric(EigenMetric::Mahalanobis),
            vec![("alice", &a1), ("alice", &a2), ("bob", &b1), ("bob", &b2)],
        )
        .expect("train mahalanobis");
        let back = EigenfaceRecognizer::from_bytes(&rec.to_bytes()).expect("decode");
        assert_eq!(back.config(), rec.config());
        assert_eq!(
            back.rank_crop(&a1),
            rec.rank_crop(&a1),
            "whitened distances must survive the round-trip"
        );
    }
}
