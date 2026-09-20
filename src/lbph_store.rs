//! Zero-dependency binary persistence for [`crate::lbph::LbphRecognizer`].
//!
//! An enrolled LBPH gallery is otherwise in-memory only: process exit loses it and
//! re-enrolment needs every original crop. This module stores the **descriptors**
//! (config + per-identity LBP histograms) in a compact, checked little-endian format,
//! so a gallery survives restarts without keeping the crops or any image codec
//! around. No serde, no third-party crate.
//!
//! Descriptors are stored verbatim as IEEE-754 `f32`: round-tripping is bit-exact,
//! which keeps chi-square distances identical after reload.
//!
//! # File layout (version 2, all integers little-endian)
//!
//! ```text
//! offset  size       field
//! 0       4          magic  b"RSLB"
//! 4       2          format version (u16, currently 2)
//! 6       4          radius (u32)
//! 10      4          face_size (u32)
//! 14      4          grid_x (u32)
//! 18      4          grid_y (u32)
//! 22      4          max_distance (f32 bits)
//! 26      4          min_margin (f32 bits)
//! 30      1          equalize (u8, 0/1)
//! 31      4          n_identities (u32)
//! per identity:
//!         4          label byte length n_l (u32), UTF-8 label (no NUL)
//!         4          n_descriptors (u32, >= 1)
//! per descriptor:
//!         4          cells (u32, must equal grid_x * grid_y)
//!         4          n_values (u32, must equal cells * BINS)
//!         4*n_values histogram f32 values
//! ```
//!
//! Version 2 stores OpenCV-`elbp_`-exact descriptors (bit 0 at the 3 o'clock
//! sample, bilinear ring interpolation, OpenCV spatial cells). Version 1 was only
//! ever written by pre-release builds (a 6 o'clock nearest-pixel convention whose
//! uniform-bin permutation is not distance-comparable), so the decoder rejects it
//! with [`LbphStoreError::UnsupportedVersion`] rather than mixing conventions.
//!
//! Decoding validates every length, the magic/version, UTF-8 labels, finite scalars,
//! config/descriptor shape agreement, unique labels, and that the blob ends exactly
//! where the format says — trailing bytes are rejected rather than ignored.

use std::fs;
use std::io;
use std::path::Path;

use crate::binio::{push_f32, push_u16, push_u32, BinError, Reader};
use crate::lbph::{LbphConfig, LbphDescriptor, LbphIdentity, LbphRecognizer, BINS};

/// File magic for an LBPH gallery blob.
const MAGIC: &[u8; 4] = b"RSLB";

/// Current on-disk format version.
///
/// v2: OpenCV-`elbp_` sampling convention (3 o'clock bit 0, bilinear ring).
/// v1 (pre-release): different neighbour order/nearest-pixel bins; rejected.
const FORMAT_VERSION: u16 = 2;

/// Fail-fast limits so a corrupt length header cannot force a pathological allocation.
const MAX_IDENTITIES: u32 = 1_000_000;
const MAX_DESCRIPTORS_PER_IDENTITY: u32 = 1_000_000;
const MAX_LABEL_BYTES: u32 = 4096;

/// Anything that can go wrong loading a stored gallery.
#[derive(Debug)]
pub enum LbphStoreError {
    /// Blob does not start with the `RSLB` magic.
    BadMagic,
    /// Blob declares a format version this build cannot read.
    UnsupportedVersion(u16),
    /// Blob ended while a field or descriptor was still expected.
    Truncated,
    /// A label is not valid UTF-8 or contained an interior NUL.
    InvalidLabel,
    /// The `equalize` byte was neither 0 nor 1.
    InvalidEqualizeByte(u8),
    /// Counts/shape/scalars in the blob are inconsistent (see message).
    InvalidConfig(String),
    /// Bytes remain after the last declared descriptor was consumed.
    TrailingBytes(usize),
    /// Filesystem error while reading or writing the gallery file.
    Io(io::Error),
}

impl std::fmt::Display for LbphStoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LbphStoreError::BadMagic => write!(f, "not an rsface LBPH gallery (bad magic)"),
            LbphStoreError::UnsupportedVersion(v) => {
                write!(f, "unsupported LBPH gallery format version {v}")
            }
            LbphStoreError::Truncated => write!(f, "truncated LBPH gallery blob"),
            LbphStoreError::InvalidLabel => write!(f, "invalid gallery label (not UTF-8)"),
            LbphStoreError::InvalidEqualizeByte(b) => {
                write!(f, "invalid equalize byte {b} (expected 0 or 1)")
            }
            LbphStoreError::InvalidConfig(msg) => write!(f, "invalid gallery contents: {msg}"),
            LbphStoreError::TrailingBytes(n) => {
                write!(f, "{n} trailing byte(s) after the gallery data")
            }
            LbphStoreError::Io(e) => write!(f, "gallery file I/O error: {e}"),
        }
    }
}

impl std::error::Error for LbphStoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LbphStoreError::Io(e) => Some(e),
            _ => None,
        }
    }
}

/// Serialize a recogniser's gallery to the version-2 binary format.
#[must_use]
pub fn encode(rec: &LbphRecognizer) -> Vec<u8> {
    let cfg = rec.config();
    let mut out = Vec::new();
    out.extend_from_slice(MAGIC);
    push_u16(&mut out, FORMAT_VERSION);
    push_u32(&mut out, cfg.radius as u32);
    push_u32(&mut out, cfg.face_size as u32);
    push_u32(&mut out, cfg.grid_x as u32);
    push_u32(&mut out, cfg.grid_y as u32);
    push_f32(&mut out, cfg.max_distance);
    push_f32(&mut out, cfg.min_margin);
    out.push(u8::from(cfg.equalize));
    push_u32(&mut out, rec.len() as u32);

    for id in rec.identities() {
        let label = id.label.as_bytes();
        push_u32(&mut out, label.len() as u32);
        out.extend_from_slice(label);
        push_u32(&mut out, id.descriptors.len() as u32);
        for desc in &id.descriptors {
            push_u32(&mut out, desc.cells() as u32);
            push_u32(&mut out, desc.as_slice().len() as u32);
            for &v in desc.as_slice() {
                push_f32(&mut out, v);
            }
        }
    }
    out
}

/// Parse and validate [`encode`] output back into a usable recogniser.
///
/// # Errors
///
/// Any [`LbphStoreError`] variant describing a malformed blob.
pub fn decode(bytes: &[u8]) -> Result<LbphRecognizer, LbphStoreError> {
    let mut r = Reader::new(bytes);
    let magic = tr(r.take(4))?;
    if magic != MAGIC {
        return Err(LbphStoreError::BadMagic);
    }
    let version = tr(r.u16())?;
    if version != FORMAT_VERSION {
        return Err(LbphStoreError::UnsupportedVersion(version));
    }

    let radius = tr(r.u32())? as usize;
    let face_size = tr(r.u32())? as usize;
    let grid_x = tr(r.u32())? as usize;
    let grid_y = tr(r.u32())? as usize;
    let max_distance = tr(r.f32())?;
    let min_margin = tr(r.f32())?;
    let equalize = match tr(r.u8())? {
        0 => false,
        1 => true,
        other => return Err(LbphStoreError::InvalidEqualizeByte(other)),
    };
    validate_config(radius, face_size, grid_x, grid_y, max_distance, min_margin)?;

    let cells_expected = grid_x
        .checked_mul(grid_y)
        .filter(|&c| c > 0)
        .ok_or_else(|| LbphStoreError::InvalidConfig("grid area overflows usize".into()))?;
    let values_expected = cells_expected
        .checked_mul(BINS)
        .ok_or_else(|| LbphStoreError::InvalidConfig("descriptor length overflows usize".into()))?;

    let n_identities = bounded_u32(&mut r, MAX_IDENTITIES)?;
    let mut identities = Vec::with_capacity(n_identities as usize);
    let mut seen = std::collections::HashSet::with_capacity(n_identities as usize);
    for _ in 0..n_identities {
        let label = read_label(&mut r)?;
        if !seen.insert(label.clone()) {
            return Err(LbphStoreError::InvalidConfig(format!(
                "duplicate identity label {label:?}"
            )));
        }
        let n_desc = bounded_u32(&mut r, MAX_DESCRIPTORS_PER_IDENTITY)?;
        if n_desc == 0 {
            return Err(LbphStoreError::InvalidConfig(format!(
                "identity {label:?} has zero descriptors"
            )));
        }
        let mut descriptors = Vec::with_capacity(n_desc as usize);
        for _ in 0..n_desc {
            let cells = tr(r.u32())? as usize;
            let n_values = tr(r.u32())? as usize;
            if cells != cells_expected {
                return Err(LbphStoreError::InvalidConfig(format!(
                    "descriptor cells {cells} != grid area {cells_expected}"
                )));
            }
            if n_values != values_expected {
                return Err(LbphStoreError::InvalidConfig(format!(
                    "descriptor length {n_values} != cells * {BINS} ({values_expected})"
                )));
            }
            let mut values = Vec::with_capacity(n_values);
            for _ in 0..n_values {
                let v = tr(r.f32())?;
                if !v.is_finite() {
                    return Err(LbphStoreError::InvalidConfig(
                        "descriptor contains NaN/infinite values".into(),
                    ));
                }
                values.push(v);
            }
            descriptors.push(LbphDescriptor::from_parts(cells, values));
        }
        identities.push(LbphIdentity { label, descriptors });
    }

    if r.remaining() != 0 {
        return Err(LbphStoreError::TrailingBytes(r.remaining()));
    }

    let config = LbphConfig {
        radius,
        face_size,
        grid_x,
        grid_y,
        max_distance,
        min_margin,
        equalize,
    };
    Ok(LbphRecognizer::from_identities(config, identities))
}

/// Read a gallery file; thin wrapper over [`decode`].
///
/// # Errors
///
/// [`LbphStoreError::Io`] for filesystem failures, or a decode error on a bad blob.
pub fn load(path: impl AsRef<Path>) -> Result<LbphRecognizer, LbphStoreError> {
    let bytes = fs::read(path).map_err(LbphStoreError::Io)?;
    decode(&bytes)
}

/// Atomically write a gallery file: encode to a sibling temp file, then rename over
/// the destination so a crash cannot leave a half-written gallery in place.
///
/// # Errors
///
/// [`LbphStoreError::Io`] for any filesystem failure.
pub fn save(rec: &LbphRecognizer, path: impl AsRef<Path>) -> Result<(), LbphStoreError> {
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
    fs::write(&tmp, encode(rec)).map_err(LbphStoreError::Io)?;
    fs::rename(&tmp, path).map_err(|e| {
        // Best-effort cleanup; ignore a failure of the cleanup itself.
        let _ = fs::remove_file(&tmp);
        LbphStoreError::Io(e)
    })?;
    Ok(())
}

/// Sanity-check the numeric config fields stored in a blob header.
fn validate_config(
    radius: usize,
    face_size: usize,
    grid_x: usize,
    grid_y: usize,
    max_distance: f32,
    min_margin: f32,
) -> Result<(), LbphStoreError> {
    if radius == 0 {
        return Err(LbphStoreError::InvalidConfig("radius must be >= 1".into()));
    }
    // Extraction samples radius pixels around every pixel; honour that minimum.
    if face_size < 2 * radius + 1 {
        return Err(LbphStoreError::InvalidConfig(format!(
            "face_size {face_size} too small for radius {radius}"
        )));
    }
    if grid_x == 0 || grid_y == 0 {
        return Err(LbphStoreError::InvalidConfig(
            "grid must be non-empty".into(),
        ));
    }
    if !max_distance.is_finite() || max_distance < 0.0 {
        return Err(LbphStoreError::InvalidConfig(format!(
            "max_distance {max_distance} must be finite and >= 0"
        )));
    }
    if !min_margin.is_finite() || min_margin < 0.0 {
        return Err(LbphStoreError::InvalidConfig(format!(
            "min_margin {min_margin} must be finite and >= 0"
        )));
    }
    Ok(())
}

/// Read a UTF-8 label with an explicit length prefix; reject interior NUL bytes so a
/// later C-friendly FFI layer cannot truncate a stored identity.
fn read_label(r: &mut Reader<'_>) -> Result<String, LbphStoreError> {
    let n = bounded_u32(r, MAX_LABEL_BYTES)? as usize;
    let raw = tr(r.take(n))?;
    if raw.contains(&0) {
        return Err(LbphStoreError::InvalidLabel);
    }
    match std::str::from_utf8(raw) {
        Ok(s) if !s.is_empty() => Ok(s.to_string()),
        Ok(_) => Err(LbphStoreError::InvalidConfig("empty identity label".into())),
        Err(_) => Err(LbphStoreError::InvalidLabel),
    }
}

fn bounded_u32(r: &mut Reader<'_>, max: u32) -> Result<u32, LbphStoreError> {
    let v = tr(r.u32())?;
    if v > max {
        Err(LbphStoreError::InvalidConfig(format!(
            "count {v} exceeds sanity limit {max}"
        )))
    } else {
        Ok(v)
    }
}

/// Map a fixed-width cursor failure onto the codec's truncation variant.
fn tr<T>(r: Result<T, BinError>) -> Result<T, LbphStoreError> {
    r.map_err(|BinError::Truncated| LbphStoreError::Truncated)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::GrayImage;

    fn cfg() -> LbphConfig {
        LbphConfig {
            radius: 1,
            face_size: 12,
            grid_x: 2,
            grid_y: 2,
            max_distance: 16.7,
            min_margin: 0.0,
            equalize: false,
        }
    }

    fn crop(seed: u8) -> GrayImage {
        let mut img = GrayImage::new(12, 12);
        for y in 0..12 {
            for x in 0..12 {
                img.as_mut_slice()[y * 12 + x] =
                    ((x as u16 + y as u16 + u16::from(seed)) % 256) as u8;
            }
        }
        img
    }

    fn trained() -> LbphRecognizer {
        let mut rec = LbphRecognizer::new(cfg());
        rec.enroll("alice", &crop(1));
        rec.enroll("alice", &crop(2));
        rec.enroll("bob", &crop(40));
        rec.enroll("bob", &crop(41));
        rec
    }

    #[test]
    fn roundtrip_preserves_config_descriptors_and_rankings() {
        let rec = trained();
        let before: Vec<(String, f32)> = rec.rank_crop(&crop(3));

        let bytes = rec.to_bytes();
        let back = LbphRecognizer::from_bytes(&bytes).expect("decode");
        assert_eq!(back.config(), rec.config());
        assert_eq!(back.len(), rec.len());
        assert_eq!(back.crop_count(), rec.crop_count());

        for (a, b) in rec.identities().iter().zip(back.identities()) {
            assert_eq!(a.label, b.label);
            assert_eq!(a.descriptors.len(), b.descriptors.len());
            for (da, db) in a.descriptors.iter().zip(&b.descriptors) {
                // Bit-exact f32 round-trip.
                assert_eq!(da.as_slice(), db.as_slice());
                assert_eq!(da.cells(), db.cells());
            }
        }
        assert_eq!(back.rank_crop(&crop(3)), before);
        assert_eq!(
            back.verify("alice", &crop(1)),
            rec.verify("alice", &crop(1))
        );
    }

    #[test]
    fn empty_gallery_roundtrips() {
        let rec = LbphRecognizer::new(cfg());
        let bytes = rec.to_bytes();
        let back = LbphRecognizer::from_bytes(&bytes).expect("decode empty");
        assert_eq!(back.len(), 0);
        assert!(back.is_empty());
        assert_eq!(back.config(), rec.config());
    }

    #[test]
    fn file_save_load_roundtrip_is_atomic_and_usable() {
        let rec = trained();
        let path = std::env::temp_dir().join(format!(
            "rsface_lbph_store_{}_{}.bin",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        rec.save(&path).expect("save");
        let back = LbphRecognizer::load(&path).expect("load");
        assert_eq!(back.crop_count(), 4);
        assert_eq!(
            back.rank_crop(&crop(3)).first().map(|(l, _)| l.as_str()),
            Some("alice")
        );
        fs::remove_file(&path).expect("cleanup");
    }

    #[test]
    fn rejects_bad_magic_truncation_and_trailing_bytes() {
        let bytes = trained().to_bytes();
        let mut bad_magic = bytes.clone();
        bad_magic[0] = b'X';
        assert!(matches!(
            LbphRecognizer::from_bytes(&bad_magic),
            Err(LbphStoreError::BadMagic)
        ));

        assert!(matches!(
            LbphRecognizer::from_bytes(&bytes[..bytes.len() - 1]),
            Err(LbphStoreError::Truncated)
        ));
        assert!(matches!(
            LbphRecognizer::from_bytes(&[]),
            Err(LbphStoreError::Truncated)
        ));

        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(matches!(
            LbphRecognizer::from_bytes(&trailing),
            Err(LbphStoreError::TrailingBytes(1))
        ));
    }

    #[test]
    fn rejects_unknown_version_and_shape_mismatch() {
        let bytes = trained().to_bytes();

        let mut future = bytes.clone();
        future[4..6].copy_from_slice(&999u16.to_le_bytes());
        assert!(matches!(
            LbphRecognizer::from_bytes(&future),
            Err(LbphStoreError::UnsupportedVersion(999))
        ));

        // v1 carried the pre-OpenCV sampling convention; mixing its bins with v2
        // would compare unrelated histograms, so it is rejected outright.
        let mut legacy = bytes.clone();
        legacy[4..6].copy_from_slice(&1u16.to_le_bytes());
        assert!(matches!(
            LbphRecognizer::from_bytes(&legacy),
            Err(LbphStoreError::UnsupportedVersion(1))
        ));

        // Header layout: magic(4) + version(2) + radius(4) + face_size(4) + grid_x at 14.
        // Claim grid_x = 3 while descriptors were encoded for a 2x2 grid -> mismatch.
        let mut tampered = bytes.clone();
        tampered[14..18].copy_from_slice(&3u32.to_le_bytes());
        match LbphRecognizer::from_bytes(&tampered) {
            Err(LbphStoreError::InvalidConfig(msg)) => assert!(msg.contains("cells")),
            other => panic!("expected InvalidConfig, got {other:?}"),
        }

        // max_distance at offset 22: NaN bits must be rejected.
        let mut nan_thr = bytes.clone();
        nan_thr[22..26].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(matches!(
            LbphRecognizer::from_bytes(&nan_thr),
            Err(LbphStoreError::InvalidConfig(_))
        ));
    }

    /// Minimal valid blob header with n_identities = 2, 1×1 grid, radius 1.
    fn header_for_two_identities() -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(MAGIC);
        b.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        b.extend_from_slice(&1u32.to_le_bytes()); // radius
        b.extend_from_slice(&3u32.to_le_bytes()); // face_size (>= 2*radius+1)
        b.extend_from_slice(&1u32.to_le_bytes()); // grid_x
        b.extend_from_slice(&1u32.to_le_bytes()); // grid_y
        b.extend_from_slice(&16.7f32.to_le_bytes());
        b.extend_from_slice(&0.0f32.to_le_bytes());
        b.push(0u8); // equalize
        b.extend_from_slice(&2u32.to_le_bytes()); // n_identities
        b
    }

    /// Append one identity with a single valid 1-cell, 59-bin descriptor.
    fn push_identity(out: &mut Vec<u8>, label: &str) {
        out.extend_from_slice(&(label.len() as u32).to_le_bytes());
        out.extend_from_slice(label.as_bytes());
        out.extend_from_slice(&1u32.to_le_bytes()); // n_descriptors
        out.extend_from_slice(&1u32.to_le_bytes()); // cells
        out.extend_from_slice(&(BINS as u32).to_le_bytes()); // n_values
        let bin = 1.0 / BINS as f32; // uniform finite values; label checks are the point
        for _ in 0..BINS {
            out.extend_from_slice(&bin.to_le_bytes());
        }
    }

    #[test]
    fn rejects_empty_and_duplicate_labels() {
        // Empty label: header (n=1 would also work, but n=2 keeps padding concerns out)
        let empty_header = {
            let mut h = header_for_two_identities();
            h.truncate(31);
            h.extend_from_slice(&1u32.to_le_bytes()); // n_identities = 1
            h
        };
        let mut empty = empty_header;
        empty.extend_from_slice(&0u32.to_le_bytes()); // zero-length label
        assert!(matches!(
            LbphRecognizer::from_bytes(&empty),
            Err(LbphStoreError::InvalidConfig(_))
        ));

        let mut dup = header_for_two_identities();
        push_identity(&mut dup, "a");
        push_identity(&mut dup, "a");
        match LbphRecognizer::from_bytes(&dup) {
            Err(LbphStoreError::InvalidConfig(msg)) => assert!(msg.contains("duplicate")),
            other => panic!("expected duplicate-label InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn missing_file_is_io_error() {
        let path = std::env::temp_dir().join("rsface_lbph_store_does_not_exist_12345.bin");
        match LbphRecognizer::load(&path) {
            Err(LbphStoreError::Io(e)) => assert_eq!(e.kind(), io::ErrorKind::NotFound),
            other => panic!("expected Io(NotFound), got {other:?}"),
        }
    }
}
