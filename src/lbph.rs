//! LBPH — Local Binary Pattern Histograms, the zero-dependency recogniser.
//!
//! ArcFace needs a 174 MB trained backbone and an ONNX runtime (ort/tract); the HAAR
//! cascade detector, by contrast, is already zero-dependency. LBPH closes the loop: it
//! gives that build **real recognition — verification, identification, multi-shot
//! enrolment — with no weights download and no third-party crate**, because the
//! "training" is a handful of histogram comparisons done at enrolment time.
//!
//! # Algorithm
//!
//! Each enrolled/probe crop is normalised to a fixed square size, then every interior
//! pixel is replaced by an 8-bit code recording which of its 8 neighbours are brighter
//! (`radius 1`). The operator has two useful invariances for free:
//!
//! * Monotone photometric changes (gain/offset changes from lighting) leave every
//!   comparison bit unchanged, so the descriptor is illumination-robust without any
//!   preprocessing.
//! * Codes are reduced to the **59 uniform patterns** (Ojala et al.): patterns with
//!   at most two 0↔1 transitions around the bit ring carry the vast majority of natural
//!   image statistics; the other 198 codes share one "miscellaneous" bin.
//!
//! The face is tiled into `grid_y × grid_x` cells (6×6 by default) and one 59-bin
//! histogram is accumulated per cell. Spatial layout is what distinguishes LBPH from a
//! texture classifier — eyes-in-cell-1 and mouth-in-cell-5 must not be interchangeable.
//! Each cell histogram is L1-normalised, making the descriptor robust to crop-size
//! differences.
//!
//! Descriptors are compared with the chi-square statistic
//! `Σ_cells Σ_bins (a-b)²/(a+b)` (0 = identical). An identity may enrol several crops;
//! matching scores against the best member, exactly as [`crate::embedding::Gallery`]
//! does for ArcFace, because one frontal shot generalises poorly across pose.
//!
//! # Accuracy envelope — be honest about this
//!
//! LBPH is a 2004 descriptor and behaves like one: strong under frontal, controlled
//! enrolment/access-control conditions, brittle to large pose, scale drift, or faces
//! enrolled once under different lighting. Measured numbers for this implementation on
//! real drama-clip faces are in `docs/recognition-lbph.md`; expect rank-1 identification
//! to be usable, the cross-session verification threshold to be set conservatively, and
//! accuracy **well below** ArcFace (whose same/different cosine margin is 0.93). For
//! uncontrolled scenes use the `ort-backend`/`tract-backend` path.
//!
//! # Reference
//!
//! Ahonen, Hadid, Pietikäinen — *Face Recognition with Local Binary Patterns* (ECCV
//! 2004). The sampler matches OpenCV's `face::LBPHFaceRecognizer` `elbp_` bit-for-bit:
//! neighbour `n` at angle `2πn/P` with column offset `r·cos` and row offset
//! `−r·sin` (bit 0 at 3 o'clock), bilinear interpolation at every radius, the same
//! epsilon-tie comparison, identical 59-label uniform lookup, and spatial cells cut
//! as fixed `interior/grid` rectangles. Descriptors from older builds with a
//! different sampling convention are not distance-compatible (the on-disk gallery
//! format was bumped to version 2 for this change).

use std::collections::HashMap;

use crate::image::GrayImage;

/// Number of neighbours sampled around each pixel (`P` in the LBPH literature).
pub const NEIGHBORS: usize = 8;

/// Number of histogram bins with the uniform mapping at [`NEIGHBORS`] = 8:
/// 58 uniform codes plus one shared bin for the 198 non-uniform codes.
pub const BINS: usize = 59;

/// Bin that every non-uniform code maps to.
const NON_UNIFORM_BIN: usize = BINS - 1;

/// Chi-square epsilon: a bin that is zero in both descriptors contributes nothing.
const CHI_EPS: f32 = 1e-10;

/// Default chi-square accept distance for the default config (6×6 grid, 120 px crops).
///
/// Conservative **low-FAR** point, calibrated on this repo's two real-face evaluation
/// sets (see `docs/recognition-lbph.md`; labels from offline ArcFace clustering, the
/// evaluated path never touches ONNX):
///
/// * 35 crops / 8 identities (85 same / 510 different pairs): FAR 0.0 %, FRR 17.6 %;
/// * 77 crops / 21 identities (195 same / 2 731 different pairs, harder pose/lighting):
///   FAR 0.0 %, FRR 50.8 % at 16.7; the closest impostor distance is 17.5 and the EER
///   point is ≈ 23.2 (FAR ≈ FRR ≈ 20 %).
///
/// Numbers above use the OpenCV-`elbp_` sampler (3 o'clock bit 0, bilinear ring) and
/// sorted-path bench order; see `docs/recognition-lbph.md`. The distributions overlap on
/// the harder gallery, so no threshold gives both low FAR and low FRR there; this
/// constant deliberately buys a zero measured false-accept rate and lets
/// `LbphMatch::BelowThreshold` reject uncertain probes. Close-set identification does
/// not need it — rank-1 LOO is 88 % (60/68) on the hard set — so prefer `rank_crop`
/// when the probe is known to be enrolled. The scale is descriptor-specific (grid and
/// crop size); recalibrate with `bench_lbph` per deployment rather than trusting it
/// blindly.
/// `f32::MAX` would be the OpenCV default ("always identify"), useless for verification.
pub const DEFAULT_MAX_DISTANCE: f32 = 16.7;

/// LBPH extraction and matching parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LbphConfig {
    /// LBP sampling radius in pixels (1 = the classic 3×3 operator).
    pub radius: usize,
    /// Face crops are resized to this square size before descriptor extraction, so
    /// detections at different pixel scales produce comparable histograms. This stands
    /// in for OpenCV's requirement that all training images share one size.
    pub face_size: usize,
    /// Histogram grid columns.
    pub grid_x: usize,
    /// Histogram grid rows.
    pub grid_y: usize,
    /// Maximum chi-square distance for an accepted identification/verification.
    pub max_distance: f32,
    /// Require the best identity to beat the runner-up by at least this chi-square
    /// margin. `0.0` disables the check. Guards near-ties between enrolled people.
    pub min_margin: f32,
    /// Apply histogram equalisation to the crop before LBP. The LBP operator is already
    /// invariant to monotone lighting changes; equalisation mainly helps when crops
    /// come from cameras with very different contrast curves. Off by default; benchmark
    /// both on your data (`bench_lbph` reports the equalised variant for comparison).
    pub equalize: bool,
}

impl Default for LbphConfig {
    /// radius 1, 6×6 histogram grid, 120 px crops.
    ///
    /// OpenCV's `face::LBPHFaceRecognizer` ships an 8×8 grid. Under the OpenCV-exact
    /// sampler the hard 21-identity gallery lands at 60/68 LOO rank-1 for both 6×6 and
    /// 8×8 at 120 px (59/68 for 8×8 at 90 px), with several other cells tying within
    /// ±4 probes — differences at that size are sampling noise, and 6×6 keeps the
    /// descriptor compact. On the easier 8-identity gallery 6×6 is 32/33 while finer
    /// grids reach 33/33. 8×8 and 10×10 remain one field away for deployments with
    /// landmark-aligned crops; the full sweep is in `docs/recognition-lbph.md`.
    fn default() -> Self {
        Self {
            radius: 1,
            face_size: 120,
            grid_x: 6,
            grid_y: 6,
            max_distance: DEFAULT_MAX_DISTANCE,
            min_margin: 0.0,
            equalize: false,
        }
    }
}

/// A face descriptor: `grid_y * grid_x` L1-normalised 59-bin cell histograms.
#[derive(Clone, Debug, PartialEq)]
pub struct LbphDescriptor {
    /// Flat `cell * BINS` histogram storage, L1-normalised per cell.
    values: Vec<f32>,
    cells: usize,
}

impl LbphDescriptor {
    /// Number of scalar bins (`grid cells × 59`).
    pub fn dim(&self) -> usize {
        self.values.len()
    }

    /// Raw histogram values, cell-major.
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    /// Number of spatial cells (`grid_y × grid_x`) this descriptor covers.
    pub fn cells(&self) -> usize {
        self.cells
    }

    /// Reconstruct a descriptor from stored parts (persistence codec only); the caller
    /// is trusted to have validated `values.len() == cells * BINS`.
    pub(crate) fn from_parts(cells: usize, values: Vec<f32>) -> Self {
        Self { values, cells }
    }

    /// Chi-square distance: 0 for identical descriptors, larger for more different ones.
    ///
    /// Returns `None` on dimension mismatch (descriptors from different grid shapes)
    /// rather than comparing a prefix, and `None` when **both** descriptors carry no
    /// histogram mass (degenerate/blank crops): the statistic would otherwise be 0/0
    /// and read as a perfect match, letting two blank probes "identify" as anything.
    pub fn chi_square(&self, other: &LbphDescriptor) -> Option<f32> {
        if self.values.len() != other.values.len() {
            return None;
        }
        if is_zero_mass(&self.values) && is_zero_mass(&other.values) {
            return None;
        }
        let d: f32 = self
            .values
            .iter()
            .zip(&other.values)
            .map(|(a, b)| {
                let denom = a + b;
                if denom < CHI_EPS {
                    0.0
                } else {
                    (a - b) * (a - b) / denom
                }
            })
            .sum();
        Some(d)
    }
}

/// True when no bin in any cell carries mass (a blank/degenerate-crop descriptor).
fn is_zero_mass(values: &[f32]) -> bool {
    values.iter().all(|&v| v == 0.0)
}

/// One enrolled identity with one or more reference descriptors.
#[derive(Clone, Debug)]
pub struct LbphIdentity {
    pub label: String,
    pub descriptors: Vec<LbphDescriptor>,
}

/// Outcome of an LBPH gallery query.
///
/// All zero-dep recognisers share the distance-valued
/// [`crate::recognizer::Recognition`] enum (its analogue for cosine
/// similarity is [`crate::embedding::MatchOutcome`]); this alias keeps the
/// per-recogniser name working while letting LBPH dispatch through
/// [`crate::recognizer::FaceRecognizer`].
pub use crate::recognizer::Recognition as LbphMatch;

/// In-memory nearest-neighbour LBPH recogniser.
///
/// Brute force by design: a default-grid descriptor is 36 cells × 59 bins = 2 124
/// `f32`s (≈ 8.3 KiB) and a chi-square comparison is cheap, so linear scan is fine
/// into the low thousands of identities.
/// The gallery can be persisted without the original crops via
/// [`LbphRecognizer::to_bytes`] / [`LbphRecognizer::save`] (format documented in
/// `docs/gallery-persistence.md`).
#[derive(Clone, Debug)]
pub struct LbphRecognizer {
    config: LbphConfig,
    identities: Vec<LbphIdentity>,
    /// label -> index into `identities`, kept so enrol is O(1) lookup.
    by_label: HashMap<String, usize>,
}

impl LbphRecognizer {
    /// Empty recogniser with the given configuration.
    pub fn new(config: LbphConfig) -> Self {
        Self {
            config,
            identities: Vec::new(),
            by_label: HashMap::new(),
        }
    }

    /// Rebuild from persisted parts; the caller (the codec in [`crate::lbph_store`])
    /// has already validated shape and rejected duplicate labels.
    pub(crate) fn from_identities(config: LbphConfig, identities: Vec<LbphIdentity>) -> Self {
        let mut by_label = HashMap::with_capacity(identities.len());
        for (i, id) in identities.iter().enumerate() {
            by_label.insert(id.label.clone(), i);
        }
        Self {
            config,
            identities,
            by_label,
        }
    }

    /// Serialize config + every enrolled descriptor to the compact binary gallery
    /// format (exact `f32` round-trip; see `docs/gallery-persistence.md`).
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        crate::lbph_store::encode(self)
    }

    /// Rebuild a recogniser from [`LbphRecognizer::to_bytes`] output, validating the
    /// whole blob before returning.
    ///
    /// # Errors
    ///
    /// [`crate::lbph_store::LbphStoreError`] for a truncated, corrupt,
    /// wrong-version, or internally inconsistent blob.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, crate::lbph_store::LbphStoreError> {
        crate::lbph_store::decode(bytes)
    }

    /// Atomically write the gallery (`to_bytes` via a temp file then rename).
    ///
    /// # Errors
    ///
    /// Filesystem errors, or [`crate::lbph_store::LbphStoreError::Io`] wrapping them.
    pub fn save(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), crate::lbph_store::LbphStoreError> {
        crate::lbph_store::save(self, path)
    }

    /// Load a recogniser written by [`LbphRecognizer::save`].
    ///
    /// # Errors
    ///
    /// Filesystem errors or any [`crate::lbph_store::LbphStoreError`].
    pub fn load(
        path: impl AsRef<std::path::Path>,
    ) -> Result<Self, crate::lbph_store::LbphStoreError> {
        crate::lbph_store::load(path)
    }

    pub fn config(&self) -> &LbphConfig {
        &self.config
    }

    pub fn len(&self) -> usize {
        self.identities.len()
    }

    pub fn is_empty(&self) -> bool {
        self.identities.is_empty()
    }

    pub fn identities(&self) -> &[LbphIdentity] {
        &self.identities
    }

    /// Total enrolled crops across all identities.
    pub fn crop_count(&self) -> usize {
        self.identities.iter().map(|i| i.descriptors.len()).sum()
    }

    /// Extract the descriptor for a grayscale face crop.
    pub fn describe(&self, crop: &GrayImage) -> LbphDescriptor {
        extract(crop, &self.config)
    }

    /// Enrol a crop under `label`, creating the identity on first use.
    pub fn enroll(&mut self, label: impl Into<String>, crop: &GrayImage) {
        let label = label.into();
        let desc = self.describe(crop);
        if let Some(&idx) = self.by_label.get(&label) {
            self.identities[idx].descriptors.push(desc);
        } else {
            self.by_label.insert(label.clone(), self.identities.len());
            self.identities.push(LbphIdentity {
                label,
                descriptors: vec![desc],
            });
        }
    }

    /// Remove an identity; returns whether it was present.
    pub fn remove(&mut self, label: &str) -> bool {
        let Some(idx) = self.by_label.remove(label) else {
            return false;
        };
        self.identities.remove(idx);
        // Vec::remove shifts every later identity, so the whole index map is rebuilt.
        self.by_label.clear();
        for (i, id) in self.identities.iter().enumerate() {
            self.by_label.insert(id.label.clone(), i);
        }
        true
    }

    /// Distance of `crop` to every identity, best member per identity, nearest first.
    pub fn rank_crop(&self, crop: &GrayImage) -> Vec<(String, f32)> {
        let probe = self.describe(crop);
        self.rank(&probe)
    }

    /// Rank an already-extracted probe descriptor.
    pub fn rank(&self, probe: &LbphDescriptor) -> Vec<(String, f32)> {
        let mut scored: Vec<(String, f32)> = self
            .identities
            .iter()
            .filter_map(|id| {
                id.descriptors
                    .iter()
                    .filter_map(|d| probe.chi_square(d))
                    // Minimum over members: closest-matching enrolled pose wins.
                    .fold(None, |acc: Option<f32>, d| {
                        Some(acc.map_or(d, |a| a.min(d)))
                    })
                    .map(|d| (id.label.clone(), d))
            })
            .collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1));
        scored
    }

    /// Identify a crop under the configured distance/margin policy.
    pub fn identify_crop(&self, crop: &GrayImage) -> LbphMatch {
        self.identify(&self.describe(crop))
    }

    /// Identify an already-extracted probe descriptor.
    pub fn identify(&self, probe: &LbphDescriptor) -> LbphMatch {
        let ranked = self.rank(probe);
        let Some((best_label, best_dist)) = ranked.first().cloned() else {
            return LbphMatch::NoCandidates;
        };

        if best_dist > self.config.max_distance {
            return LbphMatch::BelowThreshold {
                best: Some((best_label, best_dist)),
            };
        }

        let (margin, runner_up) = match ranked.get(1) {
            Some((l, d)) => (d - best_dist, Some(l.clone())),
            None => (f32::INFINITY, None),
        };
        if self.config.min_margin > 0.0 && margin < self.config.min_margin {
            return LbphMatch::Ambiguous {
                first: best_label,
                second: runner_up.unwrap_or_default(),
                margin,
            };
        }

        LbphMatch::Match {
            label: best_label,
            distance: best_dist,
            margin,
        }
    }

    /// One-to-one verification: distance to the claimed label's best enrolled crop.
    /// `None` if that label is not enrolled or shapes disagree.
    pub fn verify(&self, label: &str, crop: &GrayImage) -> Option<f32> {
        let idx = *self.by_label.get(label)?;
        let probe = self.describe(crop);
        self.identities[idx]
            .descriptors
            .iter()
            .filter_map(|d| probe.chi_square(d))
            .fold(None, |acc: Option<f32>, d| {
                Some(acc.map_or(d, |a| a.min(d)))
            })
    }
}

// ---------------------------------------------------------------------------
// Descriptor extraction
// ---------------------------------------------------------------------------

/// Extract an LBPH descriptor from a grayscale crop using `config`.
pub fn extract(img: &GrayImage, config: &LbphConfig) -> LbphDescriptor {
    let face = if img.width() == config.face_size && img.height() == config.face_size {
        None
    } else {
        Some(img.resize_bilinear(config.face_size, config.face_size))
    };
    let mut face = face.unwrap_or_else(|| clone_gray(img));
    if config.equalize {
        face.equalize_hist_inplace();
    }

    let cells = config.grid_x * config.grid_y;
    let mut hist = vec![0u32; cells * BINS];
    let r = config.radius;
    let (w, h) = (face.width(), face.height());

    if w <= 2 * r || h <= 2 * r {
        // Degenerate crop: all-zero histograms instead of an out-of-bounds panic.
        return LbphDescriptor {
            values: vec![0.0; cells * BINS],
            cells,
        };
    }

    // Neighbour ring sampled exactly like OpenCV's `elbp_` (facerec/lbph_faces.cpp):
    // neighbour n sits at angle 2 pi n / P with COLUMN offset x = r cos a and ROW
    // offset y = -r sin a — bit 0 is the 3 o'clock sample — and every neighbour at
    // every radius is bilinearly interpolated from its four surrounding pixels.
    let ring = neighbor_ring(r);

    let lut = uniform_lut();
    // Spatial cells match OpenCV's spatial_histogram: the interior (border-stripped
    // region) is split into fixed Rects of `interior / grid` pixels; the leftover
    // border of up to grid-1 pixels is dropped rather than folded into the last cell.
    let (iw, ih) = (w - 2 * r, h - 2 * r);
    let (cell_w, cell_h) = (iw / config.grid_x, ih / config.grid_y);
    let (covered_w, covered_h) = (cell_w * config.grid_x, cell_h * config.grid_y);
    for iy in r..h - r {
        for ix in r..w - r {
            // Pixels outside the OpenCV Rect coverage are not in any cell.
            let (px, py) = (ix - r, iy - r);
            if px >= covered_w || py >= covered_h {
                continue;
            }
            let center = f32::from(face.row(iy)[ix]);
            let mut code: u8 = 0;
            for (n, s) in ring.iter().enumerate() {
                let v = s.weighted_sum(&face, iy, ix);
                // OpenCV tie rule: bit set when strictly greater, or equal within one
                // f32 epsilon (its float-precision guard against 0/1 jitter).
                if v > center || (v - center).abs() < f32::EPSILON {
                    code |= 1 << n;
                }
            }
            let bin = lut[code as usize] as usize;
            let gx = px / cell_w;
            let gy = py / cell_h;
            let cell = gy * config.grid_x + gx;
            hist[cell * BINS + bin] += 1;
        }
    }

    // L1-normalise each cell independently so every spatial region votes with equal
    // weight regardless of interior pixel count.
    let mut values = vec![0.0f32; cells * BINS];
    for cell in 0..cells {
        let slice = &mut values[cell * BINS..cell * BINS + BINS];
        let total: u32 = hist[cell * BINS..cell * BINS + BINS].iter().sum();
        if total > 0 {
            let inv = 1.0 / total as f32;
            for (s, &count) in slice.iter_mut().zip(&hist[cell * BINS..cell * BINS + BINS]) {
                *s = count as f32 * inv;
            }
        }
    }

    LbphDescriptor { values, cells }
}

fn clone_gray(img: &GrayImage) -> GrayImage {
    GrayImage::from_vec(img.as_slice().to_vec(), img.width(), img.height())
}

/// One precomputed bilinear sample position on the neighbour ring.
///
/// Column offsets come from `x = r cos a` and row offsets from `y = -r sin a`
/// (OpenCV `elbp_` convention); weights are the four standard bilinear
/// coefficients `w1` (floor row, floor col) … `w4` (ceil row, ceil col).
#[derive(Clone, Copy, Debug)]
struct NeighborSample {
    fx: isize,
    cx: isize,
    fy: isize,
    cy: isize,
    w1: f32,
    w2: f32,
    w3: f32,
    w4: f32,
}

impl NeighborSample {
    /// Interpolated grey value around center `(iy, ix)`. The caller only runs
    /// over the interior region and every offset is in `[-radius, radius]`, so
    /// all four lookups stay in bounds.
    fn weighted_sum(&self, face: &GrayImage, iy: usize, ix: usize) -> f32 {
        let p = |ry: isize, rx: isize| -> f32 {
            let y = (iy as isize + ry) as usize;
            let x = (ix as isize + rx) as usize;
            f32::from(face.row(y)[x])
        };
        self.w1 * p(self.fy, self.fx)
            + self.w2 * p(self.fy, self.cx)
            + self.w3 * p(self.cy, self.fx)
            + self.w4 * p(self.cy, self.cx)
    }
}

/// Build the [`NEIGHBORS`] bilinear sample positions for `radius`.
///
/// Trig is evaluated in `f64` and the point cast to `f32`, matching OpenCV's
/// `static_cast<float>(radius * cos(...))`, so r=1 diagonals carry the same
/// 0.7071… fractional parts and r=2+ arc positions land identically too.
fn neighbor_ring(radius: usize) -> [NeighborSample; NEIGHBORS] {
    std::array::from_fn(|n| {
        let angle = 2.0 * std::f64::consts::PI * n as f64 / NEIGHBORS as f64;
        let x = (radius as f64 * angle.cos()) as f32;
        let y = (-(radius as f64) * angle.sin()) as f32;
        let fx = x.floor() as isize;
        let cx = x.ceil() as isize;
        let fy = y.floor() as isize;
        let cy = y.ceil() as isize;
        let tx = x - fx as f32;
        let ty = y - fy as f32;
        NeighborSample {
            fx,
            cx,
            fy,
            cy,
            w1: (1.0 - tx) * (1.0 - ty),
            w2: tx * (1.0 - ty),
            w3: (1.0 - tx) * ty,
            w4: tx * ty,
        }
    })
}

// ---------------------------------------------------------------------------
// Uniform pattern lookup
// ---------------------------------------------------------------------------

/// Build the 256-entry uniform-pattern lookup for [`NEIGHBORS`] = 8.
///
/// Uniform codes (≤ two bit transitions around the ring, including all-zero/all-one)
/// are assigned bins 0..58 in ascending numeric label order, matching OpenCV's
/// `LBPH::uniform_lut`; every other code maps to [`NON_UNIFORM_BIN`] (58).
pub(crate) fn uniform_lut() -> [u8; 1 << NEIGHBORS] {
    let mut lut = [NON_UNIFORM_BIN as u8; 1 << NEIGHBORS];
    let mut next_bin: u8 = 0;
    for label in 0u16..(1u16 << NEIGHBORS) {
        if is_uniform(label as u8) {
            lut[label as usize] = next_bin;
            next_bin += 1;
        }
    }
    // Exactly 58 uniform codes for P = 8; non-uniform keeps the last bin.
    debug_assert_eq!(next_bin as usize, BINS - 1);
    lut
}

/// A code is uniform when its bits have at most two 0↔1 transitions around the ring.
fn is_uniform(code: u8) -> bool {
    let mut transitions = 0u8;
    for i in 0..NEIGHBORS as u8 {
        let a = (code >> i) & 1;
        let b = (code >> ((i + 1) % NEIGHBORS as u8)) & 1;
        if a != b {
            transitions += 1;
        }
    }
    transitions <= 2
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_table_has_59_bins_and_maps_non_uniform_to_last_bin() {
        let lut = uniform_lut();
        let uniform_codes = (0u16..256).filter(|&c| is_uniform(c as u8)).count();
        assert_eq!(uniform_codes, BINS - 1); // 58
        let used: std::collections::BTreeSet<u8> = lut.iter().copied().collect();
        assert_eq!(used.len(), BINS);
        // Textbook examples.
        assert_eq!(lut[0x00], 0); // all neighbours dark
        assert!(lut[0xFF] < NON_UNIFORM_BIN as u8); // all bright: uniform
        assert!(is_uniform(0b0000_0001)); // one bright neighbour
        assert!(!is_uniform(0b0001_0011)); // three runs
        assert_eq!(lut[0b0001_0011], NON_UNIFORM_BIN as u8);
    }

    #[test]
    fn constant_image_puts_all_mass_in_zero_code_bin() {
        let img = GrayImage::new(24, 24); // all zero -> every code is 0xFF
        let cfg = LbphConfig {
            face_size: 24,
            ..LbphConfig::default()
        };
        let d = extract(&img, &cfg);
        let cells = cfg.grid_x * cfg.grid_y;
        assert_eq!(d.dim(), cells * BINS);
        for cell in 0..cells {
            for bin in 0..BINS {
                let v = d.as_slice()[cell * BINS + bin];
                if bin == lut_bin_0xff() {
                    assert!((v - 1.0).abs() < 1e-6, "bin {bin} got {v}");
                } else {
                    assert!(v.abs() < 1e-6);
                }
            }
        }
    }

    /// Bin the all-one code maps to (anything uniform, just not the non-uniform bin).
    fn lut_bin_0xff() -> usize {
        uniform_lut()[0xFF] as usize
    }

    #[test]
    fn descriptor_is_invariant_to_monotone_lighting_changes() {
        // The defining LBP property: adding an offset / scaling gain must not change
        // any comparison bit, so the histograms are bit-for-bit identical.
        let mut base = GrayImage::new(40, 40);
        for y in 0..40 {
            for x in 0..40 {
                base.as_mut_slice()[y * 40 + x] = ((x * 7 + y * 13) % 180 + 30) as u8;
            }
        }
        let mut brighter = GrayImage::new(40, 40);
        for (o, &v) in brighter.as_mut_slice().iter_mut().zip(base.as_slice()) {
            *o = (v as u16 + 40).min(255) as u8;
        }
        let cfg = LbphConfig {
            face_size: 40,
            ..LbphConfig::default()
        };
        let d1 = extract(&base, &cfg);
        let d2 = extract(&brighter, &cfg);
        assert_eq!(d1, d2);
        assert!(d1.chi_square(&d2).unwrap() < 1e-6);
    }

    #[test]
    fn chi_square_is_zero_for_identical_and_symmetric() {
        let cfg = LbphConfig::default();
        let a = gradient_crop(48, 3);
        let b = gradient_crop(48, 7);
        let da = extract(&a, &cfg);
        let db = extract(&b, &cfg);
        assert!(da.chi_square(&da).unwrap() < 1e-6);
        assert!((da.chi_square(&db).unwrap() - db.chi_square(&da).unwrap()).abs() < 1e-6);
        assert!(da.chi_square(&db).unwrap() > 0.0);
    }

    #[test]
    fn chi_square_rejects_dimension_mismatch() {
        let cfg = LbphConfig::default();
        let a = extract(&gradient_crop(40, 1), &cfg);
        let other_cfg = LbphConfig {
            grid_x: 4,
            grid_y: 4,
            face_size: 40,
            ..LbphConfig::default()
        };
        let b = extract(&gradient_crop(40, 1), &other_cfg);
        assert!(a.chi_square(&b).is_none());
    }

    #[test]
    fn neighbor_ring_matches_opencv_elbp_convention() {
        let ring = neighbor_ring(1);
        // n = 0: pure east, exact pixel, unit weight.
        assert_eq!((ring[0].fx, ring[0].cx), (1, 1));
        assert_eq!((ring[0].fy, ring[0].cy), (0, 0));
        assert!((ring[0].w1 - 1.0).abs() < 1e-6);
        // n = 1: north-east diagonal at (0.7071, -0.7071). Because OpenCV floors
        // the negative row offset, ty = y - floor(y) = 0.2929 while tx = 0.7071:
        // w2 (the NE pixel) carries 0.5, w1/w4 carry 0.2071, w3 (center) 0.0858.
        let s = &ring[1];
        assert_eq!((s.fx, s.cx), (0, 1));
        assert_eq!((s.fy, s.cy), (-1, 0));
        assert!((s.w2 - 0.5).abs() < 1e-6);
        assert!((s.w1 - 0.2071).abs() < 1e-4);
        assert!((s.w4 - 0.2071).abs() < 1e-4);
        assert!((s.w3 - 0.0858).abs() < 1e-4);
        // Diagonal samples really interpolate: no single weight is 1 or 0.
        assert!(s.w1 > 0.0 && s.w2 < 1.0 && s.w3 > 0.0 && s.w4 > 0.0);
        // n = 2: pure north (row -1); n = 4: pure west (col -1).
        assert_eq!((ring[2].fy, ring[2].cy), (-1, -1));
        assert_eq!((ring[4].fx, ring[4].cx), (-1, -1));
        // n = 6: pure south (row +1).
        assert_eq!((ring[6].fy, ring[6].cy), (1, 1));
    }

    #[test]
    fn bit_zero_is_the_three_oclock_sample() {
        // 3x3 crop: grey center on black surround, only the east pixel is white.
        // The single interior pixel's code must set bits 0 (pure east) and 1
        // (north-east blend reaches the east pixel). Old 6 o'clock convention
        // put this pattern at bits 5/6, so this pins the OpenCV orientation.
        let cfg = LbphConfig {
            face_size: 3,
            grid_x: 1,
            grid_y: 1,
            ..LbphConfig::default()
        };
        let mut img = GrayImage::new(3, 3);
        img.as_mut_slice()[4] = 100; // center (1,1)
        img.as_mut_slice()[5] = 255; // east (row 1, col 2)
        let d = extract(&img, &cfg);
        assert_eq!(d.as_slice()[uniform_lut()[0b0000_0001] as usize], 1.0);

        // Lighting the north-east pixel too flips bit 1 (its w2 weight is 0.5),
        // and no other sample crosses the center — proving the n = 1 geometry.
        let mut img2 = GrayImage::new(3, 3);
        img2.as_mut_slice()[4] = 100; // center
        img2.as_mut_slice()[5] = 255; // east
        img2.as_mut_slice()[2] = 255; // north-east (row 0, col 2)
        let d2 = extract(&img2, &cfg);
        assert_eq!(d2.as_slice()[uniform_lut()[0b0000_0011] as usize], 1.0);
    }

    #[test]
    fn chi_square_rejects_pair_of_zero_mass_descriptors() {
        let cfg = LbphConfig {
            face_size: 1,
            ..LbphConfig::default()
        };
        let blank = extract(&GrayImage::new(1, 1), &cfg);
        assert!(is_zero_mass(blank.as_slice()));
        assert_eq!(blank.chi_square(&blank), None); // no fake perfect match
                                                    // One-sided zero mass still yields a finite positive distance: each cell
                                                    // contributes its L1 mass, so the answer is Some(36.0) for a 6x6 grid.
        let other = extract(&gradient_crop(40, 3), &LbphConfig::default());
        assert_eq!(blank.dim(), other.dim()); // same 6x6 grid -> comparable
        let d = blank.chi_square(&other).unwrap();
        assert!(
            (d - 36.0).abs() < 1e-3,
            "distance to full-mass descriptor {d}"
        );
    }

    #[test]
    fn tiny_crop_does_not_panic_and_yields_zero_descriptor() {
        // face_size 1 means the 1x1 crop is used as-is: smaller than 2*radius, so the
        // border-stripped interior is empty and the descriptor is all zeros.
        let cfg = LbphConfig {
            face_size: 1,
            ..LbphConfig::default()
        };
        let img = GrayImage::new(1, 1);
        let d = extract(&img, &cfg);
        assert_eq!(d.dim(), cfg.grid_x * cfg.grid_y * BINS);
        assert!(d.as_slice().iter().all(|v| *v == 0.0));
    }

    #[test]
    fn recognizer_enrolls_multiple_shots_and_identifies_nearest() {
        let cfg = LbphConfig {
            face_size: 48,
            ..LbphConfig::default()
        };
        let mut rec = LbphRecognizer::new(cfg);
        let alice = gradient_crop(48, 3);
        let alice2 = gradient_crop(48, 3);
        let bob = gradient_crop(48, 11);
        rec.enroll("alice", &alice);
        rec.enroll("alice", &alice2);
        rec.enroll("bob", &bob);
        assert_eq!(rec.len(), 2);
        assert_eq!(rec.crop_count(), 3);

        let ranked = rec.rank_crop(&alice);
        assert_eq!(ranked[0].0, "alice");
        match rec.identify_crop(&alice) {
            LbphMatch::Match { label, .. } => assert_eq!(label, "alice"),
            other => panic!("expected Match, got {other:?}"),
        }
        // Verification returns a finite distance for an enrolled label.
        let d = rec.verify("alice", &alice).unwrap();
        assert!(d < 1e-6);
        assert!(rec.verify("nobody", &alice).is_none());
    }

    #[test]
    fn threshold_rejects_far_probe() {
        let cfg = LbphConfig {
            face_size: 48,
            max_distance: 0.0001,
            ..LbphConfig::default()
        };
        let mut rec = LbphRecognizer::new(cfg);
        rec.enroll("a", &gradient_crop(48, 3));
        match rec.identify_crop(&gradient_crop(48, 11)) {
            LbphMatch::BelowThreshold { best } => assert_eq!(best.unwrap().0, "a"),
            other => panic!("expected BelowThreshold, got {other:?}"),
        }
    }

    #[test]
    fn empty_recognizer_and_remove_work() {
        let mut rec = LbphRecognizer::new(LbphConfig::default());
        assert!(rec.is_empty());
        assert_eq!(
            rec.identify_crop(&gradient_crop(48, 1)),
            LbphMatch::NoCandidates
        );
        rec.enroll("a", &gradient_crop(48, 1));
        rec.enroll("b", &gradient_crop(48, 2));
        assert!(rec.remove("a"));
        assert!(!rec.remove("a"));
        assert_eq!(rec.len(), 1);
        // Index bookkeeping survived the removal.
        assert!(rec.verify("b", &gradient_crop(48, 2)).is_some());
    }

    /// Deterministic textured crop: concentric-square ramp parameterised by `phase`,
    /// so different phases give genuinely different LBP statistics.
    fn gradient_crop(size: usize, phase: usize) -> GrayImage {
        let mut img = GrayImage::new(size, size);
        for y in 0..size {
            for x in 0..size {
                let v = (x.wrapping_mul(phase + 1) + y.wrapping_mul(2 * phase + 1)) % 256;
                img.as_mut_slice()[y * size + x] = v as u8;
            }
        }
        img
    }
}
