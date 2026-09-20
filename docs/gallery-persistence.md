# Gallery & trained-model persistence

How enrolled galleries and trained recognisers survive a process restart, and
the exact on-disk formats behind the `save` / `load` / `to_bytes` /
`from_bytes` methods of each recogniser.

Three hand-rolled formats exist, each with its own four-byte magic:

| format | magic | persists | API |
|---|---|---|---|
| LBPH gallery | `RSLB` v2 | per-crop LBP descriptors (OpenCV-`elbp_` convention); enrolment stays incremental | `LbphRecognizer` |
| Eigenfaces model | `RSEF` v1 | mean, eigenvectors, per-crop projections, config | `EigenfaceRecognizer` |
| Fisherfaces model | `RSLD` v1 | mean, discriminant axes, per-crop projections, config | `FisherfaceRecognizer` |

Shared properties:

* **Zero dependencies.** Every codec is hand-rolled little-endian `std` only —
  no serde, no image files, no model downloads.
* **Crops are not needed.** What gets stored is extracted descriptors or a
  trained subspace plus projected coefficients, never the original face crops.
  Reloading never touches an image decoder, and subspace reload needs no
  retraining.
* **Bit-exact.** All floating point is stored as raw IEEE-754 `f32` bits.
  Distances and rankings after `load` are identical to before `save` (covered
  by round-trip tests asserting ranking equality).
* **Distinct magics.** An `RSEF` blob fed to the fisherfaces decoder (or any
  other cross-feeding) fails with `BadMagic`.

§1–§6 document the LBPH gallery format; §7–§11 document the two trained-model
formats.

## 1. LBPH usage

```rust
use rsface::lbph::{LbphConfig, LbphRecognizer};

// One process: incremental enrolment, then persist.
let mut rec = LbphRecognizer::new(LbphConfig::default());
rec.enroll("alice", &alice_crop_1);
rec.enroll("alice", &alice_crop_2);
rec.enroll("bob", &bob_crop_1);
rec.save("gallery.lbph")?; // atomic temp-file + rename

// Later process: no crops required, the gallery is immediately queryable.
let rec = LbphRecognizer::load("gallery.lbph")?;
for (label, distance) in rec.rank_crop(&probe) {
    println!("{label}: {distance:.3}");
}
```

`to_bytes` / `from_bytes` expose the same encoding as an owned `Vec<u8>` for
embedding in another container (a database row, an HTTP body, a tar entry).
The byte blob is self-describing apart from the fixed 59-bin uniform LBP
vocabulary, which is part of the recogniser definition (`BINS`).

In-memory enrolment remains incremental after loading: `enroll` on a loaded
recogniser works exactly as on a freshly built one, and a later `save` writes
the merged gallery.

## 2. File layout — version 2

All integers are little-endian. All floating point is IEEE-754 binary32,
little-endian.

| offset | size | field |
|---|---:|---|
| 0 | 4 | magic, always `b"RSLB"` |
| 4 | 2 | format version, `u16`, currently `2` |
| 6 | 4 | `radius`, `u32` |
| 10 | 4 | `face_size`, `u32` |
| 14 | 4 | `grid_x`, `u32` |
| 18 | 4 | `grid_y`, `u32` |
| 22 | 4 | `max_distance`, `f32` bits |
| 26 | 4 | `min_margin`, `f32` bits |
| 30 | 1 | `equalize`, `u8` (0 or 1) |
| 31 | 4 | `n_identities`, `u32` |

Then, for each identity:

| size | field |
|---:|---|
| 4 | label byte length `n_l`, `u32` (≤ 4096) |
| `n_l` | UTF-8 label bytes (no interior NUL) |
| 4 | `n_descriptors`, `u32` (≥ 1) |

Then, for each descriptor:

| size | field |
|---:|---|
| 4 | `cells`, `u32` — must equal `grid_x × grid_y` |
| 4 | `n_values`, `u32` — must equal `cells × 59` |
| `4 × n_values` | histogram bins, `f32` each, all finite |

The blob ends immediately after the last descriptor value. A valid file for
an empty gallery is a 35-byte header with `n_identities = 0`.

Version 2 stores descriptors extracted with the OpenCV-`elbp_` convention
(bit 0 at the 3 o'clock sample, bilinear-interpolated neighbour ring, fixed
`interior/grid` spatial cells — see [`recognition-lbph.md`](recognition-lbph.md)).
Version 1 was written only by pre-release builds using a 6 o'clock
nearest-pixel convention; its bins are a non-comparable permutation, so the
decoder rejects it with `UnsupportedVersion(1)` rather than silently mixing
distances.

### Size

A default gallery (6×6 grid, 59 bins) costs **8 504 bytes per crop** (8 496
histogram bytes + an 8-byte descriptor header), plus 8 bytes plus the label
per identity. 100 identities × 5 crops ≈ 4.3 MB — descriptors, not pixels.

## 3. Validation on decode

`from_bytes` is an untrusted-input boundary: every field is validated before a
recogniser is constructed, and construction goes through crate-internal
constructors so an invalid blob cannot produce an inconsistent in-memory
recogniser.

1. Four-byte magic, else `BadMagic`.
2. Version must be `2`, else `UnsupportedVersion(v)` — in particular legacy
   version-1 galleries (pre-release sampler convention) are rejected.
3. Every fixed-width read is bounds-checked — a short blob gives `Truncated`.
4. `equalize` byte is exactly 0 or 1.
5. Config sanity: `radius ≥ 1`, `face_size ≥ 2·radius + 1`,
   `grid_x, grid_y ≥ 1`, thresholds finite and non-negative.
6. Labels are UTF-8, NUL-free, non-empty after the length prefix (an empty
   label is rejected), and unique across the gallery.
7. Every identity has at least one descriptor; every descriptor's `cells`
   equals the header grid area and its value count equals `cells × BINS`.
8. Every stored `f32` is finite (NaN/±∞ rejected).
9. Length headers are capped at sanity limits (10⁶ identities, 10⁶
   descriptors/identity, 4096 label bytes) so a corrupt length cannot drive a
   pathological allocation.
10. No trailing bytes: anything after the declared last value gives
    `TrailingBytes(n)` rather than being silently ignored.

A decode error is returned by value (`LbphStoreError`, `std::error::Error`
implemented) — nothing is aborted, panicked, or logged-and-ignored.

## 4. Atomic saves

`save` writes the encoded blob to a hidden sibling temp file
(`.<name>.tmp` in the destination's directory) and then atomically renames it
over the target. A crash or disk-full during writing therefore leaves either
the previous complete gallery or the new one — never a truncated file at the
real path. On rename failure the temp file is best-effort removed.

The rename is atomic on a single filesystem/POSIX directory; do not point the
temp and target paths at different mounts.

## 5. Compatibility policy

* The version field is bumped for any layout *or sampling-convention* change.
  Old code reading a newer file gets `UnsupportedVersion` rather than
  misparsed data.
* Within a version, fields are append-structural only in the sense documented
  above: the decoder counts bytes, so no implicit padding exists.
* The histogram semantics are fixed to the current **uniform LBP, 59 bins,
  8 neighbours, OpenCV-`elbp_` sampler** descriptor (radius is configurable
  and stored in the header) with L1-normalised per-cell histograms. Changing
  the LBP vocabulary or the sampling convention is an algorithm change that
  gets a new format version (and would not be distance-compatible anyway) —
  this is exactly why v1 (nearest-pixel, 6 o'clock bit 0) was retired to v2
  rather than quietly reinterpreted.

## 6. Non-goals

* **Encryption / access control.** Labels and histograms are stored in the
  clear; protect the file with the surrounding filesystem permissions.
* **Partial updates / incremental file formats.** Enrolment is incremental in
  memory; persistence rewrites the file. Galleries are small (§2), and a
  full rewrite is what makes the atomic-rename guarantee trivial.
* **Cross-language interop.** The layout is documented precisely enough to
  parse elsewhere, but stability guarantees are for this crate only.

## 7. Trained-model usage — eigenfaces (`RSEF`) and Fisherfaces (`RSLD`)

The two subspace recognisers are train-once: persistence stores the trained
model itself — mean vector, projection axes, and every gallery crop's
projected coefficients — so a reload is bit-identical and requires neither the
original crops nor a retrain:

```rust
use rsface::eigenface::{EigenfaceConfig, EigenfaceRecognizer};

// One process: train, then persist.
let rec = EigenfaceRecognizer::train(EigenfaceConfig::default(), training)?;
rec.save("model.eigen")?; // atomic temp-file + rename

// Later process: query immediately, no retrain.
let rec = EigenfaceRecognizer::load("model.eigen")?;
let hits = rec.rank_crop(&probe);
```

`FisherfaceRecognizer` exposes the same four methods (its blob carries the
`RSLD` magic). Adding an identity changes the discriminant subspace, so an
enlarged gallery is persisted by calling `train` again on the full crop list
and overwriting the file — there is no incremental in-memory enrolment for
these two recognisers.

`to_bytes` / `from_bytes` expose the encoding as an owned `Vec<u8>` for the
same embedding uses as the LBPH gallery.

## 8. Trained-model layout — version 1

All integers are little-endian; all floating point is IEEE-754 binary32,
little-endian, raw bits. Let `d = face_size²` (4 096 for the default 64 px
working size) and `k` the number of projection axes.

### 8.1 Common header (both formats, offsets 0–18)

| offset | size | field |
|---|---:|---|
| 0 | 4 | magic: `b"RSEF"` or `b"RSLD"` |
| 4 | 2 | format version, `u16`, currently `1` |
| 6 | 4 | `face_size`, `u32` (1..=1024) |
| 10 | 1 | `equalize`, `u8` (0 or 1) |
| 11 | 4 | `max_distance`, `f32` bits (finite, ≥ 0) |
| 15 | 4 | `min_margin`, `f32` bits (finite, ≥ 0) |

### 8.2 Eigenfaces extension (`RSEF`, offsets 19–37)

| offset | size | field |
|---|---:|---|
| 19 | 1 | `metric`, `u8`: 0 Euclidean, 1 Mahalanobis |
| 20 | 1 | `auto_threshold`, `u8` (0 or 1) |
| 21 | 4 | `variance_kept`, `f32` bits, must be in (0, 1] |
| 25 | 4 | `max_components`, `u32` |
| 29 | 1 | suggested-threshold flag, `u8` (0 or 1) |
| 30 | 4 | suggested threshold, `f32` bits (finite ≥ 0 when flag = 1; 0.0 placeholder otherwise) |
| 34 | 4 | axis count `k`, `u32` |
| 38 | `4·d` | mean vector, exactly `d` `f32`s |
| … | per axis | one `f32` inverse eigenvalue scale (`≥ 0`, the Mahalanobis weight; 0.0 for axes outside the metric's kept tail) followed by the `d`-float unit eigenvector |

### 8.3 Fisherfaces extension (`RSLD`, offsets 19–26)

| offset | size | field |
|---|---:|---|
| 19 | 4 | `max_components`, `u32` |
| 23 | 4 | axis count `k`, `u32` (`k ≥ 1`, at most `identities − 1`) |
| 27 | `4·d` | mean vector, `d` `f32`s |
| … | per axis | `d` `f32`s: one unit discriminant axis (no per-axis scale) |

### 8.4 Member section (both formats)

| size | field |
|---:|---|
| 4 | member count `n`, `u32` (≤ 10⁶; eigenfaces requires `n ≥ 2`, Fisherfaces ≥ 2 distinct labels) |

Then, per member (crop):

| size | field |
|---:|---|
| 4 | label byte length `n_l`, `u32` (1..=4096) |
| `n_l` | UTF-8 label bytes (non-empty, no interior NUL) |
| 4 | coefficient count, `u32` — must equal `k` |
| `4·k` | projected coefficients, `f32` each, all finite |

The blob ends immediately after the last coefficient; trailing bytes are an
error.

### Size

With the default 64 px working size, one vector costs `4·d` = 16 384 bytes.

* **Eigenfaces:** 38-byte header + 16 KiB mean + `k × 16 388` bytes
  (per-axis scale included) + per crop `8 + label + 4·k`. For example a model
  with `k = 50` axes and 77 crops is ≈ 0.84 MB including projections.
* **Fisherfaces:** 27-byte header + 16 KiB mean + `k × 16 384` bytes + the
  same per-crop cost; with 21 identities `k ≤ 20`, so axes + mean stay under
  ≈ 345 KiB.

## 9. Trained-model validation on decode

`from_bytes` is an untrusted-input boundary, mirroring the LBPH decoder:

1. Four-byte magic must match the method (cross-feeding gives `BadMagic`).
2. Version must be `1`, else `UnsupportedVersion(v)`.
3. Every fixed-width read is bounds-checked — a short blob gives `Truncated`.
4. `face_size` is in `1..=1024`; every flag byte is exactly 0 or 1.
5. Thresholds are finite and non-negative; for eigenfaces `variance_kept` is
   in `(0, 1]`, the metric byte is 0 or 1, and a present suggested threshold
   is finite and non-negative.
6. Every per-axis inverse scale (eigenfaces) is finite and non-negative.
7. All vectors (mean, axes, coefficient rows) are exactly the header-declared
   length and contain only finite floats.
8. Header-driven allocation is bounded up front: `d·(k + 1)` and `n·k` must
   not exceed 67 108 864 floats; member count ≤ 10⁶ and labels ≤ 4096 bytes,
   so corrupt counts cannot force a pathological allocation.
9. Labels are UTF-8, non-empty and NUL-free; each member's coefficient count
   equals the model's axis count.
10. Structural minimums: an eigenfaces model needs ≥ 2 projected crops; a
    Fisherfaces model needs `k ≥ 1` and ≥ 2 distinct labels.
11. No trailing bytes after the last coefficient (`TrailingBytes(n)`).

Errors are returned by value via `SubspaceStoreError`
(`std::error::Error` implemented) — nothing is aborted, panicked, or
logged-and-ignored. Construction goes through crate-internal `from_parts`
constructors, so an invalid blob cannot build an inconsistent recogniser.

## 10. Atomic saves

`save` uses the same sibling-temp-plus-rename mechanism and guarantee as the
LBPH gallery (§4): a crash during writing never leaves a half-written model at
the target path on the same filesystem.

## 11. Trained-model compatibility policy

* The version field is bumped for any layout change; old code reading a newer
  file gets `UnsupportedVersion`.
* `RSEF` and `RSLD` are separate magics on purpose: the two blobs are never
  interchangeable, and a future third subspace method gets its own magic.
* The stored semantics are fixed to the current projection math (pixel-space
  unit axes; coefficients in that basis; Euclidean/Mahalanobis metric byte).
  Changing the projection or sampling convention is an algorithm change that
  gets a new version — old distances would not be comparable anyway.
