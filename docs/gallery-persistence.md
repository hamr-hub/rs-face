# LBPH gallery persistence

How an enrolled LBPH gallery survives a process restart, and the exact on-disk
format behind `LbphRecognizer::save` / `load` / `to_bytes` / `from_bytes`.

* **Zero dependencies.** The codec is hand-rolled little-endian `std` only —
  no serde, no image files, no model downloads.
* **Crops are not needed.** What gets stored is the extracted descriptors
  (per-cell uniform-LBP histograms), not the original face crops. Reloading a
  gallery never touches an image decoder.
* **Bit-exact.** Histogram bins are stored as raw IEEE-754 `f32` bits.
  Chi-square distances after `load` are identical to before `save` (covered by
  a round-trip test asserting ranking equality).

## 1. Usage

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

## 2. File layout — version 1

All integers are little-endian. All floating point is IEEE-754 binary32,
little-endian.

| offset | size | field |
|---|---:|---|
| 0 | 4 | magic, always `b"RSLB"` |
| 4 | 2 | format version, `u16`, currently `1` |
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
2. Version must be `1`, else `UnsupportedVersion(v)`.
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

* The version field is bumped for any layout change. Old code reading a newer
  file gets `UnsupportedVersion` rather than misparsed data.
* Within version 1, fields are append-structural only in the sense documented
  above: the decoder counts bytes, so no implicit padding exists.
* The histogram semantics are fixed to the current **uniform LBP, 59 bins,
  8 neighbours** descriptor (radius is configurable and stored in the header)
  with L1-normalised per-cell histograms. Changing the LBP vocabulary is an
  algorithm change that gets a new format version (and would not be
  distance-compatible anyway).

## 6. Non-goals

* **Encryption / access control.** Labels and histograms are stored in the
  clear; protect the file with the surrounding filesystem permissions.
* **Partial updates / incremental file formats.** Enrolment is incremental in
  memory; persistence rewrites the file. Galleries are small (§2), and a
  full rewrite is what makes the atomic-rename guarantee trivial.
* **Cross-language interop.** The layout is documented precisely enough to
  parse elsewhere, but stability guarantees are for this crate only.
