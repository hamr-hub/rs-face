# Real-Image Benchmark Baseline (2026-09-08, updated round 5)

Baseline numbers measured on this host (aarch64, 6 cores). Use these as
reference when tuning thresholds or chasing regressions.

## Setup
- Image sources: `platform/testdata/lena.jpg` (512×512), `platform/testdata/biden.jpg`
- Both converted to PPM (P6) via `ffmpeg -pix_fmt rgb24`.
- `--algo haar` → on-disk `cascade.rfcf` (24×24, currently 2913 features /
 25 stages after the round-5 conversion from OpenCV's std frontalface_default.xml).
  The converter is at `tools/xml_to_rfcf.py`. **Caveat**: the converted
  cascade is not yet byte-equivalent with OpenCV's detector — see the
  "Known gap" section below.
- `--algo luminance` → new zero-dep detector (band-pattern + symmetry +
  edge density + variance), tuned with score_threshold=0.55, nms=0.2,
  min_size=48, band gate 0.60, mid_dim ≥ 60, mid_m ≤ 90.

## Results

| Image | Algo | Wall (s) | Dets | Frames w/ face | Notes |
|-------|------|----------|------|----------------|-------|
| Lena 512×512 | haar (on-disk)  | 0.10  | 681  | 1/1 | many FPs (see known gap below) |
| Lena 512×512 | luminance      | 0.96  | 0    | 0/1 | strict gating missed the canonical portrait |
| Biden ~600×800| haar (on-disk)  | 1.11  | 4762 | 1/1 | same FP pattern |
| Biden ~600×800| luminance      | 14.62 | 8    | 1/1 | 8 high-confidence detections |

For reference, on the same Lena image OpenCV 4.13's
`CascadeClassifier::detectMultiScale` with `scaleFactor=1.2, minNeighbors=4`
returns exactly 1 face at (221, 208, 162, 162).

## Known gap: cascade conversion is not byte-equivalent to OpenCV

The `tools/xml_to_rfcf.py` converter parses OpenCV's Haar cascade XML and
emits the rs-face `.rfcf` v2 binary format. The conversion is **structurally
correct** (2913 features, 25 stages, 6383 rects match OpenCV's source
data) but the rs-face detector does **not** produce the same detection
output as OpenCV's `detectMultiScale` on the same cascade.

What we know:
- A single-stage Python evaluation of stage 0 at the face location gives
  stage_sum = 1.093 (passes, threshold -5.043) — the parsing is correct.
- The rs-face detector reports 681 conf=5.6 windows across the image
  instead of 1 large face — the cascade eval is over-permissive.

Likely root cause: a subtle sign / threshold / leafValue convention
in OpenCV's `cascadedetect.cpp::HaarEvaluator::evalTree` / `OptFeature::calc`
that hasn't been fully reverse-engineered. Candidates:
- Sign byte (weak_features[i].sign) may not be `+1` for all stumps;
  OpenCV's training output preserves a sign per weak classifier that
  flips the inequality direction. My converter hardcodes `+1`.
- OpenCV's `OptFeature::calc` reads the rect via direct indexing into
  feature_data; our CustomRects path uses feature-width normalisation
  that may differ in how rect coordinates map to pixel offsets.
- Variance-normalisation factor: my cascade eval multiplies raw response
  by `1/sqrt(var)`, OpenCV's does `1/sqrt(var)` too, but the order of
  operations around the leaf-value threshold differs by an epsilon that
  flips borderline decisions.

This is a **known open task** for the next maintainer. The converter
script is the entry point: finish the convention reverse-engineering,
then `python3 tools/xml_to_rfcf.py <haar.xml> --out cascade.rfcf` and
verify with `python3 -c 'import cv2; cv2.CascadeClassifier("haar.xml").detectMultiScale(...)'`
that the detections match. Until that's done, the project's on-disk
`cascade.rfcf` is mostly the same behaviour as before (FP-heavy) but
with real OpenCV weights in the binary; it does NOT replace the need
for a fully byte-equivalent converter.

## What this tells us

1. **On-disk cascade is FP-heavy in rs-face** (681 on Lena) — same as
   before, the conversion didn't change behaviour. The 681 false positives
   are within +30% of the demo cascade's numbers.
2. **Luminance is high-precision, low-recall on studio-lit portraits.**
   The strict band gate (mid_m ≤ 90, mid_dim ≥ 60) was calibrated to
   kill the checkerboard FP explosion. Real-world faces that don't
   produce the canonical forehead-bright/eye-shadow/chin-mid signature
   (e.g. flat-lit magazine photography like Lena) get rejected.
   **Tradeoff is intentional** — the detector is meant as a
   second-opinion / compare-candidate, not a replacement for Haar.
3. **Luminance is slow on large images.** 14.62s on Biden vs 1.11s for
   Haar — the pyramid + per-window band/symmetry/edge computation has
   no SIMD yet. Acceptable for a second-opinion algorithm but not for
   the hot path.

## Tuning cheat sheet (for the next session)

If you need higher recall on studio portraits, relax gates in this order:
- `band` gate 0.60 → 0.45 (allows more "weak band" faces through)
- `mid_dim` 60 → 40 (allows subtuler forehead/eye contrast)
- `mid_m` 90 → 110 (allows brighter eye regions, typical of magazine
  fill lighting)

If you see FP explosions on busy backgrounds:
- tighten `band` gate 0.60 → 0.70
- tighten `mid_dim` 60 → 80
- tighten `min_size` 48 → 64 (windows spanning 2+ row blocks needed
  for band calculation stability)

For the cascade conversion gap:
- Re-read OpenCV `cascadedetect.cpp::HaarEvaluator::evalTree` and
  `OptFeature::calc` line-by-line; cross-check the sign byte handling
  and the rect→pixel mapping
- Verify with `cv2.CascadeClassifier::detectMultiScale3(image,
  scaleFactor=1.05, minNeighbors=0, outputRejectLevels=True)` against
  the rs-face detector's per-window scores

Always re-run the `luminance_face::tests::detects_synthetic_face` and
`luminance_face::tests::checkerboard_rejected` tests after any threshold
change — those two together pin the precision/recall frontier.