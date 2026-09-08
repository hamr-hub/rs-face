# Real-Image Benchmark Baseline (2026-09-08)

Baseline numbers measured on this host (aarch64, 6 cores). Use these as
reference when tuning thresholds or chasing regressions.

## Setup
- Image sources: `platform/testdata/lena.jpg` (512×512), `platform/testdata/biden.jpg`
- Both converted to PPM (P6) via `ffmpeg -pix_fmt rgb24`.
- `--algo haar` → demo cascade (8 stages, 8 features) shipped in `cascade.rfcf`.
  Not the real OpenCV Haar — that lives in `haarcascade_frontalface_default.xml`
  and must be converted via `tools/`.
- `--algo luminance` → new zero-dep detector (band-pattern + symmetry +
  edge density + variance), tuned with score_threshold=0.55, nms=0.2,
  min_size=48, band gate 0.60, mid_dim ≥ 60, mid_m ≤ 90.

## Results

| Image | Algo | Wall (s) | Dets | Frames w/ face | Notes |
|-------|------|----------|------|----------------|-------|
| Lena 512×512 | haar (demo)    | 0.12  | 681  | 1/1 | demo cascade is over-permissive — many FPs |
| Lena 512×512 | luminance      | 0.96  | 0    | 0/1 | strict gating missed the canonical portrait |
| Biden ~600×800| haar (demo)    | 1.01  | 4762 | 1/1 | same FP explosion on a larger photo |
| Biden ~600×800| luminance      | 14.62 | 8    | 1/1 | 8 high-confidence detections |

## What this tells us

1. **Demo cascade is unreliable for precision.** Both real faces produce
   hundreds-to-thousands of overlapping detections. The real
   OpenCV Haar cascade (after XML→.rfcf conversion) is more precise; the
   README already calls this out.
2. **Luminance is high-precision, low-recall on studio-lit portraits.**
   The strict band gate (mid_m ≤ 90, mid_dim ≥ 60) was calibrated to
   kill the checkerboard FP explosion. Real-world faces that don't
   produce the canonical forehead-bright/eye-shadow/chin-mid signature
   (e.g. flat-lit magazine photography like Lena) get rejected.
   **Tradeoff is intentional** — the detector is meant as a
   second-opinion / compare-candidate, not a replacement for Haar.
3. **Luminance is slow on large images.** 14.62s on Biden vs 1.01s for
   Haar — the pyramid + per-window band/symmetry/edge computation has
   no SIMD yet. Acceptable for a second-opinion algorithm but not for
   the hot path.

## Tuning cheat sheet (for the next session)

If you need higher recall on studio portraits, relax gates in this order:
- `band` gate 0.60 → 0.45 (allows more "weak band" faces through)
- `mid_dim` 60 → 40 (allows subtler forehead/eye contrast)
- `mid_m` 90 → 110 (allows brighter eye regions, typical of magazine
  fill lighting)

If you see FP explosions on busy backgrounds:
- tighten `band` gate 0.60 → 0.70
- tighten `mid_dim` 60 → 80
- tighten `min_size` 48 → 64 (windows spanning 2+ row blocks needed
  for band calculation stability)

Always re-run the `luminance_face::tests::detects_synthetic_face` and
`luminance_face::tests::checkerboard_rejected` tests after any threshold
change — those two together pin the precision/recall frontier.