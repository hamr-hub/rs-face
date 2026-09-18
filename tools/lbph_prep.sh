#!/usr/bin/env bash
#
# lbph_prep.sh — build the real-face dataset for the zero-dep recogniser accuracy benches.
#
# Pipeline:
#   1. source drama frames (*.jpg) -> binary PPM via Pillow (lab-only; the shipped crate
#      has no JPEG decoder and its native PNG reader takes uncompressed blocks only)
#   2. prep_lbph_crops (--features ort-backend): SCRFD detection + ArcFace identity
#      clustering + plain box crops labelled by ArcFace — OFFLINE ground truth only,
#      never part of the evaluated zero-dep path
#   3. bench_lbph (DEFAULT zero-dep build): LBPH pair distances, EER, rank-1
#   4. bench_eigenface (DEFAULT zero-dep build): strict-LOO PCA variant comparison,
#      pair distances, EER, rank-1
#   5. bench_fisherface (DEFAULT zero-dep build): strict-LOO PCA->LDA comparison,
#      pair distances, EER, rank-1 (each writes its own docs/bench-results-*.md)
#
# The recogniser under test never sees an ONNX model: ArcFace only names the folders.
#
# Usage:
#   tools/lbph_prep.sh [FRAMES_DIR] [WORK_DIR]
#
# Defaults: out/rsface_demo/_final  and  out/lbph
# Requires: python3 + Pillow, models/det_10g.onnx, models/w600k_r50.onnx,
#           and a libonnxruntime shared library (set ORT_DYLIB_PATH, see README).

set -euo pipefail

FRAMES_DIR="${1:-out/rsface_demo/_final}"
WORK_DIR="${2:-out/lbph}"
FRAMES_PPM="$WORK_DIR/frames"
CROPS_DIR="$WORK_DIR/crops"

if [[ ! -d "$FRAMES_DIR" ]]; then
    echo "frames dir not found: $FRAMES_DIR" >&2
    exit 2
fi
python3 -c "import PIL" 2>/dev/null || {
    echo "python3 + Pillow required for the JPG->PPM conversion (pip install Pillow)" >&2
    exit 2
}

echo "== 1/5 converting JPG frames to binary PPM ($FRAMES_DIR -> $FRAMES_PPM)"
python3 - "$FRAMES_DIR" "$FRAMES_PPM" <<'PY'
import pathlib
import sys

from PIL import Image

src_root = pathlib.Path(sys.argv[1])
dst_root = pathlib.Path(sys.argv[2])
n = 0
for jpg in sorted(src_root.rglob("*.jpg")):
    rel = jpg.relative_to(src_root)
    dst = (dst_root / rel).with_suffix(".ppm")
    dst.parent.mkdir(parents=True, exist_ok=True)
    # P6 binary RGB, no metadata — exactly what rsface::image::codec::read_ppm parses.
    Image.open(jpg).convert("RGB").save(dst, format="PPM")
    n += 1
print(f"converted {n} frames")
PY

echo "== 2/5 offline ArcFace labelling + box crops (ort backend)"
rm -rf "$CROPS_DIR"
ORT_DYLIB_PATH="${ORT_DYLIB_PATH:-/opt/homebrew/lib/libonnxruntime.dylib}" \
    cargo run --release --features ort-backend --bin prep_lbph_crops -- \
    "$FRAMES_PPM" "$CROPS_DIR"

echo "== 3/5 zero-dep LBPH accuracy evaluation"
cargo run --release --bin bench_lbph -- "$CROPS_DIR"

echo "== 4/5 zero-dep eigenfaces accuracy evaluation (strict LOO)"
cargo run --release --bin bench_eigenface -- "$CROPS_DIR"

echo "== 5/5 zero-dep fisherfaces accuracy evaluation (strict LOO)"
cargo run --release --bin bench_fisherface -- "$CROPS_DIR"
