#!/usr/bin/env bash
#
# fetch_models.sh — download and verify the face model weights rs-face can consume.
#
# Weights are NOT committed to git: they are 0.2–275 MB binaries with licence terms
# distinct from this crate's. This script fetches them into ./models/ and verifies each
# against the SHA-256 pinned in src/models.rs.
#
# Usage:
#   tools/fetch_models.sh                 # permissively-licensed models only (default)
#   tools/fetch_models.sh --all           # also fetch research-only InsightFace weights
#   tools/fetch_models.sh --pin           # print digests to paste into src/models.rs
#   tools/fetch_models.sh --dir PATH      # target directory (default ./models)
#   tools/fetch_models.sh --verify-only   # re-verify what is already on disk
#
# LICENCE — READ THIS
#
#   YuNet (OpenCV Zoo)            Apache-2.0            commercial use OK
#   SCRFD / ArcFace (InsightFace) research use ONLY     commercial use PROHIBITED
#
# The InsightFace *code* is MIT but its *pre-trained weights* are licensed for
# non-commercial research only; shipping them in a product requires a separate licence
# from DeepInsight. That is why they are opt-in behind --all rather than fetched by
# default: the accurate models are the ones you are least free to deploy, and a default
# that silently pulled them would invite a licence violation.

set -euo pipefail

MODEL_DIR="models"
FETCH_RESEARCH=0
PIN_MODE=0
VERIFY_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)         FETCH_RESEARCH=1; shift ;;
    --pin)         PIN_MODE=1; shift ;;
    --verify-only) VERIFY_ONLY=1; shift ;;
    --dir)         MODEL_DIR="$2"; shift 2 ;;
    -h|--help)     sed -n '2,28p' "$0"; exit 0 ;;
    *)             echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

mkdir -p "$MODEL_DIR"

# --- helpers ----------------------------------------------------------------

# sha256 differs between macOS (shasum) and Linux (sha256sum).
sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | cut -d' ' -f1
  else
    echo "ERROR: neither sha256sum nor shasum found" >&2
    exit 1
  fi
}

# Pull the pinned digest for a filename straight out of src/models.rs, so the script and
# the library can never disagree about what is expected. A duplicated constant here would
# eventually drift.
pinned_digest_for() {
  local file="$1"
  awk -v f="\"$file\"" '
    $0 ~ "file_name:" && $0 ~ f { found=1 }
    found && /sha256:/ {
      if ($0 ~ /None/) { print "none"; exit }
      match($0, /"[0-9a-f]{64}"/)
      if (RSTART > 0) { print substr($0, RSTART+1, 64); exit }
      print "none"; exit
    }
  ' src/models.rs 2>/dev/null || echo "none"
}

verify() {
  local path="$1" name="$2"
  [[ -f "$path" ]] || { echo "  MISSING: $path"; return 1; }

  local actual expected
  actual="$(sha256_of "$path")"
  expected="$(pinned_digest_for "$name")"

  if [[ "$PIN_MODE" == "1" ]]; then
    printf '  %-42s sha256: %s\n' "$name" "$actual"
    return 0
  fi

  if [[ "$expected" == "none" || -z "$expected" ]]; then
    echo "  WARNING: $name has no pinned digest in src/models.rs — integrity NOT verified"
    echo "           actual sha256: $actual"
    echo "           run '$0 --pin' and paste it into the ModelSpec to pin it"
    return 0
  fi

  if [[ "$actual" == "$expected" ]]; then
    echo "  OK: $name (sha256 verified)"
    return 0
  fi

  echo "  FAILED: $name digest mismatch" >&2
  echo "    expected $expected" >&2
  echo "    actual   $actual" >&2
  echo "  Refusing to use this file. Delete it and re-run to retry the download." >&2
  return 1
}

download() {
  local url="$1" dest="$2"
  if [[ -f "$dest" ]]; then
    echo "  already present: $dest"
    return 0
  fi
  echo "  downloading $(basename "$dest") ..."
  # --fail so an HTML error page is not silently saved as a .onnx; -L to follow the
  # GitHub release redirect; write to a temp file so an interrupted transfer never
  # leaves a truncated model that a later run would treat as complete.
  curl -fL --progress-bar -o "$dest.partial" "$url"
  mv "$dest.partial" "$dest"
}

# --- permissive models ------------------------------------------------------

echo "==> Apache-2.0 models (commercial use OK)"

YUNET_URL="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_FILE="face_detection_yunet_2023mar.onnx"

if [[ "$VERIFY_ONLY" == "0" ]]; then
  download "$YUNET_URL" "$MODEL_DIR/$YUNET_FILE"
fi
verify "$MODEL_DIR/$YUNET_FILE" "$YUNET_FILE" || FAILED=1

echo "==> MiniFASNet silent-liveness models (Apache-2.0, commercial use OK)"

LIVENESS_V2_URL="https://github.com/QingHeYang/Silent-Face-Anti-Spoofing-onnx/raw/main/onnx/2.7_80x80_MiniFASNetV2.onnx"
LIVENESS_V2_FILE="2.7_80x80_MiniFASNetV2.onnx"
LIVENESS_V1SE_URL="https://github.com/QingHeYang/Silent-Face-Anti-Spoofing-onnx/raw/main/onnx/4_0_0_80x80_MiniFASNetV1SE.onnx"
LIVENESS_V1SE_FILE="4_0_0_80x80_MiniFASNetV1SE.onnx"

if [[ "$VERIFY_ONLY" == "0" ]]; then
  download "$LIVENESS_V2_URL" "$MODEL_DIR/$LIVENESS_V2_FILE"
  download "$LIVENESS_V1SE_URL" "$MODEL_DIR/$LIVENESS_V1SE_FILE"
fi
verify "$MODEL_DIR/$LIVENESS_V2_FILE" "$LIVENESS_V2_FILE" || FAILED=1
verify "$MODEL_DIR/$LIVENESS_V1SE_FILE" "$LIVENESS_V1SE_FILE" || FAILED=1

# --- research-only models ---------------------------------------------------

if [[ "$FETCH_RESEARCH" == "1" ]]; then
  cat <<'WARN'

==> InsightFace models (buffalo_l)

    *** LICENCE WARNING ***
    These weights are licensed for NON-COMMERCIAL RESEARCH USE ONLY.
    Using them in a commercial product requires a separate licence from DeepInsight.
    See https://github.com/deepinsight/insightface/tree/master/model_zoo

WARN

  BUFFALO_URL="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
  ZIP="$MODEL_DIR/buffalo_l.zip"

  if [[ "$VERIFY_ONLY" == "0" ]]; then
    download "$BUFFALO_URL" "$ZIP"

    echo "  extracting det_10g.onnx and w600k_r50.onnx ..."
    # -j flattens any directory prefix (the pack layout has varied between releases);
    # -o overwrites so a re-run is idempotent.
    unzip -o -j "$ZIP" '*det_10g.onnx' '*w600k_r50.onnx' -d "$MODEL_DIR" >/dev/null
  fi

  verify "$MODEL_DIR/det_10g.onnx" "det_10g.onnx" || FAILED=1
  verify "$MODEL_DIR/w600k_r50.onnx" "w600k_r50.onnx" || FAILED=1
else
  cat <<'SKIP'

==> Skipping InsightFace SCRFD / ArcFace weights.

    These are the highest-accuracy open models available (WIDER FACE hard AP 0.828
    vs YuNet's 0.768; LFW 99.83%) but are RESEARCH-ONLY. Re-run with --all to fetch
    them, having satisfied yourself that your use is non-commercial.

SKIP
fi

# --- summary ----------------------------------------------------------------

echo
if [[ "${FAILED:-0}" == "1" ]]; then
  echo "==> FAILED: one or more models could not be verified." >&2
  exit 1
fi

if [[ "$PIN_MODE" == "1" ]]; then
  echo "==> Paste the digests above into the matching ModelSpec in src/models.rs."
else
  echo "==> Done. Models are in $MODEL_DIR/"
  echo
  echo "    Build with an inference backend to use them:"
  echo "      cargo build --release --features tract-backend   # pure Rust, CPU"
  echo "      cargo build --release --features ort-backend     # ONNX Runtime, GPU-capable"
fi

# --- onnx runtime ------------------------------------------------------------

ORT_PATH=""
if [[ -n "${ORT_DYLIB_PATH:-}" && -f "$ORT_DYLIB_PATH" ]]; then
  ORT_PATH="$ORT_DYLIB_PATH"
elif command -v brew >/dev/null 2>&1 && brew list onnxruntime >/dev/null 2>&1; then
  ORT_PATH="$(brew --prefix onnxruntime 2>/dev/null)/lib/libonnxruntime.dylib"
fi

if [[ -n "$ORT_PATH" && -f "$ORT_PATH" ]]; then
  echo "==> Found ONNX Runtime at $ORT_PATH"
  echo "    export ORT_DYLIB_PATH=\"$ORT_PATH\" before running binaries that use --features ort-backend"
else
  cat <<'ORT'

==> ONNX Runtime (libonnxruntime) was NOT found on this machine.

    The `ort-backend` feature requires the C++ runtime at runtime. Install it via:
      brew install onnxruntime                    # macOS
      apt-get install libonnxruntime-dev          # Debian/Ubuntu
      pip install onnxruntime                     # cross-distro

    Then either export ORT_DYLIB_PATH, or put libonnxruntime.so / .dylib somewhere on
    the OS loader's default search path (e.g. /usr/local/lib on Linux).

ORT
fi
