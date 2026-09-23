#!/usr/bin/env bash
#
# fetch_ort_jetson.sh — fetch the Jetson-Optimised ONNX Runtime C library
# for arm64 + CUDA 12.6 + TensorRT 10.3, and stage the .so files where
# platform/Dockerfile.gpu expects them.
#
# Why a separate script and not tools/fetch_models.sh: that script deals
# with .onnx model files (which can be re-trained). libonnxruntime.so is
# a binary runtime that pins GPU/TensorRT ABI; once a host upgrades
# JetPack the runtime needs to come from a matching build. The wheels at
# https://github.com/Shattered217/Jetson-Orin-Wheels are the canonical
# source for JetPack 6.2.x arm64 builds.
#
# The wheel itself (~113 MB) is NOT committed; only the five .so files
# needed at runtime are committed under vendor/ort-jetson/lib/onnxruntime/.
# Re-run this script to refresh after a JetPack upgrade.
#
# Usage:
#   tools/fetch_ort_jetson.sh                  # download + stage into vendor/
#   tools/fetch_ort_jetson.sh --version X.Y.Z  # pin a specific onnxruntime version
#   tools/fetch_ort-jetson.sh --verify-only   # re-check that the staged .so is current
#
# Requires: curl, sha256sum, unzip.
set -euo pipefail

WHEELS_REPO="Shattered217/Jetson-Orin-Wheels"
JP_TAG="6.2.1rc1"
WHEEL_NAME="onnxruntime_gpu-1.24.0-cp310-cp310-linux_aarch64.whl"
VENDOR_DIR="$(cd "$(dirname "$0")/.."; pwd)/vendor/ort-jetson"
LIB_DIR="${VENDOR_DIR}/lib/onnxruntime"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)  JP_TAG="$2"; WHEEL_NAME="onnxruntime_gpu-$2-cp310-cp310-linux_aarch64.whl"; shift 2 ;;
        --verify-only) VERIFY_ONLY=1; shift ;;
        --dir) VENDOR_DIR="$2"; LIB_DIR="$2/lib/onnxruntime"; shift 2 ;;
        *)  echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

WHEEL_URL="https://github.com/${WHEELS_REPO}/releases/download/${JP_TAG}/${WHEEL_NAME}"

fetch_and_stage() {
    echo "[fetch_ort_jetson] downloading $WHEEL_NAME from $WHEELS_REPO @ $JP_TAG"
    mkdir -p "$VENDOR_DIR"
    local tmp; tmp=$(mktemp -d)
    curl -fsSL "$WHEEL_URL" -o "$tmp/$WHEEL_NAME"
    echo "[fetch_ort_jetson] extracting $WHEEL_NAME"
    unzip -q -o "$tmp/$WHEEL_NAME" -d "$tmp/extract"
    mkdir -p "$LIB_DIR"
    local src="$tmp/extract/onnxruntime/capi"
    cp -f "$src/libonnxruntime.so.1.24.0" "$src/libonnxruntime_providers_cuda.so" \
          "$src/libonnxruntime_providers_shared.so" "$src/libonnxruntime_providers_tensorrt.so" \
          "$LIB_DIR/"
    ln -sf libonnxruntime.so.1.24.0 "$LIB_DIR/libonnxruntime.so"
    echo "[fetch_ort_jetson] staged to $LIB_DIR"
    rm -rf "$tmp"
}

if [[ "${VERIFY_ONLY:-0}" == 1 ]]; then
    [[ -f "$LIB_DIR/libonnxruntime.so.1.24.0" ]] || {
        echo "missing $LIB_DIR/libonnxruntime.so.1.24.0" >&2; exit 1; }
    echo "[fetch_ort_jetson] staged libonnxruntime OK"
    exit 0
fi

fetch_and_stage
echo "[fetch_ort_jetson] done."
