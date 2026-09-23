#!/bin/sh
# Container entrypoint — links the bind-mounted host TensorRT/cuDNN/CUDA libs
# into one directory so LD_LIBRARY_PATH can be a single entry. This avoids
# baking hundreds of MB into the image (JetPack 6 libs ship on the host,
# not in nvidia/cuda:*-runtime) and means the same image runs against
# any JetPack 6.x host.
set -eu

LIB=/opt/jetson-tensorrt/lib
mkdir -p "$LIB"

# Link only real regular `.so*` files from the bind-mounted paths.
# We deliberately skip:
#   - symlinks (would create cycles)
#   - the dynamic linker itself (ld-linux-*.so.1) and the C library
#     (libc.so.6 / libm.so.6 / libpthread.so.0 / libdl.so.2 / librt.so.1 /
#     libstdc++.so.6 / libgcc_s.so.1 / ld-linux* / libnsl* / libutil* / libcrypt*
#     / libattr* / libcap* / libpcre* / libffi*). These must come from the
#     container's own /usr/lib/aarch64-linux-gnu — the host's glibc 2.35
#     (Ubuntu 22.04) is older than the container's glibc 2.39
#     (Ubuntu 24.04), so swapping it in breaks every dynamic loader.
#   - linker scripts (*.so without a SONAME, which glibc would otherwise
#     try to execute as ELFs)
# The host paths are bind-mounted read-only at:
#   /opt/jetson-tensorrt/lib-host  → /usr/lib/aarch64-linux-gnu
#   /opt/jetson-tensorrt/lib-cuda  → /usr/local/cuda-12.6/targets/aarch64-linux/lib
link_from() {
    src_dir="$1"
    [ -d "$src_dir" ] || return 0
    for f in "$src_dir"/*.so*; do
        [ -f "$f" ] || continue   # skip dirs, symlinks, missing
        case "$(basename "$f")" in
            # Core system loader + C stdlib — must come from the image,
            # never from the host's older glibc.
            ld-*|libc.so.*|libm.so.*|libpthread.so.*|libdl.so.*|librt.so.*|\
            libstdc++.so.*|libgcc_s.so.*|libnsl.so.*|libutil.so.*|libcrypt.so.*|\
            libattr.so.*|libcap.so.*|libpcre.so.*|libffi.so.*) continue ;;
          esac
        base=$(basename "$f")
        ln -sf "$f" "$LIB/$base" 2>/dev/null || true
    done
}

link_from /opt/jetson-tensorrt/lib-host
link_from /opt/jetson-tensorrt/lib-cuda

# Sanity: require libnvinfer + libonnxruntime to be resolvable through $LIB.
missing=0
for lib in libnvinfer.so.10 libnvonnxparser.so.10 libcudnn.so.9 libcudart.so.12 libcublas.so.12 libonnxruntime.so.1.24.0; do
    if [ ! -e "$LIB/$lib" ]; then
        echo "[entrypoint] WARNING: missing $lib in $LIB" >&2
        missing=$((missing + 1))
    fi
done
if [ "$missing" -gt 0 ]; then
    echo "[entrypoint] $missing lib(s) missing; ORT TensorRT/CUDA EP will fail to load." >&2
fi

# Hand off to the server binary.
exec /usr/local/bin/rsface-server "$@"