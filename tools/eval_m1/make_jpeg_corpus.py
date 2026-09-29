#!/usr/bin/env python3
"""Build a pixel-exact differential corpus for the rs-face JPEG decoder.

The existing tests in tests/jpeg_decode.rs only assert *dimensions* and
"not a uniform colour". A decoder that emits a 16x16 block mosaic satisfies
both, which is exactly how a broken decode shipped. This corpus exists so
the decoder is graded on the only thing that matters: do the pixels match a
reference implementation.

For every case we write
    <name>.jpg      the encoded input
    <name>.raw      the reference decode, PIL/libjpeg RGB, row-major
    manifest.json   { name, width, height } per case

PIL is the oracle because it is libjpeg — the implementation every other
tool agrees with. Our decoder will never be byte-identical to libjpeg (it
uses a float IDCT where libjpeg uses a fixed-point one, and a different
YCbCr->RGB rounding), so the test grades mean-absolute-error and PSNR
rather than exact equality. See tests/jpeg_oracle.rs for the thresholds.

Usage: python3 make_jpeg_corpus.py [--out data/eval_jpeg]
"""
import argparse
import io
import json
import os

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_source_images():
    """A mix of synthetic patterns (exact known content) and real photos.

    Synthetic images make a decode bug *legible* — a wrong DC coefficient
    shows up as a wrong flat block. Real photos make the numbers realistic.
    """
    srcs = {}

    # Real photographs, from the repo and the LFW split.
    for rel in (
        "platform/testdata/biden.jpg",
        "platform/testdata/lena.jpg",
        "platform/testdata/two-people.jpg",
        "data/eval_m1/split/gallery/000_George_W_Bush_0315.jpg",
    ):
        p = os.path.join(ROOT, rel)
        if os.path.exists(p):
            try:
                srcs[os.path.splitext(os.path.basename(rel))[0] + "_src"] = Image.open(p).convert("RGB")
            except Exception as e:  # pragma: no cover - corpus build aid
                print(f"  skip {rel}: {e}")

    # Synthetic: a coarse grid of distinct flat cells. If the decoder gets
    # placement or DC wrong, whole 8x8 blocks go the wrong colour, which is
    # impossible to miss.
    W, H = 64, 48
    grid = np.zeros((H, W, 3), dtype=np.uint8)
    for y in range(H):
        for x in range(W):
            grid[y, x] = ((x * 4) % 256, (y * 5) % 256, ((x + y) * 3) % 256)
    srcs["synth_grid"] = Image.fromarray(grid, "RGB")

    # Synthetic: a high-frequency checkerboard. Stresses AC coefficients and
    # is destroyed by a wrong zig-zag order.
    chk = np.zeros((H, W, 3), dtype=np.uint8)
    chk[:, :] = 255
    chk[::2, ::2] = 0
    chk[1::2, 1::2] = 0
    srcs["synth_checker"] = Image.fromarray(chk, "RGB")

    # Synthetic: smooth radial gradient. Stresses the DC/predictor path.
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt((xx - W / 2) ** 2 + (yy - H / 2) ** 2)
    g = np.clip(255 - r * 4, 0, 255).astype(np.uint8)
    srcs["synth_gradient"] = Image.fromarray(
        np.stack([g, np.flipud(g), g], axis=-1), "RGB"
    )
    return srcs


def save_ref(img, path_raw):
    rgb = img.convert("RGB")
    w, h = rgb.size
    rgb.tobytes()
    with open(path_raw, "wb") as f:
        f.write(rgb.tobytes())
    return w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "eval_jpeg"))
    args = ap.parse_args()
    out = args.out
    os.makedirs(out, exist_ok=True)

    sources = load_source_images()
    manifest = []

    # (suffix, PIL save kwargs) — covers every subsampling the decoder claims.
    variants = [
        ("q95_444", dict(quality=95, subsampling=0)),   # 4:4:4, no chroma resize
        ("q95_422", dict(quality=95, subsampling=1)),   # 4:2:2
        ("q95_420", dict(quality=95, subsampling=2)),   # 4:2:0
        ("q75_420", dict(quality=75, subsampling=2)),   # more quantisation error
        ("q30_420", dict(quality=30, subsampling=2)),   # heavy loss, still legible
        ("q95_420_restart", dict(quality=95, subsampling=2, restart_marker_rows=2)),
    ]

    for sname, src in sources.items():
        for vname, kwargs in variants:
            name = f"{sname}__{vname}"
            buf = io.BytesIO()
            src.save(buf, "JPEG", **kwargs)
            data = buf.getvalue()
            jpath = os.path.join(out, name + ".jpg")
            with open(jpath, "wb") as f:
                f.write(data)
            # Oracle: round-trip through PIL's own decoder.
            ref = Image.open(io.BytesIO(data))
            w, h = save_ref(ref, os.path.join(out, name + ".raw"))
            manifest.append(
                {"name": name, "file": name + ".jpg", "raw": name + ".raw",
                 "width": w, "height": h, "bytes": len(data)}
            )

    # Grayscale-only cases: the decoder has a separate path for these.
    for sname in ("synth_grid", "synth_gradient"):
        src = sources[sname].convert("L")
        for vname, kwargs in (("q95_gray", dict(quality=95)), ("q60_gray", dict(quality=60))):
            name = f"{sname}__{vname}"
            buf = io.BytesIO()
            src.save(buf, "JPEG", **kwargs)
            data = buf.getvalue()
            with open(os.path.join(out, name + ".jpg"), "wb") as f:
                f.write(data)
            ref = Image.open(io.BytesIO(data))
            w, h = save_ref(ref, os.path.join(out, name + ".raw"))
            manifest.append(
                {"name": name, "file": name + ".jpg", "raw": name + ".raw",
                 "width": w, "height": h, "bytes": len(data)}
            )

    # Non-MCU-aligned sizes: the padding path for the last row/column of MCUs.
    base = sources["synth_gradient"]
    for (w, h) in ((17, 13), (33, 1), (1, 33), (100, 7), (7, 100)):
        name = f"odd_{w}x{h}__q95_420"
        small = base.crop((0, 0, min(w, base.width), min(h, base.height)))
        if small.size != (w, h):
            small = Image.new("RGB", (w, h))
            small.paste(base, (0, 0))
        buf = io.BytesIO()
        small.save(buf, "JPEG", quality=95, subsampling=2)
        data = buf.getvalue()
        with open(os.path.join(out, name + ".jpg"), "wb") as f:
            f.write(data)
        ref = Image.open(io.BytesIO(data))
        rw, rh = save_ref(ref, os.path.join(out, name + ".raw"))
        manifest.append({"name": name, "file": name + ".jpg", "raw": name + ".raw",
                         "width": rw, "height": rh, "bytes": len(data)})

    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    total = sum(m["bytes"] for m in manifest)
    print(f"wrote {len(manifest)} cases ({total/1024:.0f} KB jpeg) -> {out}")


if __name__ == "__main__":
    main()
