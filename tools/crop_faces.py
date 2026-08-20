#!/usr/bin/env python3
"""Crop face thumbnails from rs-face annotated output.

Reads each `out/rsface_demo/<video>/manifest.json` written by the rs-face
Rust binary, opens the matching annotated PNG, crops every detection (with
a small pad for context), and writes face thumbnails to
`out/rsface_demo/faces/<video>/face_<idx>.jpg`. Also writes a montage of
the first N faces per video to make visual inspection easy.

Usage:
    python3 tools/crop_faces.py \
        --in-dir out/rsface_demo \
        --out-dir out/rsface_demo/faces \
        --pad 0.15 --max-per-video 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def crop_faces(in_dir: Path, out_dir: Path, pad: float, max_per_video: int) -> int:
    """Process every <video>/ subdirectory under in_dir.

    Reads `detections.jsonl` (one record per sampled frame) — the format
    written by tools/annotate_all_faces.py. Each record has the same
    bbox fields (x/y/w/h) plus a relative `image` path pointing to the
    annotated JPEG inside the same subdirectory.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for video_dir in sorted(p for p in in_dir.iterdir() if p.is_dir()):
        det_path = video_dir / "detections.jsonl"
        if not det_path.exists():
            print(f"[skip] no detections.jsonl in {video_dir}")
            continue
        records = [json.loads(line) for line in det_path.read_text().splitlines() if line.strip()]
        n_with_face = sum(1 for r in records if r["boxes"])
        n_boxes = sum(len(r["boxes"]) for r in records)
        print(f"=== {video_dir.name} ===")
        print(f"  frames_with_face={n_with_face}/{len(records)} "
              f"total_boxes={n_boxes}")
        face_out = out_dir / video_dir.name
        face_out.mkdir(parents=True, exist_ok=True)
        kept = 0
        for fi, fr in enumerate(records):
            if kept >= max_per_video:
                break
            # Older records embed a relative ``image`` field; newer ones
            # (annotate_all_faces v2) drop it because callers reconstruct
            # the path as `<stem>/frame_<idx:06d>.jpg`.
            img_rel = fr.get("image")
            if img_rel:
                img_name = Path(img_rel).name
                img_path = video_dir / img_name
            else:
                img_path = video_dir / f"frame_{fr['frame_index']:06d}.jpg"
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            fh, fw = img.shape[:2]
            for di, det in enumerate(fr["boxes"]):
                if kept >= max_per_video:
                    break
                x, y, w, h = det["x"], det["y"], det["w"], det["h"]
                # Add padding proportional to box size.
                px = int(w * pad)
                py = int(h * pad)
                x0 = max(0, x - px)
                y0 = max(0, y - py)
                x1 = min(fw, x + w + px)
                y1 = min(fh, y + h + py)
                crop = img[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                out_path = face_out / (
                    f"face_f{fr['frame_index']:06d}_d{di}_ts{fr['timestamp_ms']}.jpg"
                )
                cv2.imwrite(str(out_path), crop,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                kept += 1
                total += 1
        print(f"  -> wrote {kept} face crops to {face_out}")
    return total


def make_montage(face_root: Path, cols: int = 5, max_items: int = 30) -> None:
    """Build a single horizontal montage per video for quick inspection."""
    for video_dir in sorted(p for p in face_root.iterdir() if p.is_dir()):
        faces = sorted(p for p in video_dir.glob("face_*.jpg"))
        if not faces:
            continue
        faces = faces[:max_items]
        first = cv2.imread(str(faces[0]))
        if first is None:
            continue
        # Normalize tile size to 200px tall, keep aspect.
        tile_h = 200
        tiles = []
        for f in faces:
            img = cv2.imread(str(f))
            if img is None:
                continue
            ratio = tile_h / img.shape[0]
            tile_w = int(img.shape[1] * ratio)
            tiles.append(cv2.resize(img, (tile_w, tile_h)))
        # Pad each tile to the same width for a clean grid.
        max_w = max(t.shape[1] for t in tiles)
        padded = [cv2.copyMakeBorder(t, 0, 0, 0, max_w - t.shape[1],
                                     cv2.BORDER_CONSTANT, value=(40, 40, 40))
                  for t in tiles]
        # Lay out in cols.
        rows = (len(padded) + cols - 1) // cols
        grid = []
        for r in range(rows):
            row_tiles = padded[r * cols:(r + 1) * cols]
            while len(row_tiles) < cols:
                row_tiles.append(np.zeros((tile_h, max_w, 3), dtype=np.uint8))
            grid.append(cv2.hconcat(row_tiles))
        if grid:
            montage = cv2.vconcat(grid)
            montage_path = video_dir / "_montage.jpg"
            cv2.imwrite(str(montage_path), montage,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            print(f"  montage -> {montage_path}  ({len(padded)} faces, "
                  f"{max_w}x{tile_h} per tile, {len(grid)} rows)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", default="out/rsface_demo", type=Path,
                    help="root dir containing <video>/manifest.json pairs")
    ap.add_argument("--out-dir", default="out/rsface_demo/faces", type=Path)
    ap.add_argument("--pad", default=0.15, type=float,
                    help="fractional padding around each box")
    ap.add_argument("--max-per-video", default=30, type=int)
    args = ap.parse_args()
    if not args.in_dir.exists():
        print(f"in-dir not found: {args.in_dir}")
        return 2
    n = crop_faces(args.in_dir, args.out_dir, args.pad, args.max_per_video)
    print(f"\ntotal face crops: {n}")
    print("\n--- montages ---")
    make_montage(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())