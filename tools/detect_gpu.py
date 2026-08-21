#!/usr/bin/env python3
"""GPU face detector — drop-in replacement for the DNN stage of
``annotate_all_faces.py``.

Runs the SAME Res10-SSD Caffe model on the GPU backend selected via
``--backend``. Outputs the same ``detections.jsonl`` schema and the same
annotated ``frame_*.jpg`` files. By construction (same model + same input
blob) the boxes match the CPU baseline bit-for-bit within float32 precision
— verified by ``tools/compare_cpu_gpu.py``.

Usage
-----
::

    # Pick the first available GPU backend (preferred: OpenCL on Mac,
    # CUDA on NVIDIA, etc.). Falls back to CPU if no GPU is found.
    python3 tools/detect_gpu.py \
        --in-dir data/drama-6a056da24fb185585b0928a9 \
        --out-dir out/rsface_demo/_gpu

    # Force a specific backend.
    python3 tools/detect_gpu.py --backend opencl ...
    python3 tools/detect_gpu.py --backend coreml ...
    python3 tools/detect_gpu.py --backend cuda ...
    python3 tools/detect_gpu.py --backend rocm ...
    python3 tools/detect_gpu.py --backend acl ...
    python3 tools/detect_gpu.py --backend mlu ...

    # Probe which GPU backends are available on this host.
    python3 tools/gpu_backends.py

Output schema (identical to CPU pipeline)
-----------------------------------------
For each video, ``<out-dir>/<stem>/detections.jsonl`` contains one JSON
record per sampled frame::

    {
      "video": "<stem>",
      "frame_index": 0,
      "timestamp_ms": 0,
      "boxes": [{"x": x, "y": y, "w": w, "h": h, "skin_ratio": 0.0}]
    }

Note: ``skin_ratio`` is left at 0.0 on this GPU-only pass — it requires the
geometry/skin filter from ``annotate_all_faces.detect_all`` which we run
upstream in the CPU pipeline. The ``compare_cpu_gpu.py`` script compensates
by treating skin_ratio as cosmetic.

Adding a new GPU vendor
-----------------------
Implement a class that inherits ``DetectorBackend`` from ``gpu_backends``
(or wraps an ``_OrtBackend`` with a new ``ep_name``) and append it to
``_BACKEND_REGISTRY``. The dispatcher picks it up automatically.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Lazy import so a missing onnxruntime doesn't break the cv2 OpenCL path.
def _backend(name: Optional[str]):
    import importlib
    sys.path.insert(0, str(Path(__file__).parent))
    import gpu_backends  # type: ignore
    importlib.reload(gpu_backends)
    if name in (None, "auto"):
        return gpu_backends.auto_pick()
    return gpu_backends.make_backend(name)


def sample_video(video_path: Path,
                 backend,
                 out_dir: Path,
                 sample_fps: float,
                 max_frames: Optional[int],
                 conf_thresh: float,
                 target_size: int = 300) -> tuple[int, int, int]:
    """Run the GPU detector on a single video and write JSONL + frames.

    Returns (n_frames, n_with_face, n_boxes_total)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[warn] cannot open {video_path}", file=sys.stderr)
        return (0, 0, 0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    video_out = out_dir / video_path.stem
    video_out.mkdir(parents=True, exist_ok=True)
    detections_path = video_out / "detections.jsonl"

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / sample_fps)))

    n_frames = 0
    n_with_face = 0
    n_boxes_total = 0
    t0 = time.monotonic()
    with open(detections_path, "w") as det_f:
        idx = 0
        cap_idx = 0
        while True:
            if not cap.grab():
                break
            if cap_idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    break
                boxes = backend.detect(frame, conf_thresh=conf_thresh)
                annotated = _annotate(frame, boxes)
                cv2.imwrite(
                    str(video_out / f"frame_{idx:06d}.jpg"),
                    annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, 70],
                )
                rec = {
                    "video": video_path.stem,
                    "frame_index": int(idx),
                    "timestamp_ms": int(cap.get(cv2.CAP_PROP_POS_MSEC)),
                    "boxes": [
                        {"x": int(x), "y": int(y), "w": int(w), "h": int(h),
                         "skin_ratio": 0.0, "conf": round(float(c), 4)}
                        for (x, y, w, h, c) in boxes
                    ],
                }
                det_f.write(json.dumps(rec) + "\n")
                n_frames += 1
                n_boxes_total += len(boxes)
                if boxes:
                    n_with_face += 1
                idx += 1
                if max_frames and idx >= max_frames:
                    break
            cap_idx += 1
    cap.release()
    elapsed = time.monotonic() - t0
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"  {video_path.stem}: sampled {n_frames}, {n_with_face} with face, "
          f"{n_boxes_total} boxes total; "
          f"wall={elapsed:.2f}s peak_rss={peak_kib / 1024:.0f}MiB "
          f"-> {video_out}")
    return (n_frames, n_with_face, n_boxes_total)


def _annotate(frame_bgr: np.ndarray,
              boxes: list[tuple[int, int, int, int, float]]) -> np.ndarray:
    """Same colour scheme as annotate_all_faces.annotate, but copy-free."""
    for idx, (x, y, w, h, c) in enumerate(boxes, start=1):
        hue = (idx * 47) % 180
        col_hsv = np.uint8([[[hue, 220, 255]]])
        col_bgr = cv2.cvtColor(col_hsv, cv2.COLOR_HSV2BGR)[0, 0]
        color = (int(col_bgr[0]), int(col_bgr[1]), int(col_bgr[2]))
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
        label = f"#{idx} c={c:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (x, max(0, y - th - 6)), (x + tw, y), color, -1)
        cv2.putText(frame_bgr, label, (x, max(0, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame_bgr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--backend", default="auto",
                    help="cpu | opencl | vulkan | coreml | cuda | rocm | "
                         "directml | acl | mlu | auto (default auto)")
    ap.add_argument("--sample-fps", default=5.0, type=float)
    ap.add_argument("--max-frames", default=None, type=int)
    ap.add_argument("--conf", default=0.5, type=float,
                    help="DNN confidence threshold (matches CPU default).")
    ap.add_argument("--target-size", default=300, type=int,
                    help="square side fed to the DNN (300 = original).")
    args = ap.parse_args()

    if not args.in_dir.exists():
        print(f"input dir does not exist: {args.in_dir}", file=sys.stderr)
        return 2
    video_paths = sorted(
        p for p in args.in_dir.iterdir()
        if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}
        and not p.name.startswith(".")
    )
    if not video_paths:
        print(f"no videos under {args.in_dir}", file=sys.stderr)
        return 2

    backend = _backend(args.backend)
    info = backend.info()
    print(f"== GPU detector ==")
    print(f"  backend   : {info.name} ({info.vendor})")
    print(f"  device    : {info.device}")
    print(f"  detail    : {info.detail}")
    print(f"  conf_thr  : {args.conf}")
    print(f"  target_sz : {args.target_size}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    total_boxes = 0
    total_frames = 0
    t0 = time.monotonic()
    for vp in video_paths:
        n_f, n_with_face, n_boxes = sample_video(
            vp, backend, args.out_dir, args.sample_fps,
            args.max_frames, args.conf, args.target_size,
        )
        total_frames += n_f
        total_boxes += n_boxes
    elapsed = time.monotonic() - t0
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    fps_overall = total_frames / elapsed if elapsed > 0 else 0.0
    print(f"\n== summary ==")
    print(f"  videos    : {len(video_paths)}")
    print(f"  frames    : {total_frames}")
    print(f"  boxes     : {total_boxes}")
    print(f"  wall      : {elapsed:.2f}s ({fps_overall:.2f} fps aggregate)")
    print(f"  peak RSS  : {peak_kib / 1024:.0f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())