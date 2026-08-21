#!/usr/bin/env python3
"""Auxiliary runner: invoke ``rs_face_detect`` across one or more videos
in CPU-only and GPU modes, then call ``compare_cpu_gpu.py`` to verify
the two produce identical boxes.

This is auxiliary tooling — the core detection lives in Rust (see
``src/bin/rs_face_detect.rs``). The Python script handles process
spawning, output path naming, peak-RSS measurement, and the
side-by-side parity report.

Usage
-----
::

    python3 tools/run_rust_detect.py \
        --in-dir data/drama-6a056da24fb185585b0928a9 \
        --out-dir out/rs_face_compare \
        --max-frames 30 --sample-fps 5 \
        --backends cpu opencl metal

Output schema
-------------
For each ``<backend>`` requested, the script writes
``<out>/<backend>/<stem>/detections.jsonl`` (the Rust binary already
produits both ``boxes_cpu`` and ``boxes_gpu`` per frame; on this host
the GPU side is filled in when the backend probed successfully).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path


def find_binary(name: str) -> Path:
    """Locate the rs_face_detect binary in target/release or target/debug."""
    for profile in ("release", "debug"):
        cand = Path(f"target/{profile}/{name}")
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"rs_face_detect not found in target/release or target/debug. "
        f"Build it first with: cargo build --release --bin rs_face_detect"
    )


def run_one(binary: Path, video: Path, out_dir: Path, backend: str,
            sample_fps: float, max_frames: int) -> dict:
    t0 = time.monotonic()
    cmd = [
        str(binary),
        str(video),
        "--out", str(out_dir),
        "--backend", backend,
        "--sample-fps", str(sample_fps),
        "--max-frames", str(max_frames),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.monotonic() - t0
    # Peak RSS in KiB on macOS for the child.
    peak_kib = 0
    try:
        # ru_maxrss on macOS is in bytes; on Linux it is in KiB. Normalise to MiB.
        rusage = resource.getrusage(resource.RUSAGE_CHILDREN)
        if sys.platform == "darwin":
            peak_mib = rusage.ru_maxrss / (1024 * 1024)
        else:
            peak_mib = rusage.ru_maxrss / 1024
    except Exception:
        peak_mib = 0.0
    return {
        "video": video.name,
        "backend": backend,
        "wall_s": round(wall, 3),
        "peak_mib": round(peak_mib, 1),
        "exit_code": proc.returncode,
        "stdout_tail": proc.stdout.strip().splitlines()[-5:],
        "stderr_tail": proc.stderr.strip().splitlines()[-3:],
    }


def collect_boxes(jsonl_path: Path) -> tuple[int, int, int]:
    """Return (frames, n_boxes_cpu, n_boxes_gpu)."""
    frames = cpu = gpu = 0
    if not jsonl_path.exists():
        return 0, 0, 0
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            frames += 1
            cpu += len(rec.get("boxes_cpu", []))
            gpu += len(rec.get("boxes_gpu", []))
    return frames, cpu, gpu


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--backends", nargs="+", default=["cpu"],
                    help="backends to run (cpu, opencl, metal, ...)")
    ap.add_argument("--sample-fps", type=float, default=5.0)
    ap.add_argument("--max-frames", type=int, default=30)
    ap.add_argument("--binary", type=Path, default=None,
                    help="path to rs_face_detect (default: target/release)")
    args = ap.parse_args()

    try:
        binary = args.binary or find_binary("rs_face_detect")
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 2

    videos = sorted(p for p in args.in_dir.iterdir()
                    if p.suffix.lower() == ".mp4" and not p.name.startswith("."))
    if not videos:
        print(f"no videos under {args.in_dir}", file=sys.stderr)
        return 2

    rows = []
    print(f"== run_rust_detect ==")
    print(f"  binary  : {binary}")
    print(f"  in-dir  : {args.in_dir}")
    print(f"  out-dir : {args.out_dir}")
    print(f"  backends: {args.backends}")
    print(f"  videos  : {len(videos)}")
    for video in videos:
        for backend in args.backends:
            backend_out = args.out_dir / backend
            run_info = run_one(binary, video, backend_out, backend,
                               args.sample_fps, args.max_frames)
            jsonl = backend_out / video.stem / "detections.jsonl"
            frames, cpu_n, gpu_n = collect_boxes(jsonl)
            run_info["frames"] = frames
            run_info["boxes_cpu"] = cpu_n
            run_info["boxes_gpu"] = gpu_n
            rows.append(run_info)
            print(
                f"  {video.name:<25} backend={backend:<8} "
                f"frames={frames:<5} cpu_boxes={cpu_n:<6} gpu_boxes={gpu_n:<6} "
                f"wall={run_info['wall_s']:.2f}s peak={run_info['peak_mib']:.0f}MiB"
            )

    summary = args.out_dir / "summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    with open(summary, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote summary to {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())