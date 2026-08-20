#!/usr/bin/env python3
"""Parallel face detection across multiple videos using multiprocessing.Pool.

Each worker process handles ONE video end-to-end:
  - Opens the video
  - Samples frames at --sample-fps
  - Runs DNN + Haar detection with the same multi-stage filter pipeline
  - Writes annotated frames + detections.jsonl

This gives true OS-level parallelism (separate Python processes, separate
BLAS/OpenCV thread pools), avoiding the GIL and oversubscription that
plague threaded Python code.

Per-worker metrics (peak RSS, wall time, throughput, accuracy) are
collected and printed as a comparison table. The script also runs an
optional sequential baseline so the speedup is explicit.

Usage:
    python3 tools/parallel_detect.py \
        --in-dir out/rsface_demo/_trimmed \
        --out-dir out/rsface_demo/_annotated \
        --workers 3 \
        --sample-fps 5.0 \
        --min-side 60 \
        --max-aspect 1.6 \
        --min-skin-ratio 0.30 \
        --dnn-conf 0.5 \
        --max-frames 30
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import resource
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Lazy import of cv2 in main() to avoid paying the opencv import cost in the
# parent process. The rotation codes we accept on the CLI are also cv2 symbols
# so we map them at main() time below.
ROTATE_90_CW = 0
ROTATE_180 = 1
ROTATE_90_CCW = 2

import cv2  # used by main() for ROTATE_* constants

# Disable thread oversubscription in worker processes (caller configures
# cv2 / numpy threads at start).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Import lazily inside workers so the parent process doesn't pay the
# opencv import cost.
def _worker_init() -> None:
    """Initialise each worker process for low-memory, single-threaded work."""
    import os
    # Force single-thread BLAS / OpenMP / OpenCV inside every spawn worker.
    # multiprocessing.spawn occasionally drops env vars on macOS; set here
    # *and* via setNumThreads(1) below for defence in depth.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import cv2
    cv2.setNumThreads(1)
    import numpy as np
    # Avoid numpy's automatic multi-threaded ops.
    np.set_printoptions(precision=3)


@dataclass
class WorkerMetrics:
    video: str
    frames_total: int
    frames_with_face: int
    boxes_total: int
    wall_s: float
    throughput_fps: float
    peak_rss_mib: int


def _process_video(args_tuple: tuple[str, str, dict]) -> WorkerMetrics:
    """Worker entry — runs one video end-to-end.

    args_tuple is (video_path, out_path, kwargs) — kwargs holds the
    detection parameters as a plain dict so it pickles cleanly.
    """
    video_path, out_root, kwargs = args_tuple
    _worker_init()
    # Import here so the parent doesn't pay this cost.
    from annotate_all_faces import (
        load_cascades, load_dnn, load_eye_cascade, process_video,
    )
    import cv2

    vp = Path(video_path)
    out = Path(out_root)        # process_video() creates its own <stem> dir.
    cascades = load_cascades()
    eye_cascade = load_eye_cascade()
    dnn_net = load_dnn()

    t0 = time.perf_counter()
    n_boxes = process_video(
        vp, cascades, eye_cascade, dnn_net, out,
        sample_fps=kwargs["sample_fps"],
        min_side=kwargs["min_side"],
        min_skin_ratio=kwargs["min_skin_ratio"],
        max_aspect=kwargs["max_aspect"],
        max_frames=kwargs["max_frames"],
        dnn_conf=kwargs["dnn_conf"],
        cascade_short_side=kwargs["cascade_short_side"],
        cascade_min_neighbors=kwargs["cascade_min_neighbors"],
        cascade_scale_factor=kwargs["cascade_scale_factor"],
        dnn_input_size=kwargs["dnn_input_size"],
        dnn_input_sizes=kwargs["dnn_input_sizes"],
        try_rotations=kwargs["try_rotations"],
        use_symmetry_check=kwargs["use_symmetry_check"],
    )
    wall = time.perf_counter() - t0

    det_path = out / vp.stem / "detections.jsonl"
    frames_total = frames_with_face = 0
    if det_path.exists():
        with det_path.open() as f:
            for line in f:
                rec = json.loads(line)
                frames_total += 1
                if rec.get("boxes"):
                    frames_with_face += 1

    # Peak RSS in MiB.
    rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mib = rss_bytes / 1024 / 1024 if sys.platform != "darwin" else rss_bytes / 1024 / 1024
    if sys.platform == "darwin":
        # macOS reports bytes already.
        rss_mib = rss_bytes / (1024 * 1024)
    throughput = frames_total / wall if wall > 0 else 0.0
    return WorkerMetrics(
        video=vp.stem,
        frames_total=frames_total,
        frames_with_face=frames_with_face,
        boxes_total=n_boxes,
        wall_s=wall,
        throughput_fps=throughput,
        peak_rss_mib=int(rss_mib),
    )


def _run_sequential(jobs: list[tuple[str, str, dict]]) -> list[WorkerMetrics]:
    """Run all videos sequentially in the main process (baseline)."""
    return [_process_video(j) for j in jobs]


def _run_parallel(jobs: list[tuple[str, str, dict]],
                  workers: int) -> list[WorkerMetrics]:
    """Run all videos in a process pool of `workers` workers."""
    if workers <= 1 or len(jobs) <= 1:
        return _run_sequential(jobs)
    ctx = mp.get_context("spawn")  # clean state per worker
    with ctx.Pool(processes=workers, initializer=_worker_init) as pool:
        return pool.map(_process_video, jobs)


def _print_table(label: str, metrics: list[WorkerMetrics], total_wall: float) -> None:
    print(f"\n=== {label} ===")
    print(f"{'video':<24} {'frames':>7} {'face_frames':>11} "
          f"{'boxes':>5} {'wall_s':>7} {'fps':>7} {'peak_MiB':>9}")
    print("-" * 76)
    for m in metrics:
        print(f"{m.video:<24} {m.frames_total:>7} {m.frames_with_face:>11} "
              f"{m.boxes_total:>5} {m.wall_s:>7.2f} {m.throughput_fps:>7.2f} "
              f"{m.peak_rss_mib:>9}")
    print("-" * 76)
    print(f"{'TOTAL':<24} {sum(m.frames_total for m in metrics):>7} "
          f"{sum(m.frames_with_face for m in metrics):>11} "
          f"{sum(m.boxes_total for m in metrics):>5} "
          f"{total_wall:>7.2f}")
    if metrics:
        max_rss = max(m.peak_rss_mib for m in metrics)
        avg_rss = sum(m.peak_rss_mib for m in metrics) / len(metrics)
        print(f"{'  peak RSS (max worker)':<24}{max_rss:>9} MiB")
        print(f"{'  peak RSS (avg worker)':<24}{int(avg_rss):>9} MiB")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--workers", default=0, type=int,
                    help="parallel worker count (0=auto=min(n_videos, cpu//2))")
    ap.add_argument("--sample-fps", default=5.0, type=float)
    ap.add_argument("--min-side", default=60, type=int)
    ap.add_argument("--max-aspect", default=1.6, type=float)
    ap.add_argument("--min-skin-ratio", default=0.30, type=float)
    ap.add_argument("--dnn-conf", default=0.5, type=float)
    ap.add_argument("--max-frames", default=30, type=int)
    ap.add_argument("--cascade-short-side", default=720, type=int,
                    help="downscale frames to this short side for Haar cascade "
                         "(0 = full resolution, slower).")
    ap.add_argument("--cascade-min-neighbors", default=4, type=int,
                    help="Haar detectMultiScale minNeighbors (higher = fewer FPs).")
    ap.add_argument("--cascade-scale-factor", default=1.20, type=float,
                    help="Haar pyramid step (1.05 = dense, 1.2 = fast, 1.3 = coarser).")
    ap.add_argument("--dnn-input-size", default=300, type=int,
                    help="square side fed to Res10 SSD (300 = original, 240 = faster).")
    ap.add_argument("--dnn-input-sizes", default="", type=str,
                    help="comma-separated DNN input sizes to try (overrides --dnn-input-size).")
    ap.add_argument("--try-rotation", action="append", default=[],
                    choices=["90cw", "90ccw", "180"],
                    help="also run the detector on a rotated copy of the frame. "
                         "Can be passed multiple times.")
    ap.add_argument("--no-symmetry-check", action="store_true",
                    help="disable the bilateral symmetry FP filter.")
    ap.add_argument("--skip-baseline", action="store_true",
                    help="skip the sequential baseline run")
    args = ap.parse_args()

    if args.dnn_input_sizes.strip():
        dnn_input_sizes: tuple[int, ...] = tuple(
            int(s) for s in args.dnn_input_sizes.split(",") if s.strip()
        )
    else:
        dnn_input_sizes = (args.dnn_input_size,)
    rotation_map = {"90cw": ROTATE_90_CW,
                    "90ccw": ROTATE_90_CCW,
                    "180": ROTATE_180}
    try_rotations: tuple[int, ...] = tuple(rotation_map[r] for r in args.try_rotation)
    use_symmetry_check = not args.no_symmetry_check

    videos = sorted(p for p in args.in_dir.iterdir()
                    if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"})
    if not videos:
        print(f"no videos under {args.in_dir}", file=sys.stderr)
        return 2

    kwargs = {
        "sample_fps": args.sample_fps,
        "min_side": args.min_side,
        "min_skin_ratio": args.min_skin_ratio,
        "max_aspect": args.max_aspect,
        "max_frames": args.max_frames,
        "dnn_conf": args.dnn_conf,
        "cascade_short_side": args.cascade_short_side,
        "cascade_min_neighbors": args.cascade_min_neighbors,
        "cascade_scale_factor": args.cascade_scale_factor,
        "dnn_input_size": args.dnn_input_size,
        "dnn_input_sizes": dnn_input_sizes,
        "try_rotations": try_rotations,
        "use_symmetry_check": use_symmetry_check,
    }
    jobs = [(str(vp), str(args.out_dir), kwargs) for vp in videos]

    n_cpu = os.cpu_count() or 4
    if args.workers <= 0:
        # Default: half the CPUs, capped at the number of videos.
        workers = max(1, min(len(jobs), n_cpu // 2))
    else:
        workers = args.workers

    # Sequential baseline (optional, slow on big jobs).
    seq_metrics: list[WorkerMetrics] = []
    seq_wall = 0.0
    if not args.skip_baseline and len(jobs) > 1:
        t0 = time.perf_counter()
        seq_metrics = _run_sequential(jobs)
        seq_wall = time.perf_counter() - t0
        _print_table(f"SEQUENTIAL BASELINE (1 worker)", seq_metrics, seq_wall)

    # Parallel run.
    t0 = time.perf_counter()
    par_metrics = _run_parallel(jobs, workers)
    par_wall = time.perf_counter() - t0
    _print_table(f"PARALLEL POOL ({workers} workers)", par_metrics, par_wall)

    if seq_metrics and par_metrics:
        speedup = seq_wall / par_wall if par_wall > 0 else 0.0
        seq_throughput = sum(m.frames_total for m in seq_metrics) / seq_wall
        par_throughput = sum(m.frames_total for m in par_metrics) / par_wall
        seq_max_rss = max(m.peak_rss_mib for m in seq_metrics)
        par_max_rss = max(m.peak_rss_mib for m in par_metrics)
        seq_avg_rss = sum(m.peak_rss_mib for m in seq_metrics) / len(seq_metrics)
        par_avg_rss = sum(m.peak_rss_mib for m in par_metrics) / len(par_metrics)
        print(f"\n=== Comparison ===")
        print(f"  wall time        : {seq_wall:6.2f}s -> {par_wall:6.2f}s   "
              f"speedup {speedup:.2f}x")
        print(f"  total throughput : {seq_throughput:6.2f} -> {par_throughput:6.2f} fps")
        print(f"  peak RSS (worst) : {seq_max_rss:>6} MiB -> {par_max_rss:>6} MiB  "
              f"({'better' if par_max_rss < seq_max_rss else 'same/worse'})")
        print(f"  peak RSS (avg)   : {int(seq_avg_rss):>6} MiB -> {int(par_avg_rss):>6} MiB  "
              f"({'better' if par_avg_rss < seq_avg_rss else 'same/worse'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())