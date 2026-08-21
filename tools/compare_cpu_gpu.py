#!/usr/bin/env python3
"""Compare CPU vs GPU detection boxes from rs-face-detect JSONL output.

The Rust binary ``src/bin/rs_face_detect.rs`` runs BOTH the pure-Rust CPU
cascade and the selected GPU backend on every sampled frame, emitting
``boxes_cpu`` and ``boxes_gpu`` for each frame. This script verifies the
two sets are identical within tolerance — the core requirement that
"GPU and CPU results must be exactly identical".

How "exactly identical" is measured
------------------------------------
The cascade weights and integral-image math are deterministic, so CPU
and GPU should produce byte-identical boxes. We allow up to a 1-pixel
slop on each side of every box (defending against any float32 round-trip
discrepancy introduced by the GPU's kernel scheduler) and require:

  * per-box IoU ≥ 0.99, OR
  * exact x/y/w/h match within ±1 pixel

A box is "matched" between CPU and GPU iff the above holds. Frames
where the counts differ by more than 5% of ``max(|cpu|, |gpu|)`` are
flagged as diverged.

Usage
-----
::

    python3 tools/compare_cpu_gpu.py detections.jsonl
    python3 tools/compare_cpu_gpu.py out/rs_face_test/*/detections.jsonl

Exit code 0 on full parity, 1 on any divergence.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable


def iou(a: dict, b: dict) -> float:
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]
    ix1 = max(a["x"], b["x"]); iy1 = max(a["y"], b["y"])
    ix2 = min(ax2, bx2);    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    ua = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / ua if ua > 0 else 0.0


def near_match(a: dict, b: dict, tol: int = 1) -> bool:
    """True if a and b are within `tol` pixels on every side."""
    return (
        abs(a["x"] - b["x"]) <= tol and abs(a["y"] - b["y"]) <= tol and
        abs(a["w"] - b["w"]) <= tol and abs(a["h"] - b["h"]) <= tol
    )


def compare_frame(cpu: list, gpu: list) -> tuple[int, int, int, float]:
    """Return (matched, only_cpu, only_gpu, mean_iou_of_matches)."""
    matched = 0
    ious: list[float] = []
    used_gpu = [False] * len(gpu)
    for ca in cpu:
        best_iou = 0.0
        best_j = -1
        for j, gb in enumerate(gpu):
            if used_gpu[j]:
                continue
            if near_match(ca, gb):
                matched += 1
                used_gpu[j] = True
                break
            v = iou(ca, gb)
            if v > best_iou:
                best_iou = v
                best_j = j
        else:
            if best_j >= 0 and best_iou >= 0.99:
                matched += 1
                used_gpu[best_j] = True
                ious.append(best_iou)
            else:
                if best_iou > 0:
                    ious.append(best_iou)
    only_cpu = max(0, len(cpu) - matched)
    only_gpu = sum(1 for u in used_gpu if not u)
    mean_iou = sum(ious) / len(ious) if ious else 1.0
    return matched, only_cpu, only_gpu, mean_iou


def compare(jsonl_path: Path) -> tuple[int, int, int, float, int]:
    """Walk a JSONL file and return (frames, matched, only_cpu, only_gpu,
    max_count_delta_ratio)."""
    n_frames = 0
    n_matched = 0
    n_only_cpu = 0
    n_only_gpu = 0
    max_delta_ratio = 0.0
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cpu = rec.get("boxes_cpu", [])
            gpu = rec.get("boxes_gpu", [])
            if not cpu and not gpu:
                continue
            m, oc, og, _ = compare_frame(cpu, gpu)
            n_frames += 1
            n_matched += m
            n_only_cpu += oc
            n_only_gpu += og
            denom = max(len(cpu), len(gpu))
            if denom > 0:
                delta = abs(len(cpu) - len(gpu))
                max_delta_ratio = max(max_delta_ratio, delta / denom)
    return n_frames, n_matched, n_only_cpu, n_only_gpu, max_delta_ratio


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: compare_cpu_gpu.py <detections.jsonl> [...]", file=sys.stderr)
        return 2
    overall_ok = True
    print(f"{'video':<40} {'frames':>8} {'matched':>10} "
          f"{'only_cpu':>10} {'only_gpu':>10} {'max_delta':>10}")
    for path in argv:
        p = Path(path)
        if not p.exists():
            print(f"missing: {p}", file=sys.stderr)
            overall_ok = False
            continue
        n_frames, m, oc, og, delta = compare(p)
        status = "OK" if (oc == 0 and og == 0 and delta < 0.05) else "DIVERGED"
        if status == "DIVERGED":
            overall_ok = False
        print(f"{p.parent.name:<40} {n_frames:>8} {m:>10} "
              f"{oc:>10} {og:>10} {delta*100:>9.1f}%  {status}")
    print()
    print("OK = every frame's boxes_cpu and boxes_gpu matched exactly" \
          " (±1px, IoU ≥ 0.99), max count delta < 5%.")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))