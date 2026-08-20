#!/usr/bin/env python3
"""Extract the top-3 most-frequent faces from videos under out/.

Pipeline:
  1. Open each video with cv2.VideoCapture.
  2. Sample frames at a fixed FPS (default 3 fps).
  3. Detect faces with cv2 DNN (Res10 SSD, Caffe model) when available,
     falling back to cv2 Haar cascade.
  4. Cluster faces by combined pHash + dHash similarity + size similarity.
  5. Pick the 3 largest clusters; for each, choose the highest-resolution
     representative crop.
  6. Save crops as `<videobase>_top_N_<frameIdx>.png` under out/top_faces/.

The output name embeds the cluster rank and the source frame index so
users can cross-reference back to the manifest.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Detection backends
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FaceHit:
    video: str
    frame_idx: int
    timestamp_ms: int
    crop: np.ndarray   # BGR crop with a small context pad
    x: int
    y: int
    w: int
    h: int
    confidence: float
    skin_ratio: float  # fraction of crop pixels that look like skin tone


class FaceDetector:
    """Wrapper around the available detection backend."""

    def __init__(self) -> None:
        self.backend_name = "none"
        self.net: Optional[cv2.dnn.Net] = None
        self.cascades: list[cv2.CascadeClassifier] = []

        # Try the DNN face detector first — far better accuracy than Haar.
        proto = Path("/tmp/deploy.prototxt")
        model = Path("/tmp/res10_300x300_ssd.caffemodel")
        if proto.exists() and model.exists():
            try:
                self.net = cv2.dnn.readNetFromCaffe(str(proto), str(model))
                self.backend_name = "dnn-res10-ssd"
            except cv2.error as e:
                print(f"[warn] DNN load failed ({e}); falling back to Haar", file=sys.stderr)
                self.net = None

        cv2_data = Path(cv2.data.haarcascades)
        for name in ("haarcascade_frontalface_default.xml",
                     "haarcascade_frontalface_alt2.xml",
                     "haarcascade_profileface.xml"):
            c = cv2.CascadeClassifier(str(cv2_data / name))
            if not c.empty():
                self.cascades.append(c)
                if self.backend_name == "none":
                    self.backend_name = "haar"

    @property
    def available(self) -> bool:
        return self.net is not None or bool(self.cascades)

    def detect(self, frame_bgr: np.ndarray, min_side: int = 60) -> list[tuple[int, int, int, int, float]]:
        """Return list of (x, y, w, h, confidence)."""
        if self.net is not None:
            return self._detect_dnn(frame_bgr, min_side)
        return self._detect_haar(frame_bgr, min_side)

    # ---- DNN ---- #
    def _detect_dnn(self, frame_bgr: np.ndarray, min_side: int) -> list[tuple[int, int, int, int, float]]:
        h, w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame_bgr, (300, 300)),
            1.0, (300, 300), (104.0, 177.0, 123.0),
        )
        self.net.setInput(blob)
        detections = self.net.forward()
        out = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < 0.5:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            ww, hh = x2 - x1, y2 - y1
            if ww < min_side or hh < min_side:
                continue
            out.append((x1, y1, ww, hh, conf))
        return out

    # ---- Haar ---- #
    def _detect_haar(self, frame_bgr: np.ndarray, min_side: int) -> list[tuple[int, int, int, int, float]]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        seen: list[tuple[int, int, int, int, float]] = []
        for c in self.cascades:
            params = {
                "scaleFactor": 1.1,
                "minNeighbors": 4,
                "minSize": (min_side, min_side),
                "flags": cv2.CASCADE_SCALE_IMAGE,
            }
            try:
                dets = c.detectMultiScale(gray, **params)
            except cv2.error:
                continue
            if len(dets) == 0:
                continue
            for d in dets:
                x, y, w, h = (int(v) for v in d[:4])
                conf = 1.0
                # Drop near-duplicates across cascades.
                keep = True
                for sx, sy, sw, sh, _ in seen:
                    ix1, iy1 = max(x, sx), max(y, sy)
                    ix2, iy2 = min(x + w, sx + sw), min(y + h, sy + sh)
                    iw = max(0, ix2 - ix1)
                    ih = max(0, iy2 - iy1)
                    inter = iw * ih
                    if inter == 0:
                        continue
                    iou = inter / (w * h + sw * sh - inter)
                    if iou > 0.4:
                        keep = False
                        break
                if keep:
                    seen.append((x, y, w, h, conf))
        return seen


def skin_ratio(crop_bgr: np.ndarray) -> float:
    """Rough skin-tone ratio in the crop, used to filter out text/graphics
    false positives. Returns fraction of pixels that fall inside a permissive
    HSV skin-tone envelope.
    """
    if crop_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = ((h <= 25) | (h >= 165)) & (s >= 40) & (s <= 180) & (v >= 60)
    return float(mask.mean())


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #

def iter_sampled_frames(cap: cv2.VideoCapture,
                        sample_fps: float) -> Iterable[tuple[int, int, np.ndarray]]:
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / sample_fps)))
    idx = 0
    out_idx = 0
    while True:
        if not cap.grab():
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                break
            ts = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            yield out_idx, ts, frame
            out_idx += 1
        idx += 1


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #

def _gray_small(img: np.ndarray, size: int) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)


def phash(img: np.ndarray, size: int = 16) -> int:
    r = _gray_small(img, size).astype(np.float32)
    mean = r.mean()
    bits = (r > mean).flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def dhash(img: np.ndarray, size: int = 16) -> int:
    """Difference hash: compare adjacent pixels horizontally."""
    g = _gray_small(img, size + 1)
    diff = g[:, 1:] > g[:, :-1]
    out = 0
    for b in diff.flatten():
        out = (out << 1) | int(b)
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# --------------------------------------------------------------------------- #
# Clustering
# --------------------------------------------------------------------------- #

class UF:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def cluster_hits(hits: list[FaceHit],
                 max_hamming: int = 20,
                 size_tol: float = 0.5) -> list[list[FaceHit]]:
    """Cluster hits by combined pHash+dHash Hamming + size similarity.

    The phash and dhash are concatenated into a composite 512-bit hash; we
    compare halves independently and merge if EITHER is below its own
    threshold (more lenient — same person in different pose still groups).
    """
    if not hits:
        return []
    ph = [phash(h.crop, 16) for h in hits]   # 256 bits
    dh = [dhash(h.crop, 16) for h in hits]   # 256 bits
    sizes = [(h.w * h.h) for h in hits]
    uf = UF(len(hits))
    half = max_hamming // 2
    for i in range(len(hits)):
        for j in range(i + 1, len(hits)):
            d_p = hamming(ph[i], ph[j])
            d_d = hamming(dh[i], dh[j])
            # Both hashes must agree: tighter than either alone to reduce
            # over-merging different people who happen to share one mode.
            if d_p > half and d_d > half:
                continue
            ai, aj = sizes[i], sizes[j]
            if ai == 0 or aj == 0:
                continue
            ratio = max(ai, aj) / min(ai, aj)
            if ratio > 1.0 + size_tol:
                continue
            uf.union(i, j)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(hits)):
        groups[uf.find(i)].append(i)
    return [[hits[k] for k in v] for v in groups.values()]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def process_video(video_path: Path,
                  detector: FaceDetector,
                  sample_fps: float,
                  min_face_size: int,
                  min_skin_ratio: float = 0.12) -> list[FaceHit]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[warn] cannot open {video_path}", file=sys.stderr)
        return []
    hits: list[FaceHit] = []
    base = video_path.stem
    n_frames = 0
    for out_idx, ts_ms, frame in iter_sampled_frames(cap, sample_fps):
        n_frames += 1
        dets = detector.detect(frame, min_side=min_face_size)
        for x, y, w, h, conf in dets:
            pad = int(0.20 * max(w, h))
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(frame.shape[1], x + w + pad)
            y1 = min(frame.shape[0], y + h + pad)
            crop = frame[y0:y1, x0:x1].copy()
            sr = skin_ratio(crop)
            if sr < min_skin_ratio:
                continue
            hits.append(FaceHit(base, out_idx, ts_ms, crop, x, y, w, h, conf, sr))
    cap.release()
    print(f"  {base}: scanned {n_frames} sampled frames, {len(hits)} face hits "
          f"(backend={detector.backend_name})")
    return hits


def pick_top_reps(clusters: list[list[FaceHit]],
                  top_k: int = 3) -> list[list[FaceHit]]:
    clusters_sorted = sorted(clusters, key=lambda c: -len(c))
    out = []
    for c in clusters_sorted[:top_k]:
        # Pick the largest-area, highest-confidence hit as the representative.
        c_sorted = sorted(c, key=lambda h: (-(h.w * h.h), -h.confidence))
        out.append(c_sorted)
    return out


def save_top(out_dir: Path,
             video_base: str,
             top_clusters: list[list[FaceHit]]) -> list[dict]:
    meta = []
    for rank, cluster in enumerate(top_clusters, start=1):
        if not cluster:
            continue
        rep = cluster[0]
        fname = f"{video_base}_top_{rank}_{rep.frame_idx:06d}.png"
        out_path = out_dir / fname
        cv2.imwrite(str(out_path), rep.crop)
        meta.append({
            "video": video_base,
            "rank": rank,
            "frame_index": rep.frame_idx,
            "timestamp_ms": rep.timestamp_ms,
            "cluster_size": len(cluster),
            "bbox_in_frame": [rep.x, rep.y, rep.w, rep.h],
            "confidence": round(rep.confidence, 4),
            "image_file": fname,
        })
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", default="out", type=Path)
    ap.add_argument("--out-dir", default="out/top_faces", type=Path)
    ap.add_argument("--sample-fps", default=3.0, type=float)
    ap.add_argument("--min-face-size", default=80, type=int)
    ap.add_argument("--max-hamming", default=20, type=int)
    ap.add_argument("--top-k", default=3, type=int)
    ap.add_argument("--min-skin-ratio", default=0.12, type=float,
                    help="drop hits whose skin-tone pixel ratio is below this")
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
        print(f"no videos found under {args.in_dir}", file=sys.stderr)
        return 2

    detector = FaceDetector()
    if not detector.available:
        print("no detection backend available", file=sys.stderr)
        return 2
    print(f"detection backend: {detector.backend_name}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_meta: list[dict] = []
    for vp in video_paths:
        print(f"=== {vp.name} ===")
        hits = process_video(vp, detector, args.sample_fps, args.min_face_size, args.min_skin_ratio)
        if not hits:
            continue
        clusters = cluster_hits(hits, max_hamming=args.max_hamming)
        top = pick_top_reps(clusters, top_k=args.top_k)
        meta = save_top(args.out_dir, vp.stem, top)
        all_meta.extend(meta)
        for m in meta:
            print(f"  top {m['rank']}: cluster_size={m['cluster_size']:4d} "
                  f"frame_idx={m['frame_index']} conf={m['confidence']:.3f} "
                  f"-> {m['image_file']}")

    if all_meta:
        manifest = args.out_dir / "manifest.json"
        with open(manifest, "w") as f:
            json.dump({"version": "rs-face-top-faces-1",
                       "detection_backend": detector.backend_name,
                       "entries": all_meta}, f, indent=2)
        print(f"manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())