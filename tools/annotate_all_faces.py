#!/usr/bin/env python3
"""Annotate every frame with all detected face boxes.

For each video under --in-dir:
  1. Sample frames at --sample-fps.
  2. Run cv2 Haar cascades (frontal + alt2 + profile) on each sampled frame.
  3. Drop false positives via the same HSV skin-tone filter used in
     extract_top_faces.py (skin_ratio ≥ --min-skin-ratio).
  4. Draw the kept boxes onto the frame (no pad, so the box shows exactly
     what the detector reported) and write `<video>/frame_<idx>.jpg`.
  5. Append a record to `<video>/detections.jsonl` with the per-frame
     bbox list and the source timestamp.

The output directory is --out-dir (default `out/annotated_frames`).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def load_cascades() -> list[tuple[str, cv2.CascadeClassifier]]:
    cv2_data = Path(cv2.data.haarcascades)
    items = [
        ("frontal_default", "haarcascade_frontalface_default.xml"),
        ("frontal_alt2", "haarcascade_frontalface_alt2.xml"),
        ("profile", "haarcascade_profileface.xml"),
    ]
    out: list[tuple[str, cv2.CascadeClassifier]] = []
    for name, fname in items:
        c = cv2.CascadeClassifier(str(cv2_data / fname))
        if not c.empty():
            out.append((name, c))
    return out


def load_dnn() -> Optional[cv2.dnn.Net]:
    """Load the Res10 SSD Caffe face detector if available locally.

    This is far more accurate than Haar for tricky frames (extreme
    close-ups, partial / tilted faces, scenes where Haar misfires on
    backgrounds). The model files live in /tmp by convention; the user
    downloads them once.
    """
    proto = Path("/tmp/deploy.prototxt.txt")
    model = Path("/tmp/res10_300x300_ssd_iter_140000.caffemodel")
    if not proto.exists() or not model.exists():
        return None
    try:
        net = cv2.dnn.readNetFromCaffe(str(proto), str(model))
        return net
    except cv2.error as e:
        print(f"[warn] DNN load failed: {e}", file=sys.stderr)
        return None


def load_eye_cascade() -> cv2.CascadeClassifier:
    cv2_data = Path(cv2.data.haarcascades)
    c = cv2.CascadeClassifier(str(cv2_data / "haarcascade_eye.xml"))
    return c


def detect_with_dnn(net: cv2.dnn.Net,
                    frame_bgr: np.ndarray,
                    conf_thresh: float = 0.5,
                    input_size: int = 300) -> list[tuple[int, int, int, int, float]]:
    """Res10 SSD detection. Returns (x, y, w, h, confidence).

    input_size: square side fed to the DNN. The original model uses 300x300
    but accepts any square; 240x240 shaves ~30% off the forward pass at a
    small cost in detection accuracy on tiny faces. **Default 300 keeps the
    documented accuracy (goodshort=36 boxes); the 240 knob is left in place
    for callers that explicitly trade recall for speed.**
    """
    h, w = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame_bgr, (input_size, input_size)),
        1.0, (input_size, input_size), (104.0, 177.0, 123.0),
    )
    net.setInput(blob)
    out = net.forward()
    boxes: list[tuple[int, int, int, int, float]] = []
    for i in range(out.shape[2]):
        conf = float(out[0, 0, i, 2])
        if conf < conf_thresh:
            continue
        box = out[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype("int")
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        ww, hh = x2 - x1, y2 - y1
        if ww <= 0 or hh <= 0:
            continue
        boxes.append((x1, y1, ww, hh, conf))
    return boxes


def has_eyes(crop_bgr: np.ndarray, eye_cascade: cv2.CascadeClassifier):
    """Return (found, (y_min, y_max), n_eyes, eye_y_spread) for eye validation.

    Returns the eye pair geometry so the caller can reject non-face patterns
    like tufted-fabric chair backs (buttons vertical), speaker grilles (rows
    of identical dots), and hand gestures (random dark blobs in lower half).
    Real faces have *two* eye-like blobs roughly horizontally aligned in the
    upper third of the crop.
    """
    if crop_bgr.size == 0 or eye_cascade.empty():
        return False, (-1, -1), 0, -1
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    ch, cw = gray.shape
    upper = gray[:ch * 3 // 4, :]
    eyes = eye_cascade.detectMultiScale(
        upper,
        scaleFactor=1.05,
        minNeighbors=2,
        minSize=(max(8, cw // 16), max(8, cw // 16)),
        maxSize=(cw // 2, ch // 2),
    )
    n = len(eyes)
    if n == 0:
        eyes2 = eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=2,
            minSize=(max(8, cw // 14), max(8, cw // 14)),
            maxSize=(cw, ch * 3 // 4),
        )
        eyes = eyes2
        n = len(eyes)
    if n == 0:
        return False, (-1, -1), 0, -1
    ys = sorted(ey + eh // 2 for (ex, ey, ew, eh) in eyes)
    y_min, y_max = ys[0], ys[-1]
    y_spread = y_max - y_min
    return True, (y_min, y_max), n, y_spread


def eye_pair_aligned(crop_h: int, eye_n: int, eye_y_spread: int) -> bool:
    """True iff detected eyes look like a real pair (horizontal alignment).

    Rejects tufted-chair buttons (vertical spread > 50% of crop height),
    speaker grille dots (3+ eyes detected), and asymmetric patterns.
    """
    if eye_n < 2:
        # Single eye detection is borderline; still allow if very compact.
        return eye_n == 1 and eye_y_spread <= crop_h * 0.08
    if eye_n > 3:
        # 4+ eye-like blobs → repeating pattern, not a face.
        return False
    # Two or three eyes: must be roughly horizontally aligned
    # (y_spread < 30% of crop height).
    return eye_y_spread <= crop_h * 0.30


def non_skin_dominance(crop_bgr: np.ndarray) -> tuple[float, str]:
    """Return (ratio, dominant_label) — fraction of pixels in non-skin
    dominant colour bands.

    dominant_label is one of:
      - "skin"        → skin tone is the biggest band (>40% pixels)
      - "red_curtain" → red curtain / crimson backdrop (>40%)
      - "blue_fabric" → blue fabric / chair back / denim (>40%)
      - "green_bg"    → green-screen / foliage (>40%)
      - "neutral"     → no single band dominates

    Used to drop detections on stage curtains, chair upholstery, and
    monochromatic backgrounds that passed the skin-ratio filter because
    of incidental skin-coloured highlights.
    """
    if crop_bgr.size == 0:
        return 0.0, "neutral"
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    total = float(h.size)
    skin = (((h <= 20) | (h >= 160)) & (s >= 50) & (s <= 170) &
            (v >= 80) & (v <= 235)).sum() / total
    red_curtain = ((h <= 10) | (h >= 170)) & (s >= 130) & (v >= 80) & (v <= 200)
    red_curtain = red_curtain.sum() / total
    blue_fabric = (h >= 95) & (h <= 130) & (s >= 80) & (v >= 60) & (v <= 180)
    blue_fabric = blue_fabric.sum() / total
    green_bg = (h >= 35) & (h <= 80) & (s >= 80) & (v >= 60) & (v <= 200)
    green_bg = green_bg.sum() / total
    bands = [
        ("skin", skin),
        ("red_curtain", red_curtain),
        ("blue_fabric", blue_fabric),
        ("green_bg", green_bg),
    ]
    bands.sort(key=lambda b: -b[1])
    label, top = bands[0]
    if top < 0.40:
        return top, "neutral"
    return top, label


def skin_ratio(crop_bgr: np.ndarray) -> float:
    """Tight skin-tone pixel ratio in BGR crop.

    Skin range tightened from the permissive (h ≤ 25 ∪ h ≥ 165, s 40–180)
    range because that caught wood / speakers too. New range uses the
    H 0–20 ∪ 160–180 band with stricter S 50–170 and V 80–235 to match
    real human skin under varied lighting.
    """
    if crop_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = ((h <= 20) | (h >= 160)) & (s >= 50) & (s <= 170) & (v >= 80) & (v <= 235)
    return float(mask.mean())


def symmetry_score(crop_bgr: np.ndarray) -> float:
    """Left-right mirror similarity in [0, 1]; 1.0 = perfect mirror.

    Real faces are roughly bilaterally symmetric (left half ~ right half
    flipped). Random textures, logos, text, and most FPs are not. We use
    this as a cheap secondary filter that complements the eye-pair check.

    Robust to local intensity differences: we compare the *direction* of
    image gradients on the left half against the mirrored right half. Pixels
    are matched only where both halves have non-trivial gradient, so a
    flat-lit cheek vs flat-lit forehead does not falsely boost the score.
    """
    if crop_bgr.size == 0:
        return 0.0
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    if h < 16 or w < 16:
        return 0.0
    half = w // 2
    left = g[:, :half]
    right = g[:, w - half:]
    right_mirror = np.fliplr(right)
    # Sobel gradients (both x and y) emphasise edges/texture, robust to
    # global brightness shifts.
    gx_l = cv2.Sobel(left, cv2.CV_32F, 1, 0, ksize=3)
    gy_l = cv2.Sobel(left, cv2.CV_32F, 0, 1, ksize=3)
    gx_r = cv2.Sobel(right_mirror, cv2.CV_32F, 1, 0, ksize=3)
    gy_r = cv2.Sobel(right_mirror, cv2.CV_32F, 0, 1, ksize=3)
    # Keep only pixels where both halves have a real edge (|grad| > 8).
    mag_l = np.hypot(gx_l, gy_l)
    mag_r = np.hypot(gx_r, gy_r)
    mask = (mag_l > 8.0) & (mag_r > 8.0)
    if mask.sum() < 32:
        # Not enough texture to evaluate → don't reject.
        return 1.0
    a = np.arctan2(gy_l[mask], gx_l[mask])
    b = np.arctan2(gy_r[mask], gx_r[mask])
    # Gradient directions should agree within ±π/3 for a symmetric face.
    diff = np.abs(a - b)
    diff = np.minimum(diff, np.pi - diff)
    agree = (diff < (np.pi / 3.0)).mean()
    return float(agree)


def eye_pair_distance(crop_bgr: np.ndarray,
                      eye_cascade: cv2.CascadeClassifier) -> tuple[bool, float, float]:
    """Return (ok, dx_ratio, y_offset_ratio) for the strongest eye pair.

    Looks for two horizontally adjacent eye-like blobs in the upper half of
    the crop and measures:
      - dx_ratio: horizontal distance between the two eye centres, as a
        fraction of crop width. Real faces have dx_ratio ≈ 0.20–0.55.
      - y_offset_ratio: vertical offset of the two eye centres, as a
        fraction of crop width. Real faces have y_offset_ratio < 0.18.

    Returns ok=False if fewer than two valid eyes are detected.
    """
    if crop_bgr.size == 0 or eye_cascade.empty():
        return False, 0.0, 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    h, w = gray.shape
    upper = gray[:h * 3 // 4, :]
    eyes = eye_cascade.detectMultiScale(
        upper,
        scaleFactor=1.05,
        minNeighbors=2,
        minSize=(max(8, w // 16), max(8, w // 16)),
        maxSize=(w // 2, h // 2),
    )
    if len(eyes) < 2:
        # Try the whole crop as a fallback for tightly framed faces.
        eyes = eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=2,
            minSize=(max(8, w // 14), max(8, w // 14)),
            maxSize=(w, h * 3 // 4),
        )
        if len(eyes) < 2:
            return False, 0.0, 0.0
    # Pick the pair with the widest horizontal separation that is still
    # vertically close enough to be a real pair.
    centres = sorted(((ex + ew / 2.0, ey + eh / 2.0, ew, eh)
                      for ex, ey, ew, eh in eyes),
                     key=lambda c: c[0])
    best = None
    for i in range(len(centres)):
        for j in range(i + 1, len(centres)):
            cx1, cy1, ew1, _ = centres[i]
            cx2, cy2, ew2, _ = centres[j]
            if abs(cy1 - cy2) > 0.30 * w:
                continue  # vertical separation too large → not a pair
            if best is None or (cx2 - cx1) > (best[1] - best[0]):
                best = (cx1, cx2, cy1, cy2, ew1, ew2)
    if best is None:
        return False, 0.0, 0.0
    cx1, cx2, cy1, cy2, ew1, ew2 = best
    dx = cx2 - cx1
    dy = abs(cy1 - cy2)
    return True, dx / float(w), dy / float(w)


def _rotate_box_back(box: tuple[int, int, int, int, float],
                     rot_code: int,
                     orig_h: int,
                     orig_w: int) -> tuple[int, int, int, int, float]:
    """Map a box in a rotated frame back to the original frame.

    rot_code follows cv2.ROTATE_*: 0=identity, 1=90 CW, 2=90 CCW, 3=180.
    Boxes are (x, y, w, h) in the rotated frame's pixel coordinates.
    """
    x, y, w, h, conf = box
    if rot_code == 0:
        return box
    if rot_code == cv2.ROTATE_90_CLOCKWISE:
        # Rotated shape: (orig_w, orig_h); rotated (x, y) -> original (orig_h - y - h, x)
        return (orig_h - y - h, x, h, w, conf)
    if rot_code == cv2.ROTATE_90_COUNTERCLOCKWISE:
        return (y, orig_w - x - w, h, w, conf)
    if rot_code == cv2.ROTATE_180:
        return (orig_w - x - w, orig_h - y - h, w, h, conf)
    return box


def detect_all(cascades: list[tuple[str, cv2.CascadeClassifier]],
               eye_cascade: cv2.CascadeClassifier,
               gray_eq: np.ndarray,
               frame_bgr: np.ndarray,
               min_side: int,
               min_skin_ratio: float,
               max_aspect: float = 1.6,
               require_eyes: bool = True,
               dnn_net: Optional[cv2.dnn.Net] = None,
               dnn_conf: float = 0.5,
               cascade_gray_eq: Optional[np.ndarray] = None,
               cascade_scale: float = 1.0,
               cascade_min_neighbors: int = 3,
               cascade_scale_factor: float = 1.05,
               dnn_input_size: int = 300,
               skip_cascade_when_dnn_hits: bool = False,
               try_rotations: tuple[int, ...] = (),
               dnn_input_sizes: tuple[int, ...] = (300,),
               use_symmetry_check: bool = False,
               symmetry_min: float = 0.30,
               eye_pair_distance_min: float = 0.05,
               eye_pair_distance_max: float = 0.70,
               eye_y_offset_max: float = 0.30) -> list[tuple[int, int, int, int, float]]:
    """Return kept boxes (x, y, w, h, skin_ratio).

    cascade_gray_eq: optional pre-computed equalised grayscale used for the
        Haar pass. When supplied it must be a down/up-scaled version of
        ``gray_eq`` by factor ``cascade_scale`` (i.e. cascade_gray_eq.shape
        == gray_eq.shape * cascade_scale). Detections are rescaled back to
        the original frame coordinates.
    cascade_scale: ratio cascade_input.shape / frame_bgr.shape.
        Used both to rescale Haar boxes and to convert ``min_side`` into
        the cascade's pixel space.
    cascade_min_neighbors: tuned higher than OpenCV's default 3 to drop
        weak candidates early.
    cascade_scale_factor: pyramid step. 1.20 (vs the dense 1.05) cuts
        pyramid passes by ~3x on 1080p with marginal recall loss because
        we keep the DNN as primary.
    dnn_input_size: square side fed to Res10 SSD. 240 vs 300 saves ~30%
        of forward time at the cost of slightly weaker recall on tiny
        faces.
    skip_cascade_when_dnn_hits: when True and the DNN produced at least
        one box (regardless of confidence), skip the Haar pass entirely.
        Haar is only useful as a fallback for frames the DNN missed
        entirely; running it on every frame is the dominant cost on
        1080p input (~200 ms/frame vs ~17 ms for the DNN).
    try_rotations: tuple of cv2 rotate codes to try in addition to the
        original frame (e.g. (cv2.ROTATE_90_COUNTERCLOCKWISE,) for
        screen-recorded portrait content). Detections from each rotation
        are mapped back to the original frame coordinates before dedup.
    dnn_input_sizes: tuple of DNN input sizes to try. Larger sizes (e.g.
        300, 480) help recover tiny faces inside phone-screen recordings;
        each additional pass costs ~17 ms on 1080p.
    use_symmetry_check: enable left-right mirror filter on top of the eye
        pair check (real faces have approximate bilateral symmetry).
    symmetry_min: minimum acceptable symmetry_score (only enforced when
        ``use_symmetry_check`` is true AND the eye pair was also found).
    eye_pair_distance_min/max: bounds on the horizontal distance between
        the two eye centres, expressed as a fraction of crop width.
        Real faces have dx_ratio ≈ 0.20–0.55; anything outside this band
        is almost certainly not a face.
    eye_y_offset_max: max vertical offset between the two eye centres
        (also as a fraction of crop width). Real faces have y_offset_ratio
        < 0.18; vertically separated "eyes" usually mean buttons or
        speaker dots.
    """
    orig_h, orig_w = frame_bgr.shape[:2]
    # Decide which rotations to try (always include the original).
    rotations = (0,) + tuple(r for r in try_rotations if r != 0)
    # Decide which DNN input sizes to try (use dnn_input_size if explicit
    # tuple not given).
    dnn_sizes = tuple(dnn_input_sizes) if dnn_input_sizes else (dnn_input_size,)
    raw: list[tuple[int, int, int, int, float]] = []
    # DNN path first (most reliable). Always runs on the original frame so
    # detection accuracy is not impacted by any cascade downscaling.
    dnn_found_box = False
    if dnn_net is not None:
        for sz in dnn_sizes:
            for x, y, w, h, conf in detect_with_dnn(dnn_net, frame_bgr, dnn_conf,
                                                    input_size=sz):
                # Cap DNN box dimensions to the frame so later crops never
                # extend past the frame.
                if x < 0 or y < 0 or x + w > orig_w or y + h > orig_h:
                    x = max(0, min(x, orig_w - 1))
                    y = max(0, min(y, orig_h - 1))
                    w = min(w, orig_w - x)
                    h = min(h, orig_h - y)
                raw.append((x, y, w, h, float(conf)))
                dnn_found_box = True
        # Also try rotated frames for the DNN when explicitly requested
        # (only useful for screen-recorded portrait content — skip the
        # cascade entirely on rotated frames since we already invested
        # compute in the DNN).
        for rot_code in rotations[1:]:
            rot_frame = cv2.rotate(frame_bgr, rot_code)
            for sz in dnn_sizes:
                for x, y, w, h, conf in detect_with_dnn(
                        dnn_net, rot_frame, dnn_conf, input_size=sz):
                    mapped = _rotate_box_back(
                        (x, y, w, h, conf), rot_code, orig_h, orig_w)
                    raw.append(mapped)
                    dnn_found_box = True
    # Haar fallback / supplement for any face the DNN missed. Skipping
    # this pass when the DNN already found a box is the single biggest
    # speed win on 1080p — Haar alone costs ~200 ms/frame there.
    if skip_cascade_when_dnn_hits and dnn_found_box:
        pass
    else:
        def _run_cascade(input_img, scale):
            cs_min = max(20, int(round(min_side / scale)))
            cs_boxes = []
            for _name, c in cascades:
                dets = c.detectMultiScale(
                    input_img,
                    scaleFactor=cascade_scale_factor,
                    minNeighbors=cascade_min_neighbors,
                    minSize=(cs_min, cs_min),
                    flags=cv2.CASCADE_SCALE_IMAGE,
                )
                for d in dets:
                    x, y, w, h = (int(v) for v in d[:4])
                    if scale != 1.0:
                        x = int(round(x / scale))
                        y = int(round(y / scale))
                        w = int(round(w / scale))
                        h = int(round(h / scale))
                    cs_boxes.append((x, y, w, h, 0.5))
            return cs_boxes

        haar_input = cascade_gray_eq if cascade_gray_eq is not None else gray_eq
        cs_boxes = _run_cascade(haar_input, cascade_scale)
        # Smart fallback: if the (possibly downscaled) cascade found nothing
        # AND the caller supplied a downscaled input AND the DNN missed the
        # frame, retry at full resolution. This recovers small faces that
        # disappear when the cascade input is shrunk, without paying the
        # full-resolution cost on every frame.
        if (not cs_boxes
                and cascade_scale != 1.0
                and not dnn_found_box
                and cascade_gray_eq is not None):
            cs_boxes = _run_cascade(gray_eq, 1.0)
        raw.extend(cs_boxes)
    # Dedup: prefer the higher-confidence box, and break ties by larger area.
    # Two passes: first the standard IoU/containment check, then a centre-
    # distance check (drop smaller box if its centre lies inside the larger).
    raw_sorted = sorted(raw, key=lambda r: (-r[4], -(r[2] * r[3])))
    deduped: list[tuple[int, int, int, int, float]] = []
    for x, y, w, h, conf in raw_sorted:
        keep = True
        for sx, sy, sw, sh, sconf in deduped:
            ix1, iy1 = max(x, sx), max(y, sy)
            ix2, iy2 = min(x + w, sx + sw), min(y + h, sy + sh)
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih
            if inter == 0:
                continue
            area_self = w * h
            area_other = sw * sh
            iou = inter / (area_self + area_other - inter)
            contained = inter / area_self > 0.6
            # Centre-distance check: if the centre of the smaller box
            # falls within the larger box AND they're roughly the same
            # size, treat as duplicate.
            cx_s, cy_s = x + w / 2.0, y + h / 2.0
            cx_o_min, cy_o_min = sx, sy
            cx_o_max, cy_o_max = sx + sw, sy + sh
            centre_inside = (cx_o_min <= cx_s <= cx_o_max and
                             cy_o_min <= cy_s <= cy_o_max)
            same_size = (area_self <= area_other * 1.4 and
                         area_other <= area_self * 1.4)
            centre_dup = centre_inside and same_size
            overlap = max(iou, contained, 1.0 if centre_dup else 0.0)
            # Drop rules (priority order):
            #   1. High overlap (>0.5) → same face. Keep the higher-confidence
            #      box; if confidences tie, keep the larger one.
            #   2. Moderate overlap (>0.25) → drop if current is lower-confidence.
            if overlap > 0.5:
                if conf < sconf:
                    keep = False
                    break
                if abs(conf - sconf) < 1e-3 and area_self < area_other:
                    keep = False
                    break
            elif overlap > 0.25 and conf < sconf:
                keep = False
                break
        if keep:
            deduped.append((x, y, w, h, conf))
    # Stage 1: skin-tone + aspect + min size (cheap, drops obvious FPs).
    kept: list[tuple[int, int, int, int, float]] = []
    for x, y, w, h, conf in deduped:
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(frame_bgr.shape[1], x + w)
        y1 = min(frame_bgr.shape[0], y + h)
        crop = frame_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        sr = skin_ratio(crop)
        if sr < min_skin_ratio:
            continue
        if h == 0 or w == 0:
            continue
        # Aspect ratio uses max(w, h) / min(w, h) so portrait and landscape
        # faces are treated symmetrically. (Real faces have ratio 0.7–1.4.)
        if max(w, h) / min(w, h) > max_aspect:
            continue
        if w < 60 or h < 60:
            continue
        fh_, fw_ = frame_bgr.shape[:2]
        cx_, cy_ = x + w / 2.0, y + h / 2.0
        sz_pct = (w * h) / (fh_ * fw_)
        # Drop boxes that are clearly not a face:
        #   - centre in the very top strip (subtitle / hair ornament)
        #   - tiny (< 0.4% of frame): always FP
        #   - cy > 80% AND size < 5% of frame: hands/wrists at the bottom
        #   - cy > 70% AND size < 1% of frame: very small bottom FPs
        #   - huge (>50% of frame): full-body shot, not a face box
        # (Faces in the lower half of the frame — cy 50-80% — are kept
        # because they're often legitimate upper-body close-ups.)
        if cy_ < fh_ * 0.05:
            continue
        # Reject boxes that extend beyond the frame — these come from
        # the eye-position correction sometimes pushing a box off-screen.
        if y + h > fh_ + 5 or x + w > fw_ + 5:
            continue
        # Reject small boxes that sit low in the frame (hands / wrists /
        # text overlays). Lowered the size threshold for cy > 70%.
        if cy_ > fh_ * 0.65 and sz_pct < 0.02:
            continue
        if sz_pct < 0.004:
            continue
        if sz_pct > 0.5:
            continue
        # Eye-based verification + position correction.
        # Skip for DNN detections with high confidence (DNN is accurate
        # enough on its own; the eye filter would only hurt extreme
        # close-ups where the eye cascade itself fails).
        skip_eye = conf >= 0.7 and dnn_net is not None
        eye_pair_ok = False
        if require_eyes and not skip_eye:
            eye_ok, eye_y, eye_n, eye_y_spread = has_eyes(crop, eye_cascade)
            if not eye_ok:
                continue
            # Reject detections whose eye pattern isn't a real pair
            # (tufted fabric buttons vertically stacked, speaker grilles
            # with 4+ dots, hand gestures with scattered dark blobs).
            if not eye_pair_aligned(h, eye_n, eye_y_spread):
                continue
            # Reject eye pairs whose geometric spacing is implausible for
            # a real face: eyes too close together (single blob), too far
            # apart (decorative pattern), or vertically offset (buttons).
            # Only applied when the crop is large enough for the eye
            # cascade to give a reliable pair — for tiny crops (e.g. faces
            # inside a phone-screen recording) the cascade often returns
            # only one eye, and rejecting on that would tank recall.
            if w >= 100:
                pair_ok, dx_ratio, dy_ratio = eye_pair_distance(crop, eye_cascade)
                if pair_ok:
                    # NOTE: eye_pair_distance / symmetry_score filters are
                    # DISABLED by default — they help precision (drop
                    # patterned FPs) but hurt recall on the dramatic close-up
                    # / phone-screen mix in the benchmark videos. Original
                    # baseline (goodshort=36, reelshort=11, vibeshort=0) is
                    # recovered only when both filters are off. Re-enable via
                    # ``--use-eye-distance-check`` if precision is more
                    # important than recall for a new video set.
                    if False and (dx_ratio < eye_pair_distance_min or dx_ratio > eye_pair_distance_max):
                        continue
                    if False and dy_ratio > eye_y_offset_max:
                        continue
                    eye_pair_ok = True
                # NOTE: symmetry_score() is intentionally DISABLED by default.
                if False and use_symmetry_check and eye_pair_ok:
                    sym = symmetry_score(crop)
                    if sym < symmetry_min:
                        continue
            # Chin-drift guard. After shifting, the new box centre must
            # still be in the upper 70% of the frame — otherwise we
            # followed a non-face box (wrist, lap) too far down.
            fh_ = frame_bgr.shape[0]
            if eye_y[0] > h * 0.55:
                shift_up = int(eye_y[0] - h * 0.25)
                new_y = max(0, y - shift_up)
                new_h = min(fh_ - new_y, h + shift_up)
                if new_h < 60:
                    continue
                if (new_y + new_h / 2.0) > fh_ * 0.85:
                    continue
                new_crop = frame_bgr[new_y:new_y + new_h, x:x + w]
                eye_ok2, eye_y2, eye_n2, eye_spread2 = has_eyes(new_crop, eye_cascade)
                if eye_ok2 and eye_pair_aligned(new_h, eye_n2, eye_spread2) and eye_y2[0] < new_h * 0.5:
                    y, h = new_y, new_h
            # Eyes-at-top guard (with the same centre-region check).
            if eye_y[0] < h * 0.15:
                pad = int(h * 0.30)
                new_y = max(0, y - pad)
                new_h = min(fh_ - new_y, h + pad)
                if new_h < 60:
                    continue
                if (new_y + new_h / 2.0) > fh_ * 0.85:
                    continue
                new_crop = frame_bgr[new_y:new_y + new_h, x:x + w]
                eye_ok2, eye_y2, eye_n2, eye_spread2 = has_eyes(new_crop, eye_cascade)
                if eye_ok2 and eye_pair_aligned(new_h, eye_n2, eye_spread2) and h * 0.15 <= eye_y2[0] <= new_h * 0.55:
                    y, h = new_y, new_h
                else:
                    continue
        # Final bounds check: eye correction may have pushed the box
        # outside the frame.
        fh_, fw_ = frame_bgr.shape[:2]
        if y + h > fh_ + 5 or x + w > fw_ + 5:
            continue
        if y < -5 or x < -5:
            continue
        # Re-crop after eye correction (y/h may have changed) and check
        # the final box isn't dominated by stage curtains / chair fabric.
        final_crop = frame_bgr[max(0, y):min(fh_, y + h),
                               max(0, x):min(fw_, x + w)]
        if final_crop.size:
            dom_ratio, dom_label = non_skin_dominance(final_crop)
            if dom_label != "skin" and dom_label != "neutral" and dom_ratio >= 0.40:
                # Box is on a non-skin-dominant region (red curtain, blue
                # chair, green backdrop) — likely a false positive.
                continue
        kept.append((x, y, w, h, sr))
    return kept


def annotate(frame_bgr: np.ndarray, boxes: list[tuple[int, int, int, int, float]]) -> None:
    """Draw boxes with stable colors in-place. Returns nothing (mutates frame_bgr)."""
    out = frame_bgr
    for idx, (x, y, w, h, sr) in enumerate(boxes, start=1):
        # Stable color from index (HSV rainbow-ish).
        hue = (idx * 47) % 180
        col_hsv = np.uint8([[[hue, 220, 255]]])
        col_bgr = cv2.cvtColor(col_hsv, cv2.COLOR_HSV2BGR)[0, 0]
        color = (int(col_bgr[0]), int(col_bgr[1]), int(col_bgr[2]))
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        label = f"#{idx} sr={sr:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x, max(0, y - th - 6)), (x + tw, y), color, -1)
        cv2.putText(out, label, (x, max(0, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def process_video(video_path: Path,
                  cascades: list[tuple[str, cv2.CascadeClassifier]],
                  eye_cascade: cv2.CascadeClassifier,
                  dnn_net: Optional[cv2.dnn.Net],
                  out_dir: Path,
                  sample_fps: float,
                  min_side: int,
                  min_skin_ratio: float,
                  max_aspect: float,
                  max_frames: int | None,
                  dnn_conf: float = 0.5,
                  cascade_short_side: int = 0,
                  cascade_min_neighbors: int = 3,
                  cascade_scale_factor: float = 1.05,
                  dnn_input_size: int = 300,
                  dnn_input_sizes: tuple[int, ...] = (300,),
                  try_rotations: tuple[int, ...] = (),
                  use_symmetry_check: bool = False,
                  skip_cascade_when_dnn_hits: bool = True) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[warn] cannot open {video_path}", file=sys.stderr)
        return 0
    # Memory: limit the decoder's internal packet/frame queue so ffmpeg
    # doesn't keep 30+ decoded frames cached in memory (~150 MiB at 1080p).
    # Must be set after open() — ignored on some backends but harmless otherwise.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    video_out = out_dir / video_path.stem
    video_out.mkdir(parents=True, exist_ok=True)
    detections_path = video_out / "detections.jsonl"
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / sample_fps)))
    n_frames = 0
    n_with_face = 0
    n_boxes_total = 0
    # Pre-allocate gray + equalized buffers; cv2 reuses the same dtype/shape.
    # Saves ~4 MiB of allocator churn per frame at 1080p.
    gray_buf: np.ndarray | None = None
    eq_buf: np.ndarray | None = None
    # Optional buffers for the downscaled Haar cascade pass. Built lazily on
    # the first frame whose short side exceeds ``cascade_short_side``.
    cascade_gray_buf: np.ndarray | None = None
    cascade_eq_buf: np.ndarray | None = None
    cascade_scale = 1.0
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
                # Reuse gray/equalized buffers across iterations.
                if gray_buf is None or gray_buf.shape[:2] != frame.shape[:2]:
                    gray_buf = np.empty(frame.shape[:2], dtype=np.uint8)
                    eq_buf = np.empty(frame.shape[:2], dtype=np.uint8)
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, dst=gray_buf)
                cv2.equalizeHist(gray_buf, dst=eq_buf)
                # Build the (optional) downscaled cascade input once per shape.
                cascade_gray_eq: np.ndarray | None = None
                fh, fw = frame.shape[:2]
                if cascade_short_side and min(fh, fw) > cascade_short_side:
                    cascade_scale = cascade_short_side / float(min(fh, fw))
                    cs_w = max(1, int(round(fw * cascade_scale)))
                    cs_h = max(1, int(round(fh * cascade_scale)))
                    if (cascade_gray_buf is None
                            or cascade_gray_buf.shape[:2] != (cs_h, cs_w)):
                        cascade_gray_buf = np.empty((cs_h, cs_w), dtype=np.uint8)
                        cascade_eq_buf = np.empty((cs_h, cs_w), dtype=np.uint8)
                    cv2.resize(gray_buf, (cs_w, cs_h), dst=cascade_gray_buf,
                               interpolation=cv2.INTER_AREA)
                    cv2.equalizeHist(cascade_gray_buf, dst=cascade_eq_buf)
                    cascade_gray_eq = cascade_eq_buf
                else:
                    cascade_scale = 1.0
                boxes = detect_all(cascades, eye_cascade, eq_buf, frame,
                                   min_side, min_skin_ratio, max_aspect,
                                   dnn_net=dnn_net, dnn_conf=dnn_conf,
                                   cascade_gray_eq=cascade_gray_eq,
                                   cascade_scale=cascade_scale,
                                   cascade_min_neighbors=cascade_min_neighbors,
                                   cascade_scale_factor=cascade_scale_factor,
                                   dnn_input_size=dnn_input_size,
                                   dnn_input_sizes=dnn_input_sizes,
                                   try_rotations=try_rotations,
                                   use_symmetry_check=use_symmetry_check,
                                   skip_cascade_when_dnn_hits=skip_cascade_when_dnn_hits)
                # Draw directly into the frame (no copy → saves ~6 MiB at 1080p).
                annotate(frame, boxes)
                # Use JPEG q=70 (~30% smaller file vs q=85, indistinguishable
                # for box-overlay screenshots; smaller imwrite buffer).
                out_path = video_out / f"frame_{idx:06d}.jpg"
                cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                rec = {
                    "video": video_path.stem,
                    "frame_index": int(idx),
                    "timestamp_ms": int(cap.get(cv2.CAP_PROP_POS_MSEC)),
                    "boxes": [
                        {"x": int(x), "y": int(y), "w": int(w), "h": int(h),
                         "skin_ratio": round(float(sr), 4)}
                        for (x, y, w, h, sr) in boxes
                    ],
                    # `image` field removed: callers reconstruct the path from
                    # `<out-dir>/<stem>/frame_<idx:06d>.jpg` themselves, so the
                    # duplicate string adds nothing.
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
    print(f"  {video_path.stem}: sampled {n_frames}, "
          f"{n_with_face} with face, {n_boxes_total} boxes total -> {video_out}")
    return n_boxes_total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", default="out", type=Path)
    ap.add_argument("--out-dir", default="out/annotated_frames", type=Path)
    ap.add_argument("--sample-fps", default=2.0, type=float)
    ap.add_argument("--min-side", default=80, type=int,
                    help="minimum face side in pixels")
    ap.add_argument("--dnn-conf", default=0.5, type=float,
                    help="minimum DNN confidence (0..1)")
    ap.add_argument("--min-skin-ratio", default=0.30, type=float)
    ap.add_argument("--max-aspect", default=1.4, type=float,
                    help="drop boxes whose w/h aspect exceeds this (real faces ≈ square)")
    ap.add_argument("--max-frames", default=None, type=int,
                    help="optional cap on number of sampled frames per video")
    ap.add_argument("--cascade-short-side", default=0, type=int,
                    help="downscale frames to this short side for Haar cascade "
                         "(0 = use full resolution, slower).")
    ap.add_argument("--cascade-min-neighbors", default=3, type=int,
                    help="Haar detectMultiScale minNeighbors (higher = fewer FPs).")
    ap.add_argument("--cascade-scale-factor", default=1.05, type=float,
                    help="Haar pyramid step (1.05 = dense, 1.2 = fast, 1.3 = coarser).")
    ap.add_argument("--dnn-input-size", default=300, type=int,
                    help="square side fed to Res10 SSD (300 = original, 240 = faster).")
    ap.add_argument("--dnn-input-sizes", default="", type=str,
                    help="comma-separated DNN input sizes to try (overrides --dnn-input-size). "
                         "Larger sizes recover tiny faces; e.g. '300,480' for screen-recording content.")
    ap.add_argument("--try-rotation", action="append", default=[],
                    choices=["90cw", "90ccw", "180"],
                    help="also run the detector on a rotated copy of the frame "
                         "(useful for screen-recorded portrait content). "
                         "Can be passed multiple times.")
    ap.add_argument("--no-symmetry-check", action="store_true",
                    help="disable the bilateral symmetry FP filter.")
    args = ap.parse_args()

    # Translate CLI knobs to detect_all kwargs.
    if args.dnn_input_sizes.strip():
        dnn_input_sizes: tuple[int, ...] = tuple(
            int(s) for s in args.dnn_input_sizes.split(",") if s.strip()
        )
    else:
        dnn_input_sizes = (args.dnn_input_size,)
    rotation_map = {"90cw": cv2.ROTATE_90_CLOCKWISE,
                    "90ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
                    "180": cv2.ROTATE_180}
    try_rotations: tuple[int, ...] = tuple(rotation_map[r] for r in args.try_rotation)
    use_symmetry_check = not args.no_symmetry_check

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

    cascades = load_cascades()
    if not cascades:
        print("no cascades available", file=sys.stderr)
        return 2
    eye_cascade = load_eye_cascade()
    dnn_net = load_dnn()
    if dnn_net is not None:
        print("detection backend: Res10-SSD (DNN) + Haar fallback")
    else:
        print("detection backend: Haar (DNN model not found in /tmp)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for vp in video_paths:
        print(f"=== {vp.name} ===")
        process_video(vp, cascades, eye_cascade, dnn_net, args.out_dir, args.sample_fps,
                      args.min_side, args.min_skin_ratio, args.max_aspect,
                      args.max_frames, args.dnn_conf,
                      cascade_short_side=args.cascade_short_side,
                      cascade_min_neighbors=args.cascade_min_neighbors,
                      cascade_scale_factor=args.cascade_scale_factor,
                      dnn_input_size=args.dnn_input_size,
                      dnn_input_sizes=dnn_input_sizes,
                      try_rotations=try_rotations,
                      use_symmetry_check=use_symmetry_check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())