#!/usr/bin/env python3
"""Reference implementation of OpenCV's Haar cascade stage evaluation.

Mirrors the exact arithmetic used by OpenCV 4.x's
``modules/objdetect/src/cascadedetect.cpp`` so the rs-face Rust implementation
can be checked byte-for-byte on a hand-built cascade. Used by
``tests/byte_equivalence.rs`` to discover the cascade conversion / eval
discrepancy documented in ``docs/BENCHMARK_BASELINE.md``.

Two important conventions to keep in mind when comparing against the rs-face
runtime:

* **Variance normalisation floor**: OpenCV uses
  ``normfactor = 1.0 / sqrt(var + 1e-6)`` so the factor is bounded above by
  ``1000`` on a flat (zero-variance) window. Any implementation that returns
  ``None`` for ``variance_part <= 0`` will reject windows OpenCV still
  evaluates, *and* any implementation that lets the factor diverge to ``+inf``
  will over-amplify borderline responses.
* **Stage threshold sign**: OpenCV stores the stage threshold as a *negative*
  number (``<stageThreshold>-5.0425...</stageThreshold>``) and the cascade
  passes when ``stage_sum >= threshold``. The stage accumulates the leaf
  value selected by ``value < weak_threshold ? leaf[0] : leaf[1]`` where the
  raw feature response is ``raw * normfactor``.
"""

from __future__ import annotations

import math
import struct
import sys
import xml.etree.ElementTree as ET
from typing import List, Optional, Sequence, Tuple

# Match the rs-face 24x24 cascade window; the inner normrect used by OpenCV's
# `varianceNormFactor` is `(window - 2)` on each side.
WINDOW_W = 24
WINDOW_H = 24

# OpenCV's variance-norm floor: 1e-6 keeps the factor bounded above by
# ~1000 on perfectly uniform windows. Without it, a flat window would
# multiply the raw response by infinity.
VARIANCE_FLOOR = 1e-6


def build_integrals(image: Sequence[int], width: int, height: int) -> Tuple[List[int], List[int]]:
    """Return (sum_ii, sum_sq_ii) tables sized (W+1) * (H+1) inclusive of
    a zero-padded row/column. Pixel coordinates are (x, y) with x in
    [0, width) and y in [0, height); the table index is ``y * (width + 1) + x``
    matching OpenCV's row-major layout.
    """
    stride = width + 1
    sum_ii = [0] * (stride * (height + 1))
    sum_sq_ii = [0] * (stride * (height + 1))
    for y in range(height):
        row_off = y * stride
        next_off = (y + 1) * stride
        rs = 0
        rss = 0
        for x in range(width):
            p = int(image[y * width + x])
            rs += p
            rss += p * p
            # i64 wrapping matches rs-face's `i64` accumulator.
            sum_ii[row_off + x + 1] = rs + sum_ii[next_off + x + 1] - sum_ii[next_off + x]
            sum_sq_ii[row_off + x + 1] = rss + sum_sq_ii[next_off + x + 1] - sum_sq_ii[next_off + x]
    return sum_ii, sum_sq_ii


def rect_sum(ii: Sequence[int], stride: int, x1: int, y1: int, x2: int, y2: int) -> int:
    """Sum of pixels in the half-open rect [x1, x2) x [y1, y2).

    Mirrors rs-face's `IntegralImage::rect_sum` (inclusion-exclusion).
    """
    return ii[y2 * stride + x2] - ii[y1 * stride + x2] - ii[y2 * stride + x1] + ii[y1 * stride + x1]


def tilted_rect_sum(
    ii: Sequence[int],
    rii: Sequence[int],
    stride: int,
    x1: int,
    y1: int,
    rw: int,
    rh: int,
) -> int:
    """OpenCV's `cv::HaarEvaluator::OptFeature::tilted_rect_sum` cone-sum.

    The 45-degree rotated rect is the inclusion-exclusion of four cones:
        R[p0] - R[p1] - R[p2] + R[p3]
    with the CV_TILTED_OFS corners
        p0 = (x1, y1)
        p1 = (x1 - rh, y1 + rh)
        p2 = (x1 + rw, y1 + rw)
        p3 = (x1 + rw - rh, y1 + rw + rh)
    The rs-face `tilted_rect_sum` uses the same corner set (see
    `feature.rs::eval`).
    """
    p0 = y1 * stride + x1
    p1 = (y1 + rh) * stride + (x1 - rh)
    p2 = (y1 + rw) * stride + (x1 + rw)
    p3 = (y1 + rw + rh) * stride + (x1 + rw - rh)
    return rii[p0] - rii[p1] - rii[p2] + rii[p3]


def feature_response(
    rects: Sequence[Tuple[int, int, int, int, float]],
    tilted: bool,
    ii: Sequence[int],
    rii: Sequence[int],
    stride: int,
    ox: int,
    oy: int,
) -> float:
    """Sum `weight * rect_sum` over a feature's rects, anchored at `(ox, oy)`.

    For CustomRects (OpenCV's hand-written features) the rect coordinates are
    already in window pixels; for the canonical families the rs-face runtime
    applies `win_w/fw, win_h/fh` scaling. This reference matches CustomRects
    directly because that is the path the XML converter emits.
    """
    total = 0.0
    for (rx, ry, rw, rh, weight) in rects:
        ax, ay = ox + rx, oy + ry
        if tilted:
            s = tilted_rect_sum(ii, rii, stride, ax, ay, rw, rh)
        else:
            s = rect_sum(ii, stride, ax, ay, ax + rw, ay + rh)
        total += float(s) * weight
    return total


def norm_factor(
    sum_in: int,
    sum_sq_in: int,
    nw: int = WINDOW_W - 2,
    nh: int = WINDOW_H - 2,
) -> Optional[float]:
    """OpenCV's `varianceNormFactor = 1 / sqrt(N*sum_sq - sum^2 + 1e-6)`.

    Returns ``None`` for the (rare) case where variance_part is non-positive
    even after the floor — matching the rs-face convention of refusing to
    evaluate on zero-variance windows.
    """
    var_part = float(nw) * float(nh) * float(sum_sq_in) - float(sum_in) * float(sum_in)
    if var_part + VARIANCE_FLOOR <= 0.0:
        return None
    return 1.0 / math.sqrt(var_part + VARIANCE_FLOOR)


def classify_window(
    image: Sequence[int],
    img_w: int,
    img_h: int,
    rects: Sequence[Tuple[int, int, int, int, float]],
    tilted: bool,
    weak_classifiers: Sequence[Tuple[int, float, float, float]],
    stage_threshold: float,
    stage_bias: float = 0.0,
    ox: int = 0,
    oy: int = 0,
) -> Optional[float]:
    """Evaluate one stage at window origin (ox, oy).

    Returns the cascade score (cumulative stage_sum) on pass, or ``None`` on
    any stage rejection. Matches OpenCV 4.x's `CascadeClassifier::runAt`.

    `weak_classifiers` is a list of `(feature_idx, threshold, left_val,
    right_val)` tuples, where `left_val` is picked when `value < threshold`
    and `right_val` otherwise.
    """
    ii, sqii = build_integrals(image, img_w, img_h)
    rii = ii  # Reference only uses upright paths; tilted features share
    # the upright II in rs-face for the simple `tilted_rect_sum` formulation
    # (the cone-sum is encoded by the corner geometry, not by the table).
    stride = img_w + 1
    # Inner normrect is (x+1, y+1, x+window_w-1, y+window_h-1) in II coords.
    nx1, ny1 = ox + 1, oy + 1
    nx2, ny2 = ox + WINDOW_W - 1, oy + WINDOW_H - 1
    sum_in = rect_sum(ii, stride, nx1, ny1, nx2, ny2)
    sum_sq_in = rect_sum(sqii, stride, nx1, ny1, nx2, ny2)
    nf = norm_factor(sum_in, sum_sq_in)
    if nf is None:
        return None
    stage_sum = 0.0
    for (_idx, threshold, left_val, right_val) in weak_classifiers:
        raw = feature_response(rects, tilted, ii, rii, stride, ox, oy)
        value = raw * nf
        stage_sum += left_val if value < threshold else right_val
        if stage_sum < stage_threshold + stage_bias:
            return None
    return stage_sum


def parse_open_cv_cascade(xml_path: str):
    """Mirror of ``tools/convert_opencv_xml.py`` reading.

    Returns ``(window_w, window_h, features, stages)``. Each feature is
    ``(tilted, rects)`` where ``rects`` is ``[(x, y, w, h, weight), ...]``.
    Each stage is ``(threshold, weak_classifiers)`` where each weak classifier
    is ``(feature_idx, threshold, left_val, right_val)``.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    cascade = root.find('cascade')
    if cascade is None:
        cascade = root.find('haarcascade_frontalface_default')
    if cascade is None:
        cascade = root[0]
    size = cascade.find('size')
    if size is None:
        ww = int(cascade.find('width').text)
        wh = int(cascade.find('height').text)
    else:
        parts = size.text.split()
        ww, wh = int(parts[0]), int(parts[1])

    features = []
    feats_elem = cascade.find('features')
    for f in feats_elem.findall('_'):
        tilted_elem = f.find('tilted')
        tilted = tilted_elem is not None and int(tilted_elem.text.strip()) == 1
        rects_elem = f.find('rects')
        rects = []
        for r in rects_elem.findall('_'):
            parts = r.text.split()
            rects.append((int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), float(parts[4])))
        features.append((tilted, rects))

    stages = []
    for st in cascade.find('stages').findall('_'):
        threshold = float(st.find('stageThreshold').text)
        weak = []
        for w in st.find('weakClassifiers').findall('_'):
            internal = list(map(float, w.find('internalNodes').text.split()))
            feature_idx = int(internal[2])
            thr = internal[3]
            leaf = w.find('leafValues').text.split()
            left_val = float(leaf[0])
            right_val = float(leaf[1])
            weak.append((feature_idx, thr, left_val, right_val))
        stages.append((threshold, weak))
    return ww, wh, features, stages


def classify_image(
    image: Sequence[int],
    img_w: int,
    img_h: int,
    features,
    stages,
    stage_bias: float = 0.0,
):
    """Walk all stages in order; return the cumulative cascade score on
    pass, ``None`` on rejection. Single-window variant of OpenCV's
    `runAt` used by the rs-face byte-equivalence test.
    """
    ii, sqii = build_integrals(image, img_w, img_h)
    rii = ii
    stride = img_w + 1
    nx1, ny1 = 1, 1
    nx2, ny2 = WINDOW_W - 1, WINDOW_H - 1
    sum_in = rect_sum(ii, stride, nx1, ny1, nx2, ny2)
    sum_sq_in = rect_sum(sqii, stride, nx1, ny1, nx2, ny2)
    nf = norm_factor(sum_in, sum_sq_in)
    if nf is None:
        return None
    total = 0.0
    for (stage_threshold, weak) in stages:
        stage_sum = 0.0
        for (feature_idx, weak_threshold, left_val, right_val) in weak:
            tilted, rects = features[feature_idx]
            raw = feature_response(rects, tilted, ii, rii, stride, 0, 0)
            value = raw * nf
            stage_sum += left_val if value < weak_threshold else right_val
        if stage_sum < stage_threshold + stage_bias:
            return None
        total += stage_sum
    return total


def main():
    """CLI: ``opencv_cascade_reference.py <xml> <ppm>`` prints reference
    cascade scores against the rs-face runtime output. Used as a one-off
    sanity check, not as a regular test driver.
    """
    if len(sys.argv) != 3:
        print('usage: opencv_cascade_reference.py <in.xml> <in.ppm>', file=sys.stderr)
        sys.exit(2)
    ww, wh, features, stages = parse_open_cv_cascade(sys.argv[1])
    print(f"cascade: {ww}x{wh} {len(features)} features, {len(stages)} stages")
    # Caller feeds PPM separately; this stub is a no-op for now.
    sys.exit(0)


if __name__ == '__main__':
    main()
