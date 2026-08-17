#!/usr/bin/env python3
"""Convert an OpenCV Haar cascade XML to rs-face's `.rfcf` format.

Usage:
    python3 tools/convert_opencv_xml.py haarcascade_frontalface_default.xml cascade.rfcf

OpenCV cascade XML structure (simplified):
  <opencv_storage>
    <haarcascade_frontalface_default type_id="opencv-haar-classifier">
      <size>24 24</size>
      <stages>
        <_>
          <maxWeakCount>...</maxWeakCount>
          <stageThreshold>...</stageThreshold>
          <weakClassifiers>
            <_>
              <internalNodes>0 -1 ...</internalNodes>
              <leafValues>...</leafValues>
            </_>
            ...
          </weakClassifiers>
        </_>
        ...
      </stages>
      <features>
        <_>
          <rects>
            <_>3 7 14 4 -1.</_>
            ...
          </rects>
          <tilted>0</tilted>
        </_>
        ...
      </features>
    </haarcascade_frontalface_default>
  </opencv_storage>

The `internalNodes` field encodes a decision stump:
    leafValues = [left, right]
    if (p0 + 2*p1*left + ...) is too much detail, OpenCV uses the simple form:
        internalNodes = 0 -1 feature_index threshold 1 0 0
        (p0 is always 0 for the simple form)
"""
import struct
import sys
import xml.etree.ElementTree as ET

# Feature kind encoding in the .rfcf binary:
#   0 = VerticalEdge
#   1 = HorizontalEdge
#   2 = DiagonalEdge
#   3 = VerticalCenter
#   4 = HorizontalCenter
#   5 = CustomRects (OpenCV-style pixel-coordinate layout)
KIND_VERTICAL_EDGE = 0
KIND_HORIZONTAL_EDGE = 1
KIND_DIAGONAL_EDGE = 2
KIND_VERTICAL_CENTER = 3
KIND_HORIZONTAL_CENTER = 4
KIND_CUSTOM_RECTS = 5


def classify_feature(rects):
    """Given a list of `<_>x y w h weight</_>` rects, decide which canonical
    5-family Haar feature it matches.

    Returns (kind, fw, fh) or None if the feature doesn't fit a canonical shape
    (in which case the converter will skip it).
    """
    rs = []
    for r in rects:
        x, y, w, h, weight = [int(r[0]), int(r[1]), int(r[2]), int(r[3]), float(r[4])]
        rs.append((x, y, w, h, weight))
    # Must be exactly 2 or 3 rects.
    if len(rs) not in (2, 3):
        return None
    # Compute bounding box.
    xs = [r[0] for r in rs]
    ys = [r[1] for r in rs]
    xe = [r[0] + r[2] for r in rs]
    ye = [r[1] + r[3] for r in rs]
    fw = max(xe) - min(xs)
    fh = max(ye) - min(ys)
    if fw <= 0 or fh <= 0 or fw > 16 or fh > 16:
        return None

    # Vertical edge: top +1, bottom -1, full width.
    if len(rs) == 2:
        (x1, y1, w1, h1, p1), (x2, y2, w2, h2, p2) = rs
        if x1 == 0 and x2 == 0 and w1 == fw and w2 == fw and h1 + h2 == fh and y2 == y1 + h1 and p1 == 1 and p2 == -1:
            return KIND_VERTICAL_EDGE, fw, fh
        if y1 == 0 and y2 == 0 and h1 == fh and h2 == fh and w1 + w2 == fw and x2 == x1 + w1 and p1 == 1 and p2 == -1:
            return KIND_HORIZONTAL_EDGE, fw, fh
        # Diagonal edge (two equal tilted rectangles stacked). OpenCV uses a
        # tilted flag; we treat all tilted features as DiagonalEdge.
        if w1 == fw and w2 == fw and h1 + h2 == fh and y2 == y1 + h1:
            return KIND_DIAGONAL_EDGE, fw, fh

    # Vertical center: top +1, middle -2, bottom +1.
    if len(rs) == 3:
        (x1, y1, w1, h1, p1), (x2, y2, w2, h2, p2), (x3, y3, w3, h3, p3) = rs
        if x1 == x2 == x3 == 0 and w1 == w2 == w3 == fw and h1 + h2 + h3 == fh and y2 == y1 + h1 and y3 == y2 + h2 and p1 == 1 and p2 == -2 and p3 == 1:
            return KIND_VERTICAL_CENTER, fw, fh
        if y1 == y2 == y3 == 0 and h1 == h2 == h3 == fh and w1 + w2 + w3 == fw and x2 == x1 + w1 and x3 == x2 + w2 and p1 == 1 and p2 == -2 and p3 == 1:
            return KIND_HORIZONTAL_CENTER, fw, fh
    return None


def parse_rects(rect_texts):
    """Parse a list of `<_>x y w h weight</_>` text strings into integer rect tuples."""
    out = []
    for r in rect_texts:
        parts = r
        x, y, w, h, weight = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), float(parts[4])
        out.append((x, y, w, h, weight))
    return out


def parse_open_cv_cascade(path):
    tree = ET.parse(path)
    root = tree.getroot()
    cascade = root.find('cascade') or root.find('haarcascade_frontalface_default') or root[0]
    size = cascade.find('size')
    if size is None:
        # Newer OpenCV XMLs use <height>/<width> instead of <size>.
        w = int(cascade.find('width').text)
        h = int(cascade.find('height').text)
        ww, wh = w, h
    else:
        size = size.text.split()
        ww, wh = int(size[0]), int(size[1])
    # Parse features.
    # We emit *every* feature as CustomRects (kind=5) — the runtime supports
    # arbitrary weighted-rectangle layouts in pixel coordinates. Canonical
    # kind encoding is left as a future optimization.
    features = []
    feats_elem = cascade.find('features')
    feat_index = 0
    for f in feats_elem.findall('_'):
        rects_elem = f.find('rects')
        rects = [r.text.split() for r in rects_elem.findall('_')]
        parsed_rects = parse_rects(rects)
        if not parsed_rects:
            features.append((KIND_CUSTOM_RECTS, 0, 0, []))
            feat_index += 1
            continue
        # OpenCV Haar feature rects are in feature-local coordinates relative
        # to the 24x24 detection window. CustomRects uses pixel coordinates
        # directly, so pass them through unchanged.
        features.append((KIND_CUSTOM_RECTS, 0, 0, parsed_rects))
        feat_index += 1
    # Parse stages.
    stages = []
    for st in cascade.find('stages').findall('_'):
        threshold = float(st.find('stageThreshold').text)
        weak = []
        for w in st.find('weakClassifiers').findall('_'):
            internal = list(map(float, w.find('internalNodes').text.split()))
            # OpenCV Haar formats:
            #   old: [0, -1, feature_idx, threshold, sign, 0, 0]   (7 values)
            #   new: [0, -1, feature_idx, threshold]              (4 values; sign = -1)
            feature_idx = int(internal[2])
            thresh = internal[3]
            left_right = w.find('leafValues').text.split()
            left_val = float(left_right[0])
            right_val = float(left_right[1])
            # OpenCV's predictor: `value < threshold ? leafValues[0] : leafValues[1]`.
            # Our eval mirrors this directly: `value < threshold ? left_val : right_val`.
            # So `left_val = leafValues[0]` and `right_val = leafValues[1]` with no
            # swap. The `sign` field is preserved as a legacy marker but is no
            # longer consulted at eval time.
            sign = 1
            weak.append((feature_idx, thresh, sign, left_val, right_val))
        stages.append((threshold, weak))
    return ww, wh, features, stages


def write_rfcf(path, ww, wh, features, stages):
    with open(path, 'wb') as f:
        f.write(b'RFCF')
        # Version 2 supports CustomRects (kind=5) for arbitrary weighted-rect
        # layouts in pixel coordinates — used by the OpenCV Haar cascades.
        f.write(struct.pack('<I', 2))
        f.write(struct.pack('<I', ww))
        f.write(struct.pack('<I', wh))
        f.write(struct.pack('<I', len(features)))
        for feat in features:
            kind, fw, fh, rects = feat
            f.write(struct.pack('<B', kind))
            f.write(struct.pack('<B', fw))
            f.write(struct.pack('<B', fh))
            f.write(struct.pack('<I', len(rects)))
            for (x, y, w, h, weight) in rects:
                # Clamp to u8; OpenCV coords are 0..24 so this is always safe.
                x_b = max(0, min(255, x))
                y_b = max(0, min(255, y))
                w_b = max(1, min(255, w))
                h_b = max(1, min(255, h))
                f.write(struct.pack('<BBBB', x_b, y_b, w_b, h_b))
                f.write(struct.pack('<f', weight))
        f.write(struct.pack('<I', len(stages)))
        for threshold, weak in stages:
            f.write(struct.pack('<f', threshold))
            f.write(struct.pack('<I', len(weak)))
            for feature_idx, thresh, sign, left_val, right_val in weak:
                f.write(struct.pack('<I', feature_idx))
                f.write(struct.pack('<f', thresh))
                f.write(struct.pack('<b', sign))
                f.write(struct.pack('<f', left_val))
                f.write(struct.pack('<f', right_val))


def main():
    if len(sys.argv) != 3:
        print('usage: convert_opencv_xml.py <in.xml> <out.rfcf>', file=sys.stderr)
        sys.exit(2)
    ww, wh, features, stages = parse_open_cv_cascade(sys.argv[1])
    write_rfcf(sys.argv[2], ww, wh, features, stages)
    print(f'wrote {sys.argv[2]}: window {ww}x{wh}, {len(features)} features, {len(stages)} stages')


if __name__ == '__main__':
    main()
