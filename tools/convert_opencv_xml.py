#!/usr/bin/env python3
"""Convert an OpenCV Haar cascade XML to rs-face's `.rfcf` v3 format.

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

The `<tilted>` flag is read per feature: tilted=1 means the rects are 45
degree rotated rectangles and the runtime evaluates them against the
rotated integral table. Dropping that flag (as older versions of this
script did) turns any tilted cascade into upright garbage.

Each weak classifier's `internalNodes` line is the simple stump form:
    internalNodes = 0 -1 feature_index threshold [1 0 0]
OpenCV predicts leafValues[0] when value < threshold else leafValues[1];
the runtime mirrors that directly.
"""

import struct
import sys
import xml.etree.ElementTree as ET

# Every OpenCV-trained feature is emitted as a CustomRects feature
# (kind=5): arbitrary weighted rectangles in window-pixel coordinates.
KIND_CUSTOM_RECTS = 5

# v3 per-feature flags.
FLAG_TILTED = 1


def parse_rects(rect_texts):
    """Parse a list of `<_>x y w h weight</_>` text strings into float rect tuples."""
    out = []
    for r in rect_texts:
        x, y, w, h, weight = int(r[0]), int(r[1]), int(r[2]), int(r[3]), float(r[4])
        out.append((x, y, w, h, weight))
    return out


def parse_open_cv_cascade(path):
    tree = ET.parse(path)
    root = tree.getroot()
    cascade = root.find('cascade')
    if cascade is None:
        cascade = root.find('haarcascade_frontalface_default')
    if cascade is None:
        cascade = root[0]
    size = cascade.find('size')
    if size is None:
        # Newer OpenCV XMLs use <height>/<width> instead of <size>.
        ww = int(cascade.find('width').text)
        wh = int(cascade.find('height').text)
    else:
        parts = size.text.split()
        ww, wh = int(parts[0]), int(parts[1])

    # Every feature is a (tilted, rects) pair. Rects pass through unchanged
    # because CustomRects uses window-pixel coordinates directly.
    features = []
    feats_elem = cascade.find('features')
    for f in feats_elem.findall('_'):
        tilted_elem = f.find('tilted')
        tilted = tilted_elem is not None and int(tilted_elem.text.strip()) == 1
        rects_elem = f.find('rects')
        rects = parse_rects([r.text.split() for r in rects_elem.findall('_')])
        features.append((tilted, rects))

    stages = []
    for st in cascade.find('stages').findall('_'):
        threshold = float(st.find('stageThreshold').text)
        weak = []
        for w in st.find('weakClassifiers').findall('_'):
            internal = list(map(float, w.find('internalNodes').text.split()))
            # OpenCV Haar formats:
            #   old: [0, -1, feature_idx, threshold, sign, 0, 0]  (7 values)
            #   new: [0, -1, feature_idx, threshold]               (4 values)
            feature_idx = int(internal[2])
            thresh = internal[3]
            left_right = w.find('leafValues').text.split()
            left_val = float(left_right[0])
            right_val = float(left_right[1])
            # OpenCV: value < threshold ? leafValues[0] : leafValues[1].
            # The runtime eval mirrors this, so no leaf swap. `sign` is a
            # legacy marker, no longer consulted at eval time.
            sign = 1
            weak.append((feature_idx, thresh, sign, left_val, right_val))
        stages.append((threshold, weak))
    return ww, wh, features, stages


def write_rfcf(path, ww, wh, features, stages):
    with open(path, 'wb') as f:
        f.write(b'RFCF')
        # Version 3: each feature record carries a flags byte
        # (bit 0 = tilted 45-degree rectangles).
        f.write(struct.pack('<I', 3))
        f.write(struct.pack('<I', ww))
        f.write(struct.pack('<I', wh))
        f.write(struct.pack('<I', len(features)))
        for tilted, rects in features:
            flags = FLAG_TILTED if tilted else 0
            f.write(struct.pack('<BBBB', KIND_CUSTOM_RECTS, 0, 0, flags))
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
    tilted_count = sum(1 for tilted, _ in features if tilted)
    write_rfcf(sys.argv[2], ww, wh, features, stages)
    print(
        f'wrote {sys.argv[2]}: window {ww}x{wh}, '
        f'{len(features)} features ({tilted_count} tilted), {len(stages)} stages'
    )


if __name__ == '__main__':
    main()
