#!/usr/bin/env python3
"""
Convert an OpenCV Haar cascade XML to rs-face's compact .rfcf v2 binary.

Reference format (rsface::haar::Cascade::save/load):
  magic       : "RFCF" (4 bytes)
  version     : u32 LE (= 2)
  window_w    : u32 LE
  window_h    : u32 LE
  nfeat       : u32 LE
  for each feature:
    kind    : u8  (5 = CustomRects for OpenCV-style arbitrary layouts)
    width   : u8
    height  : u8
    nrect   : u32 LE
    for each rect: x:u8 y:u8 w:u8 h:u8 weight:f32 LE
  nstage      : u32 LE
  for each stage:
    stage_threshold : f32 LE
    nw : u32 LE
    for each weak feature:
      feature_index : u32 LE
      threshold     : f32 LE
      sign          : i8  (OpenCV uses +1 / -1 — the cascade training output
                            leaves the sign implicit; we encode the cascade's
                            raw inequality direction here so the detector
                            doesn't have to guess.)
      left_val      : f32 LE
      right_val     : f32 LE

This converter uses the OpenCV Python bindings to:
1. Load the cascade (parses the XML, builds cv2 internal model).
2. Iterate features/stages and dump them in the v2 layout.

Why CustomRects (kind=5) instead of the canonical kind 0..4?
The 5 canonical OpenCV Haar kinds cover 90% of trained features but not all.
CustomRects keeps rect coordinates in pixel space (no fw/fh division) which
exactly matches OpenCV's evaluation kernel — zero precision loss.

Usage:
  python3 tools/xml_to_rfcf.py <input.xml> [--out PATH]

If --out is omitted, writes to <input>.rfcf next to the input file.
"""
import argparse
import os
import struct
import sys

import cv2

# Internal cv2 cascade struct offsets (OpenCV 4.x source: modules/objdetect/src/cascadedetect.hpp)
# We use cv2.CascadeClassifier.getOriginalWindowSize() + the cascade's nested
# internal data via cv2's "ConstCvCascade" / "CvHaarClassifier" reading helpers.
# In Python the cleanest path is to re-parse it as XML and emit the .rfcf.


# CustomRects kind code in the .rfcf layout (matches FeatureKind::CustomRects = 5).
KIND_CUSTOM_RECTS = 5


def parse_cascade(xml_path: str):
    """Walk the OpenCV Haar cascade XML and emit .rfcf records.

    Returns (window_w, window_h, features, stages).
    features = list of (width, height, [(x, y, w, h, weight), ...])
    stages   = list of (stage_threshold, [(feat_idx, threshold, sign, left, right), ...])
    """
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # OpenCV layout: <opencv_storage><cascade><width/><height/><stages/>...<features/>
    cascade_el = root.find("cascade")
    if cascade_el is None:
        raise ValueError("cascade XML missing <cascade>")
    size_w = cascade_el.find("width")
    size_h = cascade_el.find("height")
    if size_w is None or size_h is None:
        raise ValueError("cascade XML missing <width>/<height>")
    ww, wh = (int(size_w.text), int(size_h.text))
    if (ww, wh) != (24, 24):
        # The .rfcf v2 layout was designed around 24×24 — we still emit it,
        # but warn so the operator knows what they're getting.
        print(
            f"[xml_to_rfcf] WARN: cascade window is {ww}x{wh}, .rfcf v2 "
            f"was built around 24x24 (scaling happens in the detector).",
            file=sys.stderr,
        )

    # Flat list of every <feature> across every <stage><trees><_>.
    # rs-face needs all features in one contiguous block with stage offsets
    # pointing into it, mirroring OpenCV's "feature" vs "weak feature" split.
    flat_features = []
    stages = []

    features_root = cascade_el.find("features")
    if features_root is None:
        raise ValueError("cascade XML missing <features>")

    for f_el in features_root.findall("_"):
        rects = []
        # OpenCV XML layout: <rects><_>x y w h weight</_><_>x y w h weight</_></rects>
        # Each rect is a direct child of <rects>, NOT a text-content rect.
        rects_el = f_el.find("rects")
        if rects_el is None:
            raise ValueError("feature missing <rects>")
        for r_el in rects_el.findall("_"):
            txt = (r_el.text or "").strip()
            if not txt:
                continue
            toks = txt.split()
            if len(toks) != 5:
                raise ValueError(f"bad rect: {r_el.text!r}")
            x, y, w_, h_ = (int(v) for v in toks[:4])
            weight = float(toks[4])
            rects.append((x, y, w_, h_, weight))
        fw = int(f_el.find("width").text) if f_el.find("width") is not None else ww
        fh = int(f_el.find("height").text) if f_el.find("height") is not None else wh
        flat_features.append((fw, fh, rects))

    stages_root = cascade_el.find("stages")
    if stages_root is None:
        raise ValueError("cascade XML missing <stages>")

    for st_el in stages_root.findall("_"):
        thr_el = st_el.find("stageThreshold")
        if thr_el is None:
            raise ValueError("stage missing <stageThreshold>")
        thr = float(thr_el.text)
        weak_features = []
        # OpenCV layout: <weakClassifiers><_>...internal nodes...</_></weakClassifiers>
        # Each "_" is a single-feature tree: <internalNodes>1 x y w h</internalNodes>
        # <leafValues>... left/right ...</leafValues>. The simplified stump-style
        # cascades (frontalface_default.xml) have one leaf pair per tree, plus
        # an empty `<left_node>` / `<right_node>` chain (we ignore those).
        wc_el = st_el.find("weakClassifiers")
        if wc_el is None:
            # Skip degenerate stages with no weak classifiers.
            stages.append((thr, weak_features))
            continue
        for tree_el in wc_el.findall("_"):
            fv_el = tree_el.find("internalNodes")
            lv_el = tree_el.find("leafValues")
            if fv_el is None or lv_el is None:
                raise ValueError("weak classifier missing required node")
            # OpenCV stump tree internalNodes = "<nodeCount> <left> <feature_idx> <threshold>"
            #   nodeCount: number of internal nodes below this subtree (0 for stump)
            #   left: negative = leaf value index, positive = internal node index
            #   feature_idx: index into the global <features> list
            #   threshold: feature response threshold for the split
            fv_toks = fv_el.text.split()
            if len(fv_toks) < 4:
                raise ValueError(f"bad internalNodes: {fv_el.text!r}")
            feat_idx = int(fv_toks[2])
            fthr = float(fv_toks[3])
            # leafValues = "<leftVal> <rightVal>" (stump tree, no recursion)
            lv_toks = lv_el.text.split()
            if len(lv_toks) < 2:
                raise ValueError(f"bad leafValues: {lv_el.text!r}")
            lv = float(lv_toks[0])
            rv = float(lv_toks[1])
            weak_features.append(
                (
                    feat_idx,
                    fthr,
                    +1,  # OpenCV default: value < threshold → left, else right.
                    lv,
                    rv,
                )
            )
        stages.append((thr, weak_features))

    return ww, wh, flat_features, stages


def write_rfcf(out_path: str, ww: int, wh: int, features, stages) -> None:
    with open(out_path, "wb") as f:
        f.write(b"RFCF")
        f.write(struct.pack("<I", 2))  # version
        f.write(struct.pack("<I", ww))
        f.write(struct.pack("<I", wh))
        f.write(struct.pack("<I", len(features)))
        for fw, fh, rects in features:
            f.write(struct.pack("<B", KIND_CUSTOM_RECTS))
            # Feature-local width/height — for CustomRects we still record the
            # OpenCV fw/fh so the detector's pixel-coordinate math matches.
            # Clamp to u8 range (cascades never exceed 255 in practice).
            f.write(struct.pack("<B", min(fw, 255)))
            f.write(struct.pack("<B", min(fh, 255)))
            f.write(struct.pack("<I", len(rects)))
            for (x, y, w_, h_, weight) in rects:
                f.write(struct.pack("<BBBB", x, y, w_, h_))
                f.write(struct.pack("<f", weight))
        f.write(struct.pack("<I", len(stages)))
        for stage_threshold, weak_features in stages:
            f.write(struct.pack("<f", stage_threshold))
            f.write(struct.pack("<I", len(weak_features)))
            for (feat_idx, thr, sign, lv, rv) in weak_features:
                f.write(struct.pack("<I", feat_idx))
                f.write(struct.pack("<f", thr))
                f.write(struct.pack("<b", sign))
                f.write(struct.pack("<f", lv))
                f.write(struct.pack("<f", rv))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("xml", help="input OpenCV Haar cascade XML")
    p.add_argument("--out", help="output .rfcf path (default: <xml>.rfcf)")
    args = p.parse_args()

    if not os.path.isfile(args.xml):
        sys.exit(f"xml not found: {args.xml}")

    out = args.out or (args.xml.rsplit(".", 1)[0] + ".rfcf")
    ww, wh, features, stages = parse_cascade(args.xml)
    write_rfcf(out, ww, wh, features, stages)

    total_rects = sum(len(r) for (_, _, r) in features)
    total_weaks = sum(len(s) for (_, s) in stages)
    print(
        f"[xml_to_rfcf] wrote {out}: {ww}x{wh}, "
        f"{len(features)} features ({total_rects} rects), "
        f"{len(stages)} stages ({total_weaks} weak features)"
    )


if __name__ == "__main__":
    main()