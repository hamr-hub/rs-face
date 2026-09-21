#!/usr/bin/env python3
"""M2 liveness FAR/FRR evaluation for rs-face (standalone, does not touch src/).

Pipeline mirrors src/liveness.rs + src/liveness_detector.rs exactly:
  1. detect faces with the pinned YuNet ONNX (fallback: centered box)
  2. per head (2.7x MiniFASNetV2, 4.0x MiniFASNetV1SE): expanded crop
     (scale limited to image bounds), resize to 80x80, build raw [0,255]
     BGR NCHW tensor, run ONNX, softmax the 3 logits
  3. average the two probability rows; real_score = avg[1]
     classes: [printed photo, real, screen replay]

Dataset: CASIA-FASD test frames mirrored on Hugging Face
(akahana/anti-spoofing-casiafasd, casiafasd.tar.gz), extracted under
data/eval_m2/test_img/. Labels come from the _real/_fake filename suffix;
attack subtype from the CASIA video type token:
  3/4/5/6 (+ HR_2/HR_3) = print attacks (warped / cut photo)
  7/8 (+ HR_4)          = screen-replay attacks

To limit correlation between frames of one video clip, at most --per-source
frames are sampled from each source video (deterministic seed).

Usage:
  python3 tools/eval_m2_far_frr.py                 # full eval, CPU
  python3 tools/eval_m2_far_frr.py --max-real 120 --max-fake 120
"""
import argparse
import csv
import glob
import os
import re
from collections import defaultdict

import cv2
import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
DATA_DIR = os.path.join(ROOT, "data", "eval_m2")
IMG_DIR = os.path.join(DATA_DIR, "test_img", "test_img", "color")
OUT_SCORES = os.path.join(DATA_DIR, "m2_scores.csv")
OUT_ROC = os.path.join(DATA_DIR, "m2_roc.csv")
INPUT = 80
REAL_CLASS = 1

PRINT_TOKENS = {"3", "4", "5", "6", "HR_2", "HR_3"}
SCREEN_TOKENS = {"7", "8", "HR_4"}


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def expanded_crop(img_w: int, img_h: int, box, scale: float):
    """Port of src/liveness.rs::expanded_crop; returns (x0,y0,w,h)."""
    x, y, w, h = box
    limited = min(scale, (img_w - 1) / w, (img_h - 1) / h)
    new_w, new_h = w * limited, h * limited
    cx, cy = x + w / 2, y + h / 2
    l, t = cx - new_w / 2, cy - new_h / 2
    r, b = cx + new_w / 2, cy + new_h / 2
    if l < 0:
        r -= l
        l = 0
    if t < 0:
        b -= t
        t = 0
    if r > img_w - 1:
        l -= r - (img_w - 1)
        r = img_w - 1
    if b > img_h - 1:
        t -= b - (img_h - 1)
        b = img_h - 1
    l, t = max(0, l), max(0, t)
    r, b = min(img_w - 1, r), min(img_h - 1, b)
    x0, y0 = int(np.floor(l)), int(np.floor(t))
    x1, y1 = min(img_w, int(np.ceil(r)) + 1), min(img_h, int(np.ceil(b)) + 1)
    return x0, y0, x1 - x0, y1 - y0


def token_of(filename: str) -> str:
    # e.g. 28_HR_2.avi_125_fake.jpg -> source "28_HR_2", token "HR_2"
    m = re.match(r"\d+_(.+)\.avi_\d+_", filename)
    return m.group(1)


def source_of(filename: str) -> str:
    return re.match(r"(.+?)_\d+_(?:real|fake)\.jpg", filename).group(1)


def sample_files(per_source: int, seed: int):
    by_source = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg"))):
        by_source[source_of(os.path.basename(path))].append(path)
    rng = np.random.default_rng(seed)
    reals, prints, screens = [], [], []
    for src, paths in sorted(by_source.items()):
        paths = sorted(paths)
        if len(paths) > per_source:
            idx = sorted(rng.choice(len(paths), per_source, replace=False))
            paths = [paths[i] for i in idx]
        tok = token_of(os.path.basename(paths[0]))
        is_real = paths[0].endswith("_real.jpg")
        for p in paths:
            if is_real:
                reals.append(p)
            elif tok in PRINT_TOKENS:
                prints.append(p)
            elif tok in SCREEN_TOKENS:
                screens.append(p)
    return reals, prints, screens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=2)
    ap.add_argument("--max-real", type=int, default=0, help="0 = no cap")
    ap.add_argument("--max-print", type=int, default=0)
    ap.add_argument("--max-screen", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    reals, prints, screens = sample_files(args.per_source, args.seed)
    if args.max_real:
        reals = reals[:args.max_real]
    if args.max_print:
        prints = prints[:args.max_print]
    if args.max_screen:
        screens = screens[:args.max_screen]
    attacks = [(p, "print") for p in prints] + [(p, "screen") for p in screens]
    print(f"samples: real={len(reals)} print={len(prints)} screen={len(screens)}")
    assert len(reals) >= 100 and len(attacks) >= 100, "M2 needs >=100 per class"

    # --- load models once, CPU, single process (8GB Jetson friendly) ---
    yunet = cv2.FaceDetectorYN.create(
        os.path.join(MODELS, "face_detection_yunet_2023mar.onnx"),
        "", (320, 320), score_threshold=0.6)
    heads = []
    for name in ("2.7_80x80_MiniFASNetV2.onnx",
                 "4_0_0_80x80_MiniFASNetV1SE.onnx"):
        sess = ort.InferenceSession(os.path.join(MODELS, name),
                                    providers=["CPUExecutionProvider"])
        heads.append((sess, 2.7 if "V2" in name else 4.0))

    def score(path):
        img = cv2.imread(path)  # BGR, already what the model expects
        h, w = img.shape[:2]
        yunet.setInputSize((w, h))
        _, faces = yunet.detect(img)
        detected = faces is not None and len(faces) > 0
        if detected:
            fx, fy, fw, fh = faces[0][:4].astype(int)
            box = (max(0, fx), max(0, fy), max(1, fw), max(1, fh))
        else:
            # fallback: centered face box (CASIA frames are face-centered)
            side = int(min(w, h) * 0.55)
            box = ((w - side) // 2, (h - side) // 2, side, side)
        rows = []
        for sess, scale in heads:
            cx, cy, cw, ch = expanded_crop(w, h, box, scale)
            crop = img[cy:cy + ch, cx:cx + cw]
            crop = cv2.resize(crop, (INPUT, INPUT))
            tensor = crop.transpose(2, 0, 1)[None].astype(np.float32)  # BGR raw
            logits = sess.run(None, {sess.get_inputs()[0].name: tensor})[0][0]
            rows.append(softmax(logits))
        avg = np.mean(rows, axis=0)
        return avg, detected

    records = []
    n_miss = 0
    all_items = [(p, "real", "real") for p in reals] + \
                [(p, "attack", t) for p, t in attacks]
    for i, (path, label, subtype) in enumerate(all_items):
        probs, detected = score(path)
        n_miss += not detected
        records.append({
            "path": os.path.relpath(path, ROOT),
            "label": label, "subtype": subtype,
            "yunet_hit": int(detected),
            "p_paper": probs[0], "p_real": probs[1], "p_screen": probs[2],
        })
        if (i + 1) % 200 == 0:
            print(f"  scored {i+1}/{len(all_items)}")
    print(f"YuNet misses (centered fallback used): {n_miss}/{len(records)}")

    real_scores = np.array([r["p_real"] for r in records if r["label"] == "real"])
    atk_scores = np.array([r["p_real"] for r in records if r["label"] == "attack"])
    print_scores = np.array([r["p_real"] for r in records if r["subtype"] == "print"])
    screen_scores = np.array([r["p_real"] for r in records if r["subtype"] == "screen"])

    # --- ROC sweep under the actual deployment rule from src/liveness.rs:
    # accept iff argmax == real AND p_real >= threshold ---
    pp = np.array([[r["p_paper"], r["p_real"], r["p_screen"]] for r in records])
    argmax_real = pp.argmax(axis=1) == REAL_CLASS

    def accepts(mask_idx, t):
        return argmax_real[mask_idx] & (pp[mask_idx, REAL_CLASS] >= t)

    idx_real = np.array([i for i, r in enumerate(records) if r["label"] == "real"])
    idx_atk = np.array([i for i, r in enumerate(records) if r["label"] == "attack"])
    idx_print = np.array([i for i, r in enumerate(records) if r["subtype"] == "print"])
    idx_screen = np.array([i for i, r in enumerate(records) if r["subtype"] == "screen"])

    thresholds = np.unique(np.concatenate([[0.0, 1.0], pp[:, REAL_CLASS]]))
    thresholds = np.sort(thresholds)[::-1]
    roc_rows = []
    for t in thresholds:
        far = float(np.mean(accepts(idx_atk, t)))
        frr = float(np.mean(~accepts(idx_real, t)))
        far_print = float(np.mean(accepts(idx_print, t)))
        far_screen = float(np.mean(accepts(idx_screen, t)))
        roc_rows.append((t, far, frr, 1 - frr, far_print, far_screen))

    # AUC via monotone ordering of ROC points; EER on the sweep
    fpr = np.array([r[1] for r in roc_rows])
    frr_arr = np.array([r[2] for r in roc_rows])
    tpr = 1.0 - frr_arr

    # The argmax deployment rule caps FPR at the fraction of attacks whose
    # argmax is real (~2%), so the deployment-rule curve is only the low-FPR
    # operating segment. For a headline ranking metric compute the standard
    # ROC AUC on the raw p_real score (Mann-Whitney trapezoid).
    def raw_roc(pos, neg):
        ths = np.sort(np.unique(np.concatenate([pos, neg])))[::-1]
        f = np.array([np.mean(neg >= t) for t in ths])
        tp = np.array([np.mean(pos >= t) for t in ths])
        return f, tp

    raw_fpr, raw_tpr = raw_roc(real_scores, atk_scores)
    order = np.argsort(raw_fpr)
    auc = float(np.trapezoid(raw_tpr[order], raw_fpr[order]))

    # EER on the raw-score ROC (where FPR and FRR span the full range)
    raw_frr = 1.0 - raw_tpr
    eer_i = int(np.argmin(np.abs(raw_fpr - raw_frr)))
    eer = float((raw_fpr[eer_i] + raw_frr[eer_i]) / 2)

    # best operating point: Youden's J
    j_i = int(np.argmax(tpr - fpr))
    t_youden = float(roc_rows[j_i][0])

    # M2-compliant point: FAR<=1% and FRR<=5%; pick the candidate with the
    # largest minimum slack, tie-broken by Youden J
    ok = [i for i, r in enumerate(roc_rows) if r[1] <= 0.01 and r[2] <= 0.05]
    t_m2 = None
    if ok:
        def slack(i):
            return min(0.01 - roc_rows[i][1], 0.05 - roc_rows[i][2])
        t_m2 = float(roc_rows[max(ok, key=slack)][0])

    with open(OUT_SCORES, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    with open(OUT_ROC, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "FAR", "FRR", "TPR", "FAR_print", "FAR_screen"])
        w.writerows([(f"{t:.6f}", f"{far:.6f}", f"{frr:.6f}", f"{tprv:.6f}",
                      f"{fp:.6f}", f"{fs:.6f}")
                     for t, far, frr, tprv, fp, fs in roc_rows])

    def metrics_at(t):
        return (float(np.mean(accepts(idx_atk, t))),
                float(np.mean(~accepts(idx_real, t))),
                float(np.mean(accepts(idx_print, t))),
                float(np.mean(accepts(idx_screen, t))))

    print("\n==== M2 liveness results ====")
    print(f"real N={len(real_scores)}, attack N={len(atk_scores)} "
          f"(print {len(print_scores)}, screen {len(screen_scores)})")
    print(f"AUC={auc:.4f}  EER~={eer:.4f}")
    print(f"Youden threshold={t_youden:.4f} -> FAR={metrics_at(t_youden)[0]:.4f} "
          f"FRR={metrics_at(t_youden)[1]:.4f}")
    for tag, t in (("Youden", t_youden), ("M2", t_m2)):
        if t is None:
            print(f"{tag}: no threshold meets FAR<=1% & FRR<=5%")
            continue
        far, frr, fp, fs = metrics_at(t)
        print(f"{tag} threshold={t:.4f}: FAR={far:.4f} FRR={frr:.4f} "
              f"(FAR print={fp:.4f}, screen={fs:.4f})")
    print(f"\nscores -> {os.path.relpath(OUT_SCORES, ROOT)}")
    print(f"roc    -> {os.path.relpath(OUT_ROC, ROOT)}")


if __name__ == "__main__":
    main()
