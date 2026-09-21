#!/usr/bin/env python3
"""Smoke test: verify the rs-face MiniFASNet liveness weights really run.

Standalone — does NOT import or modify src/. Mirrors the exact preprocessing
documented in src/liveness.rs:
  - 80x80 crops fed as raw [0,255] BGR NCHW tensors (no mean subtraction)
  - two heads (2.7x and 4.0x crop scales), logits -> softmax -> averaged

Usage: python3 tools/eval_m2_smoke.py
"""
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
INPUT = 80


def load_head(name: str) -> ort.InferenceSession:
    path = os.path.join(MODELS, name)
    if not os.path.exists(path):
        sys.exit(f"missing weight: {path} — run tools/fetch_models.sh first")
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def main() -> None:
    heads = [
        (load_head("2.7_80x80_MiniFASNetV2.onnx"), 2.7),
        (load_head("4_0_0_80x80_MiniFASNetV1SE.onnx"), 4.0),
    ]
    # Random probe + zeros probe: just confirms real forward passes execute.
    rng = np.random.default_rng(0)
    for probe_name, probe in (("zeros", np.zeros((1, 3, INPUT, INPUT), np.float32)),
                              ("random", rng.uniform(0, 255, (1, 3, INPUT, INPUT)).astype(np.float32))):
        rows = []
        for sess, _scale in heads:
            logits = sess.run(None, {sess.get_inputs()[0].name: probe})[0][0]
            rows.append(softmax(logits))
        avg = np.mean(rows, axis=0)
        print(f"probe={probe_name:6s} avg probs [paper, real, screen] = "
              f"{avg[0]:.4f} / {avg[1]:.4f} / {avg[2]:.4f}")
    print("OK: both MiniFASNet heads load and produce 3-class probabilities.")


if __name__ == "__main__":
    main()
