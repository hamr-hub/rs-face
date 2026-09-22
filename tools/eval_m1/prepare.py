#!/usr/bin/env python3
"""Prepare the M1 evaluation split from data/eval_m1/lfw.parquet (LFW via HF bitmind).

Deterministic protocol:
  * identities with >= 4 images only
  * pick the N identities with the most images
  * per identity: 1 gallery enrollment image, up to K probe images
    (chosen with a fixed RNG seed)
Writes JPEGs under outdir/{gallery,probes}/ and manifest.json.

Usage: python3 prepare.py [--people 70] [--probes-per-person 3] [--seed 20260921]
"""
import argparse
import io
import json
import os
import random
from collections import defaultdict

import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PARQUET = os.path.join(ROOT, "data", "eval_m1", "lfw.parquet")
OUTDIR = os.path.join(ROOT, "data", "eval_m1", "split")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--people", type=int, default=70)
    ap.add_argument("--probes-per-person", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260921)
    args = ap.parse_args()

    df = pd.read_parquet(PARQUET)
    groups = defaultdict(list)
    for _, row in df.iterrows():
        identity = row["filename"].rsplit("_", 1)[0]
        groups[identity].append((row["filename"], row["image"]["bytes"]))

    eligible = [(name, items) for name, items in groups.items() if len(items) >= 4]
    eligible.sort(key=lambda kv: (-len(kv[1]), kv[0]))
    chosen = eligible[: args.people]

    rng = random.Random(args.seed)
    manifest = {"gallery": [], "probes": []}
    for d in ("gallery", "probes"):
        os.makedirs(os.path.join(OUTDIR, d), exist_ok=True)

    for person_idx, (identity, items) in enumerate(chosen):
        items = items[:]
        rng.shuffle(items)
        enroll = items[0]
        probes = items[1 : 1 + args.probes_per_person]

        def dump(rec, split, fname_bytes):
            fname, blob = fname_bytes
            img = Image.open(io.BytesIO(blob)).convert("RGB")
            out = os.path.join(OUTDIR, split, f"{person_idx:03d}_{fname}")
            img.save(out, "JPEG", quality=95)
            rel = os.path.relpath(out, OUTDIR)
            rec.append({"file": rel, "identity": identity})

        dump(manifest["gallery"], "gallery", enroll)
        for p in probes:
            dump(manifest["probes"], "probes", p)

    with open(os.path.join(OUTDIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(
        f"people={len(chosen)} gallery={len(manifest['gallery'])} "
        f"probes={len(manifest['probes'])} out={OUTDIR}"
    )


if __name__ == "__main__":
    main()
