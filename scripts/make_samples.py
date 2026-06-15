#!/usr/bin/env python
"""Render a few PTB-XL Lead II test records as ECG strip images for the demo.

Outputs single-lead strip PNGs into assets/samples/ so the live app has
ready-to-try inputs without users needing their own ECG. PTB-XL is CC-BY 4.0;
attribution is in the README.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from src.diagnostic.config import CACHE_DIR, LEAD_TO_IDX, SUPERCLASSES
from src.digitization import render_ecg_strip

X = np.load(os.path.join(CACHE_DIR, "X_100.npy"))
Y = np.load(os.path.join(CACHE_DIR, "Y.npy"))
folds = np.load(os.path.join(CACHE_DIR, "folds.npy"))
test_idx = np.where(folds == 10)[0]

OUT = os.path.join(ROOT, "assets", "samples")
os.makedirs(OUT, exist_ok=True)

picks = {0: "normal", 2: "st_t_change", 3: "conduction_disturbance"}
for cls, label in picks.items():
    cand = [i for i in test_idx if Y[i, cls] == 1 and Y[i].sum() == 1]
    if not cand:
        continue
    i = cand[0]
    lead_ii = X[i, :, LEAD_TO_IDX["II"]]
    path = os.path.join(OUT, f"sample_{label}.png")
    render_ecg_strip(lead_ii, fs=100, save_path=path)
    print(f"wrote {path} (record {i}, true={SUPERCLASSES[cls]})")
