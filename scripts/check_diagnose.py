#!/usr/bin/env python
"""End-to-end check of the image -> diagnosis pipeline with the trained model.

Takes real PTB-XL test records, runs (a) direct single-lead diagnosis and
(b) render-to-image -> digitize -> diagnose, and prints findings vs truth.
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from src.diagnostic.config import CACHE_DIR, LEAD_TO_IDX, SUPERCLASSES
from src.diagnose import diagnose_signal, diagnose_image
from src.digitization import render_ecg_strip

X = np.load(os.path.join(CACHE_DIR, "X_100.npy"))
Y = np.load(os.path.join(CACHE_DIR, "Y.npy"))
folds = np.load(os.path.join(CACHE_DIR, "folds.npy"))
test_idx = np.where(folds == 10)[0]

for cls, name in [(0, "NORM"), (1, "MI"), (2, "STTC")]:
    cand = [i for i in test_idx if Y[i, cls] == 1 and Y[i].sum() == 1]
    if not cand:
        continue
    i = cand[0]
    true = [SUPERCLASSES[j] for j in range(5) if Y[i, j] == 1]
    leadII = X[i, :, LEAD_TO_IDX["II"]]

    res = diagnose_signal(leadII[None, :], lead_config="lead2", calibrated=True, with_saliency=True)
    print(f"\n=== record {i} | true={true} ===")
    print(f"[direct signal] top={res['top']['name']} ({res['top']['probability']:.2f}) "
          f"saliency_len={None if res['saliency'] is None else len(res['saliency'])}")

    tmp = os.path.join(tempfile.gettempdir(), f"diag_{name}.png")
    render_ecg_strip(leadII, fs=100, save_path=tmp)
    rimg = diagnose_image(tmp, lead_config="lead2", with_saliency=False)
    print(f"[via image]     calibrated={rimg['digitization']['calibrated']} "
          f"top={rimg['top']['name']} ({rimg['top']['probability']:.2f})")

print("\nEnd-to-end pipeline OK.")
