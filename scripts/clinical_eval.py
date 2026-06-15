#!/usr/bin/env python
"""Clinically-honest evaluation of the 12-lead diagnostic ensemble.

AUROC alone is not a deployment criterion. This reports, per superclass on the
test fold: AUROC, AUPRC (honest under imbalance), and - at a fixed clinical
operating point (sensitivity >= 0.90) - the resulting specificity and PPV at
both the test-set prevalence and a lower screening prevalence (the base-rate
cliff). Also reports the Brier score (calibration).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve

from src.diagnostic.config import SUPERCLASSES, SUPERCLASS_FULL, MODEL_SAVE_DIR
from src.diagnostic.data import load_split
from src.diagnostic.architectures import build_arch
from src.diagnostic.ensemble import _predict
from src.utils import get_device

TARGET_SENS = 0.90
SCREEN_PREV = 0.05


def ppv(sens, spec, prev):
    tp = sens * prev
    fp = (1 - spec) * (1 - prev)
    return tp / (tp + fp + 1e-12)


def main():
    device = get_device()
    data = load_split("12lead")
    Xte, Yte = data["test"]

    path = os.path.join(MODEL_SAVE_DIR, "ptbxl_ensemble_12lead.pt")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    probs = []
    for m in ckpt["members"]:
        model = build_arch(m["arch"], n_leads=12).to(device)
        model.load_state_dict(m["state"])
        model.eval()
        probs.append(_predict(model, Xte, device, tta=True))
    P = np.mean(probs, axis=0)

    print(f"\nClinically-honest report (12-lead ensemble, test fold, sens>={TARGET_SENS:.0%})")
    print(f"{'Class':<24}{'AUROC':>7}{'AUPRC':>7}{'Prev':>7}"
          f"{'Spec':>7}{'PPV@prev':>9}{'PPV@5%':>8}{'Brier':>7}")
    for c, name in enumerate(SUPERCLASSES):
        y, p = Yte[:, c], P[:, c]
        if y.min() == y.max():
            continue
        prev = y.mean()
        auroc = roc_auc_score(y, p)
        auprc = average_precision_score(y, p)
        brier = brier_score_loss(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        idx = int(np.argmax(tpr >= TARGET_SENS))
        spec = 1 - fpr[idx]
        ppv_prev = ppv(tpr[idx], spec, prev)
        ppv_screen = ppv(tpr[idx], spec, SCREEN_PREV)
        print(f"{SUPERCLASS_FULL[name]:<24}{auroc:>7.3f}{auprc:>7.3f}{prev:>7.2f}"
              f"{spec:>7.2f}{ppv_prev:>9.2f}{ppv_screen:>8.2f}{brier:>7.3f}")


if __name__ == "__main__":
    main()
