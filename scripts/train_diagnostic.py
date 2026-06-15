#!/usr/bin/env python
"""Train + evaluate both PTB-XL diagnostic models (single-lead and 12-lead).

Builds the signal cache on first use, then trains and evaluates each lead
configuration. Run after PTB-XL is extracted:

    python scripts/train_diagnostic.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.diagnostic.train import train_diag
from src.diagnostic.evaluate import evaluate_diag

EPOCHS = 30
results = {}

for lead in ["lead2", "12lead"]:
    print(f"\n{'='*70}\n  TRAIN  [{lead}]\n{'='*70}", flush=True)
    best_val = train_diag(lead_config=lead, epochs=EPOCHS)
    print(f"\n{'='*70}\n  EVALUATE  [{lead}]\n{'='*70}", flush=True)
    test_macro = evaluate_diag(lead_config=lead)
    results[lead] = {"val_macro_auroc": best_val, "test_macro_auroc": test_macro}

print(f"\n{'#'*70}\n  SUMMARY\n{'#'*70}")
for lead, r in results.items():
    print(f"  {lead:<8} val macro-AUROC={r['val_macro_auroc']:.4f}  "
          f"test macro-AUROC={r['test_macro_auroc']:.4f}")
