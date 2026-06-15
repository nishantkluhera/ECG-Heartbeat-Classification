#!/usr/bin/env python
"""Fast sanity check that PTB-XL extracted correctly and parses.

Validates metadata loading, diagnostic-superclass aggregation, and reading a
single WFDB record - WITHOUT building the full cache. Run after extraction:

    python scripts/check_ptbxl.py
"""
import os
import sys
import ast

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from src.diagnostic.data import find_ptbxl_root, _aggregate_superclasses, _multi_hot
from src.diagnostic.config import SUPERCLASSES


def main():
    root = find_ptbxl_root()
    print(f"PTB-XL root: {root}")

    db = pd.read_csv(os.path.join(root, "ptbxl_database.csv"), index_col="ecg_id")
    db.scp_codes = db.scp_codes.apply(ast.literal_eval)
    scp = pd.read_csv(os.path.join(root, "scp_statements.csv"), index_col=0)
    scp = scp[scp.diagnostic == 1]
    diag_map = scp.diagnostic_class.to_dict()

    labels = db.scp_codes.apply(lambda d: _aggregate_superclasses(d, diag_map))
    Y = np.stack(labels.apply(_multi_hot).values)

    print(f"records: {len(db)}")
    print(f"label matrix: {Y.shape}")
    for i, name in enumerate(SUPERCLASSES):
        print(f"  {name:<6} positives: {int(Y[:, i].sum())}")
    print(f"records with >=1 label: {int((Y.sum(1) > 0).sum())}")
    print(f"strat folds: {sorted(db.strat_fold.unique())}")

    import wfdb
    rel = db.filename_lr.iloc[0]
    sig, meta = wfdb.rdsamp(os.path.join(root, rel))
    print(f"sample record {rel}: signal {sig.shape}, fs={meta['fs']}")
    print("OK" if sig.shape == (1000, 12) else "UNEXPECTED SIGNAL SHAPE")


if __name__ == "__main__":
    main()
