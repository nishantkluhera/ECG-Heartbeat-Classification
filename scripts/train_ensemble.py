#!/usr/bin/env python
"""Train + evaluate the diagnostic ensemble (research push).

    python scripts/train_ensemble.py                      # full run (default)
    python scripts/train_ensemble.py --epochs 1 --archs resnet1d --seeds 0   # smoke test
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.diagnostic.ensemble import run_ensemble

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--leads", default="12lead")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--archs", nargs="+", default=["resnet1d", "resnet1d_wang", "inception1d"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    args = ap.parse_args()
    run_ensemble(args.leads, tuple(args.archs), tuple(args.seeds), args.epochs)
