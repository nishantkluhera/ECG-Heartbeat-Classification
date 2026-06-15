"""Evaluate the PTB-XL diagnostic model on the test fold."""
import logging
import os

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score

from src.diagnostic.config import SUPERCLASSES, SUPERCLASS_FULL, BATCH_SIZE, NUM_WORKERS, model_path
from src.diagnostic.data import make_dataloaders
from src.diagnostic.model import load_diag_model
from src.utils import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@torch.no_grad()
def evaluate_diag(lead_config: str = "12lead"):
    path = model_path(lead_config)
    if not os.path.exists(path):
        logging.error(f"No diagnostic model at {path}. Train it first.")
        return None

    device = get_device()
    model = load_diag_model(path, device=device)
    loaders = make_dataloaders(lead_config, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    test_loader = loaders["test"]

    probs, labels = [], []
    for X, y in test_loader:
        X = X.to(device, non_blocking=True)
        probs.append(torch.sigmoid(model(X)).float().cpu().numpy())
        labels.append(y.numpy())
    probs = np.concatenate(probs)
    labels = np.concatenate(labels)

    print(f"\nPTB-XL test-set results [{lead_config}]")
    print(f"{'Class':<26}{'AUROC':>8}{'F1@0.5':>9}{'Support':>9}")
    aucs = []
    for c, name in enumerate(SUPERCLASSES):
        support = int(labels[:, c].sum())
        if labels[:, c].min() != labels[:, c].max():
            auc = roc_auc_score(labels[:, c], probs[:, c])
            f1 = f1_score(labels[:, c], (probs[:, c] >= 0.5).astype(int), zero_division=0)
            aucs.append(auc)
        else:
            auc, f1 = float("nan"), float("nan")
        print(f"{SUPERCLASS_FULL[name]:<26}{auc:>8.4f}{f1:>9.4f}{support:>9}")

    macro = float(np.mean(aucs))
    print(f"\nMacro AUROC: {macro:.4f}")
    logging.info(f"[{lead_config}] test macro-AUROC = {macro:.4f}")
    return macro


if __name__ == "__main__":
    evaluate_diag("12lead")
