"""Train and evaluate a diversified diagnostic ensemble (research push).

Trains multiple architectures x seeds with ECG augmentation, selecting each
model's best epoch on the validation fold (9). Reports macro-AUROC on the test
fold (10) ONCE, for individual models, the prob-averaged ensemble, and the
ensemble with test-time augmentation. No test-fold information is used for any
selection — only averaging of independently-trained models.
"""
import copy
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

from src.diagnostic.config import (
    SUPERCLASSES, SUPERCLASS_FULL, BATCH_SIZE, MODEL_SAVE_DIR, RANDOM_STATE,
)
from src.diagnostic.data import load_split
from src.diagnostic.architectures import build_arch
from src.diagnostic.augment import ECGAugment
from src.utils import set_seed, get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class _AugDataset(Dataset):
    def __init__(self, X, Y, augment=None):
        self.X = X.astype(np.float32)
        self.Y = torch.as_tensor(Y, dtype=torch.float32)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.augment(self.X[i]) if self.augment else self.X[i]
        return torch.as_tensor(x, dtype=torch.float32), self.Y[i]


def _macro_auroc(y_true, y_prob):
    aucs = []
    for c in range(y_true.shape[1]):
        if y_true[:, c].min() != y_true[:, c].max():
            aucs.append(roc_auc_score(y_true[:, c], y_prob[:, c]))
    return float(np.mean(aucs)), aucs


def _pos_weight(Y, device):
    Y = torch.as_tensor(Y, dtype=torch.float32)
    pos = Y.sum(0)
    return ((Y.shape[0] - pos) / pos.clamp(min=1)).clamp(max=20.0).to(device)


@torch.no_grad()
def _predict(model, X, device, tta=False, bs=256):
    """Return (N, n_classes) sigmoid probabilities, optionally TTA-averaged."""
    model.eval()

    def _run(arr):
        out = []
        for i in range(0, len(arr), bs):
            xb = torch.as_tensor(arr[i:i + bs], dtype=torch.float32, device=device)
            out.append(torch.sigmoid(model(xb)).float().cpu().numpy())
        return np.concatenate(out)

    if not tta:
        return _run(X)
    # Average predictions over a few small time-shift views.
    acc = np.zeros((len(X), len(SUPERCLASSES)), dtype=np.float64)
    n = 0
    for s in (0, -20, 20, -40, 40):
        acc += _run(np.roll(X, s, axis=2))
        n += 1
    return (acc / n).astype(np.float32)


def _train_one(arch, seed, data, device, epochs, patience=8):
    set_seed(seed)
    Xtr, Ytr = data["train"]
    Xva, Yva = data["val"]
    n_leads = Xtr.shape[1]

    train_loader = DataLoader(_AugDataset(Xtr, Ytr, ECGAugment()), batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    model = build_arch(arch, n_leads=n_leads).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=_pos_weight(Ytr, device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                           factor=0.3, patience=4, min_lr=1e-5)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_auroc, best_state, no_improve = 0.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        val_prob = _predict(model, Xva, device)
        v_auroc, _ = _macro_auroc(Yva, val_prob)
        scheduler.step(v_auroc)
        if v_auroc > best_auroc:
            best_auroc, best_state, no_improve = v_auroc, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    logging.info(f"  [{arch} seed={seed}] best val macro-AUROC = {best_auroc:.4f}")
    model.load_state_dict(best_state)
    return model


def run_ensemble(lead_config="12lead",
                 archs=("resnet1d", "resnet1d_wang", "inception1d"),
                 seeds=(0, 1), epochs=30):
    device = get_device()
    logging.info(f"Loading PTB-XL [{lead_config}] ...")
    data = load_split(lead_config)
    Xte, Yte = data["test"]

    members, indiv = [], []
    for arch in archs:
        for seed in seeds:
            logging.info(f"Training {arch} (seed {seed}) ...")
            model = _train_one(arch, seed, data, device, epochs)
            prob = _predict(model, Xte, device)
            auroc, _ = _macro_auroc(Yte, prob)
            indiv.append((f"{arch}-s{seed}", auroc))
            members.append({"arch": arch, "seed": seed,
                            "state": copy.deepcopy(model.state_dict())})
            # cache test probs (plain + TTA) for ensembling
            members[-1]["prob"] = prob
            members[-1]["prob_tta"] = _predict(model, Xte, device, tta=True)

    ens = np.mean([m["prob"] for m in members], axis=0)
    ens_tta = np.mean([m["prob_tta"] for m in members], axis=0)
    ens_auroc, _ = _macro_auroc(Yte, ens)
    ens_tta_auroc, per_class = _macro_auroc(Yte, ens_tta)

    print("\n================ ENSEMBLE RESULTS (test fold 10) ================")
    print("Individual models:")
    for name, a in indiv:
        print(f"  {name:<22} macro-AUROC {a:.4f}")
    print(f"\nEnsemble (mean)         macro-AUROC {ens_auroc:.4f}")
    print(f"Ensemble + TTA          macro-AUROC {ens_tta_auroc:.4f}")
    print("\nPer-class AUROC (ensemble + TTA):")
    for c, name in enumerate(SUPERCLASSES):
        if Yte[:, c].min() != Yte[:, c].max():
            print(f"  {SUPERCLASS_FULL[name]:<26} {roc_auc_score(Yte[:, c], ens_tta[:, c]):.4f}")

    path = f"{MODEL_SAVE_DIR}/ptbxl_ensemble_{lead_config}.pt"
    torch.save({"members": [{k: m[k] for k in ("arch", "seed", "state")} for m in members],
                "lead_config": lead_config, "test_macro_auroc": ens_tta_auroc}, path)
    logging.info(f"Saved ensemble ({len(members)} models) to {path}")
    return {"individual": indiv, "ensemble": ens_auroc, "ensemble_tta": ens_tta_auroc}


if __name__ == "__main__":
    run_ensemble()
