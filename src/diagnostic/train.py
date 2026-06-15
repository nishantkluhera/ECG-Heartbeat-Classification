"""Train the PTB-XL diagnostic model (multi-label, macro-AUROC)."""
import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from src.diagnostic.config import (
    EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, EARLY_STOPPING_PATIENCE,
    LR_PATIENCE, LR_FACTOR, MIN_LR, USE_AMP, NUM_WORKERS, RANDOM_STATE,
    SUPERCLASSES, LEADS, LEAD_TO_IDX, model_path,
)
from src.diagnostic.data import make_dataloaders, load_norm_stats
from src.diagnostic.model import build_diag_model
from src.utils import set_seed, get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def macro_auroc(y_true: np.ndarray, y_prob: np.ndarray):
    """Macro AUROC over classes that have both positive and negative samples."""
    aucs = {}
    for c, name in enumerate(SUPERCLASSES):
        col = y_true[:, c]
        if col.min() != col.max():
            aucs[name] = roc_auc_score(col, y_prob[:, c])
    macro = float(np.mean(list(aucs.values()))) if aucs else float("nan")
    return macro, aucs


def _pos_weight(loader, device):
    """Inverse-frequency positive weights for BCE, from the training labels."""
    Y = loader.dataset.Y
    pos = Y.sum(dim=0)
    neg = Y.shape[0] - pos
    w = (neg / pos.clamp(min=1)).clamp(max=20.0)
    return w.to(device)


@torch.no_grad()
def _collect_logits(model, loader, device, use_amp):
    model.eval()
    logits, labels = [], []
    for X, y in loader:
        X = X.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(X)
        logits.append(out.float().cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def _fit_temperature(logits, labels):
    """Single-temperature scaling to calibrate multi-label probabilities."""
    T = nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.02, max_iter=100)
    bce = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = bce(logits / T.clamp(min=1e-2), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.clamp(min=1e-2).item())


def train_diag(lead_config: str = "12lead", epochs: int = EPOCHS):
    logging.info(f"Training PTB-XL diagnostic model [{lead_config}] ...")
    set_seed(RANDOM_STATE)
    device = get_device()

    loaders = make_dataloaders(lead_config, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    n_leads = loaders["n_leads"]
    train_loader, val_loader, test_loader = loaders["train"], loaders["val"], loaders["test"]
    logging.info(f"n_leads={n_leads} | train={len(train_loader.dataset)} "
                 f"val={len(val_loader.dataset)} test={len(test_loader.dataset)}")

    model = build_diag_model(n_leads=n_leads, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=_pos_weight(train_loader, device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LR
    )
    use_amp = USE_AMP and device.type == "cuda"
    grad_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_auroc, epochs_no_improve = 0.0, 0
    history = {"train_loss": [], "val_loss": [], "val_auroc": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for X, y in train_loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(X), y)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            running += loss.item() * X.size(0)
        train_loss = running / len(train_loader.dataset)

        val_logits, val_labels = _collect_logits(model, val_loader, device, use_amp)
        val_loss = nn.BCEWithLogitsLoss()(val_logits, val_labels).item()
        val_probs = torch.sigmoid(val_logits).numpy()
        v_macro, _ = macro_auroc(val_labels.numpy(), val_probs)
        scheduler.step(v_macro)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auroc"].append(v_macro)
        logging.info(f"Epoch {epoch:3d}/{epochs} | train_loss {train_loss:.4f} | "
                     f"val_loss {val_loss:.4f} | val_macroAUROC {v_macro:.4f} | "
                     f"lr {optimizer.param_groups[0]['lr']:.2e}")

        if v_macro > best_auroc:
            best_auroc = v_macro
            epochs_no_improve = 0
            _save_checkpoint(model, lead_config, n_leads, best_auroc, val_logits, val_labels)
            logging.info(f"  -> New best (val_macroAUROC={best_auroc:.4f}) saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                logging.info(f"Early stopping at epoch {epoch}.")
                break

    logging.info(f"Training done. Best val macro-AUROC: {best_auroc:.4f}")
    return best_auroc


def _save_checkpoint(model, lead_config, n_leads, val_auroc, val_logits, val_labels):
    """Save a self-contained checkpoint: weights, lead norm stats, and calibration T."""
    temperature = _fit_temperature(val_logits, val_labels)

    mean, std = load_norm_stats()
    if lead_config == "12lead":
        idx = list(range(len(LEADS)))
    else:  # lead2
        idx = [LEAD_TO_IDX["II"]]

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "lead_config": lead_config,
            "n_leads": n_leads,
            "lead_indices": idx,
            "lead_means": mean[idx].tolist(),
            "lead_stds": std[idx].tolist(),
            "temperature": temperature,
            "superclasses": SUPERCLASSES,
            "val_auroc": val_auroc,
        },
        model_path(lead_config),
    )


if __name__ == "__main__":
    train_diag("12lead")
