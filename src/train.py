"""Train the ECG CNN classifier (PyTorch + optional CUDA AMP)."""
import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from src.config import (
    TRAIN_CSV, TEST_CSV, MODEL_PATH, EPOCHS, BATCH_SIZE, LEARNING_RATE,
    WEIGHT_DECAY, VALIDATION_SPLIT, RANDOM_STATE, EARLY_STOPPING_PATIENCE,
    LR_PATIENCE, LR_FACTOR, MIN_LR, USE_AMP, N_FEATURES, N_CLASSES,
)
from src.data_loader import load_data, prepare_training_data, make_dataloader
from src.model import build_cnn_model
from src.utils import set_seed, get_device, plot_training_history

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=False):
    """Run one epoch. If optimizer is given -> train mode, else eval mode."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for X, y in loader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(X)
                loss = criterion(logits, y)

            if is_train:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * X.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += X.size(0)

    return total_loss / total, correct / total


def train_model(epochs: int = EPOCHS):
    """Load data, build the model, train with early stopping, and save the best."""
    logging.info("Starting model training process...")
    set_seed(RANDOM_STATE)
    device = get_device()

    # 1. Load + preprocess (also fits & saves the scaler)
    df_train, _ = load_data(TRAIN_CSV, TEST_CSV)
    if df_train is None:
        return None
    X, y, _ = prepare_training_data(df_train)

    # 2. Stratified train/validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE, stratify=y
    )
    train_loader = make_dataloader(X_tr, y_tr, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = make_dataloader(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)
    logging.info(f"Train samples: {len(X_tr)} | Validation samples: {len(X_val)}")

    # 3. Model, loss, optimizer, scheduler
    model = build_cnn_model(n_features=N_FEATURES, n_classes=N_CLASSES, device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LR
    )
    use_amp = USE_AMP and device.type == "cuda"
    grad_scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    # 4. Training loop with early stopping + best checkpoint
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_val_loss, epochs_no_improve = 0.0, float("inf"), 0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, device,
                                     optimizer=optimizer, scaler=grad_scaler, use_amp=use_amp)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, device, use_amp=use_amp)
        scheduler.step(val_loss)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        lr = optimizer.param_groups[0]["lr"]
        logging.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val_loss {val_loss:.4f} acc {val_acc:.4f} | lr {lr:.2e}"
        )

        # Save best model (by validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "n_features": N_FEATURES,
                    "n_classes": N_CLASSES,
                    "val_acc": best_val_acc,
                    "epoch": epoch,
                },
                MODEL_PATH,
            )
            logging.info(f"  -> New best model (val_acc={best_val_acc:.4f}) saved to {MODEL_PATH}")

        # Early stopping (by validation loss)
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                logging.info(f"Early stopping triggered at epoch {epoch}.")
                break

    logging.info(f"Training finished. Best validation accuracy: {best_val_acc:.4f}")
    plot_training_history(history)
    return history


if __name__ == "__main__":
    train_model()
