"""Utility helpers: reproducibility, device selection, plotting and reporting."""
import os
import logging

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless-safe backend (no display needed for saving plots)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from src.config import CLASS_NAMES, VISUALIZATION_DIR, RANDOM_STATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def set_seed(seed: int = RANDOM_STATE) -> None:
    """Seed Python, NumPy and (if available) PyTorch RNGs for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    """Return the best available torch device (CUDA GPU if present, else CPU)."""
    import torch

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        logging.info(f"Using CUDA device: {name}")
        return torch.device("cuda")
    logging.info("CUDA not available - using CPU.")
    return torch.device("cpu")


def count_parameters(model) -> int:
    """Return the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_training_history(history: dict, save_path: str = None) -> str:
    """Plot training/validation accuracy and loss curves and save to disk."""
    if save_path is None:
        save_path = os.path.join(VISUALIZATION_DIR, "training_history.png")

    epochs_range = range(1, len(history["train_acc"]) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_acc"], label="Training Accuracy")
    plt.plot(epochs_range, history["val_acc"], label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["train_loss"], label="Training Loss")
    plt.plot(epochs_range, history["val_loss"], label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.suptitle("Model Training History", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=120)
    plt.close()
    logging.info(f"Training history plot saved to {save_path}")
    return save_path


def plot_confusion_matrix(y_true, y_pred, class_names=CLASS_NAMES, save_path: str = None) -> str:
    """Plot a confusion matrix heatmap and save to disk."""
    if save_path is None:
        save_path = os.path.join(VISUALIZATION_DIR, "confusion_matrix.png")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("Actual Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    logging.info(f"Confusion matrix plot saved to {save_path}")
    return save_path


def print_classification_report(y_true, y_pred, class_names=CLASS_NAMES) -> str:
    """Build, print and return a per-class classification report."""
    # labels=range(len(class_names)) keeps the report stable even if a class
    # is absent from y_true/y_pred (e.g. on small synthetic splits).
    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    print("\nClassification Report:")
    print(report)
    return report
