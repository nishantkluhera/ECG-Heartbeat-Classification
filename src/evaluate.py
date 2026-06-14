"""Evaluate a trained ECG CNN on the held-out test set."""
import logging
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score

from src.config import (
    TRAIN_CSV, TEST_CSV, MODEL_PATH, SCALER_PATH, CLASS_NAMES, BATCH_SIZE,
)
from src.data_loader import load_data, prepare_eval_data, load_scaler, make_dataloader
from src.model import load_trained_model
from src.utils import get_device, plot_confusion_matrix, print_classification_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@torch.no_grad()
def _predict_all(model, loader, device):
    preds = []
    for X in loader:
        X = X.to(device, non_blocking=True)
        logits = model(X)
        preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(preds)


def evaluate_model():
    """Load the trained model + saved scaler, evaluate on the test set."""
    logging.info("Starting model evaluation process...")
    device = get_device()

    if not os.path.exists(MODEL_PATH):
        logging.error(f"No trained model at {MODEL_PATH}. Run training first "
                      f"(`python main.py train`).")
        return None
    if not os.path.exists(SCALER_PATH):
        logging.error(f"No saved scaler at {SCALER_PATH}. Re-run training to produce it.")
        return None

    model = load_trained_model(MODEL_PATH, device=device)
    scaler = load_scaler(SCALER_PATH)

    # We only need the test split; reuse the saved scaler so preprocessing
    # exactly matches training (no SMOTE, no re-fitting).
    _, df_test = load_data(TRAIN_CSV, TEST_CSV)
    if df_test is None:
        return None

    X_test, y_true = prepare_eval_data(df_test, scaler)
    # Dataset without labels -> loader yields features only.
    loader = make_dataloader(X_test, y=None, batch_size=BATCH_SIZE, shuffle=False)

    logging.info("Making predictions on the test set...")
    y_pred = _predict_all(model, loader, device)

    accuracy = accuracy_score(y_true, y_pred)
    logging.info(f"Test Set Accuracy: {accuracy:.4f}")

    print_classification_report(y_true, y_pred, class_names=CLASS_NAMES)
    plot_confusion_matrix(y_true, y_pred, class_names=CLASS_NAMES)

    logging.info("Evaluation process completed.")
    return accuracy


if __name__ == "__main__":
    evaluate_model()
