"""Single-heartbeat inference.

Loads the trained model and the *saved* scaler once, then classifies raw ECG
segments of shape (N_FEATURES,). Because the scaler is the same object fitted
during training, inputs are normalised exactly as the model expects.
"""
import logging
import os

import numpy as np
import torch

from src.config import MODEL_PATH, SCALER_PATH, N_FEATURES, CLASS_MAP, CLASS_NAMES
from src.data_loader import load_scaler, preprocess_signal
from src.model import load_trained_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Cached resources (loaded lazily, once).
_model = None
_scaler = None
_device = None


def load_prediction_resources() -> bool:
    """Load and cache the model + scaler. Returns True on success."""
    global _model, _scaler, _device
    if _model is not None and _scaler is not None:
        return True

    if not os.path.exists(MODEL_PATH):
        logging.error(f"No trained model at {MODEL_PATH}. Run `python main.py train` first.")
        return False
    if not os.path.exists(SCALER_PATH):
        logging.error(f"No saved scaler at {SCALER_PATH}. Re-run training to produce it.")
        return False

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = load_trained_model(MODEL_PATH, device=_device)
    _scaler = load_scaler(SCALER_PATH)
    return True


@torch.no_grad()
def predict_heartbeat(ecg_signal_segment: np.ndarray):
    """Classify a single ECG segment.

    Args:
        ecg_signal_segment: 1D array of shape (N_FEATURES,).

    Returns:
        (class_name, confidence) on success, else (None, None).
    """
    if not load_prediction_resources():
        return None, None

    try:
        x = preprocess_signal(ecg_signal_segment, _scaler)        # (1, 1, N_FEATURES)
        x = torch.as_tensor(x, dtype=torch.float32, device=_device)
        probs = torch.softmax(_model(x), dim=1)[0].cpu().numpy()
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        class_name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else CLASS_MAP.get(idx, "Unknown")
        logging.info(f"Prediction: {class_name} (confidence {confidence:.4f})")
        return class_name, confidence
    except Exception as e:  # noqa: BLE001 - surface any inference error to the caller
        logging.error(f"Error during prediction: {e}")
        return None, None


@torch.no_grad()
def predict_proba(ecg_signal_segment: np.ndarray):
    """Return the full class-probability vector for a single segment (or None)."""
    if not load_prediction_resources():
        return None
    x = preprocess_signal(ecg_signal_segment, _scaler)
    x = torch.as_tensor(x, dtype=torch.float32, device=_device)
    return torch.softmax(_model(x), dim=1)[0].cpu().numpy()


if __name__ == "__main__":
    print("Loading prediction resources...")
    if not load_prediction_resources():
        print("Failed to load resources. Train the model first: python main.py train")
        raise SystemExit(1)

    # Demo on a random segment.
    dummy = np.random.rand(N_FEATURES).astype(np.float32)
    name, conf = predict_heartbeat(dummy)
    print(f"Random segment -> {name} (confidence {conf:.2f})")

    # Demo on a real test sample, if available.
    import pandas as pd
    from src.config import TEST_CSV

    if os.path.exists(TEST_CSV):
        df = pd.read_csv(TEST_CSV, header=None)
        sample = df.iloc[100, :N_FEATURES].values
        actual = CLASS_NAMES[int(df.iloc[100, N_FEATURES])]
        name, conf = predict_heartbeat(sample)
        print(f"Test sample (actual: {actual}) -> {name} (confidence {conf:.2f})")
