"""1D CNN model definition (PyTorch) for ECG heartbeat classification."""
import logging

import torch
import torch.nn as nn

from src.config import N_FEATURES, N_CLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ECGCNN(nn.Module):
    """A 1D Convolutional Neural Network for single-heartbeat classification.

    Input  : (batch, 1, n_features)  - a single-channel ECG segment.
    Output : (batch, n_classes)      - raw logits (apply softmax for probabilities).

    Architecture: three Conv1d -> BatchNorm -> ReLU -> MaxPool -> Dropout blocks
    followed by a fully-connected classifier head. Mirrors a classic
    feature-extractor + classifier design tuned for short time-series.
    """

    def __init__(self, n_features: int = N_FEATURES, n_classes: int = N_CLASSES):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 64, kernel_size=5, padding="same"),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.2),
            # Block 2
            nn.Conv1d(64, 128, kernel_size=5, padding="same"),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3),
            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, padding="same"),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(0.3),
        )

        # Infer the flattened feature size with a dummy forward pass so the
        # classifier head adapts automatically to a different n_features.
        flat_dim = self._infer_flatten_dim(n_features)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes),
        )

    def _infer_flatten_dim(self, n_features: int) -> int:
        was_training = self.features.training
        self.features.eval()  # avoid touching BatchNorm running stats
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_features)
            flat = self.features(dummy).numel()
        if was_training:
            self.features.train()
        return flat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_cnn_model(n_features: int = N_FEATURES, n_classes: int = N_CLASSES, device=None) -> ECGCNN:
    """Construct the CNN, optionally move it to a device, and log a summary."""
    model = ECGCNN(n_features=n_features, n_classes=n_classes)
    if device is not None:
        model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Built ECGCNN with {n_params:,} trainable parameters.")
    return model


def load_trained_model(checkpoint_path: str, device=None) -> ECGCNN:
    """Load a trained model from a checkpoint saved by ``train.py``.

    Supports both the dict-style checkpoint (with metadata) and a bare
    ``state_dict``. Returns the model in eval mode on the chosen device.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        n_features = checkpoint.get("n_features", N_FEATURES)
        n_classes = checkpoint.get("n_classes", N_CLASSES)
        state_dict = checkpoint["model_state_dict"]
    else:  # bare state_dict
        n_features, n_classes, state_dict = N_FEATURES, N_CLASSES, checkpoint

    model = ECGCNN(n_features=n_features, n_classes=n_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    logging.info(f"Loaded trained model from {checkpoint_path}")
    return model
