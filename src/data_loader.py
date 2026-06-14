"""Data loading and preprocessing for the ECG classifier.

Key design point: the fitted ``StandardScaler`` is *saved to disk* during
training and *re-loaded* during evaluation / prediction. This guarantees that
inference scales inputs with exactly the statistics the model was trained on -
fixing a subtle correctness bug where prediction previously re-fit the scaler
on raw (un-resampled) data.
"""
import logging

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import (
    N_FEATURES, N_CLASSES, RANDOM_STATE, APPLY_SMOTE,
    SCALER_PATH, BATCH_SIZE, NUM_WORKERS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# --------------------------------------------------------------------------- #
# Raw I/O
# --------------------------------------------------------------------------- #
def load_data(train_path: str, test_path: str):
    """Load the train and test CSVs (no header). Returns (df_train, df_test)."""
    try:
        df_train = pd.read_csv(train_path, header=None)
        df_test = pd.read_csv(test_path, header=None)
        logging.info(f"Loaded train data with shape: {df_train.shape}")
        logging.info(f"Loaded test data with shape:  {df_test.shape}")
        return df_train, df_test
    except FileNotFoundError:
        logging.error(f"Data files not found at {train_path} or {test_path}.")
        logging.error("Download the MIT-BIH CSVs into 'data/' or run "
                      "`python main.py generate-data` to create a synthetic dataset.")
        return None, None


def split_features_labels(df: pd.DataFrame):
    """Split a dataframe into X (first N_FEATURES cols) and integer y (last col)."""
    X = df.iloc[:, :N_FEATURES].values.astype(np.float32)
    y = df.iloc[:, N_FEATURES].values.astype(np.int64)
    return X, y


# --------------------------------------------------------------------------- #
# Scaler persistence
# --------------------------------------------------------------------------- #
def save_scaler(scaler: StandardScaler, path: str = SCALER_PATH) -> None:
    joblib.dump(scaler, path)
    logging.info(f"Scaler saved to {path}")


def load_scaler(path: str = SCALER_PATH) -> StandardScaler:
    scaler = joblib.load(path)
    logging.info(f"Scaler loaded from {path}")
    return scaler


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def _reshape_for_conv(X: np.ndarray) -> np.ndarray:
    """(N, features) -> (N, 1, features) channel-first for nn.Conv1d."""
    return X.reshape(X.shape[0], 1, X.shape[1]).astype(np.float32)


def prepare_training_data(df_train: pd.DataFrame, apply_smote: bool = APPLY_SMOTE,
                          scaler_path: str = SCALER_PATH):
    """Preprocess training data: scale (fit), optionally SMOTE, reshape.

    Pipeline order: fit the scaler on the *real* training signals, transform,
    then run SMOTE in the scaled space (better distance behaviour for the
    KNN-based oversampler). The fitted scaler is persisted to ``scaler_path``.

    Returns (X, y, scaler) where X is (N, 1, N_FEATURES) float32.
    """
    X, y = split_features_labels(df_train)
    logging.info(f"Initial class distribution:\n{pd.Series(y).value_counts().sort_index()}")

    scaler = StandardScaler().fit(X)
    X = scaler.transform(X).astype(np.float32)
    save_scaler(scaler, scaler_path)

    if apply_smote:
        logging.info("Applying SMOTE to balance the training data...")
        try:
            X, y = SMOTE(random_state=RANDOM_STATE).fit_resample(X, y)
            logging.info(f"Class distribution after SMOTE:\n{pd.Series(y).value_counts().sort_index()}")
        except ValueError as e:
            logging.warning(f"SMOTE skipped ({e}). Training on the original distribution.")

    return _reshape_for_conv(X), y.astype(np.int64), scaler


def prepare_eval_data(df: pd.DataFrame, scaler: StandardScaler):
    """Preprocess evaluation/test data using a *pre-fitted* scaler (no SMOTE).

    Returns (X, y) where X is (N, 1, N_FEATURES) float32 and y is int64.
    """
    X, y = split_features_labels(df)
    X = scaler.transform(X).astype(np.float32)
    return _reshape_for_conv(X), y


def preprocess_signal(signal: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Scale and reshape a single raw ECG segment of shape (N_FEATURES,).

    Returns an array of shape (1, 1, N_FEATURES) ready for the model.
    """
    signal = np.asarray(signal, dtype=np.float32).reshape(1, -1)
    if signal.shape[1] != N_FEATURES:
        raise ValueError(f"Expected {N_FEATURES} features, got {signal.shape[1]}.")
    scaled = scaler.transform(signal).astype(np.float32)
    return scaled.reshape(1, 1, N_FEATURES)


# --------------------------------------------------------------------------- #
# torch Dataset / DataLoader
# --------------------------------------------------------------------------- #
class ECGDataset(Dataset):
    """Wraps preprocessed (X, y) arrays as a torch Dataset."""

    def __init__(self, X: np.ndarray, y: np.ndarray = None):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


def make_dataloader(X, y=None, batch_size: int = BATCH_SIZE, shuffle: bool = False,
                    num_workers: int = NUM_WORKERS) -> DataLoader:
    return DataLoader(
        ECGDataset(X, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
