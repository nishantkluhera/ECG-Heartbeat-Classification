"""Tests for preprocessing, scaler persistence and the torch Dataset."""
import os

import numpy as np
import pandas as pd
import pytest

# These pull in torch / imblearn; skip cleanly if the env isn't ready yet.
pytest.importorskip("torch")
pytest.importorskip("imblearn")

from src import data_loader as dl
from src.config import N_FEATURES, N_CLASSES


def _make_df(n_per_class: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for cls in range(N_CLASSES):
        feats = rng.random((n_per_class, N_FEATURES))
        labels = np.full((n_per_class, 1), cls)
        rows.append(np.hstack([feats, labels]))
    data = np.vstack(rows)
    rng.shuffle(data)
    return pd.DataFrame(data)


def test_split_features_labels():
    df = _make_df()
    X, y = dl.split_features_labels(df)
    assert X.shape == (len(df), N_FEATURES)
    assert X.dtype == np.float32
    assert y.dtype == np.int64
    assert set(np.unique(y)) <= set(range(N_CLASSES))


def test_prepare_training_data_and_scaler_roundtrip(tmp_path):
    df = _make_df()
    scaler_path = os.path.join(tmp_path, "scaler.joblib")
    X, y, scaler = dl.prepare_training_data(df, apply_smote=False, scaler_path=scaler_path)

    assert X.shape[1:] == (1, N_FEATURES)  # channel-first
    assert X.dtype == np.float32
    assert len(X) == len(y)
    assert os.path.exists(scaler_path)

    # Saved scaler reloads and transforms consistently.
    reloaded = dl.load_scaler(scaler_path)
    raw = df.iloc[:5, :N_FEATURES].values.astype(np.float32)
    np.testing.assert_allclose(reloaded.transform(raw), scaler.transform(raw), rtol=1e-5)


def test_prepare_training_data_smote_balances(tmp_path):
    # Imbalanced: class 0 gets many more samples than the rest.
    rng = np.random.default_rng(1)
    big = np.hstack([rng.random((100, N_FEATURES)), np.zeros((100, 1))])
    small = np.vstack([
        np.hstack([rng.random((15, N_FEATURES)), np.full((15, 1), c)])
        for c in range(1, N_CLASSES)
    ])
    df = pd.DataFrame(np.vstack([big, small]))
    scaler_path = os.path.join(tmp_path, "scaler.joblib")
    _, y, _ = dl.prepare_training_data(df, apply_smote=True, scaler_path=scaler_path)
    counts = np.bincount(y)
    assert counts.min() == counts.max()  # SMOTE balanced every class


def test_preprocess_signal_shape_and_validation(tmp_path):
    df = _make_df()
    scaler_path = os.path.join(tmp_path, "scaler.joblib")
    _, _, scaler = dl.prepare_training_data(df, apply_smote=False, scaler_path=scaler_path)

    sig = np.random.rand(N_FEATURES).astype(np.float32)
    out = dl.preprocess_signal(sig, scaler)
    assert out.shape == (1, 1, N_FEATURES)

    with pytest.raises(ValueError):
        dl.preprocess_signal(np.random.rand(N_FEATURES - 1), scaler)


def test_ecg_dataset_and_loader():
    X = np.random.rand(20, 1, N_FEATURES).astype(np.float32)
    y = np.random.randint(0, N_CLASSES, size=20)

    ds = dl.ECGDataset(X, y)
    assert len(ds) == 20
    xi, yi = ds[0]
    assert tuple(xi.shape) == (1, N_FEATURES)

    loader = dl.make_dataloader(X, y, batch_size=8, shuffle=False)
    xb, yb = next(iter(loader))
    assert tuple(xb.shape) == (8, 1, N_FEATURES)
    assert tuple(yb.shape) == (8,)

    # Without labels the loader yields features only.
    feat_loader = dl.make_dataloader(X, None, batch_size=8)
    xb_only = next(iter(feat_loader))
    assert tuple(xb_only.shape) == (8, 1, N_FEATURES)
