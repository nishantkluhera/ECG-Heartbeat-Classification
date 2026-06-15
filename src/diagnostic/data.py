"""PTB-XL loading, label aggregation, caching and splits.

Builds multi-label diagnostic-superclass targets, loads the 100 Hz WFDB
signals (cached to .npy so it is a one-time cost), and exposes fold-based
train/val/test splits standardised per lead.
"""
import os
import ast
import glob
import logging

import numpy as np
import pandas as pd

from src.diagnostic.config import (
    PTBXL_DIR, CACHE_DIR, SAMPLING_RATE, SIGNAL_LENGTH, LEADS, LEAD_TO_IDX,
    SUPERCLASSES, TRAIN_FOLDS, VAL_FOLD, TEST_FOLD,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

_SUPER_IDX = {name: i for i, name in enumerate(SUPERCLASSES)}


def find_ptbxl_root() -> str:
    """Locate the directory that contains ptbxl_database.csv under data/ptbxl/."""
    matches = glob.glob(os.path.join(PTBXL_DIR, "**", "ptbxl_database.csv"), recursive=True)
    if not matches:
        raise FileNotFoundError(
            f"ptbxl_database.csv not found under {PTBXL_DIR}. "
            f"Download/extract PTB-XL first (see scripts/get_ptbxl.py)."
        )
    return os.path.dirname(matches[0])


def _aggregate_superclasses(scp_codes: dict, diag_map: dict) -> list:
    """Map a record's scp_codes dict to its set of diagnostic superclasses."""
    out = set()
    for code in scp_codes:
        sc = diag_map.get(code)
        if sc is not None:
            out.add(sc)
    return list(out)


def _multi_hot(superclasses: list) -> np.ndarray:
    vec = np.zeros(len(SUPERCLASSES), dtype=np.float32)
    for sc in superclasses:
        if sc in _SUPER_IDX:
            vec[_SUPER_IDX[sc]] = 1.0
    return vec


def build_cache(force: bool = False) -> None:
    """Load PTB-XL once and cache X (signals), Y (labels), folds, and norm stats."""
    x_path = os.path.join(CACHE_DIR, f"X_{SAMPLING_RATE}.npy")
    y_path = os.path.join(CACHE_DIR, "Y.npy")
    f_path = os.path.join(CACHE_DIR, "folds.npy")
    n_path = os.path.join(CACHE_DIR, "norm_stats.npz")
    if not force and all(os.path.exists(p) for p in (x_path, y_path, f_path, n_path)):
        logging.info("PTB-XL cache already present; skipping rebuild.")
        return

    import wfdb  # imported here so config/tests don't require it

    root = find_ptbxl_root()
    logging.info(f"Building PTB-XL cache from {root} ...")

    db = pd.read_csv(os.path.join(root, "ptbxl_database.csv"), index_col="ecg_id")
    db.scp_codes = db.scp_codes.apply(ast.literal_eval)

    scp = pd.read_csv(os.path.join(root, "scp_statements.csv"), index_col=0)
    scp = scp[scp.diagnostic == 1]
    diag_map = scp.diagnostic_class.to_dict()  # scp code -> superclass

    db["superclasses"] = db.scp_codes.apply(lambda d: _aggregate_superclasses(d, diag_map))
    Y = np.stack(db.superclasses.apply(_multi_hot).values)
    folds = db.strat_fold.values.astype(np.int64)

    filename_col = "filename_lr" if SAMPLING_RATE == 100 else "filename_hr"
    paths = db[filename_col].values

    n = len(paths)
    X = np.zeros((n, SIGNAL_LENGTH, len(LEADS)), dtype=np.float32)
    for i, rel in enumerate(paths):
        sig, _ = wfdb.rdsamp(os.path.join(root, rel))
        # Guard against rare off-by-one lengths.
        L = min(SIGNAL_LENGTH, sig.shape[0])
        X[i, :L, :] = sig[:L, :len(LEADS)]
        if (i + 1) % 2000 == 0:
            logging.info(f"  loaded {i + 1}/{n} records")

    # Per-lead standardisation stats from the TRAINING folds only.
    train_mask = np.isin(folds, TRAIN_FOLDS)
    flat = X[train_mask].reshape(-1, len(LEADS))
    mean = flat.mean(axis=0).astype(np.float32)
    std = (flat.std(axis=0) + 1e-8).astype(np.float32)

    np.save(x_path, X)
    np.save(y_path, Y)
    np.save(f_path, folds)
    np.savez(n_path, mean=mean, std=std)
    logging.info(f"Cached PTB-XL: X{X.shape}, Y{Y.shape} -> {CACHE_DIR}")


def load_norm_stats():
    d = np.load(os.path.join(CACHE_DIR, "norm_stats.npz"))
    return d["mean"], d["std"]


def _select_leads(X: np.ndarray, lead_config: str) -> np.ndarray:
    """X is (N, L, 12). Return (N, C, L) channel-first for the chosen leads."""
    if lead_config == "12lead":
        idx = list(range(len(LEADS)))
    elif lead_config == "lead2":
        idx = [LEAD_TO_IDX["II"]]
    else:
        raise ValueError(f"Unknown lead_config '{lead_config}' (use '12lead' or 'lead2').")
    Xs = X[:, :, idx]                      # (N, L, C)
    return np.transpose(Xs, (0, 2, 1))     # (N, C, L)


def load_split(lead_config: str = "12lead"):
    """Return standardised channel-first splits for the chosen lead configuration.

    Returns dict with keys train/val/test, each a (X, Y) tuple.
    """
    build_cache()
    X = np.load(os.path.join(CACHE_DIR, f"X_{SAMPLING_RATE}.npy"))
    Y = np.load(os.path.join(CACHE_DIR, "Y.npy"))
    folds = np.load(os.path.join(CACHE_DIR, "folds.npy"))
    mean, std = load_norm_stats()

    X = (X - mean) / std                   # broadcast over (N, L, 12)
    X = _select_leads(X, lead_config).astype(np.float32)

    def split(mask):
        return X[mask], Y[mask].astype(np.float32)

    return {
        "train": split(np.isin(folds, TRAIN_FOLDS)),
        "val": split(folds == VAL_FOLD),
        "test": split(folds == TEST_FOLD),
        "n_leads": X.shape[1],
    }


# torch Dataset / DataLoader -------------------------------------------------- #
def make_dataloaders(lead_config: str, batch_size: int, num_workers: int = 0):
    import torch
    from torch.utils.data import Dataset, DataLoader

    data = load_split(lead_config)

    class _DS(Dataset):
        def __init__(self, xy):
            self.X = torch.as_tensor(xy[0], dtype=torch.float32)
            self.Y = torch.as_tensor(xy[1], dtype=torch.float32)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            return self.X[i], self.Y[i]

    pin = torch.cuda.is_available()
    loaders = {
        name: DataLoader(_DS(data[name]), batch_size=batch_size,
                         shuffle=(name == "train"), num_workers=num_workers, pin_memory=pin)
        for name in ("train", "val", "test")
    }
    loaders["n_leads"] = data["n_leads"]
    return loaders
