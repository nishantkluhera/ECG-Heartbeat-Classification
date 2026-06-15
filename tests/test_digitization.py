"""Round-trip test for ECG image digitization: signal -> image -> signal."""
import os
import tempfile

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("scipy")

from src.digitization import render_ecg_strip, digitize_ecg_image, _fit_length


def _synthetic_ecg(n=1000, fs=100, hr=72):
    """A simple QRS-like periodic signal (no neurokit dependency in tests)."""
    t = np.arange(n) / fs
    rr = 60.0 / hr
    sig = np.zeros(n)
    # R peaks as narrow gaussians, plus a small T wave.
    for k in range(int(t[-1] / rr) + 1):
        r = k * rr
        sig += 1.0 * np.exp(-((t - r) ** 2) / (2 * 0.01 ** 2))
        sig += 0.25 * np.exp(-((t - r - 0.18) ** 2) / (2 * 0.03 ** 2))
    return (sig - sig.mean()) / (sig.std() + 1e-8)


def test_round_trip_recovers_morphology():
    fs = 100
    ecg = _synthetic_ecg(1000, fs)
    tmp = os.path.join(tempfile.gettempdir(), "ecg_round_trip.png")
    render_ecg_strip(ecg, fs=fs, save_path=tmp, with_grid=True)

    res = digitize_ecg_image(tmp, target_len=len(ecg), target_fs=fs)
    assert res.signal.shape == (len(ecg),)

    a = (ecg - ecg.mean()) / (ecg.std() + 1e-8)
    b = (res.signal - res.signal.mean()) / (res.signal.std() + 1e-8)
    corr = float(np.corrcoef(a, b)[0, 1])
    assert corr > 0.8, f"round-trip correlation too low: {corr:.3f}"


def test_grid_calibration_detected():
    fs = 100
    ecg = _synthetic_ecg(1000, fs)
    tmp = os.path.join(tempfile.gettempdir(), "ecg_cal.png")
    render_ecg_strip(ecg, fs=fs, save_path=tmp, with_grid=True)
    res = digitize_ecg_image(tmp, target_len=1000, target_fs=fs)
    assert res.calibrated
    assert res.pixels_per_mm and res.pixels_per_mm > 1


def test_fit_length():
    assert len(_fit_length(np.ones(500), 1000)) == 1000
    assert len(_fit_length(np.ones(1500), 1000)) == 1000
    assert len(_fit_length(np.ones(1000), 1000)) == 1000
