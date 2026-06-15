"""Single-lead ECG image digitization: image -> 1D signal.

Converts a clean single-lead ECG strip image (screenshot / scan / clean
printout) back into a 1-D waveform that downstream models can consume:

    de-grid -> binarize -> per-column trace extraction -> grid-based
    calibration (mV / seconds) -> resample to a target sampling rate.

Real-world phone photos (skew, shadows, 12-lead layouts) are substantially
harder; this module targets clean single-lead strips and is honest about its
calibration confidence. Includes a renderer used both for testing
(signal -> image -> signal round-trip) and for the app's "try a sample" mode.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Rendering (for tests and the demo's sample mode)
# --------------------------------------------------------------------------- #
def render_ecg_strip(signal: np.ndarray, fs: int = 100, save_path: Optional[str] = None,
                     mv_per_unit: float = 1.0, with_grid: bool = True):
    """Render a 1-D ECG signal as a standard ECG-paper strip image (PNG).

    The signal is assumed to be in millivolts (after multiplying by
    ``mv_per_unit``). Uses 25 mm/s and 10 mm/mV conventions. Returns save_path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sig = np.asarray(signal, dtype=float).ravel() * mv_per_unit
    t = np.arange(len(sig)) / fs
    duration = t[-1] if len(t) else 1.0

    fig, ax = plt.subplots(figsize=(min(25, max(6, duration * 2.5)), 3.2))
    ax.plot(t, sig, color="black", linewidth=1.0)

    if with_grid:
        # Minor grid: 0.04 s / 0.1 mV; major grid: 0.2 s / 0.5 mV.
        ax.set_xticks(np.arange(0, duration + 0.04, 0.2))
        ax.set_yticks(np.arange(np.floor(sig.min() - 0.5), np.ceil(sig.max() + 0.5), 0.5))
        ax.grid(which="major", color="#e9737d", linewidth=0.8)
        ax.minorticks_on()
        ax.grid(which="minor", color="#f2b9bd", linewidth=0.4)
        ax.set_xticks(np.arange(0, duration + 0.04, 0.04), minor=True)
    ax.margins(x=0)
    ax.set_xlabel("")
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    fig.tight_layout(pad=0.2)
    if save_path is None:
        save_path = "ecg_strip.png"
    fig.savefig(save_path, dpi=100)
    plt.close(fig)
    return save_path


# --------------------------------------------------------------------------- #
# Digitization
# --------------------------------------------------------------------------- #
@dataclass
class DigitizationResult:
    signal: np.ndarray                 # 1-D recovered waveform (mV if calibrated)
    fs: int                            # sampling rate of `signal`
    calibrated: bool                   # True if grid-based mV/s calibration succeeded
    pixels_per_mm: Optional[float] = None
    info: dict = field(default_factory=dict)


def _to_gray(img: np.ndarray) -> np.ndarray:
    import cv2
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _binarize_trace(img: np.ndarray) -> np.ndarray:
    """Return a binary mask where the (dark) ECG trace is foreground (255).

    For colour images the trace is isolated as *near-black* pixels, which
    cleanly rejects the red/pink ECG grid (high in the red channel). For
    grayscale images we fall back to Otsu thresholding.
    """
    import cv2
    if img.ndim == 3:
        max_channel = img.max(axis=2)               # black trace -> low in every channel
        bw = np.where(max_channel < 120, 255, 0).astype(np.uint8)
        if bw.sum() < 0.0005 * 255 * bw.size:       # too little -> fall back to Otsu
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, bw = cv2.threshold(cv2.GaussianBlur(gray, (3, 3), 0), 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, bw = cv2.threshold(cv2.GaussianBlur(img, (3, 3), 0), 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)


def _estimate_pixels_per_mm(gray: np.ndarray) -> Optional[float]:
    """Estimate small-box (1 mm) pixel spacing from the grid via autocorrelation."""
    # Grid lines create a periodic pattern in column/row brightness.
    col_profile = 255.0 - gray.mean(axis=0)      # darker grid lines -> peaks
    col_profile = col_profile - col_profile.mean()
    if np.allclose(col_profile, 0):
        return None
    ac = np.correlate(col_profile, col_profile, mode="full")[len(col_profile) - 1:]
    # Find the first strong peak after lag 0 within a plausible mm range (3-40 px).
    lo, hi = 3, min(40, len(ac) - 1)
    if hi <= lo:
        return None
    peak = lo + int(np.argmax(ac[lo:hi]))
    return float(peak) if ac[peak] > 0 else None


def _extract_trace(bw: np.ndarray) -> np.ndarray:
    """One y-value per column (centroid of trace pixels), NaN-interpolated."""
    h, w = bw.shape
    ys = np.full(w, np.nan)
    for x in range(w):
        rows = np.where(bw[:, x] > 0)[0]
        if rows.size:
            ys[x] = rows.mean()
    # Interpolate gaps.
    valid = ~np.isnan(ys)
    if valid.sum() < 2:
        raise ValueError("Could not detect an ECG trace in the image.")
    ys[~valid] = np.interp(np.flatnonzero(~valid), np.flatnonzero(valid), ys[valid])
    # Invert (image y grows downward) so larger value = higher amplitude.
    return (h - 1) - ys


def _resample(sig: np.ndarray, n_out: int) -> np.ndarray:
    from scipy.signal import resample
    if len(sig) == n_out:
        return sig.astype(np.float32)
    return resample(sig, n_out).astype(np.float32)


def digitize_ecg_image(image, target_len: int = 1000, target_fs: int = 100,
                       paper_speed_mm_s: float = 25.0, gain_mm_mv: float = 10.0) -> DigitizationResult:
    """Digitize a single-lead ECG strip image into a 1-D waveform.

    Args:
        image: file path (str) or a BGR/grayscale numpy array.
        target_len / target_fs: output length and sampling rate.
        paper_speed_mm_s / gain_mm_mv: standard ECG calibration constants.

    Returns a DigitizationResult. If grid spacing can be estimated the signal
    is calibrated to millivolts; otherwise it is per-instance normalised and
    ``calibrated=False`` (a flag the app surfaces to the user).
    """
    import cv2
    if isinstance(image, str):
        img = cv2.imread(image, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image}")
    else:
        img = image

    gray = _to_gray(img)
    bw = _binarize_trace(img)

    # Crop to the trace's bounding box so surrounding margins/labels don't
    # distort the horizontal (time) scale.
    rows, cols = np.where(bw > 0)
    if rows.size < 2:
        raise ValueError("Could not detect an ECG trace in the image.")
    pad = 3
    r0, r1 = max(0, rows.min() - pad), min(bw.shape[0], rows.max() + pad)
    c0, c1 = max(0, cols.min() - pad), min(bw.shape[1], cols.max() + pad)
    bw = bw[r0:r1, c0:c1]
    gray = gray[r0:r1, c0:c1]

    trace = _extract_trace(bw)                      # pixel amplitude per column
    trace = trace - np.median(trace)                # baseline at 0

    ppmm = _estimate_pixels_per_mm(gray)
    calibrated = False
    if ppmm and ppmm > 1:
        # Amplitude calibration to millivolts from the grid gain. Time is mapped
        # by resampling the full strip width to the target length (standard 10 s
        # strip assumption), which avoids crop/pad misalignment.
        mv_per_px = 1.0 / (gain_mm_mv * ppmm)       # (mm/px) / (mm/mV) -> mV/px
        sig = _resample(trace * mv_per_px, target_len)
        calibrated = True
    else:
        # Fallback: shape-only, per-instance standardisation.
        sig = _resample(trace, target_len)
        sig = (sig - sig.mean()) / (sig.std() + 1e-8)

    return DigitizationResult(
        signal=sig.astype(np.float32), fs=target_fs, calibrated=calibrated,
        pixels_per_mm=ppmm,
        info={"n_columns": int(bw.shape[1]), "image_shape": tuple(gray.shape)},
    )


def _fit_length(sig: np.ndarray, target_len: int) -> np.ndarray:
    """Center-crop or zero-pad a signal to exactly target_len samples."""
    if len(sig) == target_len:
        return sig
    if len(sig) > target_len:
        start = (len(sig) - target_len) // 2
        return sig[start:start + target_len]
    out = np.zeros(target_len, dtype=np.float32)
    start = (target_len - len(sig)) // 2
    out[start:start + len(sig)] = sig
    return out
