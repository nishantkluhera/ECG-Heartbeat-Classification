"""End-to-end inference: ECG image (or signal) -> calibrated diagnostic output.

Pipeline:  image -> digitize -> standardise -> diagnostic model
           -> temperature-calibrated probabilities -> findings + Grad-CAM.

Every result carries an explicit medical disclaimer. This is an educational /
research tool, NOT a medical device and NOT a diagnosis.
"""
import logging
import os

import numpy as np
import torch

from src.diagnostic.config import (
    SUPERCLASSES, SUPERCLASS_FULL, SIGNAL_LENGTH, SAMPLING_RATE, model_path,
)
from src.diagnostic.model import load_diag_model
from src.digitization import digitize_ecg_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DISCLAIMER = (
    "EDUCATIONAL / RESEARCH USE ONLY. This tool is not a medical device, does "
    "not provide a diagnosis, and may be wrong. ECG interpretation must be done "
    "by a qualified clinician. If you have any health concern, seek professional "
    "medical care."
)

_cache = {}          # lead_config -> (model, checkpoint)
_device = None


def _get_device():
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def _load(lead_config: str):
    if lead_config in _cache:
        return _cache[lead_config]
    path = model_path(lead_config)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No diagnostic model at {path}. Train it first: "
            f"python main.py diagnose-train --leads {lead_config}"
        )
    device = _get_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = load_diag_model(path, device=device)
    _cache[lead_config] = (model, ckpt)
    return _cache[lead_config]


def _standardise(signal_leads: np.ndarray, ckpt: dict, calibrated: bool) -> np.ndarray:
    """Map a (n_leads, L) signal into the model's standardised input space.

    Calibrated (mV) signals use the training per-lead mean/std. Uncalibrated
    signals are already per-instance z-scored (~unit std), which already lives
    in the standardised space, so they are passed through.
    """
    x = np.asarray(signal_leads, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    if calibrated:
        mean = np.asarray(ckpt["lead_means"], dtype=np.float32)[:, None]
        std = np.asarray(ckpt["lead_stds"], dtype=np.float32)[:, None]
        x = (x - mean) / std
    return x[None]      # (1, n_leads, L)


def _gradcam(model, x: torch.Tensor, class_idx: int) -> np.ndarray:
    """1-D Grad-CAM over the last residual stage; returns a length-L heatmap in [0,1]."""
    activations, gradients = {}, {}

    def fwd_hook(_m, _i, out):
        activations["v"] = out.detach()

    def bwd_hook(_m, _gi, go):
        gradients["v"] = go[0].detach()

    target = model.stages[-1]
    h1 = target.register_forward_hook(fwd_hook)
    h2 = target.register_full_backward_hook(bwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        logits[0, class_idx].backward()
        act = activations["v"][0]                       # (C, L')
        grad = gradients["v"][0]                         # (C, L')
        weights = grad.mean(dim=1, keepdim=True)         # (C, 1)
        cam = torch.relu((weights * act).sum(dim=0))     # (L',)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = torch.nn.functional.interpolate(
            cam[None, None], size=x.shape[-1], mode="linear", align_corners=False
        )[0, 0]
        return cam.cpu().numpy()
    finally:
        h1.remove()
        h2.remove()


@torch.no_grad()
def _forward_probs(model, x_t, temperature):
    logits = model(x_t) / max(temperature, 1e-2)
    return torch.sigmoid(logits)[0].cpu().numpy()


def diagnose_signal(signal_leads: np.ndarray, lead_config: str = "lead2",
                    calibrated: bool = True, threshold: float = 0.5, with_saliency: bool = True):
    """Diagnose a standardisable (n_leads, L) ECG signal. Returns a result dict."""
    model, ckpt = _load(lead_config)
    device = _get_device()
    temperature = ckpt.get("temperature", 1.0)

    x = _standardise(signal_leads, ckpt, calibrated)
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    probs = _forward_probs(model, x_t, temperature)

    findings = [
        {"class": SUPERCLASSES[i], "name": SUPERCLASS_FULL[SUPERCLASSES[i]],
         "probability": float(probs[i]), "flagged": bool(probs[i] >= threshold)}
        for i in range(len(SUPERCLASSES))
    ]
    findings.sort(key=lambda d: d["probability"], reverse=True)

    saliency = None
    if with_saliency:
        top_idx = int(np.argmax(probs))
        x_grad = torch.as_tensor(x, dtype=torch.float32, device=device).requires_grad_(True)
        saliency = _gradcam(model, x_grad, top_idx)

    return {
        "findings": findings,
        "top": findings[0],
        "calibrated": calibrated,
        "lead_config": lead_config,
        "saliency": saliency,
        "disclaimer": DISCLAIMER,
    }


def diagnose_image(image, lead_config: str = "lead2", threshold: float = 0.5,
                   with_saliency: bool = True):
    """Digitize a single-lead ECG strip image and diagnose it."""
    dig = digitize_ecg_image(image, target_len=SIGNAL_LENGTH, target_fs=SAMPLING_RATE)
    result = diagnose_signal(
        dig.signal[None, :], lead_config=lead_config, calibrated=dig.calibrated,
        threshold=threshold, with_saliency=with_saliency,
    )
    result["digitization"] = {
        "calibrated": dig.calibrated,
        "pixels_per_mm": dig.pixels_per_mm,
        "signal": dig.signal,
        "fs": dig.fs,
    }
    if not dig.calibrated:
        result["warning"] = ("Grid calibration failed - amplitudes are approximate, "
                             "so confidence is reduced. Use a clearer, flatter image.")
    return result
