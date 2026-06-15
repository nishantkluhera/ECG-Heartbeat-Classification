"""Train-time ECG signal augmentation.

Operates on standardised (per-lead z-scored) signals of shape (C, L), so the
default magnitudes are relative to ~unit standard deviation. Augmentation is
applied only during training; validation/test use clean signals (with optional
test-time augmentation handled separately in the ensemble).
"""
import numpy as np


class ECGAugment:
    def __init__(self, p: float = 0.5, noise_std: float = 0.06, max_shift: int = 50,
                 scale_range=(0.85, 1.15), wander_amp: float = 0.10,
                 lead_mask_p: float = 0.10):
        self.p = p
        self.noise_std = noise_std
        self.max_shift = max_shift
        self.scale_range = scale_range
        self.wander_amp = wander_amp
        self.lead_mask_p = lead_mask_p

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """x: (C, L) float32 -> augmented (C, L) float32."""
        x = np.array(x, dtype=np.float32, copy=True)
        C, L = x.shape

        # Circular time shift (preserves the full waveform).
        if np.random.rand() < self.p and self.max_shift > 0:
            x = np.roll(x, np.random.randint(-self.max_shift, self.max_shift + 1), axis=1)

        # Additive Gaussian noise.
        if np.random.rand() < self.p:
            x = x + np.random.normal(0.0, self.noise_std, size=x.shape).astype(np.float32)

        # Per-record amplitude scaling.
        if np.random.rand() < self.p:
            x = x * np.float32(np.random.uniform(*self.scale_range))

        # Low-frequency baseline wander.
        if np.random.rand() < self.p:
            t = np.linspace(0, 2 * np.pi, L, dtype=np.float32)
            freq = np.random.uniform(0.5, 2.0)
            phase = np.random.uniform(0, 2 * np.pi)
            x = x + (self.wander_amp * np.sin(freq * t + phase)).astype(np.float32)

        # Random lead dropout (only meaningful for multi-lead input).
        if C > 1 and np.random.rand() < self.lead_mask_p:
            x[np.random.randint(0, C)] = 0.0

        return x.astype(np.float32)


def tta_views(x: np.ndarray, shifts=(0, -20, 20, -40, 40)):
    """Generate test-time-augmentation views of a (C, L) signal via small shifts."""
    return [np.roll(x, s, axis=1).astype(np.float32) for s in shifts]
