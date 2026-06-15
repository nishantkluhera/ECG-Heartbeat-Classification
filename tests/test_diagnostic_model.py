"""Tests for the PTB-XL 1D-ResNet and Grad-CAM (no dataset required)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.diagnostic.model import ResNet1D, build_diag_model, load_diag_model
from src.diagnostic.config import N_CLASSES


@pytest.mark.parametrize("n_leads", [12, 1])
def test_forward_shape(n_leads):
    model = ResNet1D(n_leads=n_leads)
    out = model(torch.randn(3, n_leads, 1000))
    assert out.shape == (3, N_CLASSES)


def test_single_train_step_reduces_loss():
    torch.manual_seed(0)
    model = build_diag_model(n_leads=1)
    x = torch.randn(16, 1, 1000)
    y = (torch.rand(16, N_CLASSES) > 0.5).float()
    crit = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = crit(model(x), y).item()
    for _ in range(8):
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < first


def test_checkpoint_roundtrip(tmp_path):
    path = str(tmp_path / "diag.pt")
    model = ResNet1D(n_leads=1)
    torch.save({"model_state_dict": model.state_dict(), "n_leads": 1,
                "lead_config": "lead2", "temperature": 1.5,
                "lead_means": [0.0], "lead_stds": [1.0]}, path)
    loaded = load_diag_model(path, device=torch.device("cpu"))
    assert not loaded.training
    x = torch.randn(1, 1, 1000)
    with torch.no_grad():
        np.testing.assert_allclose(model.eval()(x).numpy(), loaded(x).numpy(),
                                   rtol=1e-4, atol=1e-5)


def test_gradcam_shape():
    from src.diagnose import _gradcam
    model = ResNet1D(n_leads=1).eval()
    x = torch.randn(1, 1, 1000, requires_grad=True)
    cam = _gradcam(model, x, class_idx=0)
    assert cam.shape == (1000,)
    assert cam.min() >= 0.0 and cam.max() <= 1.0 + 1e-6
