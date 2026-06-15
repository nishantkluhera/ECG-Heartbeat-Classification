"""Forward-pass tests for the ensemble architectures + augmentation."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.diagnostic.architectures import build_arch, ResNet1DWang, InceptionTime1D
from src.diagnostic.config import N_CLASSES
from src.diagnostic.augment import ECGAugment, tta_views


@pytest.mark.parametrize("name", ["resnet1d", "resnet1d_wang", "inception1d"])
@pytest.mark.parametrize("n_leads", [12, 1])
def test_arch_forward(name, n_leads):
    model = build_arch(name, n_leads=n_leads)
    out = model(torch.randn(2, n_leads, 1000))
    assert out.shape == (2, N_CLASSES)


def test_arch_trains_one_step():
    model = build_arch("resnet1d_wang", n_leads=12)
    x = torch.randn(8, 12, 1000)
    y = (torch.rand(8, N_CLASSES) > 0.5).float()
    crit = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = crit(model(x), y).item()
    for _ in range(6):
        opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
    assert loss.item() < first


def test_augment_shape_and_determinism():
    x = np.random.randn(12, 1000).astype(np.float32)
    aug = ECGAugment()(x)
    assert aug.shape == (12, 1000) and aug.dtype == np.float32
    # original not mutated
    assert not np.shares_memory(aug, x)
    views = tta_views(x)
    assert len(views) == 5 and all(v.shape == (12, 1000) for v in views)
