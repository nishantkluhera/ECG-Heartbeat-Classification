"""Tests for the ECGCNN architecture and a single train step."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.model import ECGCNN, build_cnn_model, load_trained_model
from src.config import N_FEATURES, N_CLASSES


def test_forward_output_shape():
    model = ECGCNN()
    x = torch.randn(4, 1, N_FEATURES)
    out = model(x)
    assert out.shape == (4, N_CLASSES)


def test_adapts_to_custom_feature_length():
    model = ECGCNN(n_features=200, n_classes=3)
    out = model(torch.randn(2, 1, 200))
    assert out.shape == (2, 3)


def test_single_train_step_reduces_loss():
    torch.manual_seed(0)
    model = build_cnn_model()
    x = torch.randn(16, 1, N_FEATURES)
    y = torch.randint(0, N_CLASSES, (16,))
    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    first = criterion(model(x), y).item()
    for _ in range(10):  # overfit a tiny batch
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < first  # learning happened


def test_checkpoint_save_and_load(tmp_path):
    path = str(tmp_path / "model.pt")
    model = ECGCNN()
    torch.save(
        {"model_state_dict": model.state_dict(),
         "n_features": N_FEATURES, "n_classes": N_CLASSES},
        path,
    )
    loaded = load_trained_model(path, device=torch.device("cpu"))
    assert not loaded.training  # eval mode
    x = torch.randn(1, 1, N_FEATURES)
    with torch.no_grad():
        np.testing.assert_allclose(
            model.eval()(x).numpy(), loaded(x).numpy(), rtol=1e-4, atol=1e-5
        )
