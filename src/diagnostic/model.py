"""1D ResNet for multi-label PTB-XL diagnosis (configurable lead count)."""
import logging

import torch
import torch.nn as nn

from src.diagnostic.config import N_CLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _conv(in_c, out_c, k=3, stride=1):
    return nn.Conv1d(in_c, out_c, kernel_size=k, stride=stride, padding=k // 2, bias=False)


class BasicBlock1d(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = _conv(in_c, out_c, 3, stride)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = _conv(out_c, out_c, 3, 1)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_c),
            )

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet1D(nn.Module):
    """ResNet-18-style 1D CNN producing multi-label logits."""

    def __init__(self, n_leads: int = 12, n_classes: int = N_CLASSES,
                 layers=(2, 2, 2, 2), channels=(64, 128, 256, 512), dropout: float = 0.3):
        super().__init__()
        self.n_leads = n_leads
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        in_c = channels[0]
        stages = []
        for stage_idx, (n_blocks, out_c) in enumerate(zip(layers, channels)):
            for block_idx in range(n_blocks):
                stride = 2 if (block_idx == 0 and stage_idx > 0) else 1
                stages.append(BasicBlock1d(in_c, out_c, stride))
                in_c = out_c
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], n_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        return self.head(x)            # raw logits; apply sigmoid for probabilities


def build_diag_model(n_leads: int = 12, device=None) -> ResNet1D:
    model = ResNet1D(n_leads=n_leads)
    if device is not None:
        model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Built ResNet1D ({n_leads}-lead) with {n_params:,} trainable parameters.")
    return model


def load_diag_model(checkpoint_path: str, device=None) -> ResNet1D:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    n_leads = ckpt.get("n_leads", 12)
    model = ResNet1D(n_leads=n_leads).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logging.info(f"Loaded diagnostic model ({n_leads}-lead) from {checkpoint_path}")
    return model
