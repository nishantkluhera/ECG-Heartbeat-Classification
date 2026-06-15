"""Top published PTB-XL architectures for the diagnostic ensemble.

- resnet1d_wang : Wang et al. 2017 residual net (best CNN on PTB-XL superdiag, 0.930).
- inception1d   : InceptionTime (Fawaz et al. 2019).
- resnet1d      : the project's own ResNet-18-style 1D CNN (src.diagnostic.model).

A small factory (`build_arch`) returns any of them by name so the ensemble can
mix architectures for diversity.
"""
import torch
import torch.nn as nn

from src.diagnostic.config import N_CLASSES
from src.diagnostic.model import ResNet1D


# --------------------------------------------------------------------------- #
# ResNet1D (Wang 2017)
# --------------------------------------------------------------------------- #
class _WangBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, 8, padding="same", bias=False)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, 5, padding="same", bias=False)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.conv3 = nn.Conv1d(out_c, out_c, 3, padding="same", bias=False)
        self.bn3 = nn.BatchNorm1d(out_c)
        self.short = nn.Sequential(nn.Conv1d(in_c, out_c, 1, bias=False), nn.BatchNorm1d(out_c))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        return self.relu(y + self.short(x))


class ResNet1DWang(nn.Module):
    def __init__(self, n_leads=12, n_classes=N_CLASSES, filters=(64, 128, 128)):
        super().__init__()
        self.n_leads = n_leads
        blocks, in_c = [], n_leads
        for f in filters:
            blocks.append(_WangBlock(in_c, f))
            in_c = f
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(in_c, n_classes))

    def forward(self, x):
        return self.head(self.blocks(x))


# --------------------------------------------------------------------------- #
# InceptionTime (Fawaz 2019)
# --------------------------------------------------------------------------- #
class _Inception(nn.Module):
    def __init__(self, in_c, n_filters=32, kss=(39, 19, 9), bottleneck=32):
        super().__init__()
        self.use_bottleneck = in_c > 1
        b_out = bottleneck if self.use_bottleneck else in_c
        self.bottleneck = nn.Conv1d(in_c, bottleneck, 1, bias=False) if self.use_bottleneck else nn.Identity()
        self.convs = nn.ModuleList(
            [nn.Conv1d(b_out, n_filters, k, padding="same", bias=False) for k in kss])
        self.maxpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.maxconv = nn.Conv1d(in_c, n_filters, 1, bias=False)
        self.bn = nn.BatchNorm1d(n_filters * (len(kss) + 1))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b = self.bottleneck(x)
        outs = [conv(b) for conv in self.convs]
        outs.append(self.maxconv(self.maxpool(x)))
        return self.relu(self.bn(torch.cat(outs, dim=1)))


class InceptionTime1D(nn.Module):
    def __init__(self, n_leads=12, n_classes=N_CLASSES, n_filters=32, depth=6):
        super().__init__()
        self.n_leads = n_leads
        mod_out = n_filters * 4
        self.mods = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        in_c, res_in = n_leads, n_leads
        for d in range(depth):
            self.mods.append(_Inception(in_c, n_filters=n_filters))
            in_c = mod_out
            if d % 3 == 2:
                self.shortcuts.append(nn.Sequential(
                    nn.Conv1d(res_in, mod_out, 1, bias=False), nn.BatchNorm1d(mod_out)))
                res_in = mod_out
        self.relu = nn.ReLU(inplace=True)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(mod_out, n_classes))

    def forward(self, x):
        res, sc = x, 0
        for d, mod in enumerate(self.mods):
            x = mod(x)
            if d % 3 == 2:
                x = self.relu(x + self.shortcuts[sc](res))
                res, sc = x, sc + 1
        return self.head(x)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_arch(name: str, n_leads: int = 12, n_classes: int = N_CLASSES) -> nn.Module:
    name = name.lower()
    if name in ("resnet1d", "resnet"):
        return ResNet1D(n_leads=n_leads, n_classes=n_classes)
    if name in ("resnet1d_wang", "wang"):
        return ResNet1DWang(n_leads=n_leads, n_classes=n_classes)
    if name in ("inception1d", "inception"):
        return InceptionTime1D(n_leads=n_leads, n_classes=n_classes)
    raise ValueError(f"Unknown architecture '{name}'")
