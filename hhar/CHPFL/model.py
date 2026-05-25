"""HARNet - small 1D-CNN + GRU + FC for HAR / IoT sensor time-series.

Designed for ~150k parameters (vs ~798k for Cifar10CNN). The architecture is
shared between HHAR (in_ch=6, num_classes=6) and PAMAP2 (in_ch=~52,
num_classes=12) -- only in_ch and num_classes vary per dataset.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HARNet(nn.Module):
    """1D-CNN feature extractor + GRU temporal model + linear classifier."""

    def __init__(self, in_ch: int = 6, num_classes: int = 6, hidden: int = 192):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        self.gru = nn.GRU(
            input_size=64, hidden_size=hidden, num_layers=1, batch_first=True,
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_ch, T)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)                              # (B, 64, T/2)
        x = x.permute(0, 2, 1)                         # (B, T/2, 64)
        _, h = self.gru(x)                             # h: (1, B, hidden)
        h = h.squeeze(0)                                # (B, hidden)
        h = self.dropout(h)
        return self.fc(h)


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.detach().reshape(-1) for _, p in model.named_parameters()]).cpu().float()


def set_flat_params(model: nn.Module, flat: torch.Tensor) -> None:
    flat = flat.to(next(model.parameters()).device)
    idx = 0
    for _, p in model.named_parameters():
        n = p.data.numel()
        p.data.copy_(flat[idx:idx + n].view_as(p.data))
        idx += n
