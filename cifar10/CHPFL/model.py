"""CIFAR-10 CNN used by the MTGC paper (Fang et al., NeurIPS 2024).

Mirrors `client_model('cifar10')` from the official MTGC repo:
two 5x5 conv layers (64 ch) + maxpool + three FC layers (384, 192, 10).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Cifar10CNN(nn.Module):
    def __init__(self, n_class: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 5 * 5, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, n_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_flat_params(model: nn.Module) -> torch.Tensor:
    """Concatenate all `named_parameters` into a 1-D tensor (CPU, float32)."""
    return torch.cat([p.data.detach().reshape(-1) for _, p in model.named_parameters()]).cpu().float()


def set_flat_params(model: nn.Module, flat: torch.Tensor) -> None:
    """Inverse of `get_flat_params`."""
    idx = 0
    flat = flat.to(next(model.parameters()).device)
    for _, p in model.named_parameters():
        n = p.data.numel()
        p.data.copy_(flat[idx:idx + n].view_as(p.data))
        idx += n
