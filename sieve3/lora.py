"""Minimal LoRA for the last blocks of the VFM (no external dependency).

Only low-rank updates of the frozen linear layers are trained, so stage A fits in hours on one GPU and the
pretrained visual knowledge is kept; the update starts at zero (B = 0), i.e. training starts from the VFM.
"""
from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scale = float(alpha) / float(rank)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    # some model code reads these attributes of its linear layers
    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scale


def add_lora(module: nn.Module, rank: int, alpha: float) -> int:
    """Wrap every nn.Linear inside `module` (attention projections and MLP). Returns the number wrapped."""
    targets = [(parent, name, child) for parent in module.modules() for name, child in parent.named_children()
               if isinstance(child, nn.Linear)]
    for parent, name, child in targets:
        setattr(parent, name, LoRALinear(child, rank, alpha))
    return len(targets)


def find_blocks(model: nn.Module):
    """The transformer blocks: the longest nn.ModuleList of the model (encoder.layer for DINOv2/DINOv3)."""
    lists = [m for m in model.modules() if isinstance(m, nn.ModuleList)]
    if not lists:
        raise ValueError("no transformer block list found in the VFM")
    return max(lists, key=len)
