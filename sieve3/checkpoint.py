"""Saving and loading CoupledNet checkpoints (the ViT is rebuilt from its saved config: no download needed)."""
from __future__ import annotations

import torch

from .config import from_dict, to_dict
from .model import CoupledNet
from .pipeline import Fusion


def vit_config_of(net):
    vit = getattr(net.backbone, "vit", None)
    return None if vit is None else vit.config.to_dict()


def save_checkpoint(path, net, cfg, **extra):
    torch.save({"model": net.state_dict(), "config": to_dict(cfg), "vit_config": vit_config_of(net), **extra}, path)


def load_model(path, device, eval_mode: bool = True):
    """Returns (Fusion, checkpoint dict)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = from_dict(ckpt["config"])
    net = CoupledNet(cfg.model, vit_config=ckpt.get("vit_config"))
    net.load_state_dict(ckpt["model"])
    net = net.to(device)
    if eval_mode:
        net.eval()
    return Fusion(net, cfg), ckpt
