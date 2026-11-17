"""Large RGB foundation model (VFM-L) used only offline: dense features, PCA, cache.

Its cached features are the prior of stage A (relational distillation of materials, cannot-link of distant
part couplings).

Works with any HF vision transformer whose last_hidden_state ends with the patch tokens
(DINOv2, DINOv2-with-registers, DINOv3 ViT). The teacher never runs at inference.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .ops import vfm_normalize


class Teacher:
    def __init__(self, path: str, device, dtype=torch.float16):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(path, torch_dtype=dtype).to(device).eval()
        self.ps = int(getattr(self.model.config, "patch_size", 14))
        self.dim = int(getattr(self.model.config, "hidden_size"))
        self.device = device
        self.dtype = dtype
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def _forward(self, x):
        x = ((x - self.mean) / self.std).to(self.dtype)
        tokens = self.model(pixel_values=x).last_hidden_state
        hp, wp = x.shape[-2] // self.ps, x.shape[-1] // self.ps
        return tokens[0, -hp * wp:].float().T.reshape(-1, hp, wp)

    @torch.inference_mode()
    def features(self, rgb: torch.Tensor, long_side: int = 518, tile: int = 896, overlap: int = 112) -> torch.Tensor:
        """rgb [3,H,W] (any positive scale). Returns [D, hp, wp] covering the whole image.

        Images are resized so that the long side is about `long_side` pixels (upsampling small images
        gives denser tokens). If the resized image exceeds `tile`, it is processed in overlapping tiles.
        """
        x = vfm_normalize(rgb[None].float().to(self.device))
        H, W = x.shape[-2:]
        scale = long_side / max(H, W)
        Hi = max(self.ps, int(round(H * scale / self.ps)) * self.ps)
        Wi = max(self.ps, int(round(W * scale / self.ps)) * self.ps)
        x = F.interpolate(x, size=(Hi, Wi), mode="bicubic", align_corners=False, antialias=scale < 1).clamp(0, 1)
        if Hi <= tile and Wi <= tile:
            return self._forward(x)
        tile -= tile % self.ps
        overlap -= overlap % self.ps
        stride = tile - overlap
        hp, wp = Hi // self.ps, Wi // self.ps
        acc = torch.zeros(self.dim, hp, wp, device=self.device)
        cnt = torch.zeros(1, hp, wp, device=self.device)
        ys = list(range(0, max(Hi - tile, 0) + 1, stride)) or [0]
        xs = list(range(0, max(Wi - tile, 0) + 1, stride)) or [0]
        if ys[-1] + tile < Hi:
            ys.append(Hi - tile)
        if xs[-1] + tile < Wi:
            xs.append(Wi - tile)
        for y in ys:
            for xx in xs:
                patch = x[..., y:y + min(tile, Hi), xx:xx + min(tile, Wi)]
                f = self._forward(patch)
                py, px = y // self.ps, xx // self.ps
                acc[:, py:py + f.shape[1], px:px + f.shape[2]] += f
                cnt[:, py:py + f.shape[1], px:px + f.shape[2]] += 1
        return acc / cnt.clamp_min(1)


def fit_pca(samples: list, dims: int, max_tokens: int = 300_000, seed: int = 0):
    """samples: list of [D,hp,wp] tensors (CPU). Returns mean [D], basis [D,dims], explained fraction."""
    generator = torch.Generator().manual_seed(seed)
    per = max(1, math.ceil(max_tokens / max(len(samples), 1)))
    picks = []
    for f in samples:
        t = f.flatten(1).T.float()
        if t.shape[0] > per:
            t = t[torch.randperm(t.shape[0], generator=generator)[:per]]
        picks.append(t)
    X = torch.cat(picks).double()
    mean = X.mean(0)
    cov = (X - mean).T @ (X - mean) / max(X.shape[0] - 1, 1)
    evals, evecs = torch.linalg.eigh(cov)
    basis = evecs[:, -dims:].flip(-1)
    explained = float(evals[-dims:].sum() / evals.sum().clamp_min(1e-30))
    return mean.float(), basis.float(), explained


def project(f: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    D, hp, wp = f.shape
    return ((f.flatten(1).T - mean) @ basis).T.reshape(-1, hp, wp)
