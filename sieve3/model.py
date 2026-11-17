"""CoupledNet: a VFM that reads RGB and LR-HSI jointly and outputs couplings -- never spectra.

Per pixel, all outputs are relational:
  Pi    [B,K,H,W]  material slots (what, how much)                      -> material laws solved from H
  Gamma [B,M,H,W]  part codes over a global codebook: "the same point of  -> part detail solved from H,
                   the same kind of thing", wherever it is in the scene      measured through phase diversity
  beta  [B,1,H,W]  does this pixel couple to its copies at all?           -> beta = 0 falls back to materials
  g     [B,1,H,W]  gate of the observable correction (inherited from SIEVE-2)

How the VFM reads the HSI (adapter = "xattn"): in front of the LoRA-tuned ViT blocks, every patch token attends
to LR locations with a bias -w_h * ||z_q - z_k||^2, where z are per-image PCA tokens of the LR-HSI computed on
fitting cells only. It receives the visual features of its *spectral twins*. Only distances of z enter, so the
adapter is band-agnostic and invariant to the arbitrary per-image PCA basis; the output projection starts at
zero, so training starts from the pretrained VFM. HSI enters as relations, never as values to copy.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .lora import add_lora, find_blocks
from .ops import key_grid, sincos_2d, vfm_normalize


class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=3, stride=1, dilation=1, groups=1, act=True):
        pad = dilation * (k - 1) // 2
        layers = [nn.Conv2d(cin, cout, k, stride, pad, dilation=dilation, groups=groups, bias=False),
                  nn.BatchNorm2d(cout)]
        if act:
            layers.append(nn.GELU())
        super().__init__(*layers)


class InvertedResidual(nn.Module):
    def __init__(self, channels, expand=2, dilation=1):
        super().__init__()
        hidden = channels * expand
        self.body = nn.Sequential(ConvBNAct(channels, hidden, 1),
                                  ConvBNAct(hidden, hidden, 3, dilation=dilation, groups=hidden),
                                  ConvBNAct(hidden, channels, 1, act=False))

    def forward(self, x):
        return x + self.body(x)


# ----------------------------------------------------------------------------- how the VFM reads the HSI
class SpectralTwinAttention(nn.Module):
    """Tokens attend to LR locations, biased towards spectrally similar ones (distances of z only)."""

    def __init__(self, dim: int, heads: int = 4, pe_dim: int = 32, max_keys: int = 256):
        super().__init__()
        if dim % heads:
            raise ValueError(f"adapter width {dim} is not divisible by {heads} heads")
        self.heads, self.pe_dim, self.max_keys = heads, pe_dim, max_keys
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.pe_q = nn.Linear(pe_dim, dim, bias=False)
        self.pe_k = nn.Linear(pe_dim, dim, bias=False)
        self.pe_v = nn.Linear(pe_dim, dim, bias=False)
        self.spec_w = nn.Parameter(torch.zeros(heads))          # softplus(0) = 0.69 per head
        self.out = nn.Linear(dim, dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, tokens: torch.Tensor, hw, z: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
        """tokens [B,N,D] on an hq x wq grid (row-major); z [B,dz,h,w] spectral tokens (fitting cells only)."""
        B, N, D = tokens.shape
        hq, wq = hw
        hk, wk = key_grid(z.shape[-2], z.shape[-1], self.max_keys)
        z = z.float()
        zq = F.interpolate(z, size=(hq, wq), mode="bilinear", align_corners=False).flatten(2).transpose(1, 2)
        zk = F.adaptive_avg_pool2d(z, (hk, wk)).flatten(2).transpose(1, 2)            # [B,Nk,dz]
        grid = tokens.transpose(1, 2).reshape(B, D, hq, wq)
        tk = F.adaptive_avg_pool2d(grid, (hk, wk)).flatten(2).transpose(1, 2)        # visual features at keys
        pq = sincos_2d(hq, wq, self.pe_dim, tokens.device)
        pk = sincos_2d(hk, wk, self.pe_dim, tokens.device)
        dh = D // self.heads
        kn = self.norm_k(tk)
        k = (self.k(kn) + self.pe_k(pk)).view(B, -1, self.heads, dh).transpose(1, 2)    # [B,h,Nk,dh]
        v = (self.v(kn) + self.pe_v(pk)).view(B, -1, self.heads, dh).transpose(1, 2)
        wspec = F.softplus(self.spec_w).view(1, -1, 1, 1)
        zk2 = (zk ** 2).sum(-1).unsqueeze(1)                                          # [B,1,Nk]
        outs = []
        for i in range(0, N, chunk):
            x = self.norm_q(tokens[:, i:i + chunk])
            q = (self.q(x) + self.pe_q(pq[i:i + chunk])).view(B, -1, self.heads, dh).transpose(1, 2)
            with torch.autocast(device_type=tokens.device.type, enabled=False):     # distances in fp32
                zc = zq[:, i:i + chunk]
                dist = ((zc ** 2).sum(-1, keepdim=True) - 2.0 * zc @ zk.transpose(1, 2) + zk2).clamp_min(0.0)
            logits = (q @ k.transpose(-1, -2)).float() / math.sqrt(dh) - wspec * dist.unsqueeze(1)
            att = logits.softmax(-1).to(v.dtype)
            outs.append((att @ v).transpose(1, 2).reshape(B, -1, D))
        return tokens + self.out(torch.cat(outs, 1)).to(tokens.dtype)


class SpectralContrast(nn.Module):
    """Edge variant: local spectral contrast (distances of z to its 8 neighbours and to the image mean),
    invariant to the PCA basis, injected by a zero-initialised 1x1 projection."""

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(9, dim, 1), nn.GELU(), nn.Conv2d(dim, dim, 1))
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, f: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h, w = f.shape[-2:]
        zf = F.interpolate(z.float(), size=(h, w), mode="bilinear", align_corners=False)
        pad = F.pad(zf, (1, 1, 1, 1), mode="replicate")
        feats = [((zf - pad[..., dy:dy + h, dx:dx + w]) ** 2).sum(1, keepdim=True)
                 for dy in range(3) for dx in range(3) if (dy, dx) != (1, 1)]
        feats.append(((zf - zf.mean((2, 3), keepdim=True)) ** 2).sum(1, keepdim=True))
        return f + self.proj(torch.log1p(torch.cat(feats, 1)).to(f.dtype))


# ----------------------------------------------------------------------------- backbones
class TinyBackbone(nn.Module):
    """Edge backbone (~0.7M parameters): HR stem, stride-4 trunk with dilations, stride-8 context, no ViT."""
    hr_dim = 24

    def __init__(self, mcfg, width4=96, width8=128):
        super().__init__()
        self.stem = nn.Sequential(ConvBNAct(3, 24), ConvBNAct(24, 24))
        self.down = nn.Sequential(ConvBNAct(24, 48, stride=2), ConvBNAct(48, width4, stride=2))
        self.trunk4 = nn.Sequential(*[InvertedResidual(width4, 2, d) for d in (1, 2, 4, 8)])
        if mcfg.adapter == "xattn":
            self.adapter = SpectralTwinAttention(width4, mcfg.adapter_heads, mcfg.adapter_pe, mcfg.adapter_keys)
        elif mcfg.adapter == "conv":
            self.adapter = SpectralContrast(width4)
        elif mcfg.adapter == "none":
            self.adapter = None
        else:
            raise ValueError(f"unknown adapter {mcfg.adapter}")
        self.down8 = ConvBNAct(width4, width8, stride=2)
        self.trunk8 = nn.Sequential(*[InvertedResidual(width8, 2, d) for d in (1, 2, 4)])
        self.fuse = ConvBNAct(width4 + width8, mcfg.lr_dim, 1)

    def forward(self, rgb, rstd, z):
        hr = self.stem(rstd)
        f4 = self.trunk4(self.down(hr))
        if isinstance(self.adapter, SpectralTwinAttention):
            B, D, h4, w4 = f4.shape
            tokens = self.adapter(f4.flatten(2).transpose(1, 2), (h4, w4), z)
            f4 = tokens.transpose(1, 2).reshape(B, D, h4, w4)
        elif self.adapter is not None:
            f4 = self.adapter(f4, z)
        f8 = self.trunk8(self.down8(f4))
        f8 = F.interpolate(f8, size=f4.shape[-2:], mode="bilinear", align_corners=False)
        return hr, self.fuse(torch.cat([f4, f8], 1))


def build_vit(path: str, vit_config: dict | None = None):
    from transformers import AutoConfig, AutoModel  # optional dependency
    if vit_config is not None:
        try:
            conf = dict(vit_config)
            return AutoModel.from_config(AutoConfig.for_model(conf.pop("model_type"), **conf))
        except Exception as error:  # noqa: BLE001 - fall back to the pretrained weights
            if not path:
                raise
            print(f"warning: rebuilding the ViT from its saved config failed ({error}); loading {path}")
    return AutoModel.from_pretrained(path, torch_dtype=torch.float32)


class HFBackbone(nn.Module):
    """DINOv2/DINOv3 ViT: frozen, LoRA on the last blocks, spectral-twin attention injected in front of them."""
    hr_dim = 24

    def __init__(self, mcfg, vit_config=None):
        super().__init__()
        if not mcfg.hf_path and vit_config is None:
            raise ValueError("model.hf_path is required for the hf backbone")
        self.vit = build_vit(mcfg.hf_path, vit_config)
        config = self.vit.config
        self.ps = int(getattr(config, "patch_size", 14))
        dim = int(config.hidden_size)
        self.input_scale = float(mcfg.hf_input_scale)
        self.max_side = int(mcfg.hf_max_side)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        for p in self.vit.parameters():
            p.requires_grad_(False)
        blocks = list(find_blocks(self.vit))
        n_tuned = max(0, min(int(mcfg.lora_blocks), len(blocks)))
        tuned = blocks[len(blocks) - n_tuned:] if n_tuned else []
        for block in tuned:
            if mcfg.lora_rank > 0:
                add_lora(block, mcfg.lora_rank, mcfg.lora_alpha)
            else:
                for p in block.parameters():
                    p.requires_grad_(True)
        self.adapter = None
        self.post_adapter = None
        self._ctx = None
        if mcfg.adapter == "xattn":
            self.adapter = SpectralTwinAttention(dim, mcfg.adapter_heads, mcfg.adapter_pe, mcfg.adapter_keys)
            target = tuned[0] if tuned else blocks[-1]
            target.register_forward_pre_hook(self._inject, with_kwargs=True)
        elif mcfg.adapter == "conv":
            self.post_adapter = SpectralContrast(mcfg.lr_dim)
        elif mcfg.adapter != "none":
            raise ValueError(f"unknown adapter {mcfg.adapter}")
        self.stem = nn.Sequential(ConvBNAct(3, 24), ConvBNAct(24, 24))
        self.proj = nn.Sequential(nn.Conv2d(dim, mcfg.lr_dim, 1), nn.GELU(),
                                  nn.Conv2d(mcfg.lr_dim, mcfg.lr_dim, 3, padding=1))

    def _inject(self, module, args, kwargs):
        """Forward pre-hook of the first adapted block: add the spectral-twin attention to the patch tokens."""
        if self._ctx is None:
            return None
        z, hp, wp = self._ctx
        hidden = args[0] if args else kwargs["hidden_states"]
        n = hp * wp
        patches = self.adapter(hidden[:, -n:], (hp, wp), z)
        hidden = torch.cat([hidden[:, :hidden.shape[1] - n], patches.to(hidden.dtype)], 1)
        if args:
            return (hidden,) + tuple(args[1:]), kwargs
        kwargs = dict(kwargs)
        kwargs["hidden_states"] = hidden
        return args, kwargs

    def forward(self, rgb, rstd, z):
        hr = self.stem(rstd)
        H, W = rgb.shape[-2:]
        scale = min(self.input_scale, self.max_side / max(H, W))
        Hi = max(self.ps, int(round(H * scale / self.ps)) * self.ps)
        Wi = max(self.ps, int(round(W * scale / self.ps)) * self.ps)
        x = F.interpolate(vfm_normalize(rgb.float()), size=(Hi, Wi), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        hp, wp = Hi // self.ps, Wi // self.ps
        self._ctx = (z, hp, wp) if self.adapter is not None else None
        try:
            tokens = self.vit(pixel_values=x).last_hidden_state
        finally:
            self._ctx = None
        grid = tokens[:, -hp * wp:].transpose(1, 2).reshape(x.shape[0], -1, hp, wp)
        f = self.proj(grid)
        if self.post_adapter is not None:
            f = self.post_adapter(f, z)
        return hr, f


def _sqdist(proto: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """||p_k - z_n||^2 for proto [B,K,d], tokens [B,N,d] -> [B,K,N]."""
    return ((proto ** 2).sum(-1).unsqueeze(-1) - 2.0 * proto @ tokens.transpose(1, 2)
            + (tokens ** 2).sum(-1).unsqueeze(1))


# ----------------------------------------------------------------------------- the network
class CoupledNet(nn.Module):
    def __init__(self, mcfg, vit_config=None):
        super().__init__()
        self.cfg = mcfg
        if mcfg.backbone == "tiny":
            self.backbone = TinyBackbone(mcfg)
        elif mcfg.backbone == "hf":
            self.backbone = HFBackbone(mcfg, vit_config)
        else:
            raise ValueError(f"unknown backbone {mcfg.backbone}")
        if mcfg.part_source not in ("vfm", "color"):
            raise ValueError(f"unknown part_source {mcfg.part_source}")
        d, dl, dt, K = mcfg.embed_dim, mcfg.lr_dim, mcfg.slot_dim, mcfg.slots
        self.hr_trunk = nn.Sequential(ConvBNAct(dl + self.backbone.hr_dim + 3, 48), ConvBNAct(48, 48))
        self.mat_head = nn.Conv2d(48, d, 1)
        # part codes: the VFM's answer to "where else in this scene is the same point measured?"
        self.use_parts = mcfg.parts > 0 and mcfg.part_source == "vfm"
        if self.use_parts:
            self.part_head = nn.Sequential(ConvBNAct(48, 48), nn.Conv2d(48, mcfg.part_dim, 1))
            self.codebook = nn.Parameter(F.normalize(torch.randn(mcfg.parts, mcfg.part_dim), dim=1))
            self.log_tau_part = nn.Parameter(torch.tensor(math.log(0.1)))
        # coupling gate: the VFM decides *whether* a pixel couples; the measurements decide the values
        self.use_beta = mcfg.parts > 0 and mcfg.coupling_gate
        if self.use_beta:
            self.beta_head = nn.Sequential(ConvBNAct(48, 16), nn.Conv2d(16, 1, 1))
            nn.init.zeros_(self.beta_head[-1].weight)
            nn.init.constant_(self.beta_head[-1].bias, 2.0)        # start coupled (beta ~ 0.88)
        # material slots (as SIEVE-2): RGB tokens + LR spectra of fitting cells
        self.tok_norm = nn.LayerNorm(dl)
        self.to_k = nn.Linear(dl, dt, bias=False)
        self.to_v = nn.Linear(dl, dt, bias=False)
        self.to_q = nn.Linear(dt, dt, bias=False)
        self.slot_init = nn.Parameter(torch.randn(K, dt) * dt ** -0.5)
        self.slot_norm = nn.LayerNorm(dt)
        self.mlp_norm = nn.LayerNorm(dt)
        self.gru = nn.GRUCell(dt, dt)
        self.mlp = nn.Sequential(nn.Linear(dt, 2 * dt), nn.GELU(), nn.Linear(2 * dt, dt))
        self.key_out = nn.Linear(dt, d)
        self.log_tau = nn.Parameter(torch.tensor(math.log(0.1)))
        self.log_lam = nn.Parameter(torch.tensor(math.log(0.1)))
        # observable-correction gate (SIEVE-2) with the measured map as an extra input
        self.gate_head = nn.Sequential(ConvBNAct(d + 5, 32), ConvBNAct(32, 32), nn.Conv2d(32, 1, 1))
        nn.init.zeros_(self.gate_head[-1].weight)
        nn.init.constant_(self.gate_head[-1].bias, 1.0)

    # -- dense features: RGB + spectral relations (z: fitting cells only)
    def pixels(self, rgb, rstd, z, lr_size):
        hr, f = self.backbone(rgb, rstd, z)
        if tuple(f.shape[-2:]) != tuple(lr_size):
            f = F.interpolate(f, size=lr_size, mode="bilinear", align_corners=False)
        f_up = F.interpolate(f, size=rstd.shape[-2:], mode="bilinear", align_corners=False)
        t = self.hr_trunk(torch.cat([f_up, hr.to(f_up.dtype), rstd.to(f_up.dtype)], 1))
        e = F.normalize(self.mat_head(t).float(), dim=1)
        u = F.normalize(self.part_head(t).float(), dim=1) if self.use_parts else None
        beta_logit = self.beta_head(t).float() if self.use_beta else None
        return {"e": e, "f": f, "u": u, "beta_logit": beta_logit}

    # -- part codes over the global codebook
    def part_codes(self, u):
        tau = self.log_tau_part.exp().clamp(0.02, 1.0)
        return (torch.einsum("bdhw,md->bmhw", u, F.normalize(self.codebook, dim=1)) / tau).softmax(1)

    # -- material slots
    def slot_attention(self, f, z, fit, valid):
        B, _, h, w = f.shape
        dt = self.slot_init.shape[1]
        tokens = self.tok_norm(f.flatten(2).transpose(1, 2).float())
        k, v = self.to_k(tokens), self.to_v(tokens)
        zt = z.flatten(2).transpose(1, 2).float()
        fit_w = fit.float().flatten(2)
        valid_w = valid.float().flatten(2)
        slots = self.slot_init.unsqueeze(0).expand(B, -1, -1)
        lam = self.log_lam.exp()
        proto = None
        for _ in range(self.cfg.slot_iters):
            q = self.to_q(self.slot_norm(slots))
            logits = torch.einsum("bkd,bnd->bkn", q, k) / math.sqrt(dt)
            if proto is not None:
                logits = logits - lam * _sqdist(proto, zt)
            attn = logits.float().softmax(1) * valid_w
            update = torch.einsum("bkn,bnd->bkd", attn / (attn.sum(-1, keepdim=True) + 1e-6), v.float())
            slots = self.gru(update.reshape(-1, dt), slots.reshape(-1, dt)).view(B, -1, dt)
            slots = slots + self.mlp(self.mlp_norm(slots))
            weight = attn * fit_w
            proto = torch.einsum("bkn,bnd->bkd", weight, zt) / (weight.sum(-1, keepdim=True) + 1e-6)
        return F.normalize(self.key_out(slots).float(), dim=-1), proto

    def assign(self, e, keys, proto, z_hr):
        tau = self.log_tau.exp().clamp(0.02, 1.0)
        lam = self.log_lam.exp()
        logits = torch.einsum("bdhw,bkd->bkhw", e.float(), keys) / tau
        zz = (z_hr ** 2).sum(1, keepdim=True)
        zp = torch.einsum("bdhw,bkd->bkhw", z_hr.float(), proto)
        pp = (proto ** 2).sum(-1)[:, :, None, None]
        return (logits - lam * (zz - 2.0 * zp + pp)).softmax(1)

    # -- gate of the observable correction: amortised evidence + material confidence + measured map
    def gate(self, e, Pi, evidence, mu):
        K = Pi.shape[1]
        pmax = Pi.max(1, keepdim=True).values
        entropy = -(Pi * Pi.clamp_min(1e-8).log()).sum(1, keepdim=True) / math.log(max(K, 2))
        x = torch.cat([e.float(), pmax.float(), entropy.float(), evidence.float(), mu.float()], 1)
        return torch.sigmoid(self.gate_head(x).float())
