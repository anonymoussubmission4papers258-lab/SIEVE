"""Export the edge student's dense branch to ONNX (TensorRT / ONNX Runtime on the device).

Graph: (rgb [1,3,H,W], z [1,dz,H/s,W/s]) -> (e, f, u, beta_logit). z are the spectral tokens of the LR-HSI
(sieve3.spectral.spectral_tokens: a small per-image PCA, run in PyTorch/NumPy on the device). The remaining
steps -- slots, part-code softmax, the closed-form coupled solve (fp32/fp64, never quantised), gates and the
audit -- are small and run in PyTorch on the device.
  python scripts/export_onnx.py --ckpt runs/B_s0/model.pt --size 512 --out results/dense_branch.onnx
Use --static (default) for the "xattn" adapter (adaptive pooling of the keys needs fixed shapes).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sieve3.checkpoint import load_model  # noqa: E402
from sieve3.ops import standardize_rgb  # noqa: E402


class DenseBranch(nn.Module):
    def __init__(self, net, scale):
        super().__init__()
        self.net = net
        self.scale = scale

    def forward(self, rgb, z):
        rstd = standardize_rgb(rgb, torch.ones_like(rgb[:, :1]))
        h, w = rgb.shape[-2] // self.scale, rgb.shape[-1] // self.scale
        out = self.net.pixels(rgb, rstd, z, (h, w))
        e = out["e"]
        u = out["u"] if out["u"] is not None else e[:, :1] * 0
        beta = out["beta_logit"] if out["beta_logit"] is not None else e[:, :1] * 0
        return e, out["f"], u, beta


def main(args):
    fusion, _ = load_model(args.ckpt, torch.device("cpu"))
    if fusion.cfg.model.backbone != "tiny":
        raise SystemExit("ONNX export is provided for the edge (tiny) student; distil stage A first (distill_edge.py)")
    module = DenseBranch(fusion.net, args.scale).eval()
    rgb = torch.rand(1, 3, args.size, args.size)
    z = torch.randn(1, fusion.cfg.model.spectral_dims, args.size // args.scale, args.size // args.scale)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    dynamic = None if args.static else {"rgb": {2: "H", 3: "W"}, "z": {2: "h", 3: "w"}, "e": {2: "H", 3: "W"},
                                        "f": {2: "h", 3: "w"}, "u": {2: "H", 3: "W"}, "beta_logit": {2: "H", 3: "W"}}
    kwargs = {"input_names": ["rgb", "z"], "output_names": ["e", "f", "u", "beta_logit"], "opset_version": 17,
              "dynamic_axes": dynamic}
    try:                                    # torch >= 2.9 defaults to the dynamo exporter: keep the TorchScript one
        torch.onnx.export(module, (rgb, z), args.out, dynamo=False, **kwargs)
    except TypeError:                       # torch < 2.5 has no `dynamo` argument
        torch.onnx.export(module, (rgb, z), args.out, **kwargs)
    print(f"exported {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--static", action="store_true", default=True)
    p.add_argument("--dynamic", dest="static", action="store_false")
    p.add_argument("--out", default="results/dense_branch.onnx")
    main(p.parse_args())
