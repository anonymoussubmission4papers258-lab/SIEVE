"""Measure what the paper must report: inference latency/memory, parameters, FLOPs, and training speed.

Inference (deployed path, no VFM-L teacher):
  python scripts/benchmark.py --ckpt runs/B_s0/model.pt --sizes 256,512,1024 --bands 32,128 --modes fast,default
Training speed of a config (random data; prints s/iter and the hours the configured iterations need):
  python scripts/benchmark.py --config configs/stageA_vfm.json --train-step
Without --ckpt a randomly initialised network of --config is timed (same cost as a trained one).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sieve3 import audit  # noqa: E402
from sieve3.checkpoint import load_model  # noqa: E402
from sieve3.config import load_config  # noqa: E402
from sieve3.losses import objective  # noqa: E402
from sieve3.model import CoupledNet  # noqa: E402
from sieve3.ops import block_valid, random_check, standardize_rgb  # noqa: E402
from sieve3.pipeline import Fusion  # noqa: E402
from sieve3.spectral import spectral_tokens  # noqa: E402
from sieve3.trainer import make_optimizer  # noqa: E402


def build(args, device):
    if args.ckpt:
        return load_model(args.ckpt, device)[0]
    cfg = load_config(args.config, extra_files=args.extra)
    return Fusion(CoupledNet(cfg.model).to(device).eval(), cfg)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def pixel_flops(net, size, bands, s, device):
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None
    rgb = torch.rand(1, 3, size, size, device=device)
    lr = torch.rand(1, bands, size // s, size // s, device=device)
    fit = torch.ones(1, 1, size // s, size // s, device=device)
    z = spectral_tokens(lr, fit, net.cfg.spectral_dims)
    rstd = standardize_rgb(rgb, torch.ones_like(rgb[:, :1]))
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        net.pixels(rgb, rstd, z, (size // s, size // s))
    return counter.get_total_flops()


def inference(args, device, results):
    fusion = build(args, device)
    net = fusion.net
    results["params_M"] = sum(p.numel() for p in net.parameters()) / 1e6
    results["backbone"] = fusion.cfg.model.backbone
    s = args.scale
    for size in [int(v) for v in args.sizes.split(",")]:
        for bands in [int(v) for v in args.bands.split(",")]:
            rgb = torch.rand(1, 3, size, size, device=device)
            lr = torch.rand(1, bands, size // s, size // s, device=device)
            valid = torch.ones(1, 1, size, size, device=device)
            for mode in args.modes.split(","):
                for _ in range(args.warmup):
                    audit.run(fusion, rgb, lr, valid, s, 0.01, mode=mode, tile=args.tile, amp=args.amp)
                sync(device)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                times = []
                for _ in range(args.repeat):
                    start = time.perf_counter()
                    audit.run(fusion, rgb, lr, valid, s, 0.01, mode=mode, tile=args.tile, amp=args.amp)
                    sync(device)
                    times.append(time.perf_counter() - start)
                times.sort()
                run = {"size": size, "bands": bands, "mode": mode, "median_ms": 1000 * times[len(times) // 2],
                       "peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else None}
                results["runs"].append(run)
                print(json.dumps(run), flush=True)
            flops = pixel_flops(net, size, bands, s, device)
            if flops is not None:
                results[f"dense_branch_GFLOPs_{size}"] = flops / 1e9


def train_speed(args, device, results):
    cfg = load_config(args.config, extra_files=args.extra)
    t = cfg.train
    net = CoupledNet(cfg.model).to(device).train()
    fusion = Fusion(net, cfg)
    optim, _ = make_optimizer(net, t)
    s, B, P, bands = args.scale, t.batch, t.crop, args.train_bands
    amp = t.amp and device.type == "cuda"
    generator = torch.Generator().manual_seed(0)

    def one():
        ref = torch.rand(B, bands, P, P, device=device)
        rgb = ref[:, :3] * 0.5 + ref[:, 3:6] * 0.5
        lr = torch.nn.functional.avg_pool2d(ref, s) + 0.01 * torch.randn(B, bands, P // s, P // s, device=device)
        valid = torch.ones(B, 1, P, P, device=device)
        vlr = block_valid(valid, s)
        check = random_check(B, P // s, P // s, t.check_period, device, generator) * vlr
        teacher = torch.randn(B, 64, P, P, device=device)
        out = fusion.train_forward(rgb, lr, valid, vlr - check, s, 0.01, amp=amp)
        loss, _ = objective(out, lr, ref, valid, check, s, 0.01, teacher, t, generator)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

    for _ in range(args.warmup):
        one()
    sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(args.repeat):
        one()
    sync(device)
    per = (time.perf_counter() - start) / args.repeat
    results["train"] = {"config": args.config, "batch": B, "crop": P, "bands": bands, "s_per_iter": per,
                        "iterations": t.iterations, "hours_for_iterations": per * t.iterations / 3600,
                        "time_budget_hours": t.time_budget_hours,
                        "trainable_M": sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6,
                        "peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else None}
    print(json.dumps(results["train"], indent=2), flush=True)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    results = {"device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
               "runs": []}
    if args.train_step:
        train_speed(args, device, results)
    else:
        inference(args, device, results)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return results


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="")
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "stageB_edge.json"))
    p.add_argument("--extra", action="append", default=[])
    p.add_argument("--sizes", default="256,512")
    p.add_argument("--bands", default="32,128")
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--modes", default="fast,default")
    p.add_argument("--tile", type=int, default=0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeat", type=int, default=10)
    p.add_argument("--train-step", action="store_true", help="measure training speed of --config instead")
    p.add_argument("--train-bands", type=int, default=32)
    p.add_argument("--out", default="results/benchmark.json")
    return p


if __name__ == "__main__":
    main(parser().parse_args())
