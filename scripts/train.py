"""Stage A: train the joint RGB+HSI VFM that outputs couplings (one run for all training panels, one GPU).

  python scripts/train.py --config configs/stageA_vfm.json --panels configs/panels.json \
      --train-panels PCB-SR,PCB-DN,HyTex-A,LIB-HSI,Chikusei,Harvard --teacher cache/teacher --out runs/A_s0
Ablations (each is one more run; see configs/ablations/):
  --extra configs/ablations/no_coupling.json      parts = 0 (SIEVE-2 level: materials only)
  --extra configs/ablations/colour_coupling.json  part codes from RGB k-means instead of the VFM
  --extra configs/ablations/no_hsi_adapter.json   the VFM does not read the HSI
  --extra configs/ablations/self_supervised.json  no HR supervision (held-out evidence + VFM-L prior only)
  --extra configs/ablations/no_coupling_gate.json beta = 1
  --no-teacher                                    no VFM-L prior (w_rel, w_part off)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sieve3 import trainer  # noqa: E402
from sieve3.config import load_config  # noqa: E402
from sieve3.data import TrainStore, load_panels  # noqa: E402
from sieve3.losses import objective  # noqa: E402
from sieve3.model import CoupledNet  # noqa: E402
from sieve3.pipeline import Fusion  # noqa: E402


def parse_sets(items):
    out = {}
    for item in items or []:
        key, value = item.split("=", 1)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def setup(args):
    cfg = load_config(args.config, parse_sets(args.set), args.extra)
    t = cfg.train
    if args.iters:
        t.iterations = args.iters
    if args.seed is not None:
        t.seed = args.seed
    if args.hours is not None:
        t.time_budget_hours = args.hours
    torch.manual_seed(t.seed)
    torch.backends.cuda.matmul.allow_tf32 = False   # keep the closed-form normal equations in true fp32/fp64
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    panels = load_panels(args.panels, args.train_panels.split(","))
    store = TrainStore(panels, None if args.no_teacher else args.teacher, device, t.store_max_gb)
    if not args.no_teacher and not store.has_teacher:
        raise SystemExit("teacher features missing for some training images: run scripts/cache_teacher.py first "
                         "or pass --no-teacher for the prior-free ablation")
    print(f"train store: {store.total_gb:.2f} GB on {store.store_device}; panels: "
          f"{[g['panel'].name for g in store.groups]}", flush=True)
    return cfg, device, store


def main(args):
    cfg, device, store = setup(args)
    t = cfg.train
    net = CoupledNet(cfg.model).to(device)
    fusion = Fusion(net, cfg)
    amp = t.amp and device.type == "cuda"

    def step(batch, rgb, lr, fit, check, generator):
        s, sigma = batch["s"], batch["sigma"]
        out = fusion.train_forward(rgb, lr, batch["valid"], fit, s, sigma, amp=amp)
        loss, terms = objective(out, lr, batch["ref"], batch["valid"], check, s, sigma, batch["teacher"], t,
                                generator)
        terms.update({"gate_mean": float(out["g"].mean()), "beta_mean": float(out["beta"].mean()),
                      "mu_mean": float(out["mu"].mean())})
        return loss, terms

    return trainer.run(net, cfg, store, step, args.out, device,
                       extra_ckpt={"stage": "A", "train_panels": args.train_panels.split(","),
                                   "teacher": None if args.no_teacher else args.teacher})


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "stageA_vfm.json"))
    p.add_argument("--extra", action="append", default=[], help="override file(s) merged after --config")
    p.add_argument("--panels", required=True)
    p.add_argument("--train-panels", required=True)
    p.add_argument("--teacher", default="cache/teacher")
    p.add_argument("--no-teacher", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--iters", type=int, default=0)
    p.add_argument("--hours", type=float, default=None, help="time budget (overrides train.time_budget_hours)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--set", action="append", help="override, e.g. --set train.batch=8")
    return p


if __name__ == "__main__":
    main(parser().parse_args())
