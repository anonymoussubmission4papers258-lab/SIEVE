"""Stage B: distil the stage-A couplings into the edge student (tiny CNN backbone, no ViT at inference).

  python scripts/distill_edge.py --stageA runs/A_s0/model.pt --config configs/stageB_edge.json \
      --panels configs/panels.json --train-panels PCB-SR,PCB-DN,HyTex-A,LIB-HSI,Chikusei,Harvard --out runs/B_s0
The student shares the stage-A part codebook (frozen), so its part codes are directly comparable; it keeps all
physical terms (held-out evidence, HR) and adds KL(Gamma), material co-assignment and gate distillation.
The stage-A model runs online without gradients (bf16 on GPU); the VFM-L cache is not needed here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_v3 as stage_a  # noqa: E402
from sieve3 import trainer_v2 as trainer  # noqa: E402
from sieve3.checkpoint import load_model  # noqa: E402
from sieve3.losses import distill_objective  # noqa: E402
from sieve3.model import CoupledNet  # noqa: E402
from sieve3.pipeline import Fusion  # noqa: E402


def main(args):
    args.no_teacher = True                     # stage B learns from the stage-A model, not from the VFM-L cache
    cfg, device, store = stage_a.setup(args)
    t = cfg.train
    teacher, _ = load_model(args.stageA, device)
    for p in teacher.net.parameters():
        p.requires_grad_(False)
    ma, mb = teacher.cfg.model, cfg.model
    key_a = (ma.parts, ma.part_dim, ma.part_source, ma.color_codes)
    key_b = (mb.parts, mb.part_dim, mb.part_source, mb.color_codes)
    if key_a != key_b:
        raise SystemExit(f"student part codes (parts, part_dim, part_source, color_codes) = {key_b} must match "
                         f"stage A {key_a}")
    cfg.routing = teacher.cfg.routing          # identical physics: same closed-form solve as stage A
    net = CoupledNet(mb).to(device)
    if net.use_parts:
        with torch.no_grad():
            net.codebook.copy_(teacher.net.codebook)
            net.log_tau_part.copy_(teacher.net.log_tau_part)
        net.codebook.requires_grad_(False)
        net.log_tau_part.requires_grad_(False)
    fusion = Fusion(net, cfg)
    amp = t.amp and device.type == "cuda"

    def step(batch, rgb, lr, fit, check, generator):
        s, sigma = batch["s"], batch["sigma"]
        with torch.no_grad():
            out_t = teacher.train_forward(rgb, lr, batch["valid"], fit, s, sigma, amp=amp)
        out_s = fusion.train_forward(rgb, lr, batch["valid"], fit, s, sigma, amp=amp)
        loss, terms = distill_objective(out_s, out_t, lr, batch["ref"], batch["valid"], check, s, sigma, t, generator)
        terms.update({"gate_mean": float(out_s["g"].mean()), "beta_mean": float(out_s["beta"].mean())})
        return loss, terms

    return trainer.run(net, cfg, store, step, args.out, device,
                       resume=args.resume, extra_ckpt={"stage": "B", "stageA": str(args.stageA),
                                   "train_panels": args.train_panels.split(",")})


def parser():
    p = stage_a.parser()
    p.set_defaults(config=str(Path(__file__).resolve().parents[1] / "configs" / "stageB_edge.json"))
    p.add_argument("--stageA", required=True, help="stage-A checkpoint (model.pt)")
    return p


if __name__ == "__main__":
    main(parser().parse_args())
