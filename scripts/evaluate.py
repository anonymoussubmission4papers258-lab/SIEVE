"""Evaluate a stage-A or stage-B checkpoint on val/test splits, with or without the audit.

  python scripts/evaluate.py --ckpt runs/A_s0/model.pt --panels configs/panels.json \
      --eval-panels PCB-SR,PCB-DN,HyTex-A,HyTex-B,HyTex-C,LIB-HSI,Chikusei,Harvard --split test --mode default
mode: fast (VFM decisions only) | default (audited: coupling + correction tests) | strict (audited, cross-fitted)
Writes per-image rows (metrics, region counts, mean measured fraction, coupling residual, seconds) and
per-panel summaries; --save-pred also stores the HR cube and the maps (mu, beta, g, codes, residual).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sieve3 import audit  # noqa: E402
from sieve3.checkpoint import load_model  # noqa: E402
from sieve3.data import evaluation_inputs, load_panels  # noqa: E402
from sieve3.metrics import observability, reconstruction_metrics  # noqa: E402

SUMMARY_KEYS = ("rmse", "psnr", "sam", "ssim", "ergas", "seconds", "obs_joint_null", "mu_mean", "beta_mean",
                "resid_mean")


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fusion, ckpt = load_model(args.ckpt, device)
    if args.audit_statistic:
        fusion.cfg.audit.statistic = args.audit_statistic
    panels = load_panels(args.panels, args.eval_panels.split(","))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for panel in panels.values():
        for sample in panel.samples:
            if sample.split != args.split:
                continue
            inputs = evaluation_inputs(panel, sample, device)
            sigma = panel.sigma if args.sigma == "protocol" else "auto"
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            res = audit.run(fusion, inputs["rgb"], inputs["lr"], inputs["valid"], panel.scale, sigma,
                            srf=panel.srf if args.use_srf_gate else None, mode=args.mode, tile=args.tile,
                            amp=args.amp, out_device=torch.device(args.out_device) if args.out_device else None)
            if device.type == "cuda":
                torch.cuda.synchronize()
            rep = res["report"]
            row = {"panel": panel.name, "id": sample.id, "seconds": time.time() - start, "stage": ckpt.get("stage"),
                   "rgb_source": inputs["rgb_source"], "lr_source": inputs["lr_source"], "scene": rep.get("scene"),
                   "sigma": rep.get("sigma")}
            for key in ("mu_mean", "beta_mean", "gate_mean", "resid_mean", "kappa_mean", "kappa_coupling_mean"):
                if key in rep:
                    row[key] = rep[key]
            for group in ("regions_correction", "regions_coupling"):
                for name, count in rep.get(group, {}).items():
                    row[f"{group}_{name}"] = count
            if inputs["ref"] is not None:
                valid = inputs["valid"][0, 0] > 0.5
                pred = res["x"][0].to(device)
                row.update(reconstruction_metrics(pred, inputs["ref"], valid, panel.scale))
                row.update({f"obs_{k}": v for k, v in observability(pred, inputs["ref"], valid, panel.scale,
                                                                       panel.srf).items()})
            rows.append(row)
            print(json.dumps(row), flush=True)
            if args.save_pred:
                pdir = out / "pred" / panel.name
                pdir.mkdir(parents=True, exist_ok=True)
                np.save(pdir / f"{sample.id}.npy", res["x"][0].permute(1, 2, 0).cpu().numpy().astype(np.float32))
                maps = {}
                for key in ("mu", "beta", "g"):
                    if res.get(key) is not None:
                        maps[key] = res[key][0, 0].cpu().numpy().astype(np.float16)
                for key in ("code", "code_coupling"):
                    if res.get(key) is not None:
                        maps[key] = res[key][0, 0].cpu().numpy().astype(np.uint8)
                for key in ("kappa", "kappa_coupling", "resid"):
                    if res.get(key) is not None:
                        maps[key] = res[key][0, 0].cpu().numpy().astype(np.float32)
                np.savez_compressed(pdir / f"{sample.id}_maps.npz", **maps)
    if not rows:
        raise SystemExit("no samples evaluated")
    keys = sorted({k for r in rows for k in r})
    with open(out / f"{args.split}_{args.mode}.csv", "w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for name in sorted({r["panel"] for r in rows}):
        sel = [r for r in rows if r["panel"] == name]
        summary[name] = {k: float(np.mean([r[k] for r in sel if k in r])) for k in SUMMARY_KEYS
                         if any(k in r for r in sel)}
        summary[name]["n"] = len(sel)
        summary[name]["n_nonfinite_rmse"] = sum(1 for r in sel if "rmse" in r and not np.isfinite(r["rmse"]))
    (out / f"{args.split}_{args.mode}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--panels", required=True)
    p.add_argument("--eval-panels", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--mode", default="default", choices=["fast", "default", "strict"])
    p.add_argument("--tile", type=int, default=512, help="HR tile of the multi-pass engine (0 = whole image; exact)")
    p.add_argument("--amp", action="store_true", help="bf16 for the dense features (the solve stays fp32/fp64)")
    p.add_argument("--out-device", default="", help="'cpu' keeps full-resolution outputs in host memory (big scenes)")
    p.add_argument("--sigma", default="protocol", choices=["protocol", "auto"])
    p.add_argument("--audit-statistic", default="", choices=["", "z", "t"])
    p.add_argument("--use-srf-gate", action="store_true", help="scene-level RGB/HSI compatibility test (known SRF)")
    p.add_argument("--save-pred", action="store_true")
    p.add_argument("--out", default="results/eval")
    return p


if __name__ == "__main__":
    main(parser().parse_args())
