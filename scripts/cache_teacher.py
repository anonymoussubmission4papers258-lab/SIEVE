"""Cache PCA-reduced dense features of the large RGB foundation model for the training images.

Run once; training then never loads the teacher. Example:
  python scripts/cache_teacher.py --panels configs/panels.json --teacher facebook/dinov2-large \
      --out cache/teacher --panels-used PCB-SR,PCB-DN,HyTex-A,LIB-HSI,Chikusei,Harvard
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sieve3.data import load_cube, load_panels, load_rgb  # noqa: E402
from sieve3.teacher import Teacher, fit_pca, project  # noqa: E402


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    names = args.panels_used.split(",") if args.panels_used else None
    panels = load_panels(args.panels, names)
    teacher = Teacher(args.teacher, device)
    splits = set(args.splits.split(","))
    feats, keys = [], []
    start = time.time()
    for panel in panels.values():
        for sample in panel.samples:
            if sample.split not in splits or not sample.reference:
                continue
            ref = load_cube(sample.reference, panel.layout)
            if sample.rgb and panel.srf is None:
                rgb = torch.from_numpy(load_rgb(sample.rgb, ref.shape[1:]))
            elif panel.srf is not None:
                rgb = torch.einsum("kc,chw->khw", torch.from_numpy(panel.srf), torch.from_numpy(ref))
            else:
                raise FileNotFoundError(f"{panel.name}/{sample.id}: no RGB source")
            f = teacher.features(rgb, args.long_side, args.tile).cpu().half()
            feats.append(f)
            keys.append((panel.name, sample.id))
            print(f"teacher {panel.name}/{sample.id}: {tuple(f.shape)}", flush=True)
    if not feats:
        raise SystemExit("no images to cache")
    mean, basis, explained = fit_pca([f.float() for f in feats], args.dims, args.max_tokens)
    out = Path(args.out)
    for (panel, sid), f in zip(keys, feats):
        (out / panel).mkdir(parents=True, exist_ok=True)
        np.save(out / panel / f"{sid}.npy", project(f.float(), mean, basis).numpy().astype(np.float16))
    meta = {"teacher": args.teacher, "long_side": args.long_side, "dims": args.dims, "explained": explained,
            "images": len(feats), "seconds": time.time() - start, "panels": sorted({k[0] for k in keys}),
            "splits": sorted(splits)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    torch.save({"mean": mean, "basis": basis}, out / "pca.pt")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--panels", required=True)
    p.add_argument("--panels-used", default="")
    p.add_argument("--teacher", required=True, help="HF id or local folder (DINOv2/DINOv3 ViT)")
    p.add_argument("--out", default="cache/teacher")
    p.add_argument("--splits", default="train")
    p.add_argument("--long-side", type=int, default=518)
    p.add_argument("--tile", type=int, default=896)
    p.add_argument("--dims", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=300000)
    p.add_argument("--device", default="cuda")
    main(p.parse_args())
