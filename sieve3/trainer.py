"""Shared training loop of stage A (joint RGB+HSI VFM) and stage B (edge student).

Time budget: with train.time_budget_hours > 0 the speed is measured after `budget_probe` iterations and every
500 iterations after that; the total number of iterations is set to what fits into the budget (never more than
train.iterations), and the cosine schedule follows the updated total. A run therefore ends on time even when
the GPU is slower than expected.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

from .checkpoint import save_checkpoint
from .data import synthesise
from .ops import block_valid, random_check


class Budget:
    def __init__(self, hours: float, iterations: int, probe: int):
        self.hours, self.max_iters, self.probe = float(hours), int(iterations), max(1, int(probe))
        self.total = int(iterations)

    def update(self, it: int, elapsed: float) -> int:
        if self.hours <= 0 or it < self.probe or not (it == self.probe or it % 500 == 0):
            return self.total
        per_iter = elapsed / it
        allowed = int(self.hours * 3600.0 * 0.97 / max(per_iter, 1e-9))
        self.total = max(it, min(self.max_iters, allowed))
        return self.total


def lr_scale(it: int, total: int, warmup: int) -> float:
    if it <= warmup:
        return it / max(warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (it - warmup) / max(total - warmup, 1))))


def make_optimizer(net, tcfg):
    vit = [p for n, p in net.named_parameters() if p.requires_grad and n.startswith("backbone.vit")]
    other = [p for n, p in net.named_parameters() if p.requires_grad and not n.startswith("backbone.vit")]
    groups = [{"params": other, "lr": tcfg.lr}]
    if vit:
        groups.append({"params": vit, "lr": tcfg.backbone_lr})
    return torch.optim.AdamW(groups, weight_decay=tcfg.weight_decay), [g["lr"] for g in groups]


def run(net, cfg, store, step, out_dir, device, extra_ckpt=None, max_bad_steps: int = 50):
    """step(batch, rgb, lr, fit, check, generator) -> (loss, terms dict). Saves out_dir/model.pt."""
    t = cfg.train
    optim, base_lrs = make_optimizer(net, t)
    budget = Budget(t.time_budget_hours, t.iterations, t.budget_probe)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(t.seed)
    n_params = sum(p.numel() for p in net.parameters())
    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"parameters: {n_params / 1e6:.2f}M (trainable {n_train / 1e6:.2f}M)", flush=True)
    log, bad = [], 0
    start = time.time()
    net.train()
    it = 0
    total = budget.total
    while it < total:
        it += 1
        scale = lr_scale(it, total, t.warmup)
        for group, base in zip(optim.param_groups, base_lrs):
            group["lr"] = base * scale
        batch = store.sample(t.batch, t.crop, generator)
        rgb, lr = synthesise(batch)
        s = batch["s"]
        valid_lr = block_valid(batch["valid"], s)
        check = random_check(lr.shape[0], lr.shape[2], lr.shape[3], t.check_period, device, generator) * valid_lr
        fit = valid_lr - check
        loss, terms = step(batch, rgb, lr, fit, check, generator)
        optim.zero_grad(set_to_none=True)
        finite = bool(torch.isfinite(loss))
        if finite:
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(net.parameters(), t.grad_clip if t.grad_clip > 0 else float("inf"))
            finite = bool(torch.isfinite(norm))
        if not finite:                          # skip the step: parameters never see a NaN/inf update
            optim.zero_grad(set_to_none=True)
            bad += 1
            print(f"warning: non-finite loss or gradient at iteration {it} ({bad} so far); step skipped", flush=True)
            if bad > max_bad_steps:
                save_checkpoint(out / "model_last_finite.pt", net, cfg, iterations=it, aborted=True)
                raise RuntimeError("too many non-finite steps (last finite weights saved)")
            continue
        optim.step()
        total = budget.update(it, time.time() - start)
        if t.save_every > 0 and it % t.save_every == 0:
            save_checkpoint(out / "model_last.pt", net, cfg, iterations=it, seconds=time.time() - start,
                            **(extra_ckpt or {}))
        if it % t.log_every == 0 or it == 1 or it == total:
            elapsed = time.time() - start
            terms.update({"iter": it, "of": total, "panel": batch["panel"], "loss": float(loss.detach()),
                          "seconds": round(elapsed, 1), "eta_min": round(elapsed / it * (total - it) / 60, 1)})
            log.append(terms)
            print(json.dumps(terms), flush=True)
    seconds = time.time() - start
    save_checkpoint(out / "model.pt", net, cfg, seconds=seconds, iterations=it, **(extra_ckpt or {}))
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else None
    (out / "train_log.json").write_text(json.dumps({"log": log, "seconds": seconds, "iterations": it,
                                                    "planned_iterations": t.iterations, "skipped_steps": bad,
                                                    "peak_gpu_gb": peak}, indent=1))
    print(f"done: {it} iterations in {seconds / 60:.1f} min -> {out / 'model.pt'}", flush=True)
    return out / "model.pt"
