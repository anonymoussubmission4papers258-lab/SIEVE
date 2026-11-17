"""Versioned production wrapper: unchanged losses/optimizer, recoverable execution.

Budget-reduced targets are explicitly a resource yield (exit 75), not completion
of the configured iteration plan. Only a full finite plan publishes model.pt.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch

from sieve3.checkpoint_v2 import atomic_json, restore_rng, restore_training, rng_state, save_training
from sieve3.data import synthesise
from sieve3.ops import block_valid, random_check
from sieve3.trainer import Budget, lr_scale, make_optimizer


class TrainingResourceYield(SystemExit):
    def __init__(self, checkpoint):
        self.checkpoint = Path(checkpoint)
        super().__init__(75)


def run(net, cfg, store, step, out_dir, device, extra_ckpt=None, max_bad_steps=0, resume=None):
    del max_bad_steps  # A non-finite step is evidence requiring diagnosis, never silent progress.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    t = cfg.train
    if t.iterations <= 0 or t.batch <= 0 or t.log_every <= 0:
        raise ValueError("iterations, batch and log_every must be positive")
    optimizer, base_lrs = make_optimizer(net, t)
    budget = Budget(t.time_budget_hours, t.iterations, t.budget_probe)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "model.pt").exists() or (out / "TRAINING_DONE.json").exists():
        raise FileExistsError("completed output already exists; refusing to overwrite it")
    generator = torch.Generator().manual_seed(t.seed)
    it, prior_seconds, log = 0, 0.0, []
    if resume:
        state = restore_training(resume, net, cfg, optimizer, generator, device)
        it, prior_seconds = int(state["iterations"]), float(state["seconds"])
        budget.total = int(state["scheduler"]["total"])
        base_lrs = list(state["scheduler"]["base_lrs"])
        log = list(state.get("log", []))
        if state.get("completed_planned_iterations"):
            raise ValueError("this recovery checkpoint already completed the configured plan")
        if state.get("reason") == "resource_budget":
            raise ValueError("resource-yield checkpoint requires a separately approved continuation policy")
    started = time.monotonic()
    net.train()
    last_checkpoint = str(resume) if resume else None

    def elapsed():
        return prior_seconds + time.monotonic() - started

    def state(reason, complete=False):
        return {"iterations": it, "completed_updates": it, "planned_iterations": t.iterations,
                "seconds": elapsed(), "skipped_steps": 0, "reason": reason,
                "completed_planned_iterations": complete, "log": log,
                "scheduler": {"total": budget.total, "base_lrs": base_lrs,
                              "warmup": t.warmup, "last_iteration": it}}

    def publish(path, reason, complete=False):
        nonlocal last_checkpoint
        result = save_training(out / path, net, cfg, optimizer, generator, device,
                               state(reason, complete), extra_ckpt)
        last_checkpoint = str(result)
        return result

    atomic_json(out / "STARTED_V2.json", {"pid": os.getpid(), "resume": str(resume) if resume else None,
                                        "iterations": it, "planned_iterations": t.iterations,
                                        "format_version": 2})
    print(f"parameters: {sum(p.numel() for p in net.parameters()) / 1e6:.2f}M "
          f"(trainable {sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6:.2f}M)", flush=True)
    try:
        while it < budget.total:
            if not ablation_guard():break
            if (out / "STOP").exists():
                checkpoint = publish("model_user_stop.pt", "user_stop")
                atomic_json(out / "STOPPED.json", {**state("user_stop"), "checkpoint": str(checkpoint)})
                raise SystemExit(76)
            if t.time_budget_hours > 0 and elapsed() >= t.time_budget_hours * 3600.0 * 0.97:
                break
            next_it = it + 1
            for group, base in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base * lr_scale(next_it, budget.total, t.warmup)
            # On numerical failure, preserve pre-batch RNG and buffers for an exact, diagnosed retry.
            before_rng = rng_state(generator, device)
            before_buffers = {name: value.detach().clone() for name, value in net.named_buffers()}
            batch = store.sample(t.batch, t.crop, generator)
            rgb, lr = synthesise(batch)
            s = batch["s"]
            valid_lr = block_valid(batch["valid"], s)
            check = random_check(lr.shape[0], lr.shape[2], lr.shape[3], t.check_period, device, generator) * valid_lr
            optimizer.zero_grad(set_to_none=True)
            loss, terms = step(batch, rgb, lr, valid_lr - check, check, generator)
            finite = bool(torch.isfinite(loss))
            norm = None
            if finite:
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(net.parameters(), t.grad_clip if t.grad_clip > 0 else float("inf"))
                finite = bool(torch.isfinite(norm))
            if not finite:
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for name, value in net.named_buffers():
                        value.copy_(before_buffers[name])
                restore_rng(before_rng, generator, device)
                checkpoint = publish("model_numerical_stop.pt", "nonfinite_loss_or_gradient")
                atomic_json(out / "ATTENTION.json", {**state("nonfinite_loss_or_gradient"),
                    "failed_attempt": next_it, "checkpoint": str(checkpoint), "no_optimizer_step": True})
                raise FloatingPointError(f"non-finite loss/gradient at attempt {next_it}; training stopped")
            optimizer.step()
            it = next_it
            if ROLE == "qualify" and CUT and it == CUT:
                return publish("model_qualification_split.pt", "controlled_qualification_split", False)
            budget.update(it, elapsed())
            if t.save_every > 0 and it % t.save_every == 0:
                publish("model_last.pt", "periodic_recovery")
            if it % t.log_every == 0 or it == 1 or it == budget.total:
                row = dict(terms)
                row.update({"iter": it, "of": budget.total, "planned_iterations": t.iterations,
                            "panel": batch["panel"], "loss": float(loss.detach()),
                            "seconds": round(elapsed(), 3), "grad_norm": float(norm.detach())})
                log.append(row)
                atomic_json(out / "CURRENT.json", {"pid": os.getpid(), "status": "training", **row})
                print(json.dumps(row, allow_nan=False), flush=True)
        complete = it == t.iterations
        if not complete:
            checkpoint = publish("model_budget_yield.pt", "resource_budget")
            atomic_json(out / "RESOURCE_YIELD.json", {**state("resource_budget"), "checkpoint": str(checkpoint)})
            print(f"resource yield: {it}/{t.iterations} finite updates; recovery -> {checkpoint}", flush=True)
            raise TrainingResourceYield(checkpoint)
        checkpoint = publish("model.pt", "configured_iterations_complete", True)
        atomic_json(out / "train_log.json", {**state("configured_iterations_complete", True),
                    "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.device(device).type == "cuda" else None})
        atomic_json(out / "TRAINING_DONE.json", {**state("configured_iterations_complete", True),
                                                "checkpoint": str(checkpoint)})
        print(f"done: {it} finite updates in {elapsed()/60:.1f} min -> {checkpoint}", flush=True)
        return checkpoint
    except (SystemExit, FloatingPointError):
        raise
    except BaseException as error:
        # Do not label unknown, possibly partial optimizer transitions recoverable.
        atomic_json(out / "ATTENTION.json", {"pid": os.getpid(), "reason": "execution_exception",
                    "error": f"{type(error).__name__}: {error}", "last_committed_iterations": it,
                    "last_atomic_checkpoint": last_checkpoint, "completed_planned_iterations": False})
        raise

ROLE='qualify'
CUT=0
def ablation_guard():
    import shutil,psutil,datetime
    root=Path('data/sieve_mainline_ablation_20261107_v1')
    base=Path('data/sieve_cloud_v1')
    for marker in (root/'STOP',base/'HSI_RGB_OperatorFusion_Research/round6_full_20261028/configs/STOP_RECENT_BENCHMARK',base/'HSI_RGB_OperatorFusion_Research/round6_full_20261028/configs/STOP_NON_SRLF_PHASE'):
        if marker.exists():raise SystemExit(76)
    if psutil.virtual_memory().available<8*2**30 or psutil.Process().memory_info().rss>16*2**30:raise RuntimeError('RAM_RSS_guard')
    if shutil.disk_usage(base).free<10*2**30:raise RuntimeError('data10_floor')
    if ROLE=='train':
        policy=json.loads((root/'TRAINING_TIME_POLICY.json').read_bytes())
        if time.time()>min(policy['deadline'],policy['start']+policy['maximum_seconds']):return False
        if shutil.disk_usage(root).free<19*2**30+220*2**20:return False
    return True
