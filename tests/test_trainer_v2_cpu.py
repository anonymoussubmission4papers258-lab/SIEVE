"""Real CPU fixtures for atomic full recovery and honest resource/numerical stops."""
from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sieve3.config import Config
from sieve3.trainer_v2 import TrainingResourceYield, run


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Identity()
        self.weight = nn.Parameter(torch.tensor(0.7))


class Store:
    def sample(self, batch, crop, generator):
        ref = torch.rand(1, 1, 8, 8, generator=generator)
        return {"ref": ref, "valid": torch.ones(1, 1, 8, 8), "s": 2, "sigma": 0.01,
                "panel": "synthetic-only", "rgb": ref.repeat(1, 3, 1, 1)}


def seed():
    torch.manual_seed(73)
    np.random.seed(73)
    random.seed(73)


def cfg():
    value = Config()
    value.train.iterations = 6
    value.train.batch = 1
    value.train.crop = 8
    value.train.warmup = 2
    value.train.log_every = 1
    value.train.save_every = 2
    value.train.seed = 43
    return value


def make_step(net, stop_dir=None, fail_at=None):
    calls = 0

    def step(batch, rgb, lr, fit, check, generator):
        nonlocal calls
        calls += 1
        a = rgb.mean() + lr.mean() + np.random.rand() + random.random()
        loss = (net.weight * a - 1.3).square()
        if fail_at is not None and calls == fail_at:
            loss = loss * float("nan")
        if stop_dir is not None and calls == 2:
            (stop_dir / "STOP").touch()
        return loss, {"fixture": True}
    return step


def main():
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="sieve3-recovery-", dir=ROOT.parents[1] / "tmp_audit") as temporary:
        root = Path(temporary)
        seed()
        full = Toy()
        run(full, cfg(), Store(), make_step(full), root / "full", "cpu")
        full_ckpt = torch.load(root / "full/model.pt", weights_only=False)
        seed()
        resumed = Toy()
        first = root / "resumed"
        try:
            run(resumed, cfg(), Store(), make_step(resumed, first), first, "cpu")
            raise AssertionError("STOP should exit 76")
        except SystemExit as error:
            assert error.code == 76
        first_ckpt = torch.load(first / "model_user_stop.pt", weights_only=False)
        assert first_ckpt["training_state"]["iterations"] == 2
        assert set(first_ckpt["rng"]) == {"python", "numpy", "torch_cpu", "sampler"}
        (first / "STOP").unlink()
        # Deliberately disturb every RNG and parameter before restoration.
        random.random(); np.random.rand(); torch.rand(2)
        resumed.weight.data.fill_(-20)
        run(resumed, cfg(), Store(), make_step(resumed), first, "cpu", resume=first / "model_user_stop.pt")
        final = torch.load(first / "model.pt", weights_only=False)
        assert torch.equal(full.weight, resumed.weight)
        for key in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(full_ckpt["optimizer"]["state"][0][key], final["optimizer"]["state"][0][key])
        assert torch.equal(full_ckpt["rng"]["torch_cpu"], final["rng"]["torch_cpu"])
        assert torch.equal(full_ckpt["rng"]["sampler"], final["rng"]["sampler"])
        assert full_ckpt["scheduler"] == final["scheduler"]
        print("PASS exact uninterrupted vs resumed model/Adam/scheduler/CPU+sampler RNG")
        budget_cfg = cfg()
        budget_cfg.train.time_budget_hours = 1e-12
        net = Toy()
        try:
            run(net, budget_cfg, Store(), make_step(net), root / "budget", "cpu")
            raise AssertionError("budget must yield")
        except TrainingResourceYield as error:
            assert error.code == 75
        assert (root / "budget/RESOURCE_YIELD.json").exists()
        assert not (root / "budget/model.pt").exists()
        assert not (root / "budget/TRAINING_DONE.json").exists()
        print("PASS resource yield is exit 75 with complete recovery and no completion marker")
        net = Toy()
        try:
            run(net, cfg(), Store(), make_step(net, fail_at=3), root / "nonfinite", "cpu")
            raise AssertionError("non-finite must fail")
        except FloatingPointError:
            pass
        recovery = torch.load(root / "nonfinite/model_numerical_stop.pt", weights_only=False)
        assert recovery["training_state"]["iterations"] == 2
        assert not (root / "nonfinite/TRAINING_DONE.json").exists()
        assert not (root / "nonfinite/model.pt").exists()
        assert json.loads((root / "nonfinite/ATTENTION.json").read_text())["no_optimizer_step"]
        print("PASS first non-finite attempt stops without consumed update or false completion")
        try:
            run(Toy(), cfg(), Store(), lambda *a: None, root / "full", "cpu")
            raise AssertionError("must refuse overwrite")
        except FileExistsError:
            pass
        assert not list(root.rglob("*.tmp"))
        print("PASS completed output cannot be overwritten; no partial atomic files remain")
    print("ALL V2 CPU RECOVERY FIXTURES PASSED")


if __name__ == "__main__":
    main()
