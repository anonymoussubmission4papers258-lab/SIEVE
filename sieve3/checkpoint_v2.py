"""Atomic, complete training recovery for the versioned SIEVE-3 runner.

The original checkpoint reader remains compatible with these dictionaries.
"""
from __future__ import annotations

import json
import os
import random
import uuid
from pathlib import Path

import numpy as np
import torch

from .checkpoint import vit_config_of
from .config import to_dict


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf8", newline="\n") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def rng_state(generator, device):
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch_cpu": torch.get_rng_state(), "sampler": generator.get_state()}
    if torch.device(device).type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state, generator, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    generator.set_state(state["sampler"])
    if "torch_cuda" in state:
        if torch.device(device).type != "cuda":
            raise ValueError("CUDA recovery state requires a CUDA device")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_training(path, net, cfg, optimizer, generator, device, training_state, extra=None):
    """Publish only after the entire payload has been written, flushed and closed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    payload = {"format_version": 2, "model": net.state_dict(), "config": to_dict(cfg),
               "vit_config": vit_config_of(net), "optimizer": optimizer.state_dict(),
               "scheduler": {"kind": "original_dynamic_cosine", **training_state["scheduler"]},
               "rng": rng_state(generator, device), "training_state": training_state,
               "iterations": training_state["iterations"], "seconds": training_state["seconds"],
               **(extra or {})}
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def restore_training(path, net, cfg, optimizer, generator, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format_version") != 2:
        raise ValueError("not a full version-2 recovery checkpoint; model-only checkpoints cannot resume")
    if checkpoint["config"] != to_dict(cfg):
        raise ValueError("resume config differs from the frozen recovery configuration")
    if checkpoint["scheduler"]["kind"] != "original_dynamic_cosine":
        raise ValueError("unrecognised scheduler recovery")
    net.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    restore_rng(checkpoint["rng"], generator, device)
    return checkpoint["training_state"]
