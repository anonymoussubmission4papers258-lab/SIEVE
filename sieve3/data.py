'Dataset adapter for the SIEVE project manifests, deterministic observations and a GPU-resident train store.\n\nPanel config (one entry of configs/panels.json):\n{\n  "name": "PCB-SR",\n  "manifest": "path/to/manifest.json",            # {"protocol": path, "samples": [{"id","split","reference","valid",...}]}\n  "protocol": "optional override of manifest[\'protocol\']",   # {"observation": {"scale","noise_sigma","srf"}}\n  "scale": 4, "sigma": 0.01,                      # optional overrides\n  "srf": null,                                    # optional override; null -> use protocol; "none" -> no SRF (e.g. RGB+SWIR)\n  "reference_layout": "HWC",                      # reference cubes are H x W x B in the SIEVE project\n  "path_map": [["<LOCAL_PATH>", "/new/root"]],     # optional prefix remapping\n  "split_map": {"validation": "val"},             # optional\n  "obs_files": {"rgb": "rgb.npy", "lr": "hsi.npy"},   # file names inside a row\'s "folder"\n  "observation_manifests": {"val": "path/val_observations/manifest.json"}\n}\nIf no observed LR file is found the LR-HSI is synthesised deterministically (seeded by panel and id);\ncomparisons with baselines must use identical observations, so point obs_files at the shared ones.\n'
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .ops import lcm, observable_hr, pool


@dataclass
class Sample:
    panel: str
    id: str
    split: str
    reference: str | None
    valid: str | None = None
    rgb: str | None = None
    lr: str | None = None


@dataclass
class Panel:
    name: str
    scale: int
    sigma: float
    srf: np.ndarray | None
    samples: list
    layout: str = "HWC"


def _remap(path, path_map, base):
    if path is None:
        return None
    p = str(path)
    for old, new in path_map:
        if p.replace("\\", "/").lower().startswith(old.replace("\\", "/").lower()):
            p = new + p.replace("\\", "/")[len(old.replace("\\", "/")):]
            break
    if os.sep == "/":
        p = p.replace("\\", "/")
    q = Path(p)
    if not q.is_absolute() and base is not None:
        q = Path(base) / q
    return str(q)


def load_panel(entry: dict) -> Panel:
    manifest = Path(entry["manifest"])
    meta = json.loads(manifest.read_text(encoding="utf8"))
    path_map = entry.get("path_map", [])
    base = manifest.parent
    protocol = {}
    protocol_path = entry.get("protocol") or meta.get("protocol")
    if protocol_path:
        protocol = json.loads(Path(_remap(protocol_path, path_map, base)).read_text(encoding="utf8"))
    obs = protocol.get("observation", {})
    scale = int(entry.get("scale", obs.get("scale", 4)))
    sigma = float(entry.get("sigma", obs.get("noise_sigma", 0.01)))
    srf = entry.get("srf")                      # null/absent -> protocol SRF; "none" -> no SRF (e.g. RGB+SWIR)
    srf = obs.get("srf") if srf is None else srf
    srf = None if srf is None or (isinstance(srf, str) and srf.lower() == "none") else np.asarray(srf, np.float32)
    files = {"rgb": "rgb.npy", "lr": "hsi.npy"}
    files.update(entry.get("obs_files", {}))
    split_map = entry.get("split_map", {})
    obs_rows = {}
    for split, opath in entry.get("observation_manifests", {}).items():
        opath = Path(_remap(opath, path_map, None))
        for row in json.loads(opath.read_text(encoding="utf8"))["samples"]:
            obs_rows[(split, str(row["id"]))] = (row, opath.parent)
    samples = []
    for row in meta["samples"]:
        split = split_map.get(row.get("split", "train"), row.get("split", "train"))
        sid = str(row["id"])
        sample = Sample(entry["name"], sid, split, _remap(row.get("reference"), path_map, base),
                        _remap(row.get("valid"), path_map, base))
        src, sbase = obs_rows.get((split, sid), (row, base))
        folder = src.get("folder")
        if folder:
            fdir = Path(_remap(folder, path_map, sbase))
            for key in ("rgb", "lr"):
                if files.get(key) and (fdir / files[key]).exists():
                    setattr(sample, key, str(fdir / files[key]))
        for key in ("rgb", "lr"):
            if src.get(key):
                setattr(sample, key, _remap(src[key], path_map, sbase))
        samples.append(sample)
    return Panel(entry["name"], scale, sigma, srf, samples, entry.get("reference_layout", "HWC"))


def load_panels(path: str, names=None) -> dict:
    entries = json.loads(Path(path).read_text(encoding="utf8"))["panels"]
    out = {}
    for entry in entries:
        if names and entry["name"] not in names:
            continue
        out[entry["name"]] = load_panel(entry)
    missing = set(names or []) - set(out)
    if missing:
        raise KeyError(f"panels not found in {path}: {sorted(missing)}")
    return out


# ----------------------------------------------------------------------------- arrays
def load_cube(path: str, layout: str = "HWC") -> np.ndarray:
    a = np.load(path, mmap_mode="r")
    if a.ndim != 3:
        raise ValueError(f"{path}: expected a 3-D cube, got {a.shape}")
    a = np.asarray(a, dtype=np.float32)
    return np.ascontiguousarray(np.moveaxis(a, -1, 0) if layout.upper() == "HWC" else a)


def load_valid(path, hw) -> np.ndarray:
    if not path or not Path(path).exists():
        return np.ones(hw, bool)
    v = np.asarray(np.load(path)).astype(bool)
    if v.ndim == 3:
        v = v.all(-1) if v.shape[:2] == tuple(hw) else v.all(0)
    if v.shape != tuple(hw):
        raise ValueError(f"{path}: valid mask {v.shape} does not match {hw}")
    return v


def load_rgb(path, hw) -> np.ndarray:
    a = np.asarray(np.load(path), np.float32)
    if a.shape == (3, *hw):
        return a
    if a.shape == (*hw, 3):
        return np.ascontiguousarray(np.moveaxis(a, -1, 0))
    raise ValueError(f"{path}: RGB shape {a.shape} does not match {hw}")


def load_lr(path, C, hw_lr) -> np.ndarray:
    a = np.asarray(np.load(path), np.float32)
    if a.shape == (C, *hw_lr):
        return a
    if a.shape == (*hw_lr, C):
        return np.ascontiguousarray(np.moveaxis(a, -1, 0))
    raise ValueError(f"{path}: LR-HSI shape {a.shape} does not match {(C, *hw_lr)}")


def sanitise(cube: np.ndarray, valid: np.ndarray):
    """Replace non-finite values by 0 and mark those pixels invalid. cube [C,H,W], valid [H,W] bool."""
    finite = np.isfinite(cube).all(0)
    return np.nan_to_num(cube, nan=0.0, posinf=0.0, neginf=0.0), valid & finite


def seed_of(*parts) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little") & ((1 << 63) - 1)


def evaluation_inputs(panel: Panel, sample: Sample, device) -> dict:
    """Full-image tensors for evaluation: observed files when available, else deterministic synthesis.

    Images are cropped to multiples of the scale; non-finite pixels are zeroed and marked invalid.
    """
    s = panel.scale
    ref_np = None
    if sample.reference and Path(sample.reference).exists():
        ref_np = load_cube(sample.reference, panel.layout)
    if ref_np is None and not (sample.rgb and sample.lr):
        raise FileNotFoundError(f"{panel.name}/{sample.id}: neither reference nor observations found")
    if ref_np is not None:
        C, H0, W0 = ref_np.shape
    else:
        rgb_shape = np.load(sample.rgb, mmap_mode="r").shape
        H0, W0 = (rgb_shape[1:] if rgb_shape[0] == 3 else rgb_shape[:2])
        C = None
    H, W = s * (H0 // s), s * (W0 // s)
    valid_np = load_valid(sample.valid, (H0, W0))[:H, :W]
    ref = None
    if ref_np is not None:
        ref_np, valid_np = sanitise(np.ascontiguousarray(ref_np[:, :H, :W]), valid_np)
        ref = torch.from_numpy(ref_np).to(device)
    if sample.rgb:
        rgb_np, valid_np = sanitise(np.ascontiguousarray(load_rgb(sample.rgb, (H0, W0))[:, :H, :W]), valid_np)
        rgb = torch.from_numpy(rgb_np).to(device)[None]
        rgb_source = "observed"
    elif panel.srf is not None:
        rgb = torch.einsum("kc,bchw->bkhw", torch.from_numpy(panel.srf).to(device), ref[None])
        rgb_source = "srf"
    else:
        raise FileNotFoundError(f"{panel.name}/{sample.id}: no observed RGB and no SRF")
    if sample.lr:
        if C is None:
            shape = np.load(sample.lr, mmap_mode="r").shape
            C = shape[0] if tuple(shape[1:]) == (H0 // s, W0 // s) else shape[-1]
        lr_np = load_lr(sample.lr, C, (H0 // s, W0 // s))
        bad = ~np.isfinite(lr_np).all(0)
        if bad.any():
            valid_np = valid_np & ~np.kron(bad, np.ones((s, s), bool))[:H, :W]
        lr = torch.from_numpy(np.nan_to_num(lr_np, nan=0.0, posinf=0.0, neginf=0.0)).to(device)[None]
        lr_source = "observed"
    valid = observable_hr(torch.from_numpy(valid_np).to(device)[None, None].float(), s)
    if not sample.lr:
        generator = torch.Generator().manual_seed(seed_of(panel.name, sample.id, "lr-observation"))
        noise = torch.randn((1, C, H // s, W // s), generator=generator).to(device)
        lr = pool(ref[None] * valid, s) + panel.sigma * noise
        lr_source = "synthesised"
    return {"ref": ref, "rgb": rgb.float(), "lr": lr.float(), "valid": valid, "rgb_source": rgb_source,
            "lr_source": lr_source}


# ----------------------------------------------------------------------------- training store
class TrainStore:
    """Training cubes of several panels kept resident (GPU if they fit, else pinned CPU).

    Every iteration draws one panel, random aligned crops with flips/rotations, and synthesises fresh
    observations on the GPU, so there is no data-loader bottleneck.
    """

    def __init__(self, panels: dict, teacher_dir: str | None, device, max_gb: float = 8.0, splits=("train",)):
        self.device = device
        self.groups = []
        items_all = []
        for panel in panels.values():
            items = []
            for sample in panel.samples:
                if sample.split not in splits or not sample.reference:
                    continue
                ref = load_cube(sample.reference, panel.layout)
                C, H, W = ref.shape
                ref, valid = sanitise(ref, load_valid(sample.valid, (H, W)))
                rgb = load_rgb(sample.rgb, (H, W)) if (panel.srf is None and sample.rgb) else None
                if panel.srf is None and rgb is None:
                    raise FileNotFoundError(f"{panel.name}/{sample.id}: training needs an SRF or an observed RGB")
                if rgb is not None:
                    rgb, valid = sanitise(rgb, valid)
                teach = None
                if teacher_dir:
                    tpath = Path(teacher_dir) / panel.name / f"{sample.id}.npy"
                    if tpath.exists():
                        teach = np.load(tpath)
                items.append({"ref": ref, "valid": valid, "rgb": rgb, "teach": teach, "id": sample.id})
            if not items:
                continue
            step = lcm(panel.scale, 8)
            side = min(min(it["ref"].shape[1], it["ref"].shape[2]) for it in items)
            self.groups.append({"panel": panel, "items": items, "max_side": side - side % step, "step": step,
                                "pixels": sum(it["ref"].shape[1] * it["ref"].shape[2] for it in items)})
            items_all.extend(items)
        if not self.groups:
            raise ValueError("no training samples found")
        total = sum(it["ref"].nbytes + (it["rgb"].nbytes if it["rgb"] is not None else 0) for it in items_all)
        self.store_device = torch.device(device) if total / 2 ** 30 <= max_gb else torch.device("cpu")
        for group in self.groups:
            for it in group["items"]:
                it["ref"] = self._put(torch.from_numpy(it["ref"]))
                it["valid"] = self._put(torch.from_numpy(it["valid"]).float()[None])
                it["rgb"] = None if it["rgb"] is None else self._put(torch.from_numpy(it["rgb"]))
                it["teach"] = None if it["teach"] is None else self._put(torch.from_numpy(it["teach"]).half())
        self.has_teacher = all(it["teach"] is not None for g in self.groups for it in g["items"])
        self.total_gb = total / 2 ** 30

    def _put(self, t):
        t = t.contiguous()
        return t.to(self.store_device) if self.store_device.type != "cpu" else t.pin_memory() if torch.cuda.is_available() else t

    def sample(self, batch: int, crop: int, generator: torch.Generator):
        weights = torch.tensor([g["pixels"] for g in self.groups], dtype=torch.float64)
        group = self.groups[int(torch.multinomial(weights, 1, generator=generator))]
        panel, s, step = group["panel"], group["panel"].scale, group["step"]
        P = min(crop, group["max_side"])
        P = max(step, P - P % step)
        areas = torch.tensor([it["ref"].shape[1] * it["ref"].shape[2] for it in group["items"]], dtype=torch.float64)
        refs, valids, rgbs, teach = [], [], [], []
        for _ in range(batch):
            it = group["items"][int(torch.multinomial(areas, 1, generator=generator))]
            C, H, W = it["ref"].shape
            y0 = s * int(torch.randint(0, (H - P) // s + 1, (1,), generator=generator))
            x0 = s * int(torch.randint(0, (W - P) // s + 1, (1,), generator=generator))
            k = int(torch.randint(0, 4, (1,), generator=generator))
            flip = bool(torch.randint(0, 2, (1,), generator=generator))

            def aug(t):
                t = torch.rot90(t, k, (-2, -1))
                return torch.flip(t, (-1,)) if flip else t

            refs.append(aug(it["ref"][:, y0:y0 + P, x0:x0 + P].to(self.device, non_blocking=True)))
            valids.append(aug(it["valid"][:, y0:y0 + P, x0:x0 + P].to(self.device, non_blocking=True)))
            if it["rgb"] is not None:
                rgbs.append(aug(it["rgb"][:, y0:y0 + P, x0:x0 + P].to(self.device, non_blocking=True)))
            if it["teach"] is not None:
                ys = (torch.arange(y0, y0 + P, device=self.device, dtype=torch.float32) + 0.5) / H * 2 - 1
                xs = (torch.arange(x0, x0 + P, device=self.device, dtype=torch.float32) + 0.5) / W * 2 - 1
                gy, gx = torch.meshgrid(ys, xs, indexing="ij")
                grid = torch.stack([gx, gy], -1)[None]
                feat = F.grid_sample(it["teach"].to(self.device).float()[None], grid, mode="bilinear",
                                     padding_mode="border", align_corners=False)[0]
                teach.append(aug(feat))
        valid = observable_hr(torch.stack(valids).float(), s)
        out = {"ref": torch.stack(refs).float(), "valid": valid, "s": s, "sigma": panel.sigma, "panel": panel.name,
               "rgb": torch.stack(rgbs).float() if rgbs else None,
               "teacher": torch.stack(teach) if len(teach) == batch else None,
               "srf": None if panel.srf is None else torch.from_numpy(panel.srf).to(self.device)}
        return out


def synthesise(batch: dict, generator=None):
    """Fresh observations for a training batch: R = X S^T (or observed RGB), H = A X + sigma * n."""
    ref, s = batch["ref"], batch["s"]
    if batch["rgb"] is not None:
        rgb = batch["rgb"]
    else:
        rgb = torch.einsum("kc,bchw->bkhw", batch["srf"], ref)
    clean = pool(ref * batch["valid"], s)
    noise = torch.randn(clean.shape, generator=generator, device="cpu").to(ref.device) if generator is not None\
        else torch.randn_like(clean)
    return rgb, clean + batch["sigma"] * noise
