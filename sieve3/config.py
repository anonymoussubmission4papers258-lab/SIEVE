"""Configuration dataclasses with JSON loading and strict key checking."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass


@dataclass
class ModelConfig:
    # ---- visual foundation model (VFM) that reads RGB and LR-HSI jointly
    backbone: str = "tiny"          # "hf": DINOv2/DINOv3 ViT + LoRA (stage A); "tiny": edge CNN (stage B)
    hf_path: str = ""               # local folder or HF id of the ViT (e.g. facebook/dinov2-small)
    hf_input_scale: float = 2.0     # ViT input resolution relative to the HR image
    hf_max_side: int = 896          # cap of the ViT input long side (<= 64x64 patch tokens at patch 14)
    lora_blocks: int = 4            # last ViT blocks adapted (LoRA); the spectral adapter sits in front of them
    lora_rank: int = 8              # 0 = full fine-tuning of those blocks
    lora_alpha: float = 16.0
    adapter: str = "xattn"          # how the VFM reads the LR-HSI: "xattn" (spectral-twin attention) | "conv" | "none"
    adapter_heads: int = 4
    adapter_keys: int = 256         # LR cells pooled to at most this many attention keys
    adapter_pe: int = 32            # width of the sinusoidal position encodings
    # ---- coupling heads (the VFM outputs couplings, never spectra)
    embed_dim: int = 32             # material embedding
    lr_dim: int = 64                # LR-grid token width
    slot_dim: int = 64
    slots: int = 32                 # K material slots per image
    slot_iters: int = 3
    spectral_dims: int = 8          # per-image PCA dimensions of the LR-HSI tokens (band-agnostic, distances only)
    parts: int = 128                # M global part codes; 0 = no coupling (SIEVE-2 level)
    part_dim: int = 16
    part_source: str = "vfm"        # "vfm" | "color" (k-means colour codes: ablation "what colour can link")
    color_codes: int = 16           # M for part_source = "color"
    coupling_gate: bool = True      # beta from the network; False -> beta = 1 everywhere


@dataclass
class RoutingConfig:
    ridge: float = 1e-3             # ridge on the column-normalised material design
    part_prior: float = 0.15        # prior std of part detail relative to RMS(H) (Bayesian ridge of delta)
    ridge_part: float = 1e-3        # floor of the normalised part ridge
    part_rank: int = 4              # spectral rank of part detail (leading directions of the material residual); 0 = all bands
    shade_lo: float = 0.05          # clamp of the shading factor (mean RGB / image mean)
    shade_hi: float = 20.0
    local_radius: int = 2           # observable correction: LR window radius of the local regression
    local_eps: float = 1e-2
    local_smooth: int = 1
    local_min_count: int = 4


@dataclass
class TrainConfig:
    iterations: int = 30000
    time_budget_hours: float = 0.0  # > 0: the measured speed caps the iterations so the run ends within the budget
    budget_probe: int = 50          # iterations before the first speed measurement
    batch: int = 8
    crop: int = 128
    lr: float = 1e-3
    backbone_lr: float = 2e-4       # LoRA / ViT parameters
    weight_decay: float = 1e-4
    warmup: int = 500
    check_period: int = 2           # held-out density 1/period^2 per iteration
    # physical terms (spectra are always solved from H; these only shape couplings and gates)
    w_ho: float = 1.0               # held-out LR evidence of the coupled candidate x_c
    w_ho_material: float = 0.25     # held-out LR evidence of the material-level candidate x_m (the fallback)
    w_gate: float = 1.0             # held-out LR evidence of the gated output (trains g)
    w_hr: float = 1.0               # HR supervision of the output x (set 0 for the self-supervised variant)
    w_hr_coupled: float = 0.5       # relative weight of the HR term on x_c
    # VFM-L prior terms (cached teacher features)
    w_rel: float = 0.5              # relational distillation of the material embedding
    w_part: float = 1.0             # cannot-link: distant pixels share a part code only if the VFM-L sees the same kind
    w_usage: float = 0.1            # codebook usage of coupled pixels
    part_min_dist: int = 12         # pixels; only pairs at least this far apart enter w_part
    part_pairs: int = 4096
    teacher_sim_lo: float = 0.2     # VFM-L cosine below lo: coupling forbidden; above hi: allowed
    teacher_sim_hi: float = 0.6
    distill_anchors: int = 256
    distill_targets: int = 1024
    tau_student: float = 0.1
    tau_teacher: float = 0.07
    # stage B (edge student) distillation from the stage-A model
    w_kd_part: float = 1.0          # KL of the part codes (shared codebook)
    w_kd_material: float = 1.0      # co-assignment of material slots on random pixel pairs
    w_kd_gates: float = 1.0         # beta and g
    kd_pairs: int = 4096
    amp: bool = True
    grad_clip: float = 1.0
    seed: int = 0
    log_every: int = 100
    save_every: int = 2000          # periodic checkpoint (model_last.pt); 0 = only at the end
    store_max_gb: float = 8.0       # training cubes kept on the GPU up to this size, else pinned CPU


@dataclass
class AuditConfig:
    folds: int = 4                  # must be a perfect square (interleaved folds)
    region: int = 8                 # region side in LR cells
    alpha: float = 0.05
    kappa_min: float = 0.05         # minimum observable fraction of a disagreement for it to be testable
    min_cells: int = 4              # minimum check cells per region and fold
    statistic: str = "z"            # "z": known sigma (controlled protocol); "t": self-normalised (real data)
    compat_threshold: float = 9.0   # scene-level RGB/HSI compatibility (known SRF only)


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)


def _coerce(current, value):
    """Keep JSON numbers consistent with the declared field type (e.g. 4000.0 -> 4000)."""
    if isinstance(current, bool):
        return bool(value)
    if isinstance(current, (int, float)) and isinstance(value, (int, float)) and not isinstance(value, bool):
        return type(current)(value)
    return value


def _merge(target, values: dict):
    for key, value in values.items():
        if key.startswith("_"):
            continue                  # "_comment" entries
        if not hasattr(target, key):
            raise KeyError(f"unknown config key: {key}")
        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(target, key, _coerce(current, value))
    return target


def merge_file(cfg: Config, path: str) -> Config:
    with open(path, encoding="utf8") as handle:
        return _merge(cfg, json.load(handle))


def load_config(path: str | None = None, overrides: dict | None = None, extra_files=None) -> Config:
    """Defaults <- config file <- extra override files (in order, e.g. configs/ablations/*.json) <- --set."""
    cfg = Config()
    if path:
        merge_file(cfg, path)
    for extra in extra_files or []:
        merge_file(cfg, extra)
    if overrides:
        _merge(cfg, overrides)
    return cfg


def to_dict(cfg: Config) -> dict:
    return asdict(cfg)


def from_dict(values: dict) -> Config:
    return _merge(Config(), values)
