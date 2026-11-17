# SIEVE

Anonymous research code for **SIEVE: Spectral Evidence-Gated Visual Priors for
Hyperspectral–RGB Fusion**.

This release includes the learned model, training and distillation code,
observation-only inference, separate metric implementations, frozen-estimator
reference kernels, experiment configurations and checkpoint
archives. It does not contain the manuscript, personal machine logs, raw datasets,
virtual environments, or access credentials.

## Start here

```bash
python -m pip install -r requirements.txt
python tests/test_reference_numpy.py
python tests/test_torch.py
```

Download `SIEVE_checkpoints_anonymous.zip` from this repository's Releases and
extract it at the repository root. `checkpoints_manifest.json` lists hashes,
training steps, completion status and model roles. The archive includes **13
checkpoint snapshots**, including historical partial/recovery states; these are
not 13 independent completed experiments.

| Checkpoint | Role | Updates |
|---|---|---:|
| `stageA_30000.pt` | Primary learned SIEVE, four-domain joint training | 30,000 |
| `stageB_20000.pt` | Secondary distilled student | 20,000 |
| `ablation_full.pt` | PCB matched-budget full model | 24,300 |
| `ablation_no_coupling.pt` | PCB matched-budget removal | 24,300 |
| `ablation_no_hsi_adapter.pt` | PCB matched-budget removal | 24,300 |

The primary checkpoint contains its ViT configuration and complete weights:
inference reconstructs the model locally without downloading DINO weights.

## Inference

Input NPZ files contain `h` (LR-HSI, HWC), `r` (RGB, HWC), `valid` (HW), and
optionally `srf` (3 × bands). The reference is never an inference input.

```bash
python scripts/infer.py --checkpoint checkpoints/stageA_30000.pt \
  --input data/example_input.npz --scale 4 --sigma 0.01 \
  --output runs/example --device cuda
```

To reproduce the fixed 79-input, 32-band comparison, provide the exact original
NPZ observations and targets under the relative paths in the manifest:

```bash
python scripts/infer.py --checkpoint checkpoints/stageA_30000.pt \
  --manifest protocols/matched79.json --data-root data \
  --metric legacy32 --output runs/matched79 --device cuda
```

The manifest checks original input/target SHA256 values. No observations are
regenerated and no reference is read until after prediction. Outputs remain raw
FP32; display clipping is not applied to scientific metrics. See
`docs/DATA_AND_PROTOCOLS.md` for native versus matched-32 evaluation.

## Training and distillation

Use a new output directory and your own data paths. Exact saved configurations
are in `configs/checkpoints/`; the generic example configurations are templates,
not substitutes for the reported recipes.

```bash
python scripts/cache_teacher.py --panels configs/panels.json \
  --teacher facebook/dinov2-small --out cache/teacher \
  --panels-used PCB-SR,Chikusei-native,Harvard-native,HyTexiLa-native
python scripts/train_v3.py --config configs/checkpoints/stageA_30000.json \
  --panels configs/panels.json \
  --train-panels PCB-SR,Chikusei-native,Harvard-native,HyTexiLa-native \
  --teacher cache/teacher --out runs/new_stageA
python scripts/distill_edge_v3.py --stageA checkpoints/stageA_30000.pt \
  --config configs/checkpoints/stageB_20000.json --panels configs/panels.json \
  --train-panels PCB-SR,Chikusei-native,Harvard-native,HyTexiLa-native \
  --teacher cache/teacher --out runs/new_stageB
```

Training reproduction is not rerun by preparing this archive. The secondary
student originally resumed from 18,257 to 20,000 updates with an explicitly
extended cosine horizon. The original interrupted branch and final branch are
both retained. See `docs/REPRODUCIBILITY.md` for scope and limitations.

## Model identities and scope

Only our method and its checkpoints are included. Benchmark result tables,
prediction figures, baseline weights and dataset files are deliberately excluded.
The primary learned model is `stageA_30000.pt`; the secondary student and PCB
ablation checkpoints have separate names. `legacy_frozen/` contains our earlier
frozen-feature estimation kernels and must not be confused with learned SIEVE.

The three PCB training ablations are single-split, single-seed, low-budget
experiments. Joint double-removal training was budget-blocked before production;
there is no checkpoint for that condition.

## Review use

Code and checkpoint metadata have been anonymized in the release copy; original
numeric tensors are verified unchanged. Third-party attribution is retained.
See `docs/ANONYMITY.md`, `THIRD_PARTY_NOTICES.md` and `RELEASE_VALIDATION.json`.

For CVPR 2027, include the code archive directly in the supplementary submission;
do not substitute a link to this repository for self-contained supplementary
files. Preparing this repository does not submit anything to OpenReview.
