# Data and evaluation protocols

Raw third-party datasets and derived teacher-feature caches are not included.
Obtain the original Chikusei, PCB, Harvard, HyTexiLa, SpectralWaste and LIB-HSI
data under each dataset provider's terms. No redistribution permission for those
datasets is asserted by this release.

`configs/panels_example.json` documents the training data adapter. Copy it to
`configs/panels.json`, provide local relative manifests, and use the same frozen
train/validation/test splits and observations as the comparison. A training
manifest has `samples` containing `id`, `split`, `reference`, `valid`, `rgb`, `lr`;
all paths are relative to that manifest. RGB/LR-HSI references point to NPY arrays.
The optional protocol specifies scale, noise sigma and SRF. Training can synthesize
observations when missing; benchmark inference deliberately requires the exact
recorded observations and does not silently synthesize substitutes.

## Matched 79 views

`protocols/matched79.json` records the fixed 192 × 192, 32-band input/target hashes
and relative file names. NPZ input keys: `h`, `r`, `srf`, `valid`. NPZ target keys:
`x`, `valid`. Arrays use HWC except HW masks and 3 × bands SRF matrices.

This protocol uses `metrics/legacy32.py`, including the original masked SAM and
uniform-window SSIM conventions. It is distinct from the native protocol. The
selected primary checkpoint is Stage A at 30,000 updates; it is not selected
using these test results. Equal-scene means are reported per panel.

## Native six-domain results

Native counts are Chikusei 1, PCB 11, Harvard 11, HyTexiLa 66, SpectralWaste 171
and LIB-HSI 75: 335 primary test views. The model trained jointly on 98 scenes
from the first four domains. SpectralWaste and LIB-HSI are zero-shot transfers.
Native benchmark result archives are intentionally not included.

Use `--metric native` with `scripts/infer.py` and an equivalent relative manifest
for native observations. `metrics/native.py` uses FP64 accumulation, Gaussian
11 × 11 SSIM (sigma 1.5), and a zero prediction against nonzero reference counts
as 90 degrees in SAM. It reports the arithmetic mean of per-scene scores, not
PSNR calculated from pooled MSE. No raw prediction clipping is allowed.

SpectralWaste's observed RGB and train-only empirical affine color relationship
must not be represented as a physical SRF. The four-center native baseline
subsets on the last three domains differ from full-scene transfer summaries;
do not combine their counts or call their training budgets matched.

## PCB ablations

The three trained variants use 33 PCB training scenes, seed 0, 24,300 steps and
the same nine validation scenes: pcb03/06/28/32/36/41/42/45/47. Their original
SIEVE metric implementation is `sieve3/metrics.py` (uniform-7 SSIM and its original
SAM handling), not the native Gaussian-11 metric. These are validation ablations,
not six-domain test experiments. No claimed score exists for joint double removal.
