# Matched PCB Stage-A ablations

The archived training entry and guarded trainer are included as source evidence.
They require their original controller callback and are not standalone launchers.
Use the portable `scripts/train_v3.py` with a frozen `configs/checkpoints/` config
to run the same model/loss/optimizer equations in a new output directory.

Completed arms: full, no_coupling, no_hsi_adapter; each seed 0, PCB 33 training
scenes, batch 8, crop 128, 24,300 updates. Neither this budget nor its uniform-7
SSIM metric is interchangeable with the primary native experiment.
The double-removal arm failed budget admission before production; no trained
double-removal checkpoint or score is available. Inference-only interventions
share the full checkpoint and are not retrained ablations.
