# Historical frozen-feature estimation kernels

These kernels implement the earlier measurement/graph/covariance estimator family.
They are not the learned DINOv2 + LoRA model. They take caller-provided tensors and
configuration; importing a kernel does not dispatch an experiment. The learned
model used for the matched-79 row is `sieve3` with `stageA_30000.pt`.
