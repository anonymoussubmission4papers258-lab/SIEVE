# Reproducibility and release transformations

The `sieve3` implementation is copied from the source pinned during the completed
primary evaluation. Model equations, slot/codebook heads, spectral-twin attention,
LoRA layers, measurement solve, losses and audit algorithms are retained. Frozen
reference kernels are separately named under `legacy_frozen/`.

The portable inference wrapper is new packaging code. Its observation-only call
to `audit.run`, primary checkpoint, default mode, 256 tile, FP32 and TF32-off settings
match the accepted trained-model comparison. Scientific references are only
opened by the separate scoring function. File paths are portable; machine-specific
process ownership, disk serials, prior task deadlines and desktop guards are not
redistributed as reproduction requirements.

The original training scripts and the guarded PCB training source are provided.
The original experiments ran with versioned resource controllers and strict
budgets. Generic training scripts are not a claim that copying one command exactly
reconstructs the original host environment or the paused-and-resumed student
history. Model configurations and complete numeric recovery states are supplied
to make those differences inspectable. Never use partial historical states as
completed-model results.

Checkpoint export retains every model tensor, optimizer value, scheduler value,
Python/NumPy/Torch RNG state, step counter, numeric training log and cost. Only
string metadata that embeds local paths is made portable. Exported files therefore
have new file hashes; both source and release hashes appear in the manifest.
The loader uses PyTorch's restricted `weights_only` mode plus the specific NumPy
containers needed for RNG states. Load only verified files from this release.

The archive preparation did not retrain or rescore the experiments. Its validation
covers source syntax, existing mathematical tests, anonymous metadata, strict
model loading, exported tensor equality and a synthetic CPU inference smoke test.
It does not claim a fresh rerun of every training or test scene.

LLM assistance was used in developing code, experimental tooling and this release's
documentation/packaging. Authors remain responsible for scientific claims and
should give the full required research-specific disclosure in their submission.
