# Full Tongji static-fusion ablation

This study is launched with:

```bash
uv run python scripts/tongji_study.py launch --config configs/tongji_ablation.yaml
```

The study root is created under `outputs/tongji-static-ablation-<Hong-Kong timestamp>/`.
Its `report.md` updates during execution and contains the final comparison. The
controller and workers run through an isolated Pueue daemon; two experiment slots
are configured with `pueue parallel --group experiments 2`. Closing the chat does
not stop the daemon. The controller prevents automatic system sleep while active
and restores the normal sleep policy on completion; explicit shutdown or sleep
still interrupts computation.

## Variants

- `cycle`: the current architecture; static tokens join cycle-level attention.
- `early`: reuse the same independently encoded static tokens in within-cycle
  attention **as well as** cycle-level attention. No additional parameters are
  introduced. This isolates adding the earlier interaction path, rather than
  simultaneously removing the existing fusion route.

All other architectural dimensions remain fixed in the configured search. Five
observation encoders retain 1 -> 8 -> 4 nonlinear maps and independent UNK values.
Matched variants have equal parameter counts (37,297 for the current vocabulary).

## Protocol

All 130 cells are split by cell, stratified within 11 material/protocol groups:
72 train, 29 validation, 29 test with split seed 2026. Each group has cells in all
three partitions. No cell identity crosses partitions. All eligible anchors over
the full recorded cell histories are enumerated each epoch; normalization is fit
once using the training cells only. Validation/test views use seed 2027 and are
identical across architectures and final training seeds.

Each variant independently searches six learning-rate/weight-decay combinations:
LR {0.0003, 0.001, 0.003}, weight decay {0, 0.0001}. All six receive two full-data
epochs. Each variant's two best candidates then receive six epochs from scratch.
The best refined validation candidate is selected separately per architecture.
No coarse or refinement trial instantiates/evaluates the test dataset.

Selected configurations are trained with paired seeds 42,43,44 for up to 20 epochs,
with validation early-stopping patience 5 and minimum 8 epochs. The scheduler may
reduce the SAME final epoch budget for both variants based on measured epoch time
and the morning deadline. Actual completed epochs, failed jobs, and chosen
hyperparameters are reported, rather than substituting incomplete runs silently.
The saved YAML files preserve the original study deadline. Set `deadline_local` to a future time before launching a new study.

Primary selection metric is equal-cell-weight validation MSE: each anchor's valid
future-label MSE is averaged within its cell, then averaged over cells. This avoids
long-lived cells dominating hyperparameter selection. Reports include cell-macro
RMSE/MAE, anchor-weighted metrics, per-cell results and material-family breakdowns.
Seed variation is descriptive, not evidence of robustness across new cell splits.

Input sampling stays at 1--6 historical charge segments, 16--64 points per segment,
with a fixed 32-cycle prediction horizon and no synthetic sensor masking. This is
not an entire trajectory-to-EOL benchmark or held-out-material/protocol evaluation.

## Execution and provenance

Windows local GPU use is authorized, including GPU 0. The supplied config uses the
RTX 4060 Laptop GPU with two concurrent jobs; farady Linux GPU 0/1 remain forbidden.
GPU reserve must remain at least 20%; the pipeline pilot and formal memory check
are saved as JSON. W&B remains offline for every worker.

Configuration and Python source are frozen before queuing. Runtime package files
are represented by copies of pyproject.toml and uv.lock. Source/data hashes,
splits, normalization and full settings are recorded. Trial result.json files
include software versions, peak allocated CUDA memory, durations and checkpoints.
No original data or existing notebook is modified. Access tokens are excluded from
the environment sent to this study's Pueue tasks, and raw Pueue status JSON must
never be printed because it can contain environment values.

The launcher also supports a pipeline test configuration with limited batches:

```bash
uv run python scripts/tongji_study.py launch --config configs/tongji_ablation_pilot.yaml
```

Pilot outputs are separate from formal experimental results and are not part of
model selection in the formal study.
