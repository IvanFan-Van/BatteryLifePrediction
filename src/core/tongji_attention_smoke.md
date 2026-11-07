# Tongji hierarchical attention smoke experiment

The initial model has **37,297 trainable parameters** with the committed dataset vocabulary.

## Architecture

- Five independent observation encoders: Linear(1, 8), SiLU, Linear(8, 4).
  Every observation field has its own learned UNK vector.
- Time has a separate 1 -> 8 -> 8 encoder; point features project to width 32.
- One within-cycle Transformer block, 2 heads, FFN width 64, summarizes each
  charge segment through a learned CYCLE token. Only genuine cycle slots are packed.
- Every static field has an independent encoder: a separate categorical table
  (its row 0 is a trainable field-specific UNK), or an independent numerical MLP
  and UNK. Eight width-32 static tokens are included in cycle-level attention.
- Two cycle-level Transformer blocks fuse STATE, static field tokens, and cycle
  tokens with true cycle ID and anchor-relative position encodings. A static- and
  history-conditioned STATE query projects to a 16-dimensional predictive state.
- A shared residual MLP dynamics predicts one state increment per real future
  cycle, conditioned on the known protocol and static context. A shared MLP head
  reads SOH without sigmoid clipping. Horizon is initially 32 cycles.
- Main loss is future-trajectory MSE, normalized per sample's valid target count.
  There is no extra reconstruction, latent constraint, monotonicity, or discount loss.

**Deferred ablation:** repeat static field tokens inside the first, within-cycle
attention stage so materials can influence individual-segment compression.
This experiment uses static fusion in the second stage only; it is not implemented
as an additional mode yet. A fair ablation should reuse splits, view seeds,
normalization, dynamics/head settings, and comparable parameter counts.

## Data and interpretation

Train/validation: `Tongji1_CY25-025_1--4` (331 recorded cycles).
Test: `Tongji1_CY25-025_1--5` (338 recorded cycles).
They share the recorded materials, 3.5 Ah rated capacity, voltage limits, 25 C
ambient setpoint, 0.25C charge / 1C discharge protocol, and post-charge rest.
Unknown electrolyte/separator fields are unknown in both; no claim is made that
unreported manufacturing differences are absent.

All eligible anchors over the full cell history are enumerated, not capped at 100.
A fixed split places 264 anchors in training and 66 in validation. Each training
anchor gets a new random history/segment view each epoch. Validation and the 337
independent test-cell anchors use fixed views. Input has 1--6 historical cycles,
16--64 contiguous charging samples per cycle; sensor masking is initially off.
Anchors without any reliable future label are excluded. Historical observations
and future labels can overlap across training/validation anchors of the same cell;
this validation is a smoke diagnostic, not independent-cell generalization.

Numerical normalization is fitted only on the training cell, never on test data or
SOH. All available charge measurements of that training cell are used for fitting;
validation is consequently same-cell validation, not a held-out preprocessing set.
No total cell lifespan or future measured curves are model inputs. Known fixed
protocol values drive future dynamics. Predictions cover a fixed 32-cycle window,
not an entire lifetime-to-EOL rollout yet.

## Run

```bash
uv run python -m unittest discover -s tests -v
uv run python -m src.core.train --config configs/tongji_pilot.yaml
uv run python scripts/queue_tongji.py --config configs/tongji_smoke.yaml
# To target an explicitly selected local daemon:
uv run python scripts/queue_tongji.py --config configs/tongji_smoke.yaml
# Resume an interrupted run with its saved effective configuration:
uv run python -m src.core.train --resume outputs/<run-id>
```

Pilot uses 1 epoch and 2 batches per stage. Full smoke uses all batches for 30
epochs. Best checkpoint is selected only by validation MSE; test is evaluated once
after fitting. Fixed-view train, validation, and test metrics then use that same
best checkpoint. Training epoch losses are separately available in CSV/TensorBoard.

The provided configs use CPU with four threads. GPU runs require explicit
CUDA_VISIBLE_DEVICES selecting permitted physical GPU 2/3 per AGENTS.md. W&B is
forced offline in both environment and logger settings, with no model upload.

Every run writes config.yaml, splits.json, normalization.json, provenance.json,
source/, gradient_check.json, checkpoints/, metrics.json, per-stage predictions.npz,
learning_curves.png, trajectories.png, and offline W&B/TensorBoard/CSV logs.
Queue submission saves config and Python source before enqueueing; tasks run that
frozen source so later workspace edits cannot change a queued experiment.
Generated outputs stay under ignored outputs/. Original data and notebook are untouched.

Metric definitions: MSE and MAE average within each sample's valid horizon, then
average over samples. RMSE is the square root of that aggregate MSE. SOH is a
fraction; multiply RMSE/MAE by 100 to express errors in percentage points.
