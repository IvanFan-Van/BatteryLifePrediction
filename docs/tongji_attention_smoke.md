# Tongji hierarchical attention curriculum pilot

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
  reads SOH. The curriculum pilot uses a four-cycle horizon.
- Main loss is future-trajectory MSE, normalized per sample's valid target count.
  There is no extra reconstruction, latent constraint, monotonicity, or discount loss.

The model supports static fusion at cycle level (`cycle`) and at both
sample-point and cycle levels (`early`). The curriculum pilot uses `cycle`.

## Data and interpretation

The pilot trains on `Tongji1_CY25-025_1--4` and tests on the independent cell
`Tongji1_CY25-025_1--5`. It considers observation anchors through cycle 6.
Epoch 0 uses all available observations. Epoch 1 retains half the available
cycles and half of each selected charge segment, and keeps 90% of each point's
valid readings rounded down. Evaluation uses a fixed view from the final stage.

Validation holds out forecast anchors from the training cell. History and future
labels can overlap across the training and validation anchors. The pilot fits
normalization from the training cell. For independent-cell validation, set
`data.cell_ids.val` using the [configuration guide](config.md).

## Run

```bash
uv run python -m unittest discover -s tests -v
uv run --locked python -m src.core.train --config configs/tongji_curriculum_pilot.yaml
uv run --locked python scripts/queue_tongji.py --config configs/tongji_curriculum_pilot.yaml
uv run --locked python -m src.core.train --resume outputs/<run-id>
```

The CPU pilot runs two epochs with limited batches. Validation MSE selects the
checkpoint used for the final fixed-view train, validation, and test metrics.
Training epoch losses are available in CSV and TensorBoard logs.

Each run writes the effective config, splits, normalization, gradient diagnostics,
checkpoints, metrics, and prediction plots under `outputs/`. The queue command
also saves a copy of the Python source before submitting to Pueue. W&B runs offline.

Metric definitions: MSE and MAE average within each sample's valid horizon, then
average over samples. RMSE is the square root of that aggregate MSE. SOH is a
fraction; multiply RMSE/MAE by 100 to express errors in percentage points.
