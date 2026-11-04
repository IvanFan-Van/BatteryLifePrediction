# Tongji data pipeline

## Prepare and inspect

From the project root:

```bash
uv sync --locked
uv run python scripts/preprocess_tongji.py --output data/tongji_v1
uv run python -m unittest discover -s tests -v
```

The preprocessing command reads trusted BatteryLife `.pkl` files from
`raw_data/Tongji`. It refuses to overwrite an existing output directory; choose a
new version directory when changing rules. Open `notebooks/Tongji.ipynb` with the
project virtual environment and run all cells. Each run picks a random cell and
cycle, records its seed, compares original and extracted curves, and displays
random training views, padding, masks, future labels, and a masked MSE example.
The original notebook's exploratory source is retained in its final markdown cell.

## Offline contract

`schema.json` defines ordered field names and units; `category_mapping.json`
contains per-field categorical IDs (`<UNK>` is 0); `manifest.json` is written last
and marks a complete conversion. Category IDs enumerate dataset metadata, not
learned statistics. Numerical values remain in physical units; fit normalization
on training cells only. A future unseen category maps to `<UNK>`.

Each `cells/<cell_id>/` contains:

| File | Shape | Meaning |
| --- | --- | --- |
| `values.npy` | `(L,5)` | Concatenated charge V, I, Qcharge, Qdischarge, temperature |
| `value_valid.npy` | `(L,5)` | Finite, provided measurements; missing numbers filled with 0 |
| `time_in_cycle.npy` | `(L,)` | Seconds relative to full-cycle start, float64 |
| `raw_sample_index.npy` | `(L,)` | Index in the original full cycle, for QA |
| `cycle_offsets.npy` | `(C+1,)` | Half-open boundaries in the concatenated point arrays |
| `cycle_id.npy` | `(C,)` | Original cycle numbers, including gaps |
| `static_categorical.npy` | `(5,)` | Form, cathode, anode, electrolyte, separator IDs |
| `static_numeric.npy` | `(3,)` | Nominal Ah, upper voltage limit, lower voltage limit |
| `static_valid.npy` | `(8,)` | Validity in categorical-then-numeric order |
| `conditions.npy` | `(C,4)` | Ambient degC, charge C-rate, discharge C-rate, post-charge rest s |
| `conditions_valid.npy` | `(C,4)` | Protocol information availability |
| `soh.npy` / `soh_valid.npy` | `(C,)` | Discharge capacity / nominal Ah and label reliability |
| `metadata.json` | -- | Raw SHA-256, source, physical metadata and per-cycle label issues |

L is the sum of selected charging points; C is the number of source cycle records.
The longest contiguous run with `I > 0.005 * nominal_capacity_Ah` is retained as
the charge observation (threshold configurable). This suppresses idle-current
noise and preserves the original sample order; it can omit charge points below
that threshold. There is no resampling, interpolation, smoothing, or SOH clipping.
Temperature is genuinely missing for all local Tongji cells and remains unknown.

SOH uses the maximum positive cumulative discharge capacity only if there are at
least two discharging points, finite discharge voltage/capacity, a counter starting
within 5% of nominal Ah from zero, no counter decrease exceeding 0.1% of nominal
Ah, and discharge reaching the specified cutoff within 0.05 V (configurable).
Failures retain the cycle but invalidate its label. These checks are heuristic
quality controls, not an independent confirmation of the measurement. Full
source curves must remain available for manual review.

Conditions are published fixed protocol setpoints decoded from filenames and
[the original experiment](https://www.nature.com/articles/s41467-022-29837-w),
not future sensor measurements. Ambient temperature is not measured cell
surface temperature. Only post-charge rest is included, not an invented complete
rest schedule. BatteryLife's upstream time normalization may compress gaps;
`time_in_cycle` is not a raw calendar clock and cannot recover rest duration.

## PyTorch sampling

```python
from torch.utils.data import DataLoader
from battery_data import TongjiDataset, collate_tongji

train = TongjiDataset(
    'data/tongji_v1', cell_ids=['Tongji1_CY25-025_1--1'],
    samples_per_cell=128, history_cycles=(1, 8), segment_points=(16, 128),
    horizon=32, max_anchor_cycle=100,
    channel_mask_probability=0.0, seed=42,
)
loader = DataLoader(train, batch_size=4, collate_fn=collate_tongji,
                    num_workers=0)
batch = next(iter(loader))
```

An index selects a cell/view slot. Training calls draw new views; slots are not
independent cells. A random anchor k is always included, and earlier eligible
cycles are sampled without replacement. Each selected cycle yields one contiguous
random charge segment. Minimum segment size is enforced; cells with no eligible
anchor produce a clear error. `max_anchor_cycle=None` removes the early-cycle cap.

Future targets are exactly cycle IDs k+1..k+H. Missing source records, unreliable
SOH, and out-of-range future labels have `target_valid=False` and finite zero
placeholders. At least one future label must be valid. Targets use full discharge
records; input extraction never uses future measured curves. Future protocols
repeat the known cell protocol; history protocols also cover unobserved cycles
between selected observations. This implementation is for constant-protocol Tongji
experiments, not arbitrary changing real-world operating conditions.

Batch values have shape `(B,Nmax,Tmax,5)`, padded only to local batch maxima.

* `point_valid (B,N,T)`: real point versus PAD.
* `value_valid (B,N,T,5)`: source measurement exists and is finite.
* `value_visible (B,N,T,5)`: this augmentation permits the measurement.

With channel augmentation disabled, visibility is True at every genuine point,
even where a natural measurement is missing. Optional augmentation hides entire
channels independently per segment; hidden numerical values are scrubbed to 0.
PAD has zero values and all three masks False. The model must distinguish real
unknown measurements (UNK embedding) from PAD (excluded from attention/pooling).
`cycle_valid = point_valid.any(-1)` is derived. There are no embeddings in this
Dataset; learn them in the model.

`history_u_valid`, `future_u_valid`, `static_valid`, and `target_valid` describe
availability in their own arrays, not randomized augmentation. History protocols
are separately padded; invalid entries never contribute. Diagnostic
`raw_sample_index` and `segment_bounds` are for audit, not intended model inputs.

Use `deterministic=True` for fixed per-index validation/test views. Set epoch with
`train.set_epoch(epoch)` for reproducible training sequences; if using persistent
workers, recreate the loader after changing epoch. DataLoader worker seeds are
respected. `dataset.close()` releases its cached mappings on Windows.

Split by cell before generating views. Same-cell fixed-view validation is useful
for a smoke run but is not independent-cell generalization. Keep test cells out
of normalization fitting and model selection. Only consume labels through their
validity mask; do not expose true end-of-life or the number of remaining recorded
cycles as model inputs.
