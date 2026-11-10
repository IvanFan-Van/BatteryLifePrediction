# Tongji data pipeline

## Prepare and inspect

From the project root:

```bash
uv sync --locked
uv run python scripts/preprocess_tongji.py --output data/processed/tongji_v1
uv run python -m unittest discover -s tests -v
```

The preprocessing command reads trusted BatteryLife `.pkl` files from
`data/raw/Tongji`. It refuses to overwrite an existing output directory; choose a
new version directory when changing rules. Open `notebooks/analysis/Tongji.ipynb` with the
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

Use [the Dataset guide](dataset.md) to build `TongjiTaskDataset`, inspect masks,
and load variable-length observations. Future labels follow real cycle IDs
`last_cycle_id + 1` through `last_cycle_id + forecast_horizon_cycles`. Missing records, invalid SOH, and out-of-range cycles
have `target_soh_valid=False`. Known protocol settings provide the future conditions.
The sample diagnostics `raw_point_position` and `segment_bounds` trace observations
back to the source.

## 按预测任务索引与观测课程

[小规模课程配置](../configs/tongji_curriculum_pilot.yaml) 使用两块真实电芯、
四步 SOH 标签和两个 epoch。带 `data.observation_schedule` 的训练配置使用
`TongjiTaskDataset`。`len(dataset.tasks)` 是可用的 `(cell_id, last_cycle_position)`
任务数，`len(dataset)` 是任务数乘以 `samples_per_task`。
`dataset[i]` 固定读取 epoch 0 的第 `i` 份样本，`dataset[i, 1]` 读取
同一个样本下标在 epoch 1 的观测。组 batch 使用 `collate_tongji`。

`cycle_fraction` 和 `segment_fraction` 用 `0.5` 这类小数表示比例。循环配额向上取整，
并始终保留最后观测循环。连续片段长度向下取整，至少保留一个点。
`value_keep: 4` 最多保留四个有效读数。`value_keep: 0.9` 按每个采样点的有效读数数目
计算保留配额并向下取整。`value_keep: 1.0` 保留全部有效读数，`value_keep: 1` 最多保留一个。
课程只改变训练输入。训练评估、验证和测试全程使用最后一个阶段的固定视图。

可以用 `TongjiTaskDataset.from_config("configs/tongji_curriculum_pilot.yaml")`
单独创建 `data.cell_ids.train` 中所有电芯的 Dataset。训练、验证和测试的
切分由训练入口负责。每组可配置多个电芯。
完整配置模板见 [config.md](config.md)。

从项目根目录运行以下命令可以检查配置校验、mask、padding 和多进程重放，
也可以单独跑两轮 CPU pilot。

~~~powershell
uv run --locked python -m unittest discover -s tests -p test_tongji_tasks.py -v
uv run --locked python -m src.core.train --config configs/tongji_curriculum_pilot.yaml
~~~
