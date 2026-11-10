# Tongji 训练配置

将下面的模板复制为项目根目录的 `config.yaml`。模板用两块 Tongji 电芯、
四步 SOH 标签和两个 CPU epoch 跑通训练、验证与测试。

```yaml
experiment:
    project: battery-life-prediction
    group: tongji-attention-single-cell
    name: tongji-observation-curriculum-pilot
seed: 42
data:
    # 数据加载与切分的通用参数。
    batch_size: 2
    num_workers: 0
    val_task_fraction: 0.2
    split_seed: 2026
    eval_seed: 2027

    # TongjiTaskDataset 的参数。
    root: data/processed/tongji_v1
    cell_ids:
        train: [Tongji1_CY25-025_1--4]
        test: [Tongji1_CY25-025_1--5]
    forecast_horizon_cycles: 4  # 每个任务预测的 SOH 轮数。
    max_cycle_position: 6  # 每个电芯从 1 开始计数。null 表示没有上限。
    samples_per_task: 1  # 每个训练任务每轮的样本数。
    max_epochs: 2  # 同时用于日程和 Trainer。

    # 运行时观测规则，epoch 从 0 开始。
    observation_schedule:
        - start_epoch: 0
          cycle_fraction: 1.0
          segment_fraction: 1.0
          value_keep: 1.0
        - start_epoch: 1
          cycle_fraction: 0.5
          segment_fraction: 0.5
          value_keep: 0.9
model:
    d_model: 32
    state_dim: 16
    feature_dim: 4
    scalar_hidden: 8
    time_dim: 8
    heads: 2
    feedforward_dim: 64
    point_layers: 1
    cycle_layers: 2
    dropout: 0.0
    context_dim: 16
    dynamics_hidden: 64
    head_hidden: 32
    time_scale_s: 10000.0
    cycle_scale: 100.0
    transition_scale: 0.1
    initial_soh_bias: 0.9
    static_fusion: cycle
optimizer:
    lr: 0.003
    weight_decay: 0.0001
trainer:
    accelerator: cpu
    cpu_threads: 4
    limit_train_batches: 2
    limit_val_batches: 1
    limit_test_batches: 2
    gradient_clip_val: 1.0
    evaluation_batches: 2
```

在项目根目录运行以下命令。

```powershell
uv run --locked python -m src.core.train --config config.yaml
```

在 Notebook 里可以直接读取同一个配置文件。

```python
from src.data import TongjiTaskDataset

dataset = TongjiTaskDataset.from_config("config.yaml")
sample = dataset[0, 1]
```

`from_config` 默认读取 `data.cell_ids.train` 列表中的所有电芯。
要单独查看其他电芯，可以传入 `cell_ids=["Tongji1_CY25-025_1--5"]`。
这个入口也接受已读取的配置字典。

训练入口使用 `cell_ids.train` 中的电芯做训练并拟合归一化，用
`cell_ids.test` 中的独立电芯做测试。每组都可以包含多个电芯。配置
`cell_ids.val` 时，三组电芯必须两两独立。省略它时，按 `data.val_task_fraction`
从每块训练电芯划出部分预测起点做验证。此时训练和验证可能共享历史片段或未来标签。

也可以直接传入 Dataset 的参数。

```python
dataset = TongjiTaskDataset(
    root="data/processed/tongji_v1",
    cell_ids=["Tongji1_CY25-025_1--4"],
    forecast_horizon_cycles=4,
    max_cycle_position=6,
    samples_per_task=1,
    max_epochs=1,
    observation_schedule=[
        dict(start_epoch=0, cycle_fraction=0.5, segment_fraction=0.5, value_keep=0.9),
    ],
)
```

第一个阶段须从 epoch 0 开始，各阶段的 `start_epoch` 依次增加且小于
`data.max_epochs`。TongjiTaskDataset 的训练总轮数也来自这个字段。
训练评估、验证和测试使用最后一个阶段的固定观测，每个任务一份。

## 旧配置迁移

| 旧字段 | 当前字段 |
| --- | --- |
| `data.horizon` | `data.forecast_horizon_cycles` |
| `data.val_ratio` | `data.val_task_fraction` |
| `data.max_last_observed_cycle` | `data.max_cycle_position` |

第三项需要根据实验设计重新取值。旧值限制真实循环号，新值限制每个电芯的
循环记录位置。例如循环号从 2 开始且连续时，旧上限 6 对应新位置 5。
新实验保存的切分按位置记录。历史实验可用原始配置、切分和对应的代码版本恢复。
