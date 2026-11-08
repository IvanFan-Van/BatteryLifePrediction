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
    val_ratio: 0.2
    split_seed: 2026
    eval_seed: 2027

    # TongjiTaskDataset 的参数。
    root: data/processed/tongji_v1
    cell_ids:
        train: [Tongji1_CY25-025_1--4]
        test: [Tongji1_CY25-025_1--5]
    horizon: 4  # 每个任务预测的 SOH 轮数。
    max_last_observed_cycle: 6  # 真实循环编号。null 表示没有上限。
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
`cell_ids.val` 时，三组电芯必须两两独立。省略它时，按 `data.val_ratio`
从每块训练电芯划出部分预测起点做验证。此时训练和验证可能共享历史片段或未来标签。

也可以直接传入 Dataset 的参数。

```python
dataset = TongjiTaskDataset(
    root="data/processed/tongji_v1",
    cell_ids=["Tongji1_CY25-025_1--4"],
    horizon=4,
    max_last_observed_cycle=6,
    max_epochs=1,
    observation_schedule=[
        dict(start_epoch=0, cycle_fraction=0.5, segment_fraction=0.5, value_keep=0.9),
    ],
)
```

第一个阶段须从 epoch 0 开始，各阶段的 `start_epoch` 依次增加且小于
`data.max_epochs`。TongjiTaskDataset 的训练总轮数也来自这个字段。
验证和测试使用最后一个阶段的固定视图。
