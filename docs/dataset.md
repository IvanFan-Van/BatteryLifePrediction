# TongjiTaskDataset 使用指南

`TongjiTaskDataset` 用于从处理后的 Tongji 电芯数据构造 SOH 预测样本。
下面的示例从项目根目录运行，使用项目的 Python 环境和
`data/processed/tongji_v1` 下的处理数据。目录中应有 `manifest.json`、
`schema.json` 和 `cells/`。完整训练配置见 [config.md](config.md)，
处理数据的字段和来源见 [tongji_data.md](tongji_data.md)。
示例按顺序运行，后面的代码块会复用前面创建的 Dataset 和变量。

## 创建 Dataset

### 1. 从 YAML 文件读取

```python
from src.data import TongjiTaskDataset

config_path = "configs/tongji_curriculum_pilot.yaml"
dataset = TongjiTaskDataset.from_config(config_path)
print(len(dataset), dataset.max_epochs)
```

`from_config` 默认选取 `data.cell_ids.train` 里的全部电芯，并读取
`data.root`、`data.horizon`、`data.max_last_observed_cycle`、
`data.max_epochs`、`data.observation_schedule` 和顶层的 `seed`。
构造得到的是这些电芯的所有合格预测任务。需要应用训练、验证和测试切分时，
使用后面的 `build_datasets` 示例。

### 2. 传入已读取的配置字典

这种方式适合在 Notebook 里复用配置，再指定要检查的电芯或预测起点。
下面只读取测试电芯的第 4 和第 6 轮，随机种子使用 `data.eval_seed`。

```python
from src.utils.files import read_yaml

config = read_yaml(config_path)
cell_id = "Tongji1_CY25-025_1--5"

subset = TongjiTaskDataset.from_config(
    config,
    cell_ids=[cell_id],
    anchor_cycle_ids={cell_id: [4, 6]},
    seed=config["data"]["eval_seed"],
)
print(subset.tasks)
subset.close()
```

`cell_ids` 接受多个电芯 ID。传入 `anchor_cycle_ids` 时，需要为每个选中的
电芯提供一组真实循环编号。也可以把这些覆盖参数直接和 YAML 路径一起传给
`from_config`。

### 3. 直接传入构造参数

这种方式适合独立的检查脚本。观测日程直接传入一组字典。

```python
direct = TongjiTaskDataset(
    root="data/processed/tongji_v1",
    cell_ids=["Tongji1_CY25-025_1--4"],
    horizon=4,
    max_last_observed_cycle=6,
    max_epochs=1,
    observation_schedule=[
        {
            "start_epoch": 0,
            "cycle_fraction": 0.5,
            "segment_fraction": 0.5,
            "value_keep": 0.9,
        },
    ],
    seed=42,
)
print(direct.tasks)
direct.close()
```

| 参数 | 含义 |
| --- | --- |
| `root` | 处理后的数据目录。相对路径从运行时的工作目录解析。 |
| `cell_ids` | 电芯 ID 集合，至少包含一个电芯，且 ID 不能重复。 |
| `horizon` | 预测未来多少轮 SOH，默认 32。 |
| `max_last_observed_cycle` | 最后观测循环的真实编号上限，包含上限本身。默认 `None`。 |
| `anchor_cycle_ids` | 可选的起点白名单，格式为 `{cell_id: [cycle_id, ...]}`。 |
| `max_epochs` | 可取样的 epoch 总数，合法索引为 `0` 到 `max_epochs - 1`。 |
| `observation_schedule` | 按 `start_epoch` 排序的阶段配置，第一个阶段从 epoch 0 开始。 |
| `seed` | 观测视图的随机种子，直接构造时默认 42。 |

日程里的 `cycle_fraction` 决定保留多少轮，向上取整并保留最后观测轮。
`segment_fraction` 决定每轮取多长的连续片段，向下取整且至少留一个点。
`value_keep` 为小数时按每点实际有效读数的数量计算保留个数并向下取整，
为整数时表示最多保留几个。比如 `0.9` 表示保留 90%，`1.0` 表示全保留，
整数 `1` 表示最多保留一个。

## 查看任务与样本

一个任务由 `(cell_id, last_observed_cycle)` 标识。`len(dataset)` 是任务数，
`dataset.tasks[i]` 是第 `i` 项任务，`anchor_cycles(cell_id)` 返回该电芯
的可用预测起点。一个起点必须有可用的充电采样点，并且之后 `horizon` 轮内
至少有一个有效 SOH 标签。

`dataset[i]` 使用 epoch 0，`dataset[i, epoch]` 使用指定 epoch。
同一任务和 epoch 在相同配置、数据和 seed 下可以重放。换 epoch 时，抽中的
历史循环、片段和读数可能变化。标签始终对齐到真实循环号
`last_observed_cycle + 1` 至 `last_observed_cycle + horizon`。

```python
import torch

train_cell = "Tongji1_CY25-025_1--4"
i = dataset.tasks.index((train_cell, 6))
full = dataset[i]
sparse = dataset[i, 1]

print(dataset.anchor_cycles(train_cell))
print(full["cycle_id"].tolist(), sparse["cycle_id"].tolist())
print(sparse["segment_bounds"].tolist())
print([tuple(segment.shape) for segment in sparse["values"]])
print(sparse["target_cycle_id"][sparse["target_valid"]])
print(sparse["target_soh"][sparse["target_valid"]])

torch.testing.assert_close(full["target_cycle_id"], sparse["target_cycle_id"])
torch.testing.assert_close(full["target_soh"], sparse["target_soh"])
torch.testing.assert_close(full["target_valid"], sparse["target_valid"])
torch.testing.assert_close(sparse["value_visible"], dataset[i, 1]["value_visible"])
```

| 字段 | 如何解读 |
| --- | --- |
| `cell_id`、`anchor_cycle` | 电芯 ID 和最后观测轮的真实编号。 |
| `cycle_id`、`values` | 抽中的循环号和对应的充电片段。每个片段形状为 `(T_i, F)`，当前 Tongji 有五个通道。 |
| `value_valid`、`value_visible` | 前者标记原始有效读数，后者标记本次保留的读数。形状与各片段的 `values` 一致。 |
| `time_in_cycle` | 每个采样点相对完整循环起点的秒数。 |
| `segment_bounds`、`raw_sample_index` | 前者是处理后单轮充电片段内的左闭右开边界，后者用于回查原始循环数组。 |
| `static_*`、`history_*`、`future_u*` | 电芯静态属性、历史工况和预测区间的已知工况，以及对应的可用性。 |
| `target_cycle_id`、`target_soh`、`target_valid` | 长度为 `horizon` 的目标编号、SOH 和标签有效性。计算误差时只使用 `target_valid` 为真的位置。 |

## 用 DataLoader 组 batch

### 固定视图，检查 padding 和 mask

单条样本的循环数和片段长度可以不同。`collate_tongji` 把它们补齐到当前
batch 的最大长度，并生成 `point_valid` 和 `cycle_valid`。用普通整数索引的
DataLoader 会读取 epoch 0。下面显式请求 epoch 1 的两条任务，便于观察补齐。

```python
from torch.utils.data import DataLoader

from src.data import collate_tongji

loader = DataLoader(
    dataset,
    batch_size=2,
    sampler=[(0, 1), (i, 1)],
    collate_fn=collate_tongji,
    num_workers=0,
)
batch = next(iter(loader))

print(batch["values"].shape)       # (B, N_max, T_max, F)
print(batch["point_valid"].shape)  # (B, N_max, T_max)
print(batch["cycle_valid"])       # (B, N_max)
print(batch["target_soh"].shape)   # (B, horizon)

assert torch.equal(batch["cycle_valid"], batch["point_valid"].any(dim=-1))
assert not batch["value_visible"][~batch["point_valid"]].any()
```

只想手动拼几条样本时，可以直接调用
`collate_tongji([dataset[0, 1], dataset[i, 1]])`。

### 随 epoch 切换观测日程

独立 PyTorch 循环可以用 `TongjiTaskSampler`。它按 epoch 打乱任务顺序，
每个任务在一轮完整遍历中访问一次，并把 `(index, epoch)` 交给 Dataset。
在开始遍历 DataLoader 前调用 `set_epoch`。使用 sampler 时，由 sampler
决定顺序，DataLoader 保持 `shuffle=False`。

```python
from src.data import TongjiTaskSampler

sampler = TongjiTaskSampler(dataset, seed=config["seed"])
train_loader = DataLoader(
    dataset,
    batch_size=config["data"]["batch_size"],
    sampler=sampler,
    collate_fn=collate_tongji,
    num_workers=0,
)

for epoch in range(dataset.max_epochs):
    sampler.set_epoch(epoch)
    for batch in train_loader:
        # 在这里调用模型并计算 loss。
        assert batch["target_valid"].any(dim=-1).all()
```

如果想顺序遍历一个固定阶段，可以传
`TongjiTaskSampler(dataset, fixed_epoch=dataset.max_epochs - 1)`。

### 使用项目的训练、验证和测试切分

项目训练入口用 `build_datasets` 构造任务切分，再通过 `make_loader` 应用
配置中的 batch 大小和 worker 数。显式提供 `data.cell_ids.val` 时，训练、
验证和测试电芯两两独立。省略它时，从每块训练电芯划出一部分预测起点做验证。

```python
from src.data.loading import build_datasets, make_loader

datasets, splits = build_datasets(config)
try:
    train_loader = make_loader(datasets["train"], config, shuffle=True)
    val_loader = make_loader(datasets["val"], config)
    test_loader = make_loader(datasets["test"], config)

    train_loader.sampler.set_epoch(1)
    print(next(iter(train_loader))["values"].shape)
    print(next(iter(val_loader))["target_soh"].shape)
    print(next(iter(test_loader))["cell_id"])
finally:
    for split_dataset in datasets.values():
        split_dataset.close()
```

`make_loader` 给验证和测试使用最后一个阶段的固定视图。
在项目的 Lightning 训练入口中，训练 sampler 会收到当前 epoch。
Notebook 中可以先用 `num_workers=0` 来检查样本。用完后调用
`dataset.close()` 释放 Dataset 当前缓存的内存映射，在 Windows 上移动或删除
处理数据之前尤其有用。

```python
dataset.close()
```
