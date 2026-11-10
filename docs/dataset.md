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
print("任务数", len(dataset.tasks), "每轮样本数", len(dataset))
```

`from_config` 默认选取 `data.cell_ids.train` 里的全部电芯，并读取
`data.root`、`data.forecast_horizon_cycles`、`data.max_cycle_position`、`data.samples_per_task`、
`data.max_epochs`、`data.observation_schedule` 和顶层的 `seed`。
构造得到的是这些电芯的所有合格预测任务。需要应用训练、验证和测试切分时，
使用后面的 `build_datasets` 示例。

### 2. 传入已读取的配置字典

这种方式适合在 Notebook 里复用配置，再指定要检查的电芯或预测起点。
下面只读取测试电芯的第 4 和第 6 条循环记录，随机种子使用 `data.eval_seed`。

```python
from src.utils.files import read_yaml

config = read_yaml(config_path)
cell_id = "Tongji1_CY25-025_1--5"

subset = TongjiTaskDataset.from_config(
    config,
    cell_ids=[cell_id],
    cycle_positions_by_cell={cell_id: [4, 6]},
    seed=config["data"]["eval_seed"],
)
print(subset.tasks)
subset.close()
```

`cell_ids` 接受多个电芯 ID。传入 `cycle_positions_by_cell` 时，需要为每个选中的
电芯提供一组从 1 开始的循环位置。也可以把这些覆盖参数直接和 YAML 路径一起传给
`from_config`。

### 3. 直接传入构造参数

这种方式适合独立的检查脚本。观测日程直接传入一组字典。

```python
direct = TongjiTaskDataset(
    root="data/processed/tongji_v1",
    cell_ids=["Tongji1_CY25-025_1--4"],
    forecast_horizon_cycles=4,
    max_cycle_position=6,
    samples_per_task=3,
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
| `forecast_horizon_cycles` | 预测未来多少轮 SOH，默认 32。 |
| `max_cycle_position` | 最后观测位置的上限。`50` 表示只观察前 50 条记录。默认 `None`。 |
| `cycle_positions_by_cell` | 可选的任务位置筛选，格式为 `{cell_id: [position, ...]}`。 |
| `samples_per_task` | 每个任务每轮的样本数，默认 1。`from_config` 也可显式覆盖 YAML 中的值。 |
| `max_epochs` | 可取样的 epoch 总数，合法索引为 `0` 到 `max_epochs - 1`。 |
| `observation_schedule` | 按 `start_epoch` 排序的阶段配置，第一个阶段从 epoch 0 开始。 |
| `seed` | 观测视图的随机种子，直接构造时默认 42。 |

日程里的 `cycle_fraction` 决定保留多少轮，向上取整并保留最后观测轮。
`segment_fraction` 决定每轮取多长的连续片段，向下取整且至少留一个点。
`value_keep` 为小数时按每点实际有效读数的数量计算保留个数并向下取整，
为整数时表示最多保留几个。比如 `0.9` 表示保留 90%，`1.0` 表示全保留，
整数 `1` 表示最多保留一个。

## 查看任务与样本

一个任务由 `(cell_id, last_cycle_position)` 标识。每个电芯的循环按真实编号
排序后，从 1 开始确定位置。空充电片段和不合格的任务不会改变其他循环的位置。
例如，真实循环号为 `[4, 6, 7, 10]` 时，位置 `3` 对应编号 `7`。配置
`max_cycle_position: 3` 时，编号 `10` 的记录仍可提供 SOH 标签。

| API | 含义 |
| --- | --- |
| `dataset.tasks[task_position]` | 任务表的一项。Python 下标 `task_position` 从 0 开始。 |
| `dataset.cycle_positions(cell_id)` | 该电芯在当前 Dataset 中可用的最后观测位置。任务需要有充电采样点和至少一个有效未来标签。 |
| `len(dataset.tasks)` | 任务数。 |
| `len(dataset)` | 一轮样本数，等于任务数乘以 `dataset.samples_per_task`。 |
| `dataset[sample_position, epoch]` | 指定 epoch 的一份样本。`dataset[sample_position]` 默认使用 epoch 0。 |

样本下标从 0 开始。用 `divmod(sample_position, dataset.samples_per_task)`
得到任务下标和任务内的观测下标。例如，每个任务有 100 份观测时，
`dataset[137, 2]` 读取 epoch 2、任务下标 1、观测下标 37。
在相同配置、数据和 seed 下，相同索引可以重放。改变观测下标或 epoch
可以生成另一份随机观测，同一任务的 SOH 标签保持相同。标签按真实循环号
对齐，从 `last_cycle_id + 1` 到 `last_cycle_id + forecast_horizon_cycles`。

```python
import torch

train_cell = "Tongji1_CY25-025_1--4"
task_position = dataset.tasks.index((train_cell, 6))
sample_position = task_position * dataset.samples_per_task
full = dataset[sample_position]
sparse = dataset[sample_position, 1]

print(dataset.cycle_positions(train_cell))
print(sparse["last_cycle_position"], sparse["last_cycle_id"])
print(full["cycle_id"].tolist(), sparse["cycle_id"].tolist())
print(sparse["segment_bounds"].tolist())
print([tuple(segment.shape) for segment in sparse["values"]])
print(sparse["target_cycle_id"][sparse["target_soh_valid"]])
print(sparse["target_soh"][sparse["target_soh_valid"]])

torch.testing.assert_close(full["target_cycle_id"], sparse["target_cycle_id"])
torch.testing.assert_close(full["target_soh"], sparse["target_soh"])
torch.testing.assert_close(full["target_soh_valid"], sparse["target_soh_valid"])
torch.testing.assert_close(sparse["value_visible"], dataset[sample_position, 1]["value_visible"])
```

| 字段 | Batch 形状 | 如何解读 |
| --- | --- | --- |
| `cell_id` | 长度 `B` 的列表 | 电芯 ID。 |
| `last_cycle_position` | `(B,)` | 最后观测循环的位置，从 1 开始。 |
| `last_cycle_id` | `(B,)` | 最后观测循环的真实编号。 |
| `cycle_id` | `(B,N)` | 本次抽中的真实循环编号。 |
| `values` | `(B,N,T,F)` | 每点的电压、电流、充电容量、放电容量和温度。 |
| `value_valid` | `(B,N,T,F)` | 源数据是否提供了有效读数。 |
| `value_visible` | `(B,N,T,F)` | 本次观测是否提供该读数。 |
| `point_valid` | `(B,N,T)` | 是否是真实采样点。 |
| `cycle_valid` | `(B,N)` | 是否是真实循环片段，由 `point_valid.any(dim=-1)` 得到。 |
| `time_in_cycle_s` | `(B,N,T)` | 相对完整循环起点的秒数。 |
| `raw_point_position` | `(B,N,T)` | 原始完整循环中的采样点下标，从 0 开始。 |
| `segment_bounds` | `(B,N,2)` | 处理后单轮充电数组中的起止下标，左闭右开。 |
| `static_categorical` | `(B,5)` | 外形、正极、负极、电解液、隔膜的类别 ID。 |
| `static_numeric` | `(B,3)` | 额定容量、上限电压、下限电压。 |
| `static_valid` | `(B,8)` | 静态属性是否已知，按类别字段、数值字段排列。 |
| `history_cycle_id` | `(B,L)` | 历史工况对应的真实循环编号。 |
| `history_u` | `(B,L,4)` | 历史工况设定。 |
| `history_u_valid` | `(B,L,4)` | 历史工况设定是否已知。 |
| `future_u` | `(B,H,4)` | 预测区间的工况设定。 |
| `future_u_valid` | `(B,H,4)` | 预测区间的工况设定是否已知。 |
| `target_cycle_id` | `(B,H)` | 预测区间内的真实循环编号。 |
| `target_soh` | `(B,H)` | SOH 监督标签。 |
| `target_soh_valid` | `(B,H)` | 是否有有效的 SOH 标签，用于 loss 和指标。 |

`N`、`T` 和 `L` 分别是批内补齐后的循环数、片段点数和历史工况长度。
`F` 是观测通道数，`H` 是 `forecast_horizon_cycles`。
单条样本里的充电片段仍为变长数组，`point_valid` 和 `cycle_valid` 在组批时生成。
工况的四个通道依次为环境温度、充电倍率、放电倍率、充电后静置时长。

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
    sampler=[(0, 1), (sample_position, 1)],
    collate_fn=collate_tongji,
    num_workers=0,
)
batch = next(iter(loader))

print(batch["values"].shape)       # (B, N_max, T_max, F)
print(batch["point_valid"].shape)  # (B, N_max, T_max)
print(batch["cycle_valid"])       # (B, N_max)
print(batch["target_soh"].shape)   # (B, forecast_horizon_cycles)

assert torch.equal(batch["cycle_valid"], batch["point_valid"].any(dim=-1))
assert not batch["value_visible"][~batch["point_valid"]].any()
```

只想手动拼几条样本时，可以直接调用
`collate_tongji([dataset[0, 1], dataset[sample_position, 1]])`。

### 随 epoch 切换观测日程

独立 PyTorch 循环可以用 `TongjiTaskSampler`。它在每个 epoch 将全部样本
下标一起打乱，再把 `(sample_position, epoch)` 交给 Dataset。DataLoader
按这个顺序组批，允许一批只覆盖部分任务。完整遍历且保留尾批时，每个任务恰好
贡献 `samples_per_task` 份。训练配置的 `trainer.limit_train_batches`
可以提前截断本轮。在开始遍历前调用 `set_epoch`，并让 DataLoader 的
`shuffle` 使用默认值 `False`。

```python
from src.data import TongjiTaskSampler

multi_view_dataset = TongjiTaskDataset.from_config(config, samples_per_task=3)
sampler = TongjiTaskSampler(multi_view_dataset, seed=config["seed"])
train_loader = DataLoader(
    multi_view_dataset,
    batch_size=config["data"]["batch_size"],
    sampler=sampler,
    collate_fn=collate_tongji,
    num_workers=0,
)

for epoch in range(multi_view_dataset.max_epochs):
    sampler.set_epoch(epoch)
    for batch in train_loader:
        # 在这里调用模型并计算 loss。
        assert batch["target_soh_valid"].any(dim=-1).all()

multi_view_dataset.close()
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

`build_datasets` 给训练评估、验证和测试各保留每任务一份样本。`make_loader` 给这些集合使用最后一个阶段的固定观测。
在项目的 Lightning 训练入口中，训练 sampler 会收到当前 epoch。
Notebook 中可以先用 `num_workers=0` 来检查样本。用完后调用
`dataset.close()` 释放 Dataset 当前缓存的内存映射，在 Windows 上移动或删除
处理数据之前尤其有用。

```python
dataset.close()
```
