# 状态估计先融合字段，再汇总时间步和循环

- **`StateEstimator(batch)` 用截至最后观测轮的输入估计当前状态 `z_k`，形状为 `(B, state_dim)`。**
  - **字段 Attention 在每个采样点上融合观测、时间、静态属性、历史工况和循环距离。** 数值和类别 token 分别投影到公共宽度 `D = d_model`。静态属性每条样本只编码一次，再广播到该样本的真实点。各字段加入各自的可学习类型向量。
  - **时间 Attention 把一轮的采样点压缩为一个循环摘要。** 每个点的 `[POINT]` 输出加上由真实采样秒数得到的位置向量，再与该轮的 `[CYCLE]` 一起交互。仅组批时补齐的点由 `point_valid` 屏蔽，真实点上的缺失观测保留字段专属的缺失 token。
  - **循环 Attention 沿真实循环编号建模，观测轮和仅有工况的轮都各占一个位置。** 按 `history_cycle_id` 对齐工况，按 `cycle_id` 放入观测摘要。没有观测的历史轮复用同一个字段 Attention，用静态属性、工况和循环距离生成摘要。每轮摘要再加上由真实循环距离得到的位置向量，供 `[STATE]` 汇总。
    - `N`、`T`、`L` 分别是批内补齐后的观测轮数、点数和历史工况长度。`K = L + 1` 是批内历史时间轴的最大长度，包含最后观测轮。每条样本按实际起止编号屏蔽多余位置，历史长度可为 0。
    - 当前 Tongji 的 `history_u` 止于最后观测轮之前。最后观测轮使用四个缺失工况 token，未来工况从单独的路径进入 Dynamics。
    - 图中字段和时间张量用补齐形状表示。执行时，字段层只打包真实采样点，时间层只打包真实观测轮。

  ```mermaid
  flowchart TB
      O["观测值 (B,N,T,5)<br/>各字段 ScalarEncoder → (B,N,T,5,D)"]
      T["采样时间 (B,N,T)<br/>ScalarEncoder → (B,N,T,1,D)"]
      S["静态类别 (B,5) + 静态数值 (B,3)<br/>独立类别表和 ScalarEncoder → (B,8,D)"]
      C["历史工况 (B,L,4) + 循环距离<br/>按真实循环号对齐并编码 → (B,K,5,D)"]

      O --> F
      T --> F
      S --> F
      C --> F

      subgraph A["① 字段 Attention · 每个采样点单独交互"]
          F["沿字段轴拼接，加入字段类型编码<br/>[POINT] + 5 观测 + 1 时间 + 8 静态 + 4 工况 + 1 距离<br/>(B,N,T,20,D)"]
          F --> FA["沿 20 个 token 做 Self-Attention<br/>(B,N,T,20,D) → (B,N,T,20,D)"]
          FA --> P["取每个采样点的 [POINT]<br/>(B,N,T,D)"]
      end

      subgraph W["② 时间 Attention · 每个循环单独交互"]
          P --> TS["加实际采样时间的位置编码，再加 [CYCLE]<br/>(B,N,1+T,D)"]
          TS --> TA["沿 1+T 个 token 做 Self-Attention<br/>(B,N,1+T,D) → (B,N,1+T,D)"]
          TA --> V["取每轮 [CYCLE]<br/>观测循环摘要 (B,N,D)"]
      end

      S --> HC
      C --> HC
      HC["只有历史工况的循环<br/>[POINT] + 8 静态 + 4 工况 + 1 距离<br/>复用字段 Attention，取 [POINT] → 每轮一个 D 维向量"]

      subgraph X["③ 循环 Attention · 每条样本沿循环轴交互"]
          V --> CT
          HC --> CT
          CT["按真实循环号填入摘要或工况表示<br/>每个循环一个 token (B,K,D)"]
          CT --> XS["加真实循环距离的位置编码，再加 [STATE]<br/>(B,1+K,D)"]
          XS --> XA["沿 1+K 个 token 做 Self-Attention<br/>(B,1+K,D) → (B,1+K,D)"]
          XA --> Z["取 [STATE] 并投影<br/>当前状态 z_k (B,D_z)"]
      end

      U["未来工况 (B,H,4)<br/>复用工况字段编码器 → (B,H,4,D) → (B,H,4D)"]
      S --> SC["按字段顺序展开<br/>静态向量 c (B,8D)"]
      Z --> DY["Dynamics 入口<br/>z_k (B,D_z) · c (B,8D) · 未来工况 (B,H,4D)"]
      SC --> DY
      U --> DY
  ```

- **`model.state_estimator` 控制三个 Attention 模块的宽度、深度和状态维度。**
  - `d_model: 16` 是投影后的公共 token 宽度，必须能被 `num_heads` 整除。
  - `state_dim: 16` 是 `[STATE]` 经线性投影后的状态宽度。
  - `num_heads: 2` 是每个 Attention block 的头数。三个模块共用此设置。
  - `feedforward_dim: 64` 是每个 Transformer block 内部前馈网络的隐层宽度。
  - `field_layers: 2`、`time_layers: 2`、`cycle_layers: 2` 分别控制字段、时间和循环模块的 block 数量。默认每阶段使用两块，后一块的摘要 token 可以读取前一块已完成交互的 token。
  - `dropout: 0.0` 是三个模块内的 dropout 概率，评估模式下关闭。

- **`encode_for_dynamics(batch)` 额外提供推演需要的上下文。**
  - `state` 为 `(B, state_dim)`，来自同一条历史估计路径。
  - `static` 为 `(B, 8D)`，按字段顺序展开八个投影后的静态 token。
  - `future_conditions` 为 `(B, H, 4D)`，逐步展开四个投影后的工况 token。历史和未来共用工况字段编码器。

- **所有连续字段各自通过一个两层 MLP，输出长度由 `model.embedding.scalar` 统一控制。**
  - `output_dim: 8` 是每个标量输出的向量长度，记为 `D_num`。改变它会改变观测、静态数值、工况和位置向量的宽度。
  - `hidden_dim: 16` 是 MLP 隐层的宽度。默认结构为 `Linear(1, 16) → GELU → Linear(16, 8)`。
  - `activation: gelu` 设置两层线性变换之间的激活函数，可选 `gelu`、`relu`、`silu`。
  - `ScalarEncoder` 保留所有输入轴，并追加 `D_num` 维。其构造参数 `mean` 和 `std` 分别设置中心和尺度，独立使用时默认是 0 和 1。掩码为 `False` 的位置返回该字段专属的可学习缺失向量。
  - `NumericFeatureEncoder` 为最后一轴的每个字段创建独立的 `ScalarEncoder`。`encode_tokens` 用 `stack` 保留字段轴 `(..., F, D_num)`。直接调用模块则按原字段顺序展开为 `(..., F * D_num)`。传入的 `mean` 和 `std` 序列决定字段数及各自的归一化参数。

- **每个静态类别字段使用独立的查表模块，并保留自己的 `<UNK>` 向量。**
  - `model.embedding.categorical.output_dim: 8` 是每个类别字段的 embedding 长度，记为 `D_cat`。
  - `CategoricalEncoder` 的 `known_ids` 决定哪些类别有独立向量。ID 0、缺失类别和训练电芯中未见的类别都映射到可学习的 `<UNK>`，即使该类别已经出现在整个数据集的类别映射中。
  - `TongjiInputEncoder.from_config` 从 `data.root` 读取处理数据，只用 `data.cell_ids.train` 中的电芯收集已知类别和拟合数值归一化。归一化统计覆盖这些电芯的完整处理数据。若验证集按同一电芯的预测起点切分，验证起点所属电芯的数据也在统计范围内。

- **`TongjiInputEncoder` 将 batch 编码为六组张量，保留采样点轴、循环轴及工况序列轴。**
  - **`observations` 为 `(B, N, T, 5 * D_num)`，默认最后一维为 40。** 同一采样点的五个观测向量首尾拼接。真实点上以 `value_visible` 判断读数是否可用，`point_valid=False` 的 padding 点输出全零。
  - **`static` 为 `(B, 5 * D_cat + 3 * D_num)`，默认最后一维为 64。** 每条样本只编码一份静态属性。向量按五个类别、三个连续属性的顺序拼接，`static_valid` 标记各字段是否已知。
  - **`time` 为 `(B, N, T, D_num)`，用于循环内的时间位置。** `model.embedding.time_scale_s: 10000.0` 是正数尺度，单位为秒。将 `time_in_cycle_s` 除以该值后，再用独立的标量 MLP 编码。padding 点输出全零。
  - **`cycle_distance` 为 `(B, N, D_num)`，用于循环间的位置。** `model.embedding.cycle_distance_scale: 100.0` 是正数尺度，单位为循环数。先计算真实编号差 `last_cycle_id - cycle_id`，除以该尺度后再编码。padding 循环输出全零。
  - **`history_conditions` 与 `future_conditions` 分别为 `(B, L, 4 * D_num)` 和 `(B, H, 4 * D_num)`，默认最后一维为 32。** 历史和未来共享同一组四字段编码器。各字段的缺失由 `history_u_valid` 或 `future_u_valid` 标记，历史长度 `L` 可以为 0。需要聚合历史工况时，可以用 `history_u_valid.any(dim=-1)` 选出至少有一项已知工况的步骤。

- **从 YAML 构造状态估计器，再将数据加载器产出的 batch 直接传给它。**
  - [`configs/basic.yaml`](../configs/basic.yaml) 包含 embedding 和状态估计器参数。字段数及顺序由处理数据的 schema 决定。
  - 以下示例在项目根目录运行，输出三组 Dynamics 输入的形状。只需要当前状态时，可以直接调用 `estimator(batch)`。

    ```python
    import torch

    from src.core.state_estimator import StateEstimator
    from src.data.loading import build_datasets, make_loader
    from src.utils.files import read_yaml

    config = read_yaml("configs/basic.yaml")
    estimator = StateEstimator.from_config(config).eval()
    datasets, _ = build_datasets(config)
    try:
        batch = next(iter(make_loader(datasets["test"], config)))
        with torch.inference_mode():
            inputs = estimator.encode_for_dynamics(batch)
        print({name: tuple(value.shape) for name, value in inputs.items()})
    finally:
        for dataset in datasets.values():
            dataset.close()
    ```
