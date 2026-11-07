本项目是一个以论文研究、模型开发和实验验证为核心的机器学习科研项目。

主要技术栈：

- **Environment & Dependencies:** uv
- **Deep Learning:** PyTorch, PyTorch Lightning
- **Experiment Tracking:** Weights & Biases (W&B), TensorBoard
- **Version Control:** Git
- **Task Scheduling:** Pueue
- **Configuration:** YAML

所有开发工作应遵循以下原则：

1. **Reproducibility:** 实验应当能够被复现。
2. **Traceability:** 每次实验应能追溯到代码、配置、数据和环境。
3. **Reliability:** 保证主分支稳定，避免破坏已有功能。
4. **Comparability:** 保证实验之间具有可比较性。
5. **Safety:** 保护原始数据、运行中的实验和服务器资源。


`main` 是项目的稳定主分支。

- `main` 必须始终保持可运行。
- 共享基础设施应优先维护在 `main`，例如：
  - Data loading and preprocessing
  - Training pipeline
  - Logging and experiment tracking
  - Checkpoint and resume mechanisms
  - Evaluation utilities
  - Shared model components
- 不允许直接向 `main` 合并未经验证的实验代码。
- 合并前至少运行相关测试和一次 smoke test。
- 实验分支不会自动获得 `main` 的新提交。必要时通过 merge 或 rebase 同步共享功能。

当需要引入新的模型架构或实验性组件时，创建独立的 Git 分支。

例如：

```bash
git switch -c exp/new_attention
git switch -c exp/latent_dynamics
git switch -c exp/physics_informed_loss
```

规则：

- 新模型架构或需要修改代码的重大方法实验应使用独立分支。
- 已支持的模型组件开关、超参数调整等，优先通过配置文件进行，不必创建新分支。
- 同一模型的不同学习率、batch size、hidden dimension 等实验，使用不同配置和实验日志追踪。
- 实验得到明确结论后，决定保留、合并还是放弃该实现。
- 稳定且可复用的组件可以合并到 main。
- 不随意删除包含重要科研结果的分支或提交。

**重要：运行中的实验与 Git 分支**

Pueue 保存任务命令和工作目录，但不会自动冻结 Git 代码。

如果运行中或排队中的任务使用当前工作目录：

- 不得随意切换分支、修改其依赖的代码或删除文件。
- 不得假设任务会使用入队时的 Git 版本。
- 如果需要同时运行不同代码版本，优先使用独立的 `git worktree`。
- 对于需要严格复现的实验，应使用固定 Git commit 的独立工作目录，并确保依赖环境也被隔离或固定。

采用 Conventional Commits：

| Type | 用途 | Example |
|---|---|---|
| `feat` | 增加新功能 | `feat(logging): add WandB logger` |
| `fix` | 修复错误 | `fix(data): prevent validation data leakage` |
| `refactor` | 代码重构 | `refactor(model): extract backbone factory` |
| `test` | 增加或修改测试 | `test(train): add one-batch smoke test` |
| `docs` | 修改文档 | `docs: document baseline reproduction` |
| `chore` | 常规维护 | `chore: update gitignore` |
| `build` | 构建与依赖管理 | `build: add Lightning dependency` |
| `ci` | CI/CD 相关修改 | `ci: run pytest on pull requests` |
| `perf` | 性能优化 | `perf(data): enable persistent workers` |

Commit 应当尽量保持原子性。

- 一个 commit 应对应一个清晰、可解释的逻辑改动。
- 同一功能所必需的代码、测试和配置修改可以放在同一个 commit。
- 无关的依赖更新、重构或功能修改应尽量分开。
- 不提交实验输出、模型权重、临时数据或敏感凭据。
- 未经用户许可，不执行 force push、破坏性 reset 或删除重要历史的操作。

实验配置优先使用 YAML 文件管理。

- 不应将学习率、batch size、模型层数等实验参数硬编码到训练逻辑中。
- 已支持的超参数变化应通过配置文件实现。
- 配置应明确包含模型、训练、数据、随机种子以及必要的评估参数。
- 实验应保存最终实际生效的配置，而不只是默认配置。
- 不同实验不应共享可被覆盖的输出目录。
- 不在实验运行期间修改该实验实际读取的配置快照。


farady（Linux）服务器有四张 NVIDIA RTX 4090。以下 GPU 编号限制仅适用于 farady。

**在 farady 上只允许使用物理 GPU 2 和 GPU 3（最后两张）。**

farady 上严格禁止使用物理 GPU 0 和 GPU 1，除非用户明确修改此规则。

Windows 本地开发环境允许使用本机所有 GPU，包括 GPU 0。可以在显存余量至少 20% 的前提下并行运行实验。

启动训练时必须显式设置 `CUDA_VISIBLE_DEVICES`。

使用物理 GPU 2：

```bash
CUDA_VISIBLE_DEVICES=2 uv run --locked python train.py
```

使用物理 GPU 3：

```bash
CUDA_VISIBLE_DEVICES=3 uv run --locked python train.py
```

同时使用两张 GPU：

```bash
CUDA_VISIBLE_DEVICES=2,3 uv run --locked python train.py
```

注意：`CUDA_VISIBLE_DEVICES` 会重新映射进程内的 CUDA 设备编号。
例如 `CUDA_VISIBLE_DEVICES=2` 时，PyTorch 中的 `cuda:0` 对应物理 GPU 2。

规则：

- 默认每次实验使用一张 GPU。
- 需要多 GPU 训练时，必须明确配置 Lightning 的 `devices` 和 `strategy`。
- 未经用户授权，不自动启动大规模或长时间的多 GPU 实验。
- 提交任务前检查相关 GPU 的占用情况。

**所有正式、长时间运行的训练实验都应通过 Pueue 启动。**

允许直接执行短时间的单元测试、smoke test 和调试命令。

目前 pueue 的配置如下:
```json
{
    "default": {"status":"Running","parallel_tasks":1},
    "gpu2": {"status":"Running","parallel_tasks":1},
    "gpu3": {"status":"Running","parallel_tasks":1}
}
```

- `gpu2` 对应物理 GPU 2。
- `gpu3` 对应物理 GPU 3。
- 默认每个 GPU 任务组最多同时运行一个训练任务。
- 如果显存运行, 可以适当增加一个 GPU 任务组的最大并发数. 但是确保最终显存能够有 20% 余量. 即当并行任务使用的显存已经超过 80% 时不允许再添加任务并行数量. 

Pueue 不会自动保存任务添加时的配置文件内容。

因此，**正式实验入队前必须创建配置快照，并让任务使用快照，而不是原始配置文件。**

以下示例在项目根目录执行，将 `config.yaml` 的快照交给 GPU 2 队列：

```bash
# 1. 创建临时目录
mkdir -p ./tmp/pueue-config

# 2. 生成唯一的配置文件
snapshot=$(mktemp "$PWD/tmp/pueue-config/config.<run_id>.yaml")

# 3. 保存当前配置
cp -- config.yaml "$snapshot"

# 4. 提交任务
if PUEUE_CONFIG_SNAPSHOT="$snapshot" \
   pueue add -g gpu2 --label "baseline-gpu2" -- \
   'CUDA_VISIBLE_DEVICES=2 uv run --locked python train.py --config "$PUEUE_CONFIG_SNAPSHOT"; rc=$?; rm -f -- "$PUEUE_CONFIG_SNAPSHOT"; exit $rc'
then
    echo "Experiment queued successfully."
else
    rm -f -- "$snapshot"
    echo "Failed to enqueue experiment." >&2
fi
```

此流程依赖 Pueue 在提交任务时保存的环境变量，使每个任务使用自己的配置快照。

执行要求：

1. 在创建快照前，确认配置源文件存在。
2. `mkdir`、`mktemp` 或 `cp` 失败时，不得继续提交任务。
3. 将示例中的 `config.yaml` 替换为实际配置文件。
4. 任务必须读取快照文件。
5. 训练脚本应将最终生效的配置复制到对应的持久化实验目录中。
6. 快照在正常完成或训练返回错误后删除，并保留训练程序的退出状态。
7. 如果 Pueue 添加任务失败，立即删除尚未使用的快照。
8. 若任务被强制终止，可能留下快照；清理前必须确认没有排队或运行中的任务仍依赖它。
9. 不要提前删除排队任务的配置快照。
10. 如果配置中的相对路径依赖 YAML 文件所在目录，必须先确认移动到临时目录不会改变路径解析行为。

注意：此流程只冻结了配置文件，不保证 Git 代码、依赖文件或数据版本不变。

### 6.3 Task Monitoring

查看任务：

```bash
pueue status
```

查看任务输出：

```bash
pueue log <task_id>
```

实时跟踪输出：

```bash
pueue follow <task_id>
```

`pueue follow` 会持续占用当前终端，可以通过 `Ctrl+C` 退出跟踪。正常情况下不会终止后台训练任务。

更多 Pueue 命令参考：
[本地 Pueue 使用指南](docs/pueue_manual)


所有实验使用的原始数据均存放于项目的 `raw_data` 目录。

**原始数据默认只读。**

- 未经用户明确许可，不得修改、覆盖或删除 `raw_data` 中的数据。
- 如果确实需要改变原始数据，应先告知用户并获得允许。
- 获得允许后，应创建独立副本，并在副本上进行操作。
- 数据预处理的输出应与原始数据分离。
- 不得无意覆盖已有预处理结果。

同样应保护 `runs/`、日志和 checkpoint 等实验产物。未经用户许可，不执行破坏性批量清理。