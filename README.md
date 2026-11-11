# Battery life prediction

The project trains a hierarchical state-space model to forecast Tongji SOH
trajectories from sampled charge segments, cell metadata, and operating conditions.
Run the following commands from the repository root.

## 1. Set up the environment

With `uv` installed, sync the dependencies from `uv.lock`. The project requires
Python 3.12 or later.

```bash
uv sync --locked
```

## 2. Preprocess the Tongji data

Place the trusted BatteryLife Tongji `.pkl` files directly in
`data/raw/Tongji/` and run the preprocessing command.

```bash
uv run --locked python scripts/preprocess_tongji.py --raw-dir data/raw/Tongji --output data/processed/tongji_v1
```

The command reads the raw files and writes a separate processed dataset.
The output directory must be new. If an existing dataset has
`"complete": true` in `manifest.json`, you can reuse it. To preprocess again,
choose a new output directory and point `data.root` in the training YAML at it.

## 3. Train the model

`configs/basic.yaml` is a limited-batch CPU example. It reads
`data/processed/tongji_v1` and specifies the training and test cell IDs under
`data.cell_ids`. If you prepared a subset of Tongji, use IDs listed in its
`manifest.json` and keep the training and test cells disjoint.

For Bash on Linux or macOS, use the following command.

```bash
CUDA_VISIBLE_DEVICES=-1 uv run --locked python -m src.core.train --config configs/basic.yaml
```

For Windows PowerShell, set the environment variable on a separate line.

```powershell
$env:CUDA_VISIBLE_DEVICES = '-1'
uv run --locked python -m src.core.train --config configs/basic.yaml
```

The trainer prints the run directory under `outputs/`. It stores the
effective configuration, data split, checkpoints, metrics, and predictions.
W&B logs are stored offline.

Submit long-running experiments through Pueue using a configuration snapshot
and an explicit `CUDA_VISIBLE_DEVICES` value. On `farady`, use only physical
GPU 2 or 3. The repository instructions include the submission procedure.
[Repository instructions](AGENTS.md) · [Pueue guide](docs/pueue_manual.md)

[Dataset guide](docs/dataset.md) · [Model architecture and configuration](docs/model.md)
