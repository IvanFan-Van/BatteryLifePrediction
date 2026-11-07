# Battery life prediction

The project trains a hierarchical state-space model to forecast Tongji SOH
trajectories from sampled charge segments and cell metadata.

| Directory | Contents |
| --- | --- |
| configs/ | Flat YAML files for smoke runs and full-dataset studies |
| data/raw/ | Original datasets, including BatteryLife Tongji pickles in Tongji/ |
| data/processed/ | Prepared datasets, including tongji_v1/ |
| docs/ | Pueue and Tongji workflow guides |
| notebooks/ | Jupyter notebooks for inspecting processed cells and batches |
| outputs/ | Experiment reports, logs, predictions, and checkpoints |
| scripts/ | Data preparation and experiment queue entry points |
| src/ | Data pipeline, Lightning callbacks, utilities, and training code |
| tests/ | Unit tests for processing, sampling, training, and study utilities |

From the project root:

~~~bash
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
uv run --locked python scripts/queue_tongji.py --config configs/tongji_pilot.yaml
~~~

The last command queues the limited-batch training pilot in the default Pueue
daemon and prints its task ID and output directory. The same training entry
point is available as `python -m src.core.train`. W&B logs are offline.

If the processed Tongji dataset does not yet exist, generate it from
`data/raw/Tongji` with `uv run --locked python scripts/preprocess_tongji.py`.
Preprocessing refuses to overwrite an existing destination. Use `Tongji.ipynb`
to inspect cells and `Tongji_usage.ipynb` for a Dataset/DataLoader example.

On an existing Linux clone where `raw_data` is a symlink, move the symlink
itself to `data/raw` once the destination is free. Its target directory
stays in place. If the link uses a relative target, recreate it relative to
`data/` and verify that it resolves to the original dataset directory.

[Pueue guide](docs/pueue_manual.md) ·
[Data contract](docs/tongji_data.md) ·
[Smoke training](docs/tongji_attention_smoke.md) ·
[Full Tongji comparison](docs/tongji_full_ablation.md)
