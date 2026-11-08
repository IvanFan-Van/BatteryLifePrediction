"""Cell-stratified partitions and dataset construction for Tongji studies."""

from collections import defaultdict

import numpy as np

from src.data import TongjiDataset
from src.utils.files import read_json
from src.utils.paths import processed_data_path


def stratified_cell_split(root, seed=2026, val_ratio=0.2, test_ratio=0.2):
    manifest = read_json(processed_data_path(root) / "manifest.json")
    groups = defaultdict(list)
    for cell in manifest["cells"]:
        # Family + ambient temperature + charging/discharging protocol.
        groups[cell["cell_id"].split("--")[0]].append(cell["cell_id"])
    rng = np.random.default_rng(seed)
    split = dict(train=[], val=[], test=[], groups={})
    for group, cells in sorted(groups.items()):
        if len(cells) < 3:
            raise ValueError(
                f"{group} needs >=3 cells for a three-way stratified split"
            )
        cells = rng.permutation(sorted(cells)).tolist()
        nval = max(1, round(len(cells) * val_ratio))
        ntest = max(1, round(len(cells) * test_ratio))
        if nval + ntest >= len(cells):
            nval = ntest = 1
        parts = dict(
            val=cells[:nval],
            test=cells[nval : nval + ntest],
            train=cells[nval + ntest :],
        )
        split["groups"][group] = parts
        for name in ["train", "val", "test"]:
            split[name].extend(parts[name])
    for name in ["train", "val", "test"]:
        split[name].sort()
    if set(split["train"]) & set(split["val"]) or set(split["test"]) & (
        set(split["train"]) | set(split["val"])
    ):
        raise AssertionError("Cell split overlap")
    split["seed"] = seed
    split["protocol"] = (
        "Disjoint cells, stratified within material-family and experimental-protocol groups; all recorded cells used."
    )
    return split


def study_datasets(config, splits, include_test=False):
    d = config["data"]
    shared = dict(
        root=d["root"],
        history_cycles=tuple(d["history_cycles"]),
        segment_points=tuple(d["segment_points"]),
        horizon=d["horizon"],
        max_anchor_cycle=None,
        enumerate_anchors=True,
        cache_size=d.get("cache_size", 128),
    )
    datasets = dict(
        train=TongjiDataset(
            **shared,
            cell_ids=splits["train"],
            seed=config["seed"],
            channel_mask_probability=d["channel_mask_probability"],
        ),
        val=TongjiDataset(
            **shared, cell_ids=splits["val"], seed=d["eval_seed"], deterministic=True
        ),
    )
    if include_test:
        datasets["train_eval"] = TongjiDataset(
            **shared, cell_ids=splits["train"], seed=d["eval_seed"], deterministic=True
        )
        datasets["test"] = TongjiDataset(
            **shared, cell_ids=splits["test"], seed=d["eval_seed"], deterministic=True
        )
    return datasets
