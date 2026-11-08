"""Cell-level selection, explicit anchor splits, and training-only scaling."""
from functools import partial

import numpy as np
from torch.utils.data import DataLoader

from src.data import (
    TongjiTaskDataset,
    TongjiTaskSampler,
    collate_tongji,
    load_cell,
)
from src.utils.files import read_json
from src.utils.paths import processed_data_path


def fit_normalization(root, training_cell_ids):
    """Fit on training cells only, never on SOH or the held-out test cell."""
    cells_dir = processed_data_path(root) / 'cells'
    groups = [('feature', 'values', 'value_valid'),
              ('static', 'static_numeric', 'static_valid'),
              ('condition', 'conditions', 'conditions_valid')]
    result = {}
    for prefix, key, valid_key in groups:
        sums = squares = counts = None
        for cell_id in training_cell_ids:
            cell = load_cell(cells_dir / cell_id)
            x = np.asarray(cell[key], dtype=np.float64)
            valid = np.asarray(cell[valid_key])
            if prefix == 'static':
                x = x[None]
                valid = valid[-x.shape[-1]:][None]
            safe = np.where(valid, x, 0)
            if sums is None:
                sums = np.zeros(x.shape[-1]); squares = sums.copy(); counts = sums.copy()
            sums += safe.sum(axis=0)
            squares += (safe * safe).sum(axis=0)
            counts += valid.sum(axis=0)
        mean = sums / np.maximum(counts, 1)
        variance = np.maximum(squares / np.maximum(counts, 1) - mean**2, 0)
        std = np.sqrt(variance)
        std[std < 1e-6] = 1.0
        result[f'{prefix}_mean'] = mean.tolist()
        result[f'{prefix}_std'] = std.tolist()
    return result


def _split_anchors(anchors, ratio, rng):
    if not 0 < ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    shuffled = rng.permutation(anchors)
    nval = max(1, round(len(anchors) * ratio))
    if nval >= len(anchors):
        raise ValueError("Not enough anchors for train/validation split")
    return sorted(shuffled[nval:].tolist()), sorted(shuffled[:nval].tolist())


def build_datasets(config, saved_splits=None):
    data = config["data"]
    if "observation_schedule" not in data:
        raise ValueError("Tongji training requires data.observation_schedule")
    if "max_epochs" in config.get("trainer", {}):
        raise ValueError("Set max_epochs under data")
    make_dataset = partial(TongjiTaskDataset.from_config, config)
    partitions = data["cell_ids"]
    if not isinstance(partitions, dict) or not {"train", "test"} <= partitions.keys():
        raise ValueError("cell_ids must define train and test lists")
    for name, ids in partitions.items():
        if name not in ("train", "val", "test"):
            raise ValueError(f"Unknown cell_ids group: {name}")
        if (
            not isinstance(ids, list)
            or not ids
            or not all(isinstance(cid, str) for cid in ids)
        ):
            raise ValueError(f"cell_ids.{name} must be a nonempty list of cell IDs")
        if len(set(ids)) != len(ids):
            raise ValueError(f"cell_ids.{name} contains duplicate cell IDs")

    train_cells = sorted(partitions["train"])
    test_cells = sorted(partitions["test"])
    independent_val = "val" in partitions
    val_cells = sorted(partitions["val"]) if independent_val else train_cells
    if set(test_cells) & set(train_cells):
        raise ValueError("Test cell IDs must be disjoint from training")
    if independent_val and set(val_cells) & (set(train_cells) | set(test_cells)):
        raise ValueError("Validation cell IDs must be disjoint from training and test")

    cells = {
        "train_cells": train_cells,
        "val_cells": val_cells,
        "test_cells": test_cells,
    }
    if saved_splits is None:
        train_anchors = val_anchors = None
        if independent_val:
            note = "Disjoint train, validation and test cells."
        else:
            train_anchors, val_anchors = {}, {}
            rng = np.random.default_rng(data["split_seed"])
            candidate = make_dataset(cell_ids=train_cells)
            try:
                for cell_id in train_cells:
                    train_anchors[cell_id], val_anchors[cell_id] = _split_anchors(
                        candidate.anchor_cycles(cell_id), data["val_ratio"], rng
                    )
            finally:
                candidate.close()
            note = "Same-cell held-out anchors. Historical observations and future labels may overlap."
        saved_splits = dict(
            cells, train_anchors=train_anchors, val_anchors=val_anchors, note=note
        )
    else:
        if any(saved_splits.get(key) != value for key, value in cells.items()):
            raise ValueError("Configured cell lists differ from the saved splits")
        train_anchors = saved_splits["train_anchors"]
        val_anchors = saved_splits["val_anchors"]
        if not independent_val and any(
            set(train_anchors[cell_id]) & set(val_anchors[cell_id])
            for cell_id in train_cells
        ):
            raise ValueError("Train and validation anchor IDs overlap")

    eval_seed = data["eval_seed"]
    return {
        "train": make_dataset(cell_ids=train_cells, anchor_cycle_ids=train_anchors),
        "train_eval": make_dataset(
            cell_ids=train_cells, anchor_cycle_ids=train_anchors, seed=eval_seed
        ),
        "val": make_dataset(
            cell_ids=val_cells, anchor_cycle_ids=val_anchors, seed=eval_seed
        ),
        "test": make_dataset(cell_ids=test_cells, seed=eval_seed),
    }, saved_splits


def make_loader(dataset, config, shuffle=False):
    data = config["data"]
    # Use the final observation policy for every fixed-view evaluation.
    fixed_epoch = None if shuffle else dataset.max_epochs - 1
    sampler = TongjiTaskSampler(
        dataset, seed=config["seed"], fixed_epoch=fixed_epoch
    )
    return DataLoader(
        dataset,
        batch_size=data["batch_size"],
        sampler=sampler,
        num_workers=data["num_workers"],
        collate_fn=collate_tongji,
        persistent_workers=False,
    )


def vocabulary_sizes(root):
    root = processed_data_path(root)
    mapping = read_json(root / 'category_mapping.json')
    schema = read_json(root / 'schema.json')
    return [len(mapping[field]) for field in schema['static_categorical']]
