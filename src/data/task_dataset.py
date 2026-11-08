"""A finite index of Tongji forecast tasks and reproducible observation views."""

import hashlib
import json

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from src.utils.files import read_json
from src.utils.paths import processed_data_path

from .observation import read_percent
from .tongji import load_cell, tensor


def sample_visible(valid, keep, rng):
    available = valid.sum(axis=1, keepdims=True)
    if type(keep) is int:
        count = np.minimum(available, keep)
    else:
        count = available * read_percent(keep) // 100

    if np.array_equal(count, available):
        return valid.copy()

    # Randomize the order of valid channels independently at each point.
    order = np.argsort(np.where(valid, rng.random(valid.shape), np.inf), axis=1)
    within_quota = np.arange(valid.shape[1]) < count
    visible = np.zeros_like(valid)
    np.put_along_axis(visible, order, within_quota, axis=1)
    return visible


class TongjiTaskDataset(Dataset):
    """Index tasks by (cell ID, last observed cycle); use (index, epoch) for views."""

    def __init__(
        self,
        root,
        cell_ids,
        schedule,
        *,
        horizon=32,
        max_last_observed_cycle=None,
        anchor_cycle_ids=None,
        seed=42,
    ):
        root = processed_data_path(root)
        manifest = read_json(root / "manifest.json")
        if not manifest.get("complete"):
            raise ValueError("Preprocessing is incomplete")
        known = {cell["cell_id"] for cell in manifest["cells"]}
        cell_ids = sorted(cell_ids)
        if not cell_ids or len(cell_ids) != len(set(cell_ids)):
            raise ValueError("cell_ids must be nonempty and unique")
        if not set(cell_ids) <= known:
            raise ValueError(f"Unknown cells: {set(cell_ids) - known}")
        if type(horizon) is not int or horizon < 1:
            raise ValueError("horizon must be a positive integer")

        self.cells_dir = root / "cells"
        self.schedule = schedule
        self.horizon = horizon
        self.seed = seed
        self._cached_cell_id = None
        self._cached_cell = None

        tasks = []
        for cell_id in cell_ids:
            cell = load_cell(self.cells_dir / cell_id)
            ids = cell["cycle_id"]
            anchors = ids[np.diff(cell["cycle_offsets"]) > 0]
            valid_targets = ids[cell["soh_valid"]]
            has_target = np.searchsorted(
                valid_targets, anchors, side="right"
            ) < np.searchsorted(valid_targets, anchors + horizon, side="right")
            eligible = anchors[has_target]
            if max_last_observed_cycle is not None:
                eligible = eligible[eligible <= max_last_observed_cycle]
            if anchor_cycle_ids is not None:
                requested = set(anchor_cycle_ids[cell_id])
                if not requested <= set(eligible):
                    raise ValueError(f"{cell_id}: requested anchors are not eligible")
                eligible = [k for k in eligible if k in requested]
            if len(eligible) == 0:
                raise ValueError(f"{cell_id}: no eligible forecasting tasks")
            tasks.extend((cell_id, int(k)) for k in eligible)
        self.tasks = tuple(tasks)

    def anchor_cycles(self, cell_id):
        return np.array([k for cid, k in self.tasks if cid == cell_id], dtype=np.int64)

    def close(self):
        self._cached_cell_id = None
        self._cached_cell = None

    def __getstate__(self):
        return dict(self.__dict__, _cached_cell_id=None, _cached_cell=None)

    def _load(self, cell_id):
        if cell_id != self._cached_cell_id:
            self._cached_cell = load_cell(self.cells_dir / cell_id)
            self._cached_cell_id = cell_id
        return self._cached_cell

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, key):
        index, epoch = key if isinstance(key, tuple) else (key, 0)
        if not 0 <= index < len(self):
            raise IndexError(index)
        cell_id, k = self.tasks[index]
        phase = self.schedule.for_epoch(epoch)
        seed_data = json.dumps([self.seed, cell_id, k, epoch]).encode("utf-8")
        digest = hashlib.blake2b(seed_data, digest_size=8).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "little"))

        cell = self._load(cell_id)
        ids = cell["cycle_id"]
        offsets = cell["cycle_offsets"]
        anchor_row = int(np.searchsorted(ids, k))
        previous = np.flatnonzero(np.diff(offsets[: anchor_row + 1]) > 0)
        eligible_count = len(previous) + 1
        count = (eligible_count * phase.cycle_percent + 99) // 100
        selected = rng.choice(previous, size=count - 1, replace=False)
        rows = sorted([*selected, anchor_row])

        sample = dict(
            cell_id=cell_id,
            anchor_cycle=k,
            cycle_id=tensor(ids[rows]),
            values=[],
            value_valid=[],
            value_visible=[],
            time_in_cycle=[],
            raw_sample_index=[],
            segment_bounds=[],
        )
        for row in rows:
            start, stop = offsets[row : row + 2]
            total = int(stop - start)
            length = max(1, total * phase.segment_percent // 100)
            left = int(rng.integers(0, total - length + 1))
            part = slice(start + left, start + left + length)

            valid = cell["value_valid"][part]
            visible = sample_visible(valid, phase.value_keep, rng)
            sample["values"].append(tensor(np.where(visible, cell["values"][part], 0)))
            sample["value_valid"].append(tensor(valid))
            sample["value_visible"].append(tensor(visible))
            sample["time_in_cycle"].append(tensor(cell["time_in_cycle"][part]))
            sample["raw_sample_index"].append(tensor(cell["raw_sample_index"][part]))
            sample["segment_bounds"].append([left, left + length])

        target_ids = k + np.arange(1, self.horizon + 1, dtype=np.int64)
        positions = np.searchsorted(ids, target_ids).clip(max=len(ids) - 1)
        target_valid = (ids[positions] == target_ids) & cell["soh_valid"][positions]
        history_ids = np.arange(int(ids[rows[0]]), k, dtype=np.int64)
        u = cell["conditions"][anchor_row]
        u_valid = cell["conditions_valid"][anchor_row]

        sample.update(
            segment_bounds=tensor(sample["segment_bounds"]),
            static_categorical=tensor(cell["static_categorical"]),
            static_numeric=tensor(cell["static_numeric"]),
            static_valid=tensor(cell["static_valid"]),
            history_cycle_id=tensor(history_ids),
            history_u=tensor(np.tile(u, (len(history_ids), 1))),
            history_u_valid=tensor(np.tile(u_valid, (len(history_ids), 1))),
            future_u=tensor(np.tile(u, (self.horizon, 1))),
            future_u_valid=tensor(np.tile(u_valid, (self.horizon, 1))),
            target_cycle_id=tensor(target_ids),
            target_soh=tensor(np.where(target_valid, cell["soh"][positions], 0)),
            target_valid=tensor(target_valid),
        )
        return sample


class TongjiTaskSampler(Sampler):
    """Send the epoch with each task so workers can replay views independently."""

    def __init__(self, dataset, *, seed=42, fixed_epoch=None):
        self.dataset = dataset
        self.seed = seed
        self.fixed_epoch = fixed_epoch
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        if self.fixed_epoch is not None:
            return ((index, self.fixed_epoch) for index in range(len(self)))
        epoch = self.epoch
        generator = torch.Generator().manual_seed(self.seed + epoch)
        order = torch.randperm(len(self), generator=generator).tolist()
        return ((index, epoch) for index in order)

    def __len__(self):
        return len(self.dataset)
