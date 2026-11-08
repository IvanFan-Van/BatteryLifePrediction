"""A finite index of Tongji forecast tasks and reproducible observation views."""

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from src.utils.files import read_json, read_yaml
from src.utils.paths import processed_data_path

ARRAY_NAMES = (
    "values",
    "value_valid",
    "time_in_cycle",
    "raw_sample_index",
    "cycle_offsets",
    "cycle_id",
    "static_categorical",
    "static_numeric",
    "static_valid",
    "conditions",
    "conditions_valid",
    "soh",
    "soh_valid",
)


def load_cell(directory):
    """Memory-map one preprocessed cell; never alter these arrays in place."""
    directory = Path(directory)
    return {
        name: np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in ARRAY_NAMES
    }


def tensor(array):
    # Copy prevents tensors from writing into read-only memory-mapped data.
    return torch.from_numpy(np.array(array, copy=True))


def _read_fraction(value, field):
    if type(value) not in (int, float) or not 0 < value <= 1:
        raise ValueError(f"{field} must be a fraction in (0, 1], got {value!r}")
    return float(value)


@dataclass(frozen=True)
class ObservationPhase:
    start_epoch: int
    cycle_fraction: float
    segment_fraction: float
    value_keep: int | float


@dataclass(frozen=True)
class ObservationSchedule:
    max_epochs: int
    phases: tuple[ObservationPhase, ...]

    @classmethod
    def from_phases(cls, items, *, max_epochs, feature_count):
        if type(max_epochs) is not int or max_epochs < 1:
            raise ValueError("max_epochs must be a positive integer")

        if not isinstance(items, list) or not items:
            raise ValueError("observation_schedule must be a nonempty list")

        phases = []
        previous_start = -1
        for item in items:
            start = item["start_epoch"]
            if type(start) is not int or not previous_start < start < max_epochs:
                raise ValueError("start_epoch must increase and be within max_epochs")
            keep = item["value_keep"]
            if type(keep) is int:
                if not 1 <= keep <= feature_count:
                    raise ValueError(f"value_keep must be between 1 and {feature_count}")
            else:
                keep = _read_fraction(keep, "value_keep")

            phases.append(
                ObservationPhase(
                    start_epoch=start,
                    cycle_fraction=_read_fraction(item["cycle_fraction"], "cycle_fraction"),
                    segment_fraction=_read_fraction(item["segment_fraction"], "segment_fraction"),
                    value_keep=keep,
                )
            )
            previous_start = start

        if phases[0].start_epoch != 0:
            raise ValueError("The first observation phase must start at epoch 0")
        return cls(max_epochs, tuple(phases))

    def for_epoch(self, epoch):
        if type(epoch) is not int or not 0 <= epoch < self.max_epochs:
            raise ValueError(f"Epoch out of range: {epoch}")
        return next(phase for phase in reversed(self.phases) if phase.start_epoch <= epoch)


def sample_visible(valid, keep, rng):
    available = valid.sum(axis=1, keepdims=True)
    if type(keep) is int:
        count = np.minimum(available, keep)
    else:
        # Exact decimal rounding at quota boundaries such as 100 * 0.29.
        fraction = Fraction(str(keep))
        quota = np.array([math.floor(n * fraction) for n in range(valid.shape[1] + 1)])
        count = quota[available]

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

    @classmethod
    def from_config(cls, config, *, cell_ids=None, anchor_cycle_ids=None, seed=None):
        """Build a dataset from a YAML path or an already loaded config."""
        if not isinstance(config, Mapping):
            config = read_yaml(config)
        data = config["data"]
        if "observation_schedule" not in data:
            raise ValueError("TongjiTaskDataset requires data.observation_schedule")
        return cls(
            root=data["root"],
            cell_ids=data["cell_ids"]["train"] if cell_ids is None else cell_ids,
            observation_schedule=data["observation_schedule"],
            max_epochs=data["max_epochs"],
            horizon=data["horizon"],
            max_last_observed_cycle=data.get("max_last_observed_cycle"),
            anchor_cycle_ids=anchor_cycle_ids,
            seed=config["seed"] if seed is None else seed,
        )

    def __init__(
        self,
        root,
        cell_ids,
        *,
        observation_schedule,
        max_epochs,
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
        if isinstance(cell_ids, str):
            raise ValueError("cell_ids must be a collection of cell IDs")
        cell_ids = sorted(cell_ids)
        if not cell_ids or len(cell_ids) != len(set(cell_ids)):
            raise ValueError("cell_ids must be nonempty and unique")
        if not set(cell_ids) <= known:
            raise ValueError(f"Unknown cells: {set(cell_ids) - known}")
        if type(horizon) is not int or horizon < 1:
            raise ValueError("horizon must be a positive integer")

        self.cells_dir = root / "cells"
        schema = read_json(root / "schema.json")
        self._schedule = ObservationSchedule.from_phases(
            observation_schedule,
            max_epochs=max_epochs,
            feature_count=len(schema["features"]),
        )
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

    @property
    def max_epochs(self):
        return self._schedule.max_epochs

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, key):
        index, epoch = key if isinstance(key, tuple) else (key, 0)
        if not 0 <= index < len(self):
            raise IndexError(index)
        cell_id, k = self.tasks[index]
        phase = self._schedule.for_epoch(epoch)
        seed_data = json.dumps([self.seed, cell_id, k, epoch]).encode("utf-8")
        digest = hashlib.blake2b(seed_data, digest_size=8).digest()
        rng = np.random.default_rng(int.from_bytes(digest, "little"))

        cell = self._load(cell_id)
        ids = cell["cycle_id"]
        offsets = cell["cycle_offsets"]
        anchor_row = int(np.searchsorted(ids, k))
        previous = np.flatnonzero(np.diff(offsets[: anchor_row + 1]) > 0)
        eligible_count = len(previous) + 1
        count = math.ceil(eligible_count * Fraction(str(phase.cycle_fraction)))
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
        segment_fraction = Fraction(str(phase.segment_fraction))
        for row in rows:
            start, stop = offsets[row : row + 2]
            total = int(stop - start)
            length = max(1, math.floor(total * segment_fraction))
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


def collate_tongji(samples):
    """Batch-local zero padding. True always means valid/visible.

    cycle_valid is derived from point_valid; it is not an independent mask.
    A genuine point with no valid channels remains a genuine point.
    """
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    batch_size = len(samples)
    n = max(len(s["values"]) for s in samples)
    t = max(len(v) for s in samples for v in s["values"])
    f = samples[0]["values"][0].shape[-1]
    batch = dict(
        cell_id=[s["cell_id"] for s in samples],
        values=torch.zeros(batch_size, n, t, f),
        value_valid=torch.zeros(batch_size, n, t, f, dtype=torch.bool),
        value_visible=torch.zeros(batch_size, n, t, f, dtype=torch.bool),
        point_valid=torch.zeros(batch_size, n, t, dtype=torch.bool),
        time_in_cycle=torch.zeros(batch_size, n, t, dtype=torch.float64),
        raw_sample_index=torch.full((batch_size, n, t), -1, dtype=torch.int64),
        cycle_id=torch.zeros(batch_size, n, dtype=torch.int64),
        segment_bounds=torch.zeros(batch_size, n, 2, dtype=torch.int64),
    )
    for b, sample in enumerate(samples):
        count = len(sample["values"])
        batch["cycle_id"][b, :count] = sample["cycle_id"]
        batch["segment_bounds"][b, :count] = sample["segment_bounds"]
        for j, value in enumerate(sample["values"]):
            length = len(value)
            batch["point_valid"][b, j, :length] = True
            for key in [
                "values",
                "value_valid",
                "value_visible",
                "time_in_cycle",
                "raw_sample_index",
            ]:
                batch[key][b, j, :length] = sample[key][j]
    batch["cycle_valid"] = batch["point_valid"].any(dim=-1)
    history_length = max(len(s["history_cycle_id"]) for s in samples)
    du = samples[0]["future_u"].shape[-1]
    batch["history_cycle_id"] = torch.zeros(
        batch_size, history_length, dtype=torch.int64
    )
    batch["history_u"] = torch.zeros(batch_size, history_length, du)
    batch["history_u_valid"] = torch.zeros(
        batch_size, history_length, du, dtype=torch.bool
    )
    for b, sample in enumerate(samples):
        length = len(sample["history_cycle_id"])
        for key in ["history_cycle_id", "history_u", "history_u_valid"]:
            batch[key][b, :length] = sample[key]
    for key in [
        "static_categorical",
        "static_numeric",
        "static_valid",
        "future_u",
        "future_u_valid",
        "target_cycle_id",
        "target_soh",
        "target_valid",
    ]:
        batch[key] = torch.stack([s[key] for s in samples])
    batch["anchor_cycle"] = torch.tensor([s["anchor_cycle"] for s in samples])
    return batch
