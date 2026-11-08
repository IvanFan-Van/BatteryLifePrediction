"""Regression tests for task identity, exact masks, and epoch-aware sampling."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data import (
    TongjiTaskDataset,
    TongjiTaskSampler,
    collate_tongji,
)
from src.data.loading import build_datasets, make_loader
from src.data.dataset import ObservationSchedule, sample_visible


CONFIG = """
seed: 42
data:
    batch_size: 2
    num_workers: 0
    split_seed: 2026
    eval_seed: 2027
    val_ratio: 0.25
    max_epochs: 2
    horizon: 3
    max_last_observed_cycle: 6
    observation_schedule:
        - start_epoch: 0
          cycle_fraction: 1.0
          segment_fraction: 1.0
          value_keep: 1.0
        - start_epoch: 1
          cycle_fraction: 0.5
          segment_fraction: 0.29
          value_keep: 0.9
"""


def parse_schedule(config, feature_count):
    return ObservationSchedule.from_phases(
        config["data"]["observation_schedule"],
        max_epochs=config["data"]["max_epochs"],
        feature_count=feature_count,
    )


def write_fixture(root, cell_ids):
    ids = np.array([2, 4, 5, 6, 9], dtype=np.int64)
    lengths = [3, 6, 5, 100, 2]
    values = np.ones((sum(lengths), 5), dtype=np.float32)
    valid = np.ones_like(values, dtype=bool)
    valid[:, 4] = False
    values[:, 4] = 0
    indices = np.concatenate([np.arange(n) for n in lengths])
    arrays = dict(
        values=values, value_valid=valid, time_in_cycle=indices.astype(np.float64),
        raw_sample_index=indices, cycle_offsets=np.r_[0, np.cumsum(lengths)],
        cycle_id=ids, static_categorical=np.zeros(5, dtype=np.int64),
        static_numeric=np.ones(3, dtype=np.float32), static_valid=np.ones(8, dtype=bool),
        conditions=np.ones((5, 4), dtype=np.float32),
        conditions_valid=np.ones((5, 4), dtype=bool),
        soh=np.array([1, .95, .9, .85, .8], dtype=np.float32),
        soh_valid=np.array([True, True, False, True, True]),
    )
    for cell_id in cell_ids:
        cell = root / "cells" / cell_id
        cell.mkdir(parents=True)
        for name, array in arrays.items():
            np.save(cell / f"{name}.npy", array)
    (root / "manifest.json").write_text(
        json.dumps({"complete": True, "cells": [{"cell_id": cell_id} for cell_id in cell_ids]}),
        encoding="utf-8",
    )
    (root / "schema.json").write_text(
        json.dumps({"features": ["voltage", "current", "charge", "discharge", "temperature"]}),
        encoding="utf-8",
    )


class TongjiTaskTests(unittest.TestCase):
    def test_config_validation_and_channel_counts(self):
        config = yaml.safe_load(CONFIG)
        channels = np.ones((2, 6), dtype=bool)
        for keep, expected in ((0.9, 5), (4, 4), (1.0, 6)):
            with self.subTest(keep=keep):
                config["data"]["observation_schedule"][0]["value_keep"] = keep
                schedule = parse_schedule(config, feature_count=6)
                visible = sample_visible(channels, schedule.for_epoch(0).value_keep,
                                         np.random.default_rng(42))
                np.testing.assert_array_equal(visible.sum(axis=1), [expected, expected])

        for invalid in (0, 0.0, 7, 4.0, True, float("nan"), "0.9", "90.0%", "101%"):
            with self.subTest(invalid=invalid):
                config["data"]["observation_schedule"][0]["value_keep"] = invalid
                with self.assertRaises(ValueError):
                    parse_schedule(config, feature_count=6)

        config = yaml.safe_load(CONFIG)
        phase = parse_schedule(config, feature_count=5).for_epoch(1)
        self.assertEqual(phase.cycle_fraction, 0.5)
        self.assertEqual(phase.segment_fraction, 0.29)

        for field in ("cycle_fraction", "segment_fraction", "value_keep"):
            with self.subTest(field=field):
                invalid_config = yaml.safe_load(CONFIG)
                invalid_config["data"]["observation_schedule"][1][field] = "90%"
                with self.assertRaisesRegex(ValueError, field):
                    parse_schedule(invalid_config, feature_count=5)

        channels = np.ones((1, 100), dtype=bool)
        self.assertEqual(sample_visible(channels, 0.29, np.random.default_rng(42)).sum(), 29)

        config["data"]["observation_schedule"][1]["start_epoch"] = 0
        with self.assertRaises(ValueError):
            parse_schedule(config, feature_count=5)

    def test_task_targets_masks_padding_and_worker_replay(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root, ["A"])
            config = yaml.safe_load(CONFIG)
            config["data"].update(root=str(root), cell_ids={"train": ["A"]})
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

            dataset = TongjiTaskDataset.from_config(config_path)
            direct = TongjiTaskDataset(
                root, ["A"], observation_schedule=config["data"]["observation_schedule"],
                max_epochs=2, horizon=3, max_last_observed_cycle=6,
            )
            subset = TongjiTaskDataset.from_config(config, cell_ids=["A"], anchor_cycle_ids={"A": [6]})
            try:
                self.assertEqual(dataset.max_epochs, 2)
                self.assertEqual(direct.tasks, dataset.tasks)
                self.assertEqual(subset.tasks, (("A", 6),))
                torch.testing.assert_close(dataset[3, 1]["values"], direct[3, 1]["values"])
                torch.testing.assert_close(dataset[3, 1]["value_visible"], subset[0, 1]["value_visible"])
                self.assertEqual(dataset.tasks, (("A", 2), ("A", 4), ("A", 5), ("A", 6)))
                full, sparse = dataset[3], dataset[3, 1]
                self.assertEqual(full["cycle_id"].tolist(), [2, 4, 5, 6])
                self.assertEqual(len(sparse["cycle_id"]), 2)
                self.assertEqual(sparse["cycle_id"][-1].item(), 6)
                self.assertEqual(len(sparse["values"][-1]), 29)
                self.assertEqual(sparse["target_cycle_id"].tolist(), [7, 8, 9])
                self.assertEqual(sparse["target_valid"].tolist(), [False, False, True])
                for field in ("target_cycle_id", "target_soh", "target_valid"):
                    torch.testing.assert_close(full[field], sparse[field])
                for raw, exists, visible in zip(
                    sparse["values"], sparse["value_valid"], sparse["value_visible"]
                ):
                    self.assertFalse(visible[~exists].any())
                    torch.testing.assert_close(visible.sum(-1), exists.sum(-1) * 90 // 100)
                    self.assertFalse(raw[~visible].any())

                sampler = TongjiTaskSampler(dataset, seed=42)
                sampler.set_epoch(1)
                keys = list(sampler)
                self.assertEqual(sorted(keys), [(index, 1) for index in range(len(dataset))])
                self.assertEqual(list(sampler), keys)
                fixed = TongjiTaskSampler(dataset, fixed_epoch=1)
                fixed.set_epoch(0)
                self.assertEqual(list(fixed), [(index, 1) for index in range(len(dataset))])

                batch = next(iter(DataLoader(
                    dataset, batch_size=2, sampler=[(0, 1), (3, 1)],
                    collate_fn=collate_tongji, num_workers=1,
                    multiprocessing_context="spawn",
                )))
                expected = collate_tongji([dataset[0, 1], sparse])
                for field in ("values", "value_visible", "point_valid", "target_soh"):
                    torch.testing.assert_close(batch[field], expected[field])
                self.assertFalse(batch["point_valid"][0, 1].any())
            finally:
                dataset.close()
                direct.close()
                subset.close()


    def test_multiple_cells_and_saved_splits(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture(root, ["A", "B", "C", "D", "E"])
            config = yaml.safe_load(CONFIG)
            partitions = {"train": ["A", "B"], "val": ["C"], "test": ["D", "E"]}
            config["data"].update(root=str(root), cell_ids=partitions)

            for independent_val in (True, False):
                with self.subTest(independent_val=independent_val):
                    if not independent_val:
                        partitions.pop("val")
                    datasets, splits = build_datasets(config)
                    try:
                        self.assertEqual(splits["train_cells"], ["A", "B"])
                        self.assertEqual(splits["test_cells"], ["D", "E"])
                        self.assertEqual(splits["val_cells"], ["C"] if independent_val else ["A", "B"])
                        for name in ("train", "val", "test"):
                            self.assertEqual(
                                {cell_id for cell_id, _ in datasets[name].tasks},
                                set(splits[f"{name}_cells"]),
                            )
                        self.assertFalse(set(datasets["train"].tasks) & set(datasets["val"].tasks))
                        self.assertEqual(make_loader(datasets["train"], config).batch_size, 2)
                        replay, _ = build_datasets(config, json.loads(json.dumps(splits)))
                        try:
                            self.assertEqual(replay["train"].tasks, datasets["train"].tasks)
                            self.assertEqual(replay["val"].tasks, datasets["val"].tasks)
                        finally:
                            for dataset in replay.values():
                                dataset.close()
                    finally:
                        for dataset in datasets.values():
                            dataset.close()

            partitions["val"] = ["D"]
            with self.assertRaisesRegex(ValueError, "Validation cell IDs"):
                build_datasets(config)


if __name__ == "__main__":
    unittest.main()
