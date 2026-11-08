"""Regression tests for task identity, exact masks, and epoch-aware sampling."""

import json
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data import (
    ObservationSchedule,
    TongjiTaskDataset,
    TongjiTaskSampler,
    collate_tongji,
)
from src.data.task_dataset import sample_visible


CONFIG = """
trainer:
  max_epochs: 2
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


class TongjiTaskTests(unittest.TestCase):
    def test_config_validation_and_channel_counts(self):
        config = yaml.safe_load(CONFIG)
        channels = np.ones((2, 6), dtype=bool)
        for keep, expected in ((0.9, 5), (4, 4), (1.0, 6)):
            with self.subTest(keep=keep):
                config["observation_schedule"][0]["value_keep"] = keep
                schedule = ObservationSchedule.from_config(config, feature_count=6)
                visible = sample_visible(channels, schedule.for_epoch(0).value_keep,
                                         np.random.default_rng(42))
                np.testing.assert_array_equal(visible.sum(axis=1), [expected, expected])

        for invalid in (0, 0.0, 7, 4.0, True, float("nan"), "0.9", "90.0%", "101%"):
            with self.subTest(invalid=invalid):
                config["observation_schedule"][0]["value_keep"] = invalid
                with self.assertRaises(ValueError):
                    ObservationSchedule.from_config(config, feature_count=6)

        config = yaml.safe_load(CONFIG)
        phase = ObservationSchedule.from_config(config, feature_count=5).for_epoch(1)
        self.assertEqual(phase.cycle_fraction, 0.5)
        self.assertEqual(phase.segment_fraction, 0.29)

        for field in ("cycle_fraction", "segment_fraction", "value_keep"):
            with self.subTest(field=field):
                invalid_config = yaml.safe_load(CONFIG)
                invalid_config["observation_schedule"][1][field] = "90%"
                with self.assertRaisesRegex(ValueError, field):
                    ObservationSchedule.from_config(invalid_config, feature_count=5)

        channels = np.ones((1, 100), dtype=bool)
        self.assertEqual(sample_visible(channels, 0.29, np.random.default_rng(42)).sum(), 29)

        config["observation_schedule"][1]["start_epoch"] = 0
        with self.assertRaises(ValueError):
            ObservationSchedule.from_config(config, feature_count=5)

    def test_task_targets_masks_padding_and_worker_replay(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            cell = root / "cells" / "A"
            cell.mkdir(parents=True)
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
            for name, array in arrays.items():
                np.save(cell / f"{name}.npy", array)
            (root / "manifest.json").write_text(
                json.dumps({"complete": True, "cells": [{"cell_id": "A"}]}), encoding="utf-8"
            )

            schedule = ObservationSchedule.read(StringIO(CONFIG), feature_count=5)
            dataset = TongjiTaskDataset(root, ["A"], schedule, horizon=3)
            try:
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


if __name__ == "__main__":
    unittest.main()
