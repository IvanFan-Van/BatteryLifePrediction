"""Behavior checks: physical label validity, sparse-cycle alignment, no hidden-value leak."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from battery_data.tongji import TongjiDataset, collate_tongji
from scripts.preprocess_tongji import extract_cycle, protocol


class TongjiDataTests(unittest.TestCase):
    def test_extraction_keeps_charge_and_rejects_incomplete_discharge(self):
        cycle = dict(current_in_A=[0, 1, 1, 0, -1, -1, 0],
                     voltage_in_V=[3, 3.5, 4.2, 4.1, 3.8, 2.5, 3],
                     charge_capacity_in_Ah=[0, 0.4, 1, 1, 0, 0, 0],
                     discharge_capacity_in_Ah=[0, 0, 0, 0, 0.001, 0.9, 0.9],
                     time_in_s=[100, 110, 120, 130, 140, 150, 160])
        result = extract_cycle(cycle, 1.0, 2.5)
        np.testing.assert_array_equal(result['raw_sample_index'], [1, 2])
        np.testing.assert_array_equal(result['time_in_cycle'], [10, 20])
        self.assertTrue(result['soh_valid'])
        self.assertAlmostEqual(result['soh'], 0.9)
        self.assertFalse(result['value_valid'][:, 4].any())
        self.assertTrue(np.isfinite(result['values']).all())
        cycle['voltage_in_V'][5] = 3.5
        incomplete = extract_cycle(cycle, 1.0, 2.5)
        self.assertFalse(incomplete['soh_valid'])
        self.assertIn('discharge_cutoff_not_reached', incomplete['label_issues'])
        self.assertEqual(len(incomplete['values']), 2)

    def test_protocol_rates_are_decoded_not_decimal_guessed(self):
        np.testing.assert_allclose(protocol('Tongji1_CY25-025_1--1'), [25, .25, 1, 1800])
        np.testing.assert_allclose(protocol('Tongji3_CY25-05_4--3'), [25, .5, 4, 3600])

    def test_sampling_sparse_targets_padding_and_determinism(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cell = root / 'cells' / 'test'
            cell.mkdir(parents=True)
            ids = np.array([2, 4, 5, 8], np.int64)
            lengths = [4, 5, 6, 7]
            values = np.ones((sum(lengths), 5), np.float32)
            valid = np.ones_like(values, bool)
            valid[:, 4] = False
            values[:, 4] = 0
            arrays = dict(values=values, value_valid=valid,
                          time_in_cycle=np.concatenate([np.arange(n, dtype=float) for n in lengths]),
                          raw_sample_index=np.concatenate([np.arange(n) for n in lengths]),
                          cycle_offsets=np.r_[0, np.cumsum(lengths)], cycle_id=ids,
                          static_categorical=np.array([1, 0]), static_numeric=np.array([1.0], np.float32),
                          static_valid=np.array([True, False, True]),
                          conditions=np.tile(np.array([25, .5, 1, 1800], np.float32), (4, 1)),
                          conditions_valid=np.ones((4, 4), bool),
                          soh=np.array([1, .9, .8, .7], np.float32),
                          soh_valid=np.ones(4, bool))
            for key, array in arrays.items():
                np.save(cell / f'{key}.npy', array)
            (root / 'manifest.json').write_text(json.dumps(dict(complete=True, cells=[dict(cell_id='test')])))
            ds = TongjiDataset(root, history_cycles=(1, 2), segment_points=(2, 4),
                               horizon=4, max_anchor_cycle=2, deterministic=True)
            a, repeated = ds[0], ds[0]
            self.assertEqual(a['anchor_cycle'], 2)
            torch.testing.assert_close(a['values'][0], repeated['values'][0])
            torch.testing.assert_close(a['target_cycle_id'], torch.tensor([3, 4, 5, 6]))
            torch.testing.assert_close(a['target_valid'], torch.tensor([False, True, True, False]))
            torch.testing.assert_close(a['target_soh'], torch.tensor([0, .9, .8, 0]))
            samples = [ds[i] for i in range(8)]
            # Force a shorter view to ensure the padding branch is exercised.
            for key in ['values', 'value_valid', 'value_visible', 'time_in_cycle', 'raw_sample_index']:
                samples[0][key][0] = samples[0][key][0][:1]
            batch = collate_tongji(samples)
            pad = ~batch['point_valid']
            self.assertTrue(pad.any())
            self.assertFalse(batch['value_valid'][pad].any())
            self.assertFalse(batch['value_visible'][pad].any())
            self.assertFalse(batch['values'][pad].any())
            self.assertFalse(batch['value_valid'][..., 4].any())
            torch.testing.assert_close(batch['cycle_valid'], batch['point_valid'].any(-1))
            ds_masked = TongjiDataset(root, segment_points=(2, 4), horizon=4,
                                     max_anchor_cycle=2, channel_mask_probability=1,
                                     deterministic=True)
            masked = collate_tongji([ds_masked[0]])
            self.assertTrue(masked['point_valid'].any())
            self.assertTrue(masked['value_valid'][..., :4].any())
            self.assertFalse(masked['value_visible'].any())
            self.assertFalse(masked['values'].any())
            # Exercise Windows spawn workers / dataset serialization.
            loader = DataLoader(ds, batch_size=2, num_workers=2, collate_fn=collate_tongji)
            worker_batch = next(iter(loader))
            direct_batch = collate_tongji([ds[0], ds[1]])
            torch.testing.assert_close(worker_batch['values'], direct_batch['values'])
            ds.close()
            ds_masked.close()


if __name__ == '__main__':
    unittest.main()
