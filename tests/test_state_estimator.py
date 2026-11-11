"""Regression tests for staged interaction and Tongji timeline alignment."""

import unittest

import torch

from src.core.embeddings import TongjiInputEncoder
from src.core.state_estimator import StateEstimator


class StateEstimatorTests(unittest.TestCase):
    def setUp(self):
        embedding = {
            "scalar": {"output_dim": 4, "hidden_dim": 8, "activation": "gelu"},
            "categorical": {"output_dim": 3},
            "time_scale_s": 10.0,
            "cycle_distance_scale": 10.0,
        }
        normalization = {}
        for name, count in (("feature", 5), ("static", 3), ("condition", 4)):
            normalization[f"{name}_mean"] = [0.0] * count
            normalization[f"{name}_std"] = [1.0] * count
        with torch.random.fork_rng():
            torch.manual_seed(42)
            self.estimator = StateEstimator(
                TongjiInputEncoder(embedding, normalization, [{1}] * 5),
                d_model=8,
                state_dim=6,
                num_heads=2,
                feedforward_dim=16,
                field_layers=2,
                time_layers=2,
                cycle_layers=2,
                dropout=0.0,
            ).eval()

        # Sample 0 observes cycles 0 and 3. Sample 1 has no history before cycle 7.
        point_valid = torch.tensor(
            [[[True, True], [True, False]], [[True, False], [False, False]]]
        )
        self.batch = {
            "values": torch.arange(40, dtype=torch.float32).reshape(2, 2, 2, 5) / 10,
            "value_visible": point_valid.unsqueeze(-1).expand(-1, -1, -1, 5),
            "point_valid": point_valid,
            "cycle_valid": point_valid.any(dim=-1),
            "time_in_cycle_s": torch.ones(2, 2, 2, dtype=torch.float64),
            "cycle_id": torch.tensor([[0, 3], [7, 0]]),
            "last_cycle_id": torch.tensor([3, 7]),
            "static_categorical": torch.ones(2, 5, dtype=torch.long),
            "static_numeric": torch.ones(2, 3),
            "static_valid": torch.ones(2, 8, dtype=torch.bool),
            "history_cycle_id": torch.tensor([[0, 1, 2], [0, 0, 0]]),
            "history_u": torch.ones(2, 3, 4),
            "history_u_valid": torch.tensor([True, False])[:, None, None].expand(
                2, 3, 4
            ),
        }

    def _field_outputs(self, batch):
        outputs = []
        hook = self.estimator.field_attention.register_forward_hook(
            lambda _module, _args, result: outputs.append(result)
        )
        try:
            with torch.inference_mode():
                state = self.estimator(batch)
        finally:
            hook.remove()
        return state, outputs

    def test_static_attributes_change_observed_point_tokens(self):
        _, baseline = self._field_outputs(self.batch)
        changed = self.batch["static_numeric"].clone()
        changed[0, 0] = 7.0
        _, modified = self._field_outputs(dict(self.batch, static_numeric=changed))
        self.assertEqual([tuple(x.shape) for x in baseline], [(4, 8), (2, 8)])
        self.assertFalse(torch.allclose(baseline[0][:3], modified[0][:3]))
        torch.testing.assert_close(baseline[0][3], modified[0][3])

    def test_history_only_cycle_changes_state_without_changing_observed_points(self):
        baseline, baseline_fields = self._field_outputs(self.batch)
        changed = self.batch["history_u"].clone()
        changed[0, 1, 0] = 9.0
        changed[1] = float("nan")
        modified, modified_fields = self._field_outputs(
            dict(self.batch, history_u=changed)
        )
        torch.testing.assert_close(baseline_fields[0], modified_fields[0])
        self.assertFalse(torch.allclose(baseline_fields[1][0], modified_fields[1][0]))
        self.assertFalse(torch.allclose(baseline[0], modified[0]))
        torch.testing.assert_close(baseline[1], modified[1])
