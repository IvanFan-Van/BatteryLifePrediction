"""Checks for Tongji protocol decoding and physical label validity."""

import unittest

import numpy as np

from src.data.preprocess_tongji import extract_cycle, protocol


class TongjiDataTests(unittest.TestCase):
    def test_extraction_keeps_charge_and_rejects_incomplete_discharge(self):
        cycle = {
            "current_in_A": [0, 1, 1, 0, -1, -1, 0],
            "voltage_in_V": [3, 3.5, 4.2, 4.1, 3.8, 2.5, 3],
            "charge_capacity_in_Ah": [0, 0.4, 1, 1, 0, 0, 0],
            "discharge_capacity_in_Ah": [0, 0, 0, 0, 0.001, 0.9, 0.9],
            "time_in_s": [100, 110, 120, 130, 140, 150, 160],
        }
        result = extract_cycle(cycle, 1.0, 2.5)
        np.testing.assert_array_equal(result["raw_sample_index"], [1, 2])
        np.testing.assert_array_equal(result["time_in_cycle"], [10, 20])
        self.assertTrue(result["soh_valid"])
        self.assertAlmostEqual(result["soh"], 0.9)  # type: ignore
        self.assertFalse(result["value_valid"][:, 4].any())
        self.assertTrue(np.isfinite(result["values"]).all())
        cycle["voltage_in_V"][5] = 3.5
        incomplete = extract_cycle(cycle, 1.0, 2.5)
        self.assertFalse(incomplete["soh_valid"])
        self.assertIn("discharge_cutoff_not_reached", incomplete["label_issues"])
        self.assertEqual(len(incomplete["values"]), 2)

    def test_protocol_rates_are_decoded_not_decimal_guessed(self):
        np.testing.assert_allclose(
            protocol("Tongji1_CY25-025_1--1"), [25, 0.25, 1, 1800]
        )
        np.testing.assert_allclose(protocol("Tongji3_CY25-05_4--3"), [25, 0.5, 4, 3600])


if __name__ == "__main__":
    unittest.main()
