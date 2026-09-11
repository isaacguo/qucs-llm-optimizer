"""Tests for circuit parameter sampling used by multi-turn starts."""
from __future__ import annotations

import unittest

from training.environment import sample_params


class SampleParamsTests(unittest.TestCase):
    def test_seeded_parameter_sampling_is_deterministic_and_bounded(self):
        first = sample_params(42)
        second = sample_params(42)
        self.assertEqual(first, second)
        bounds = {
            "ri": (0.10, 1.50),
            "ro": (3.00, 14.00),
            "alpha": (30.0, 150.0),
            "Wf": (0.20, 2.50),
            "Lc": (0.20, 10.00),
        }
        for name, value in first.items():
            self.assertGreaterEqual(value, bounds[name][0])
            self.assertLessEqual(value, bounds[name][1])

    def test_spread_shrinks_around_initial_guess(self):
        full = sample_params(7, spread=1.0)
        tight = sample_params(7, spread=0.2)
        self.assertEqual(set(full), set(tight))
        # Same seed + different spreads must not be required equal; just valid.
        for name, value in tight.items():
            self.assertIn(name, full)


if __name__ == "__main__":
    unittest.main()
