"""Tests for trainable start headroom filtering."""
from __future__ import annotations

import unittest

from training.common.goals import GoalSpec
from training.grpo.starts import is_start_too_deep, sample_params_with_headroom


class StartHeadroomTests(unittest.TestCase):
    def test_too_deep_when_within_margin_of_target(self):
        self.assertTrue(is_start_too_deep(-28.0, target_depth_db=-30.0, headroom_db=5.0))
        self.assertTrue(is_start_too_deep(-30.0, target_depth_db=-30.0, headroom_db=5.0))
        self.assertFalse(is_start_too_deep(-20.0, target_depth_db=-30.0, headroom_db=5.0))

    def test_resample_skips_deep_starts(self):
        goal = GoalSpec(5e9, (4e9, 6e9), target_depth_db=-30.0)
        calls = {"n": 0}
        dbs = [-32.0, -31.0, -12.0]

        def cost_fn(params):
            db = dbs[min(calls["n"], len(dbs) - 1)]
            calls["n"] += 1
            mag = 10 ** (db / 20.0)
            return {"total_cost": mag, "best_s21_mag": mag, "best_freq_hz": 5e9}

        params, initial_db, seed_used, n_resamples = sample_params_with_headroom(
            start_seed=10,
            goal=goal,
            spread=0.35,
            headroom_db=5.0,
            max_tries=5,
            cost_fn=cost_fn,
        )
        self.assertAlmostEqual(initial_db, -12.0, places=3)
        self.assertEqual(n_resamples, 2)
        self.assertEqual(seed_used, 12)
        self.assertIsInstance(params, dict)


if __name__ == "__main__":
    unittest.main()
