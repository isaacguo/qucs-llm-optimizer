"""Tests for shared goal distribution and is_goal_met."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from training.goals import (
    GoalDistributionConfig,
    GoalSpec,
    is_goal_met,
    load_goal_distribution,
    sample_goal_from_distribution,
)


class LoadGoalDistributionTests(unittest.TestCase):
    def test_loads_coldstart_defaults(self):
        path = Path("configs/goal_distribution.yaml")
        dist = load_goal_distribution(path)
        self.assertEqual(dist.freq_min_hz, 1e9)
        self.assertEqual(dist.freq_max_hz, 10e9)
        self.assertEqual(dist.depth_db_max, -55.0)
        self.assertLess(dist.depth_db_min, dist.depth_db_max)
        self.assertFalse(dist.heldout_enabled)


class SampleFromDistributionTests(unittest.TestCase):
    def test_samples_inside_freq_and_depth_ranges(self):
        dist = GoalDistributionConfig(
            freq_min_hz=1e9,
            freq_max_hz=10e9,
            depth_db_min=-80.0,
            depth_db_max=-55.0,
            band_half_width_hz=1e9,
            heldout_enabled=False,
            freq_bin_hz=1e9,
            depth_bin_db=5.0,
        )
        for seed in range(20):
            goal = sample_goal_from_distribution(seed, dist)
            self.assertGreaterEqual(goal.target_freq_hz, 1e9)
            self.assertLessEqual(goal.target_freq_hz, 10e9)
            self.assertGreaterEqual(goal.target_depth_db, -80.0)
            self.assertLessEqual(goal.target_depth_db, -55.0)


class IsGoalMetTests(unittest.TestCase):
    def test_meets_own_target_depth(self):
        goal = GoalSpec(5.5e9, (4.5e9, 6.5e9), target_depth_db=-70.0)
        # |S21| for -70 dB ≈ 3.162e-4
        self.assertTrue(is_goal_met({"total_cost": 3.0e-4}, goal))
        self.assertFalse(is_goal_met({"total_cost": 1.0e-3}, goal))

    def test_minus_60_fails_minus_70_target(self):
        goal = GoalSpec(5.5e9, (4.5e9, 6.5e9), target_depth_db=-70.0)
        # -60 dB ≈ 1e-3
        self.assertFalse(is_goal_met({"total_cost": 1.0e-3}, goal))


if __name__ == "__main__":
    unittest.main()
