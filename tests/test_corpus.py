"""Tests for corpus gate, index, and coverage."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.corpus import (
    append_index,
    coverage_counts,
    gate_run,
    load_index,
)
from training.goals import GoalDistributionConfig


def _state_with_goal(depth_db: float, best_total_cost: float) -> dict:
    return {
        "goal": {
            "target_freq_hz": 5.5e9,
            "band_hz": [4.5e9, 6.5e9],
            "target_depth_db": depth_db,
        },
        "history": [
            {
                "iteration": 0,
                "intent": None,
                "cost": {"total_cost": 0.1},
            },
            {
                "iteration": 1,
                "intent": {"ro": "decrease_strong"},
                "cost": {"total_cost": best_total_cost},
            },
        ],
    }


class GateRunTests(unittest.TestCase):
    def test_eligible_when_own_target_met(self):
        # -72 dB-ish vs -70 target
        state = _state_with_goal(-70.0, 2.5e-4)
        corpus = gate_run(state)
        self.assertTrue(corpus["eligible"])
        self.assertEqual(corpus["goal_met_iteration"], 1)

    def test_reject_when_only_minus_60_vs_minus_70(self):
        state = _state_with_goal(-70.0, 1.0e-3)
        corpus = gate_run(state)
        self.assertFalse(corpus["eligible"])


class IndexTests(unittest.TestCase):
    def test_append_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "index.jsonl"
            append_index(path, {"run_id": "a", "goal": {"target_depth_db": -60}})
            append_index(path, {"run_id": "b", "goal": {"target_depth_db": -65}})
            rows = load_index(path)
        self.assertEqual([r["run_id"] for r in rows], ["a", "b"])


class CoverageTests(unittest.TestCase):
    def test_bins_count_hard_successes(self):
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
        index = [
            {
                "goal": {
                    "target_freq_hz": 5.2e9,
                    "target_depth_db": -62.0,
                    "band_hz": [4.2e9, 6.2e9],
                }
            }
        ]
        counts = coverage_counts(index, dist)
        self.assertEqual(sum(counts.values()), 1)


if __name__ == "__main__":
    unittest.main()
