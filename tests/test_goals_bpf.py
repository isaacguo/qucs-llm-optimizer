# tests/test_goals_bpf.py
from __future__ import annotations
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from jobs.bpf5_agent.goals import (  # noqa: E402
    BpfGoalSpec,
    default_goal,
    goal_from_dict,
    is_goal_met,
    split_bands,
    validate_goal,
)


class BpfGoalTests(unittest.TestCase):
    def test_default_goal_passband(self):
        g = default_goal()
        self.assertEqual(g.f_low_hz, 135e6)
        self.assertEqual(g.f_high_hz, 165e6)
        self.assertEqual(g.sweep_hz, (0.0, 300e6))

    def test_validate_rejects_inverted_band(self):
        g = BpfGoalSpec(f_low_hz=200e6, f_high_hz=100e6)
        with self.assertRaises(ValueError):
            validate_goal(g)

    def test_split_bands_with_guard(self):
        g = BpfGoalSpec(
            f_low_hz=135e6,
            f_high_hz=165e6,
            stopband_guard_hz=10e6,
            sweep_hz=(0.0, 300e6),
        )
        pb, slo, shi = split_bands(g)
        self.assertEqual(pb, (135e6, 165e6))
        self.assertEqual(slo, (0.0, 125e6))
        self.assertEqual(shi, (175e6, 300e6))

    def test_is_goal_met_requires_both_gates(self):
        g = default_goal()
        cost_ok = {
            "passband_min_s21_db": -0.5,
            "stopband_max_s21_db": -25.0,
            "has_passband_samples": True,
            "has_stopband_samples": True,
        }
        self.assertTrue(is_goal_met(cost_ok, g))
        cost_bad_pb = dict(cost_ok, passband_min_s21_db=-3.0)
        self.assertFalse(is_goal_met(cost_bad_pb, g))

    def test_goal_from_dict_roundtrip_fields(self):
        g = goal_from_dict(
            {
                "f_low_hz": 100e6,
                "f_high_hz": 120e6,
                "passband_il_max_db": -1.5,
                "stopband_atten_min_db": -30.0,
                "stopband_guard_hz": 5e6,
                "sweep_hz": [0.0, 300e6],
            }
        )
        self.assertEqual(g.f_low_hz, 100e6)
        self.assertEqual(g.sweep_hz, (0.0, 300e6))


if __name__ == "__main__":
    unittest.main()
