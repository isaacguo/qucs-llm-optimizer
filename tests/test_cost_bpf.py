# tests/test_cost_bpf.py
from __future__ import annotations
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from jobs.bpf5_agent.cost import evaluate  # noqa: E402
from jobs.bpf5_agent.goals import default_goal  # noqa: E402
from qucs_sim import SimResult  # noqa: E402


def _db(mag: float) -> float:
    return 20.0 * math.log10(mag)


class CostBpfTests(unittest.TestCase):
    def test_total_cost_passband_and_stopband(self):
        # freqs: stop, pass, pass, stop
        freq = [50e6, 140e6, 150e6, 250e6]
        # |S21|: leaky stop, good pass, good pass, leaky stop
        s21 = [0.2 + 0j, 0.95 + 0j, 0.9 + 0j, 0.15 + 0j]
        s11 = [0.1 + 0j] * 4
        res = SimResult(freq_hz=freq, s11=s11, s21=s21)
        report = evaluate(res, default_goal(), lam=1.0)
        # passband worst (1 - 0.9) = 0.1; stopband max |S21| = 0.2
        self.assertAlmostEqual(report.total_cost, 0.1 + 0.2, places=9)
        self.assertAlmostEqual(report.passband_min_s21_db, _db(0.9), places=6)
        self.assertAlmostEqual(report.stopband_max_s21_db, _db(0.2), places=6)
        self.assertTrue(report.has_passband_samples)
        self.assertTrue(report.has_stopband_samples)
        # peak |S21| is at 140 MHz (0.95)
        self.assertEqual(report.s21_peak_freq_hz, 140e6)

    def test_empty_passband_raises(self):
        res = SimResult(freq_hz=[10e6, 20e6], s11=[0j, 0j], s21=[0.1j, 0.1j])
        with self.assertRaises(ValueError):
            evaluate(res, default_goal())

    def test_both_stopbands_empty_raises(self):
        # sweep only inside passband+guard so stop bands empty
        g = default_goal()
        # override via goal with tiny sweep equal to passband
        from jobs.bpf5_agent.goals import BpfGoalSpec
        tight = BpfGoalSpec(
            f_low_hz=140e6,
            f_high_hz=160e6,
            stopband_guard_hz=50e6,
            sweep_hz=(140e6, 160e6),
        )
        res = SimResult(
            freq_hz=[150e6],
            s11=[0j],
            s21=[0.99 + 0j],
        )
        with self.assertRaises(ValueError):
            evaluate(res, tight)


if __name__ == "__main__":
    unittest.main()
