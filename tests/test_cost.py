"""Tests for notch-at-target cost metrics (shunt butterfly on 2-port)."""
from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cost import TARGET_NOTCH_HZ, evaluate, s21_db  # noqa: E402
from intent import INITIAL_GUESS  # noqa: E402
from qucs_sim import SimResult, simulate  # noqa: E402


def _has_qucs_tools() -> bool:
    from qucs_sim import resolve_qucs_s, resolve_qucsator

    try:
        resolve_qucs_s()
        resolve_qucsator()
        return True
    except FileNotFoundError:
        return False


class CostUnitTests(unittest.TestCase):
    def test_evaluate_uses_s21_at_target_as_total_cost(self):
        freqs = [3e9, 4e9, 5e9, 5.5e9, 6e9, 7e9]
        s21 = [0.9 + 0j, 0.4 + 0j, 0.2 + 0j, 0.05 + 0j, 0.35 + 0j, 0.85 + 0j]
        s11 = [0.1 + 0j] * len(freqs)
        res = SimResult(freq_hz=freqs, s11=s11, s21=s21)
        report = evaluate(res, band_hz=(4e9, 6e9), target_hz=5.5e9)
        self.assertAlmostEqual(report.total_cost, 0.05)
        self.assertAlmostEqual(report.target_freq_hz, 5.5e9)
        self.assertAlmostEqual(report.target_s21_mag, 0.05)
        self.assertAlmostEqual(report.best_freq_hz, 5.5e9)
        self.assertAlmostEqual(report.best_s21_mag, 0.05)
        self.assertAlmostEqual(report.stopband_max_s21, 0.4)
        self.assertAlmostEqual(report.passband_low_mean, 0.9)
        self.assertAlmostEqual(report.passband_high_mean, 0.85)

    def test_s21_db_conversion(self):
        self.assertAlmostEqual(s21_db(10 ** -3.5), -70.0, places=5)
        self.assertTrue(math.isinf(s21_db(0.0)))


@unittest.skipUnless(_has_qucs_tools(), "qucs-s / qucsator_rf not available")
class CostIntegrationTests(unittest.TestCase):
    def test_initial_guess_reports_target_metric(self):
        with tempfile.TemporaryDirectory() as td:
            res = simulate(dict(INITIAL_GUESS), workdir=Path(td))
        report = evaluate(res, (4e9, 6e9), target_hz=TARGET_NOTCH_HZ)
        self.assertEqual(len(res.s21), len(res.freq_hz))
        self.assertEqual(report.target_freq_hz, TARGET_NOTCH_HZ)
        # Baseline notch is well below 5.5 GHz, so |S21|@5.5 is not tiny.
        self.assertGreater(report.total_cost, 0.05)
        self.assertLess(report.total_cost, 1.0)


if __name__ == "__main__":
    unittest.main()
