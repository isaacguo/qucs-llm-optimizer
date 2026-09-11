from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qucs_sim import render_schematic  # noqa: E402
from tasks.butterworth_bpf5 import INITIAL_GUESS, TEMPLATE_PATH  # noqa: E402


class BpfTemplateTests(unittest.TestCase):
    def test_render_contains_topology_and_plots(self):
        text = render_schematic(
            dict(INITIAL_GUESS),
            f0_hz=150e6,
            sweep_start_hz=0.0,
            sweep_stop_hz=300e6,
            sweep_points=301,
            template_path=TEMPLATE_PATH,
        )
        self.assertIn("<Qucs Schematic", text)
        self.assertIn("Pac P1 ", text)
        self.assertIn("Pac P2 ", text)
        for n in range(1, 6):
            self.assertIn(f"<L L{n} ", text)
            self.assertIn(f"<C C{n} ", text)
        self.assertIn("163.9296 nH", text)
        self.assertIn("6.8675 pF", text)
        self.assertIn("S21_dB=dB(S[2,1])", text)
        self.assertIn("S11_dB=dB(S[1,1])", text)
        self.assertIn("<Rect ", text)
        self.assertIn("<Smith ", text)
        self.assertGreaterEqual(text.count("<Polar "), 2)
        self.assertNotIn("MRSTUB", text)
        self.assertNotIn("SUBST", text)


if __name__ == "__main__":
    unittest.main()
