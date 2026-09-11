from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qucs_sim import SimResult, parse_dataset, render_schematic, simulate  # noqa: E402
from tasks.butterworth_bpf5 import INITIAL_GUESS, TEMPLATE_PATH  # noqa: E402


def _has_qucs_tools() -> bool:
    from qucs_sim import resolve_qucs_s, resolve_qucsator

    try:
        resolve_qucs_s()
        resolve_qucsator()
        return True
    except FileNotFoundError:
        return False


class BpfTemplateTests(unittest.TestCase):
    def test_parse_dataset_accepts_pure_real_sparams(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "circuit.dat"
            path.write_text(
                "<Qucs Dataset 1.0.7>\n"
                "<indep frequency 2>\n"
                "  +0.00000000000000000000e+00\n"
                "  +1.00000000000000000000e+06\n"
                "</indep>\n"
                "<dep S[1,1] frequency>\n"
                "  +1.00000000000000000000e+00\n"
                "  +9.00000000000000000000e-01-j1.00000000000000000000e-01\n"
                "</dep>\n"
                "<dep S[2,1] frequency>\n"
                "  +0.00000000000000000000e+00\n"
                "  +1.00000000000000000000e-01+j2.00000000000000000000e-01\n"
                "</dep>\n"
            )
            res = parse_dataset(path)
        self.assertEqual(res.s11[0], 1 + 0j)
        self.assertEqual(res.s21[0], 0 + 0j)
        self.assertEqual(res.s11[1], complex(0.9, -0.1))
        self.assertEqual(res.s21[1], complex(0.1, 0.2))

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

    def test_simulate_forwards_template_path_to_render(self):
        captured: dict = {}

        def fake_render(
            params,
            f0_hz,
            sweep_start_hz,
            sweep_stop_hz,
            sweep_points,
            template_path=None,
        ):
            captured["template_path"] = template_path
            return "<Qucs Schematic 25.2.0>\n</Qucs Schematic>\n"

        fake_result = SimResult(freq_hz=[0.0], s11=[0j], s21=[0j])
        with tempfile.TemporaryDirectory() as td:
            with (
                patch("qucs_sim.render_schematic", side_effect=fake_render),
                patch("qucs_sim.export_netlist_from_sch"),
                patch("qucs_sim.export_layout_svg") as mock_layout,
                patch("qucs_sim.run_qucsator"),
                patch("qucs_sim.parse_dataset", return_value=fake_result),
            ):
                res = simulate(
                    dict(INITIAL_GUESS),
                    f0_hz=150e6,
                    sweep_start_hz=0.0,
                    sweep_stop_hz=300e6,
                    sweep_points=61,
                    workdir=Path(td),
                    export_layout=False,
                    template_path=TEMPLATE_PATH,
                )
            mock_layout.assert_not_called()
        self.assertIs(captured["template_path"], TEMPLATE_PATH)
        self.assertEqual(res, fake_result)

    @unittest.skipUnless(_has_qucs_tools(), "qucs-s / qucsator_rf not available")
    def test_simulate_bpf_returns_sweep_length(self):
        with tempfile.TemporaryDirectory() as td:
            res = simulate(
                dict(INITIAL_GUESS),
                f0_hz=150e6,
                sweep_start_hz=0.0,
                sweep_stop_hz=300e6,
                sweep_points=61,
                workdir=Path(td),
                export_layout=False,
                template_path=TEMPLATE_PATH,
            )
            self.assertEqual(len(res.freq_hz), 61)
            self.assertEqual(len(res.s21), 61)
            self.assertFalse((Path(td) / "layout.svg").exists())
            net = (Path(td) / "circuit.net").read_text()
            self.assertTrue("L1" in net and "C1" in net)


if __name__ == "__main__":
    unittest.main()
