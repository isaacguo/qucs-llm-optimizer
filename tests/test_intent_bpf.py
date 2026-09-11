# tests/test_intent_bpf.py
from __future__ import annotations
import importlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from intent import apply_intent  # noqa: E402
from jobs.bpf5_agent.task import BOUNDS, INITIAL_GUESS, VARIABLES  # noqa: E402
from task_registry import clear_plugins_for_tests  # noqa: E402
from tasks import ensure_builtin_tasks, get_task  # noqa: E402


class IntentBpfTests(unittest.TestCase):
    def setUp(self):
        clear_plugins_for_tests()
        ensure_builtin_tasks()
        importlib.import_module("jobs.bpf5_agent.register").register()

    def test_get_task_bpf(self):
        t = get_task("butterworth_bpf5")
        self.assertEqual(t.variables, VARIABLES)
        self.assertFalse(t.export_layout)

    def test_increase_clamps_to_bound(self):
        params = dict(INITIAL_GUESS)
        hi = BOUNDS["L1"][1]
        params["L1"] = hi - 10.0
        out = apply_intent(
            params,
            {"L1": "increase_strong"},
            iteration=0,
            variables=VARIABLES,
            bounds=BOUNDS,
            step_mode="range",
        )
        self.assertEqual(out["L1"], hi)

    def test_unknown_intent_key_raises(self):
        with self.assertRaises(ValueError):
            apply_intent(
                dict(INITIAL_GUESS),
                {"L1": "hold", "ro": "increase"},
                iteration=0,
                variables=VARIABLES,
                bounds=BOUNDS,
            )

    def test_butterfly_default_still_works(self):
        from intent import INITIAL_GUESS as BF_INIT
        out = apply_intent(dict(BF_INIT), {"ro": "decrease"}, iteration=0)
        self.assertLess(out["ro"], BF_INIT["ro"])

    def test_relative_slight_step_on_small_L2_is_usable(self):
        """Range-fraction steps make L2≈6.56 nH jump ~80 nH; relative must be ≪ 20 nH."""
        params = dict(INITIAL_GUESS)
        self.assertAlmostEqual(params["L2"], 6.5577, places=4)
        out = apply_intent(
            params,
            {"L2": "increase_slight"},
            iteration=0,
            variables=VARIABLES,
            bounds=BOUNDS,
            step_mode="relative",
        )
        delta = out["L2"] - params["L2"]
        self.assertGreater(delta, 0.0)
        self.assertLess(delta, 20.0)
        # ~4% of 6.56 ≈ 0.26 nH (order-of-magnitude check)
        self.assertLess(delta, 1.0)

    def test_relative_slight_decrease_does_not_slam_to_lo(self):
        params = dict(INITIAL_GUESS)
        seed = params["L2"]
        out = apply_intent(
            params,
            {"L2": "decrease_slight"},
            iteration=0,
            variables=VARIABLES,
            bounds=BOUNDS,
            step_mode="relative",
        )
        lo = BOUNDS["L2"][0]
        self.assertGreater(out["L2"], lo)
        self.assertLess(out["L2"], seed)
        self.assertGreater(out["L2"], seed - 20.0)

    def test_range_mode_unchanged_for_butterfly_sized_step(self):
        from intent import INITIAL_GUESS as BF_INIT, BOUNDS as BF_BOUNDS
        out = apply_intent(
            dict(BF_INIT),
            {"ro": "decrease_slight"},
            iteration=0,
            step_mode="range",
        )
        lo, hi = BF_BOUNDS["ro"]
        expected_step = 0.04 * (hi - lo)  # slight, iteration 0, decay=1
        self.assertAlmostEqual(out["ro"], round(BF_INIT["ro"] - expected_step, 4), places=4)

    def test_bpf_task_defaults_to_relative_via_get_task(self):
        t = get_task("butterworth_bpf5")
        self.assertEqual(t.step_mode, "relative")
        out = apply_intent(
            dict(INITIAL_GUESS),
            {"L2": "increase_slight"},
            iteration=0,
            variables=t.variables,
            bounds=t.bounds,
            step_mode=t.step_mode,
        )
        self.assertLess(abs(out["L2"] - INITIAL_GUESS["L2"]), 20.0)


if __name__ == "__main__":
    unittest.main()
