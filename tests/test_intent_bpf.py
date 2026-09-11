# tests/test_intent_bpf.py
from __future__ import annotations
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intent import apply_intent  # noqa: E402
from tasks import get_task  # noqa: E402
from tasks.butterworth_bpf5 import BOUNDS, INITIAL_GUESS, VARIABLES  # noqa: E402


class IntentBpfTests(unittest.TestCase):
    def test_get_task_bpf(self):
        t = get_task("butterworth_bpf5")
        self.assertEqual(t.variables, VARIABLES)
        self.assertFalse(t.export_layout)

    def test_increase_clamps_to_bound(self):
        params = dict(INITIAL_GUESS)
        params["L1"] = 1990.0
        out = apply_intent(
            params,
            {"L1": "increase_strong"},
            iteration=0,
            variables=VARIABLES,
            bounds=BOUNDS,
        )
        self.assertEqual(out["L1"], 2000.0)

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


if __name__ == "__main__":
    unittest.main()
