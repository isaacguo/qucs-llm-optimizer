# tests/test_common_prompts.py
from __future__ import annotations

import unittest

from training.common.goals import GoalSpec
from training.common.prompts import SYSTEM_PROMPT, TurnRecord, build_multiturn_prompt


class CommonPromptsTests(unittest.TestCase):
    def test_system_prompt_mentions_butterfly(self):
        self.assertIn("butterfly", SYSTEM_PROMPT.lower())

    def test_build_multiturn_prompt_returns_system_and_user(self):
        goal = GoalSpec(target_freq_hz=5.5e9, band_hz=(4.5e9, 6.5e9), target_depth_db=-70.0)
        messages = build_multiturn_prompt(
            goal,
            turn_index=0,
            params={"ri": 0.30, "ro": 8.00, "alpha": 90.0, "Wf": 0.60, "Lc": 3.00},
            cost={
                "total_cost": 1e-3,
                "best_s21_mag": 1e-3,
                "best_freq_hz": 5.5e9,
            },
            history=[],
        )
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIsInstance(TurnRecord, type)


if __name__ == "__main__":
    unittest.main()
