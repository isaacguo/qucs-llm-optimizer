"""Tests for BPF tuning-skills system prompt + skill-following policy (v4)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from jobs.bpf5_agent.skills import (  # noqa: E402
    choose_intent_from_skills,
    format_skills_reasoning,
    load_skills_system_prompt,
    peak_margin_hz,
)
from jobs.bpf5_agent.goals import BpfGoalSpec  # noqa: E402


class BpfTuningSkillsTests(unittest.TestCase):
    def test_load_skills_mentions_bw_aware_margin(self):
        text = load_skills_system_prompt()
        self.assertIn("Skill A", text)
        self.assertIn("Narrower BW", text)
        self.assertIn("3 MHz", text)
        self.assertIn("passband recovery", text.lower())

    def test_peak_margin_caps_at_3mhz_for_wide_bw(self):
        self.assertAlmostEqual(peak_margin_hz(5e6), 2.5e6)
        self.assertAlmostEqual(peak_margin_hz(15e6), 3e6)  # not 7.5 MHz
        self.assertAlmostEqual(peak_margin_hz(40e6), 3e6)

    def test_wide_bw_peak_just_outside_still_centers(self):
        """BW=15 regression: peak 5 MHz above window must center, not narrow."""
        goal = BpfGoalSpec(
            f_low_hz=43.6e6,
            f_high_hz=58.6e6,
            passband_il_max_db=-1.0,
            stopband_atten_min_db=-20.0,
        )
        cost = {
            "passband_min_s21_db": -84.0,
            "stopband_max_s21_db": -18.0,
            "s21_peak_freq_hz": 64e6,
            "goal_met": False,
        }
        intent, diag = choose_intent_from_skills(cost, goal, iteration=20)
        self.assertIn("center down", diag.lower())
        self.assertTrue(intent["L1"].startswith("increase"))

    def test_near_outside_prefers_slight_center_not_narrow(self):
        goal = BpfGoalSpec(
            f_low_hz=120e6,
            f_high_hz=130e6,
            passband_il_max_db=-1.0,
            stopband_atten_min_db=-20.0,
        )
        # 2 MHz above hi with margin=2.5 → near
        cost = {
            "passband_min_s21_db": -5.0,
            "stopband_max_s21_db": 0.0,
            "s21_peak_freq_hz": 132e6,
            "goal_met": False,
        }
        intent, diag = choose_intent_from_skills(cost, goal, iteration=4)
        self.assertIn("center down", diag.lower())
        self.assertNotIn("narrow", diag.lower())

    def test_format_skills_reasoning_is_sft_cot_not_stub(self):
        goal = BpfGoalSpec(
            f_low_hz=202.5e6,
            f_high_hz=207.5e6,
            passband_il_max_db=-1.0,
            stopband_atten_min_db=-20.0,
        )
        cost = {
            "passband_min_s21_db": -10.2,
            "stopband_max_s21_db": 0.0,
            "s21_peak_freq_hz": 222e6,
            "goal_met": False,
        }
        intent, diag = choose_intent_from_skills(cost, goal, iteration=3)
        text = format_skills_reasoning(cost, goal, intent, diag, iteration=3)
        self.assertNotIn("System skills were present", text)
        self.assertNotIn("intent=", text)
        self.assertIn("Observe metrics", text)
        self.assertIn("Diagnose with Skill A", text)
        self.assertIn("center-down", text.lower())
        self.assertIn(diag.split("—")[0].strip()[:20], text)

    def test_center_up_when_peak_far_below_window(self):
        goal = BpfGoalSpec(f_low_hz=120e6, f_high_hz=130e6)
        cost = {
            "passband_min_s21_db": -40.0,
            "stopband_max_s21_db": 0.0,
            "s21_peak_freq_hz": 80e6,
            "goal_met": False,
        }
        intent, diag = choose_intent_from_skills(cost, goal, iteration=0)
        self.assertIn("center up", diag.lower())
        self.assertTrue(intent["L1"].startswith("decrease"))
        self.assertTrue(intent["C2"].startswith("decrease"))

    def test_pb_ok_sb_bad_peak_inside_avoids_strong(self):
        goal = BpfGoalSpec(
            f_low_hz=120e6,
            f_high_hz=130e6,
            passband_il_max_db=-1.0,
            stopband_atten_min_db=-20.0,
        )
        cost = {
            "passband_min_s21_db": -0.2,
            "stopband_max_s21_db": -2.0,
            "s21_peak_freq_hz": 125e6,
            "goal_met": False,
        }
        intent, diag = choose_intent_from_skills(cost, goal, iteration=5)
        self.assertIn("narrow", diag.lower())
        self.assertNotEqual(intent["L1"], "increase_strong")
        self.assertEqual(intent["L1"], "increase")  # normal
        self.assertEqual(intent["C2"], "increase")

    def test_sb_ok_pb_collapsed_widens(self):
        goal = BpfGoalSpec(
            f_low_hz=120e6,
            f_high_hz=130e6,
            passband_il_max_db=-1.0,
            stopband_atten_min_db=-20.0,
        )
        cost = {
            "passband_min_s21_db": -15.0,
            "stopband_max_s21_db": -40.0,
            "s21_peak_freq_hz": 125e6,
            "goal_met": False,
        }
        intent, diag = choose_intent_from_skills(cost, goal, iteration=10)
        self.assertIn("widen", diag.lower())
        self.assertIn("recover", diag.lower())
        self.assertTrue(intent["L1"].startswith("decrease"))
        self.assertTrue(intent["C2"].startswith("decrease"))

    def test_oscillation_switches_to_narrow(self):
        goal = BpfGoalSpec(f_low_hz=120e6, f_high_hz=130e6)
        prev = {
            "passband_min_s21_db": -5.0,
            "stopband_max_s21_db": 0.0,
            "s21_peak_freq_hz": 100e6,
            "goal_met": False,
        }
        cost = {
            "passband_min_s21_db": -5.0,
            "stopband_max_s21_db": 0.0,
            "s21_peak_freq_hz": 160e6,
            "goal_met": False,
        }
        intent, diag = choose_intent_from_skills(
            cost, goal, iteration=3, prev_cost=prev
        )
        self.assertIn("oscillation", diag.lower())
        self.assertTrue(intent["L1"].startswith("increase"))  # narrow: series L up

    def test_sb_ok_pb_near_miss_widens_slight(self):
        goal = BpfGoalSpec(
            f_low_hz=120e6,
            f_high_hz=130e6,
            passband_il_max_db=-1.0,
            stopband_atten_min_db=-20.0,
        )
        cost = {
            "passband_min_s21_db": -1.8,
            "stopband_max_s21_db": -40.0,
            "s21_peak_freq_hz": 125e6,
            "goal_met": False,
        }
        intent, diag = choose_intent_from_skills(cost, goal, iteration=10)
        self.assertIn("widen", diag.lower())
        self.assertEqual(intent["L1"], "decrease_slight")
        self.assertEqual(intent["C2"], "decrease_slight")

    def test_finish_line_keeps_slight_narrow_while_sb_short(self):
        goal = BpfGoalSpec(
            f_low_hz=152.5e6,
            f_high_hz=157.5e6,
            passband_il_max_db=-1.0,
            stopband_atten_min_db=-20.0,
        )
        cost = {
            "passband_min_s21_db": -0.0,
            "stopband_max_s21_db": -19.65,
            "s21_peak_freq_hz": 155e6,
            "goal_met": False,
        }
        intent, diag = choose_intent_from_skills(
            cost,
            goal,
            iteration=11,
            prev_thinking="Skill A/C: narrow-BW (slight), peak=inside.",
        )
        self.assertIn("slight narrow", diag.lower())
        self.assertEqual(intent["L1"], "increase_slight")

    def test_finish_line_keeps_widen_when_sb_has_margin(self):
        """CF=171 style: SB deep, PB −1.3 after widen — keep slight widen."""
        goal = BpfGoalSpec(
            f_low_hz=163.5e6,
            f_high_hz=178.5e6,
            passband_il_max_db=-1.0,
            stopband_atten_min_db=-20.0,
        )
        cost = {
            "passband_min_s21_db": -1.34,
            "stopband_max_s21_db": -21.12,
            "s21_peak_freq_hz": 170e6,
            "goal_met": False,
        }
        intent, diag = choose_intent_from_skills(
            cost,
            goal,
            iteration=12,
            prev_thinking="Skill A/E: widen-BW (slight) to recover passband.",
        )
        self.assertIn("widen", diag.lower())
        self.assertEqual(intent["L1"], "decrease_slight")

    def test_finish_line_hold_when_sb_barely_ok_after_widen(self):
        goal = BpfGoalSpec(
            f_low_hz=185.83e6,
            f_high_hz=190.83e6,
            passband_il_max_db=-1.0,
            stopband_atten_min_db=-20.0,
        )
        cost = {
            "passband_min_s21_db": -1.2,
            "stopband_max_s21_db": -20.3,
            "s21_peak_freq_hz": 186e6,
            "goal_met": False,
        }
        intent, diag = choose_intent_from_skills(
            cost,
            goal,
            iteration=20,
            prev_thinking="Skill F finish: slight widen once.",
        )
        self.assertIn("hold", diag.lower())
        self.assertTrue(all(v == "hold" for v in intent.values()))

    def test_peak_inside_off_center_nudges_before_narrow(self):
        goal = BpfGoalSpec(
            f_low_hz=151.88e6,
            f_high_hz=166.88e6,
            passband_il_max_db=-1.0,
            stopband_atten_min_db=-20.0,
        )
        cost = {
            "passband_min_s21_db": -8.5,
            "stopband_max_s21_db": -5.0,
            "s21_peak_freq_hz": 153e6,  # low edge of 15 MHz window
            "goal_met": False,
        }
        intent, diag = choose_intent_from_skills(cost, goal, iteration=5)
        self.assertIn("center up", diag.lower())


if __name__ == "__main__":
    unittest.main()
