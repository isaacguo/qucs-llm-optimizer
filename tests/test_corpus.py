"""Tests for corpus gate, index, and coverage."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.corpus import (
    append_index,
    coverage_counts,
    export_from_index,
    export_multiturn_sft_records,
    gate_run,
    load_index,
)
from training.goals import GoalDistributionConfig, GoalSpec
from training.rollout import SYSTEM_PROMPT, TurnRecord, build_multiturn_prompt


def _params(**overrides) -> dict:
    base = {"ri": 0.3, "ro": 8.0, "alpha": 90.0, "Wf": 0.6, "Lc": 3.0}
    base.update(overrides)
    return base


def _cost(total_cost: float, *, best_s21_mag: float = 0.01, best_freq_hz: float = 3.3e9) -> dict:
    return {
        "total_cost": total_cost,
        "best_s21_mag": best_s21_mag,
        "best_freq_hz": best_freq_hz,
    }


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


def _multiturn_fixture_state() -> dict:
    """Baseline + two tuning steps with full cost/params for prompt rebuild."""
    return {
        "goal": {
            "target_freq_hz": 5.5e9,
            "band_hz": [4.5e9, 6.5e9],
            "target_depth_db": -60.0,
        },
        "history": [
            {
                "iteration": 0,
                "intent": None,
                "params": _params(ro=8.0),
                "cost": _cost(0.1),
                "thinking": "Baseline only.",
            },
            {
                "iteration": 1,
                "intent": {"ro": "decrease_strong"},
                "params": _params(ro=5.0),
                "cost": _cost(0.05),
                "thinking": (
                    "Notch is too low. Shrink ro hard. "
                    "Ignore extra sentence four. "
                    "And five as well."
                ),
            },
            {
                "iteration": 2,
                "intent": {"alpha": "increase_slight"},
                "params": _params(ro=5.0, alpha=92.0),
                "cost": _cost(0.04),
                "thinking": "Fine tune alpha.\nSecond line.\nThird line.\nFourth line dropped.",
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


class MultiturnSftExportTests(unittest.TestCase):
    def test_export_isomorphic_to_build_multiturn_prompt(self):
        state = _multiturn_fixture_state()
        records = export_multiturn_sft_records(state)
        self.assertEqual(len(records), 2)

        goal = GoalSpec(
            target_freq_hz=5.5e9,
            band_hz=(4.5e9, 6.5e9),
            target_depth_db=-60.0,
        )
        baseline = state["history"][0]
        expected_prompt = build_multiturn_prompt(
            goal,
            turn_index=0,
            params=baseline["params"],
            cost=baseline["cost"],
            history=[],
            history_window=8,
            initial_db=None,
            best_db=None,
        )
        first = records[0]
        self.assertEqual(first["messages"][0]["content"], SYSTEM_PROMPT)
        self.assertIs(SYSTEM_PROMPT, expected_prompt[0]["content"])
        self.assertEqual(first["messages"][1]["content"], expected_prompt[1]["content"])
        assistant = first["messages"][2]["content"]
        self.assertIn("<reasoning>", assistant)
        self.assertIn("</reasoning>", assistant)
        self.assertIn('<intent>{"ro": "decrease_strong"}</intent>', assistant)
        # At most 3 sentences in compressed reasoning.
        reasoning = assistant.split("<reasoning>", 1)[1].split("</reasoning>", 1)[0]
        self.assertLessEqual(len([s for s in reasoning.replace("\n", ". ").split(". ") if s.strip()]), 3)
        self.assertIn("messages", first)
        self.assertIn("meta", first)

        # Second turn: history must include first completed TurnRecord facts.
        from cost import s21_db

        prior = TurnRecord(
            turn_index=0,
            prompt=[],
            completion_text="",
            valid=True,
            intent={"ro": "decrease_strong"},
            params_before=baseline["params"],
            params_after=state["history"][1]["params"],
            db_before=s21_db(baseline["cost"]["total_cost"]),
            db_after=s21_db(state["history"][1]["cost"]["total_cost"]),
        )
        mid = state["history"][1]
        expected_second = build_multiturn_prompt(
            goal,
            turn_index=1,
            params=mid["params"],
            cost=mid["cost"],
            history=[prior],
            history_window=8,
            initial_db=s21_db(baseline["cost"]["total_cost"]),
            best_db=min(
                s21_db(baseline["cost"]["total_cost"]),
                s21_db(mid["cost"]["total_cost"]),
            ),
        )
        self.assertEqual(records[1]["messages"][1]["content"], expected_second[1]["content"])

    def test_export_from_index_ignores_unlisted_runs(self):
        state = _multiturn_fixture_state()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            listed = root / "run_a"
            unlisted = root / "run_b"
            listed.mkdir()
            unlisted.mkdir()
            (listed / "state.json").write_text(json.dumps(state))
            (unlisted / "state.json").write_text(json.dumps(state))
            index_path = root / "index.jsonl"
            append_index(
                index_path,
                {
                    "run_id": "run_a",
                    "run_dir": "run_a",
                    "goal": state["goal"],
                    "best_db": -26.0,
                    "n_steps": 2,
                    "goal_met_iteration": 2,
                },
            )
            records = export_from_index(index_path, root)
        # Only listed run contributes (2 tuning turns).
        self.assertEqual(len(records), 2)
        self.assertTrue(all(r["meta"].get("run_id") == "run_a" for r in records))


if __name__ == "__main__":
    unittest.main()
