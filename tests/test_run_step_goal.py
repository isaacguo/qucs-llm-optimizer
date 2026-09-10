"""Goal-conditioned RunState and run_step helpers."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import run_step
from state import RunState
from training.goals import GoalSpec


@dataclass
class FakeCost:
    total_cost: float
    best_s21_mag: float
    best_freq_hz: float
    mean_cost: float = 0.5
    low_edge_cost: float = 0.5
    center_cost: float = 0.5
    high_edge_cost: float = 0.5
    stopband_max_s21: float = 0.5
    worst_freq_hz: float = 6e9
    target_freq_hz: float = 7e9
    target_s21_mag: float = 0.1


class RunStateGoalPersistenceTests(unittest.TestCase):
    def test_round_trip_persists_goal_and_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_a"
            state = RunState(run_dir)
            goal = {
                "target_freq_hz": 7e9,
                "band_hz": [6e9, 8e9],
                "target_depth_db": -60.0,
            }
            initial = {"ri": 0.5, "ro": 1.0, "alpha": 45.0, "Wf": 0.2, "Lc": 2.0}
            state.set_run_meta(
                goal=goal,
                start_seed=42,
                initial_params=initial,
                corpus={"eligible": False, "reason": "pending"},
            )

            reloaded = RunState(run_dir)
            self.assertEqual(reloaded.goal["target_freq_hz"], 7e9)
            self.assertEqual(reloaded.goal["target_depth_db"], -60.0)
            self.assertEqual(reloaded.goal["band_hz"], [6e9, 8e9])
            self.assertEqual(reloaded.start_seed, 42)
            self.assertEqual(reloaded.initial_params, initial)
            self.assertEqual(reloaded.corpus["reason"], "pending")
            raw = json.loads((run_dir / "state.json").read_text())
            self.assertEqual(raw["goal"]["target_freq_hz"], 7e9)
            self.assertEqual(raw["start_seed"], 42)


class GoalHelpersTests(unittest.TestCase):
    def test_goal_from_state_uses_stored_goal(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = RunState(Path(tmp))
            state.set_run_meta(
                goal={
                    "target_freq_hz": 7e9,
                    "band_hz": [6e9, 8e9],
                    "target_depth_db": -60.0,
                }
            )
            goal = run_step._goal_from_state(state)
            self.assertIsInstance(goal, GoalSpec)
            self.assertEqual(goal.target_freq_hz, 7e9)
            self.assertEqual(goal.band_hz, (6e9, 8e9))
            self.assertEqual(goal.target_depth_db, -60.0)

    def test_goal_from_state_falls_back_to_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = RunState(Path(tmp))
            goal = run_step._goal_from_state(state)
            self.assertEqual(goal.target_freq_hz, run_step.TARGET_NOTCH_HZ)
            self.assertEqual(goal.band_hz, tuple(run_step.TARGET_BAND_HZ))
            self.assertEqual(goal.target_depth_db, run_step.TARGET_DEPTH_DB)

    def test_format_report_shows_goal_freq_and_depth(self):
        goal = GoalSpec(
            target_freq_hz=7e9,
            band_hz=(6e9, 8e9),
            target_depth_db=-60.0,
        )
        entry = {
            "iteration": 0,
            "params": {"ri": 1.0, "ro": 3.0, "alpha": 40.0, "Wf": 0.3, "Lc": 2.0},
            "cost": {
                "total_cost": 0.1,
                "target_s21_mag": 0.1,
                "target_freq_hz": 7e9,
                "best_s21_mag": 0.05,
                "best_freq_hz": 7.1e9,
                "mean_cost": 0.2,
                "low_edge_cost": 0.3,
                "center_cost": 0.2,
                "high_edge_cost": 0.3,
            },
            "intent": None,
            "note": "",
        }
        text = run_step.format_report(entry, goal=goal)
        self.assertIn("7.00 GHz", text)
        self.assertIn("-60 dB", text)
        thresh = 10 ** (-60.0 / 20.0)
        self.assertIn(f"{thresh:.3e}", text)
        self.assertIn("6.0 GHz", text)  # band low edge label
        self.assertIn("8.0 GHz", text)  # band high edge label

    def test_evaluate_for_run_passes_goal_band_and_target(self):
        fake_res = object()
        goal = GoalSpec(7e9, (6e9, 8e9), target_depth_db=-60.0)
        with mock.patch.object(
            run_step,
            "evaluate",
            return_value=FakeCost(0.1, 0.05, 7e9, target_freq_hz=7e9),
        ) as ev:
            cost = run_step._evaluate_for_run(fake_res, goal)
        ev.assert_called_once_with(fake_res, (6e9, 8e9), target_hz=7e9)
        self.assertEqual(cost["target_freq_hz"], 7e9)
        self.assertAlmostEqual(cost["total_cost"], 0.1)


class InitPreservesGoalTests(unittest.TestCase):
    def test_init_preserves_preassigned_goal_and_initial_params(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "preassigned"
            run_dir.mkdir()
            goal = {
                "target_freq_hz": 7e9,
                "band_hz": [6e9, 8e9],
                "target_depth_db": -60.0,
            }
            initial = {"ri": 0.8, "ro": 2.5, "alpha": 35.0, "Wf": 0.25, "Lc": 2.5}
            (run_dir / "state.json").write_text(
                json.dumps(
                    {
                        "iteration": -1,
                        "params": None,
                        "history": [],
                        "goal": goal,
                        "start_seed": 99,
                        "initial_params": initial,
                    },
                    indent=2,
                )
            )

            captured = {}

            def fake_simulate(params, workdir=None):
                captured["params"] = dict(params)
                return {"ok": True}

            with mock.patch.object(run_step, "RUNS_ROOT", tmp_path):
                with mock.patch.object(run_step, "simulate", side_effect=fake_simulate):
                    with mock.patch.object(
                        run_step,
                        "evaluate",
                        return_value=FakeCost(0.1, 0.05, 7e9, target_freq_hz=7e9),
                    ) as ev:
                        run_step.cmd_init(
                            mock.Mock(
                                run="preassigned",
                                note="",
                                thinking="",
                                thinking_file="",
                                goal_json="",
                            )
                        )
                        ev.assert_called_once()
                        self.assertEqual(ev.call_args.args[1], (6e9, 8e9))
                        self.assertEqual(ev.call_args.kwargs["target_hz"], 7e9)

            self.assertEqual(captured["params"], initial)
            state = RunState(run_dir)
            self.assertEqual(state.goal["target_freq_hz"], 7e9)
            self.assertEqual(state.goal["target_depth_db"], -60.0)
            self.assertEqual(state.start_seed, 99)
            self.assertEqual(state.initial_params, initial)
            self.assertGreaterEqual(state.iteration, 0)

    def test_init_goal_json_stores_goal(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            goal = {
                "target_freq_hz": 4.5e9,
                "band_hz": [3.5e9, 5.5e9],
                "target_depth_db": -55.0,
            }
            with mock.patch.object(run_step, "RUNS_ROOT", tmp_path):
                with mock.patch.object(run_step, "simulate", return_value={"ok": True}):
                    with mock.patch.object(
                        run_step,
                        "evaluate",
                        return_value=FakeCost(
                            0.1, 0.05, 4.5e9, target_freq_hz=4.5e9, worst_freq_hz=4e9
                        ),
                    ):
                        run_step.cmd_init(
                            mock.Mock(
                                run="fromjson",
                                note="",
                                thinking="",
                                thinking_file="",
                                goal_json=json.dumps(goal),
                            )
                        )
            state = RunState(tmp_path / "fromjson")
            self.assertEqual(state.goal["target_freq_hz"], 4.5e9)
            self.assertEqual(state.goal["target_depth_db"], -55.0)


if __name__ == "__main__":
    unittest.main()
