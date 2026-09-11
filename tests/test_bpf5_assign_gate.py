# tests/test_bpf5_assign_gate.py
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from jobs.bpf5_agent.assign import assign_rollout, make_activity_id, make_rollout_id
from jobs.bpf5_agent.buckets import MAX_STEPS
from jobs.bpf5_agent.gate import gate_rollout, on_success
from jobs.bpf5_agent.progress import ProgressStore


class TestMakeIds(unittest.TestCase):
    def test_make_activity_id_format(self):
        now = datetime(2026, 9, 12, 2, 15, 30, tzinfo=ZoneInfo("Asia/Singapore"))
        self.assertEqual(make_activity_id(now), "bpf_agent_20260912_021530")

    def test_make_rollout_id_filesystem_safe(self):
        rid = make_rollout_id(3, 112.5, 5.0)
        self.assertEqual(rid, "rollout_b03_cf112p5_bw5")
        self.assertNotIn("/", rid)
        self.assertNotIn(".", rid)
        self.assertNotIn(" ", rid)


class TestAssignRollout(unittest.TestCase):
    def test_assign_creates_nested_dir_and_meta(self):
        with tempfile.TemporaryDirectory() as td:
            runs_root = Path(td)
            activity_id = "bpf_agent_20260912_021530"
            goal = {"f_low_hz": 110e6, "f_high_hz": 115e6}
            rollout_dir = assign_rollout(
                runs_root,
                activity_id,
                bucket=3,
                goal_dict=goal,
                cf_mhz=112.5,
                bw_mhz=5.0,
            )
            expected = runs_root / activity_id / "rollout_b03_cf112p5_bw5"
            self.assertEqual(rollout_dir, expected)
            self.assertTrue((rollout_dir / "state.json").is_file())
            state = json.loads((rollout_dir / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["task"], "butterworth_bpf5")
            self.assertEqual(state["goal"], goal)
            self.assertEqual(state["iteration"], -1)
            self.assertEqual(state["history"], [])

    def test_assign_adds_unique_suffix_when_dir_exists(self):
        with tempfile.TemporaryDirectory() as td:
            runs_root = Path(td)
            activity_id = "act"
            goal = {"f_low_hz": 110e6, "f_high_hz": 115e6}
            first = assign_rollout(
                runs_root,
                activity_id,
                bucket=3,
                goal_dict=goal,
                cf_mhz=112.5,
                bw_mhz=5.0,
            )
            second = assign_rollout(
                runs_root,
                activity_id,
                bucket=3,
                goal_dict=goal,
                cf_mhz=112.5,
                bw_mhz=5.0,
            )
            self.assertEqual(first.name, "rollout_b03_cf112p5_bw5")
            self.assertNotEqual(first, second)
            self.assertTrue(second.name.startswith("rollout_b03_cf112p5_bw5"))
            self.assertTrue(second.is_dir())
            self.assertTrue((second / "state.json").is_file())
            self.assertEqual(
                json.loads((second / "state.json").read_text(encoding="utf-8"))["history"],
                [],
            )


class TestGateRollout(unittest.TestCase):
    def _write_success_artifacts(
        self,
        rollout: Path,
        *,
        history: list[dict],
        task: str = "butterworth_bpf5",
        goal: dict | None = None,
        with_completions: bool = True,
    ) -> None:
        rollout.mkdir(parents=True, exist_ok=True)
        if goal is None:
            goal = {"f_low_hz": 1e8, "f_high_hz": 1.1e8}
        data = {
            "iteration": history[-1]["iteration"] if history else -1,
            "params": None,
            "history": history,
            "task": task,
            "goal": goal,
        }
        (rollout / "state.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        if with_completions:
            (rollout / "completions.jsonl").write_text(
                json.dumps({"iteration": history[-1]["iteration"] if history else 0}) + "\n",
                encoding="utf-8",
            )

    def test_gate_success_when_goal_met_within_max_steps(self):
        with tempfile.TemporaryDirectory() as td:
            rollout = Path(td) / "rollout"
            self._write_success_artifacts(
                rollout,
                history=[
                    {"iteration": 0, "cost": {"goal_met": False}, "params": {}, "intent": None, "note": ""},
                    {"iteration": 5, "cost": {"goal_met": True}, "params": {}, "intent": None, "note": ""},
                ],
            )
            self.assertTrue(gate_rollout(rollout, max_steps=MAX_STEPS))

    def test_gate_fail_when_goal_never_met(self):
        with tempfile.TemporaryDirectory() as td:
            rollout = Path(td) / "rollout"
            self._write_success_artifacts(
                rollout,
                history=[
                    {"iteration": 0, "cost": {"goal_met": False}, "params": {}, "intent": None, "note": ""},
                    {"iteration": 19, "cost": {"goal_met": False}, "params": {}, "intent": None, "note": ""},
                ],
            )
            self.assertFalse(gate_rollout(rollout, max_steps=MAX_STEPS))

    def test_gate_fail_when_goal_met_after_max_steps(self):
        with tempfile.TemporaryDirectory() as td:
            rollout = Path(td) / "rollout"
            self._write_success_artifacts(
                rollout,
                history=[
                    {"iteration": 21, "cost": {"goal_met": True}, "params": {}, "intent": None, "note": ""},
                ],
            )
            self.assertFalse(gate_rollout(rollout, max_steps=MAX_STEPS))

    def test_gate_success_at_exactly_max_steps(self):
        with tempfile.TemporaryDirectory() as td:
            rollout = Path(td) / "rollout"
            self._write_success_artifacts(
                rollout,
                history=[
                    {
                        "iteration": MAX_STEPS,
                        "cost": {"goal_met": True},
                        "params": {},
                        "intent": None,
                        "note": "",
                    },
                ],
            )
            self.assertTrue(gate_rollout(rollout, max_steps=MAX_STEPS))

    def test_gate_fail_baseline_only_goal_met(self):
        with tempfile.TemporaryDirectory() as td:
            rollout = Path(td) / "rollout"
            self._write_success_artifacts(
                rollout,
                history=[
                    {"iteration": 0, "cost": {"goal_met": True}, "params": {}, "intent": None, "note": ""},
                ],
            )
            self.assertFalse(gate_rollout(rollout))

    def test_gate_fail_without_completions_jsonl(self):
        with tempfile.TemporaryDirectory() as td:
            rollout = Path(td) / "rollout"
            self._write_success_artifacts(
                rollout,
                history=[
                    {"iteration": 3, "cost": {"goal_met": True}, "params": {}, "intent": None, "note": ""},
                ],
                with_completions=False,
            )
            self.assertFalse(gate_rollout(rollout))

    def test_gate_fail_wrong_task(self):
        with tempfile.TemporaryDirectory() as td:
            rollout = Path(td) / "rollout"
            self._write_success_artifacts(
                rollout,
                history=[
                    {"iteration": 3, "cost": {"goal_met": True}, "params": {}, "intent": None, "note": ""},
                ],
                task="butterfly_stub",
            )
            self.assertFalse(gate_rollout(rollout))

    def test_gate_fail_missing_goal(self):
        with tempfile.TemporaryDirectory() as td:
            rollout = Path(td) / "rollout"
            rollout.mkdir(parents=True, exist_ok=True)
            data = {
                "iteration": 3,
                "params": None,
                "history": [
                    {"iteration": 3, "cost": {"goal_met": True}, "params": {}, "intent": None, "note": ""},
                ],
                "task": "butterworth_bpf5",
            }
            (rollout / "state.json").write_text(json.dumps(data), encoding="utf-8")
            (rollout / "completions.jsonl").write_text("{}\n", encoding="utf-8")
            self.assertFalse(gate_rollout(rollout))

    def test_gate_expected_goal_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            rollout = Path(td) / "rollout"
            goal = {"f_low_hz": 1e8, "f_high_hz": 1.1e8}
            self._write_success_artifacts(
                rollout,
                history=[
                    {"iteration": 3, "cost": {"goal_met": True}, "params": {}, "intent": None, "note": ""},
                ],
                goal=goal,
            )
            self.assertFalse(
                gate_rollout(
                    rollout,
                    expected_goal={"f_low_hz": 2e8, "f_high_hz": 2.1e8},
                )
            )
            self.assertTrue(gate_rollout(rollout, expected_goal=goal))

    def test_on_success_records_progress(self):
        with tempfile.TemporaryDirectory() as td:
            store = ProgressStore(Path(td) / "progress.json")
            on_success(store, bucket=7)
            data = store.load()
            self.assertEqual(data["counts"][7], 1)


if __name__ == "__main__":
    unittest.main()
