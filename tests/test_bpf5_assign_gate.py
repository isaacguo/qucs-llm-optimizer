# tests/test_bpf5_assign_gate.py
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from jobs.bpf5_agent.assign import assign_rollout, make_activity_id, make_rollout_id
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


class TestGateRollout(unittest.TestCase):
    def _write_state(self, path: Path, history: list[dict], iteration: int | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "iteration": iteration if iteration is not None else (history[-1]["iteration"] if history else -1),
            "params": None,
            "history": history,
            "task": "butterworth_bpf5",
            "goal": {"f_low_hz": 1e8, "f_high_hz": 1.1e8},
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def test_gate_success_when_goal_met_within_max_steps(self):
        with tempfile.TemporaryDirectory() as td:
            rollout = Path(td) / "rollout"
            self._write_state(
                rollout / "state.json",
                [
                    {"iteration": 0, "cost": {"goal_met": False}, "params": {}, "intent": None, "note": ""},
                    {"iteration": 5, "cost": {"goal_met": True}, "params": {}, "intent": None, "note": ""},
                ],
            )
            self.assertTrue(gate_rollout(rollout, max_steps=20))

    def test_gate_fail_when_goal_never_met(self):
        with tempfile.TemporaryDirectory() as td:
            rollout = Path(td) / "rollout"
            self._write_state(
                rollout / "state.json",
                [
                    {"iteration": 0, "cost": {"goal_met": False}, "params": {}, "intent": None, "note": ""},
                    {"iteration": 19, "cost": {"goal_met": False}, "params": {}, "intent": None, "note": ""},
                ],
            )
            self.assertFalse(gate_rollout(rollout, max_steps=20))

    def test_gate_fail_when_goal_met_after_max_steps(self):
        with tempfile.TemporaryDirectory() as td:
            rollout = Path(td) / "rollout"
            self._write_state(
                rollout / "state.json",
                [
                    {"iteration": 21, "cost": {"goal_met": True}, "params": {}, "intent": None, "note": ""},
                ],
            )
            self.assertFalse(gate_rollout(rollout, max_steps=20))

    def test_gate_success_at_exactly_max_steps(self):
        with tempfile.TemporaryDirectory() as td:
            rollout = Path(td) / "rollout"
            self._write_state(
                rollout / "state.json",
                [
                    {"iteration": 20, "cost": {"goal_met": True}, "params": {}, "intent": None, "note": ""},
                ],
            )
            self.assertTrue(gate_rollout(rollout, max_steps=20))

    def test_on_success_records_progress(self):
        with tempfile.TemporaryDirectory() as td:
            store = ProgressStore(Path(td) / "progress.json")
            on_success(store, bucket=7)
            data = store.load()
            self.assertEqual(data["counts"][7], 1)


if __name__ == "__main__":
    unittest.main()
