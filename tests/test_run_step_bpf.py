"""Task dispatch: butterworth_bpf5 via run_step --task."""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import run_step
from qucs_sim import SimResult
from state import RunState
from tasks import get_task


def _synthetic_bpf_sim() -> SimResult:
    # stop + pass + pass + stop (covers both stopbands for default_goal)
    freq = [50e6, 140e6, 150e6, 250e6]
    s21 = [0.2 + 0j, 0.95 + 0j, 0.9 + 0j, 0.15 + 0j]
    s11 = [0.1 + 0j] * 4
    return SimResult(freq_hz=freq, s11=s11, s21=s21)


def _init_args(run: str, *, task: str = "butterworth_bpf5", goal_json: str = "") -> mock.Mock:
    return mock.Mock(
        run=run,
        note="",
        thinking="",
        thinking_file="",
        goal_json=goal_json,
        task=task,
    )


class RunStateTaskPersistenceTests(unittest.TestCase):
    def test_set_run_meta_persists_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "t"
            state = RunState(run_dir)
            self.assertIsNone(state.task)
            state.set_run_meta(task="butterworth_bpf5")
            reloaded = RunState(run_dir)
            self.assertEqual(reloaded.task, "butterworth_bpf5")


class BpfInitDispatchTests(unittest.TestCase):
    def test_init_bpf_sets_task_and_bpf_cost_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sim_kwargs = {}

            def fake_simulate(params, **kwargs):
                sim_kwargs.update(kwargs)
                sim_kwargs["params"] = dict(params)
                return _synthetic_bpf_sim()

            with mock.patch.object(run_step, "RUNS_ROOT", tmp_path):
                with mock.patch.object(run_step, "simulate", side_effect=fake_simulate):
                    buf = io.StringIO()
                    with redirect_stdout(buf):
                        run_step.cmd_init(_init_args("bpf_init"))

            state = RunState(tmp_path / "bpf_init")
            self.assertEqual(state.task, "butterworth_bpf5")
            self.assertIn("f_low_hz", state.goal)
            self.assertIn("passband_min_s21_db", state.history[0]["cost"])
            self.assertIn("stopband_max_s21_db", state.history[0]["cost"])
            self.assertIn("total_cost", state.history[0]["cost"])

            task = get_task("butterworth_bpf5")
            self.assertEqual(sim_kwargs.get("export_layout"), task.export_layout)
            self.assertEqual(Path(sim_kwargs["template_path"]), task.template_path)
            self.assertEqual(sim_kwargs.get("sweep_points"), task.sweep_points)
            self.assertEqual(set(sim_kwargs["params"]), set(task.variables))
            self.assertFalse(sim_kwargs.get("export_layout", True))

    def test_init_task_mismatch_exits(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            run_dir = tmp_path / "mismatch"
            run_dir.mkdir()
            (run_dir / "state.json").write_text(
                json.dumps(
                    {
                        "iteration": -1,
                        "params": None,
                        "history": [],
                        "task": "butterworth_bpf5",
                    }
                )
            )
            with mock.patch.object(run_step, "RUNS_ROOT", tmp_path):
                err = io.StringIO()
                with redirect_stderr(err):
                    with self.assertRaises(SystemExit) as cm:
                        run_step.cmd_init(_init_args("mismatch", task="butterfly_stub"))
            self.assertEqual(cm.exception.code, 1)
            self.assertIn("task mismatch", err.getvalue())

    def test_butterfly_init_default_task(self):
        from dataclasses import dataclass

        @dataclass
        class FakeCost:
            total_cost: float = 0.1
            best_s21_mag: float = 0.1
            best_freq_hz: float = 5.5e9
            mean_cost: float = 0.5
            low_edge_cost: float = 0.5
            center_cost: float = 0.5
            high_edge_cost: float = 0.5
            stopband_max_s21: float = 0.5
            worst_freq_hz: float = 6e9
            target_freq_hz: float = 5.5e9
            target_s21_mag: float = 0.1

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with mock.patch.object(run_step, "RUNS_ROOT", tmp_path):
                with mock.patch.object(run_step, "simulate", return_value={"ok": True}):
                    with mock.patch.object(
                        run_step, "evaluate", return_value=FakeCost()
                    ):
                        with redirect_stdout(io.StringIO()):
                            run_step.cmd_init(_init_args("bfly", task="butterfly_stub"))
            state = RunState(tmp_path / "bfly")
            self.assertEqual(state.task, "butterfly_stub")
            self.assertIn("target_freq_hz", state.goal)


class BpfFormatTests(unittest.TestCase):
    def test_format_report_and_observation_bpf(self):
        task = get_task("butterworth_bpf5")
        entry = {
            "iteration": 0,
            "params": dict(task.initial_guess),
            "cost": {
                "total_cost": 0.3,
                "passband_min_s21_db": -0.5,
                "stopband_max_s21_db": -18.0,
                "passband_mean_s21_db": -0.2,
                "s11_passband_max_db": -15.0,
                "has_passband_samples": True,
                "has_stopband_samples": True,
                "f_low_hz": 135e6,
                "f_high_hz": 165e6,
            },
            "intent": None,
            "note": "baseline",
        }
        text = run_step.format_report(entry, task="butterworth_bpf5")
        self.assertIn("passband", text.lower())
        self.assertIn("stopband", text.lower())
        self.assertIn("total_cost", text.lower())
        self.assertIn("-0.5", text)
        self.assertIn("-18", text)

        obs = run_step.format_observation(entry, task="butterworth_bpf5")
        self.assertIn("L3", obs)
        self.assertIn("C3", obs)
        self.assertIn("bounds", obs.lower())
        # sample L and C bound edges from task tables
        self.assertIn("1.0", obs)  # L lower
        self.assertIn("2000.0", obs)  # L upper


class BpfStepDispatchTests(unittest.TestCase):
    def test_step_uses_bpf_variables_and_simulate_kwargs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            calls = []

            def fake_simulate(params, **kwargs):
                calls.append({"params": dict(params), **kwargs})
                return _synthetic_bpf_sim()

            with mock.patch.object(run_step, "RUNS_ROOT", tmp_path):
                with mock.patch.object(run_step, "simulate", side_effect=fake_simulate):
                    with redirect_stdout(io.StringIO()):
                        run_step.cmd_init(_init_args("bpf_step"))
                        run_step.cmd_step(
                            mock.Mock(
                                run="bpf_step",
                                intent='{"L3":"decrease","C3":"increase"}',
                                note="tune",
                                thinking="",
                                thinking_file="",
                            )
                        )

            self.assertEqual(len(calls), 2)
            self.assertIn("L3", calls[1]["params"])
            self.assertNotIn("ro", calls[1]["params"])
            self.assertFalse(calls[1].get("export_layout", True))
            state = RunState(tmp_path / "bpf_step")
            self.assertEqual(state.iteration, 1)
            self.assertIn("passband_min_s21_db", state.history[1]["cost"])


if __name__ == "__main__":
    unittest.main()
