"""Task dispatch: butterworth_bpf5 via run_step --task."""
from __future__ import annotations

import importlib
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
from task_registry import clear_plugins_for_tests
from tasks import ensure_builtin_tasks, get_task


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


class _BpfPluginRegistered(unittest.TestCase):
    """Re-register BPF after clear_plugins_for_tests (order-independent)."""

    def setUp(self):
        clear_plugins_for_tests()
        ensure_builtin_tasks()
        importlib.import_module("jobs.bpf5_agent.register").register()


class RunStateTaskPersistenceTests(unittest.TestCase):
    def test_set_run_meta_persists_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "t"
            state = RunState(run_dir)
            self.assertIsNone(state.task)
            state.set_run_meta(task="butterworth_bpf5")
            reloaded = RunState(run_dir)
            self.assertEqual(reloaded.task, "butterworth_bpf5")


class RunStepNoBpfHardcodingTests(unittest.TestCase):
    def test_run_step_does_not_import_jobs_bpf_modules_at_top_level(self):
        import run_step as rs

        # 顶层加载后，sys.modules 可以有 jobs（因 discover），但 run_step 源码不得出现
        src = Path(rs.__file__).read_text(encoding="utf-8")
        self.assertNotIn("from cost_bpf", src)
        self.assertNotIn("from goals_bpf", src)
        self.assertNotIn("from bpf_tuning_skills", src)
        self.assertNotIn("_is_bpf", src)  # 用 get_plugin 有无代替
        self.assertNotIn("from jobs.bpf5_agent", src)
        self.assertNotIn("import jobs.bpf5_agent", src)


class BpfInitDispatchTests(_BpfPluginRegistered):
    def test_init_bpf_sets_task_and_bpf_cost_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sim_kwargs = {}

            def fake_simulate(params, **kwargs):
                sim_kwargs.update(kwargs)
                sim_kwargs["params"] = dict(params)
                return _synthetic_bpf_sim()

            with mock.patch.object(run_step, "RUNS_ROOT", tmp_path):
                with mock.patch(
                    "jobs.bpf5_agent.plugin.simulate", side_effect=fake_simulate
                ):
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


class BpfFormatTests(_BpfPluginRegistered):
    def test_format_report_and_observation_bpf(self):
        from jobs.bpf5_agent.goals import default_goal

        task = get_task("butterworth_bpf5")
        goal = default_goal()
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
                "goal_met": False,
            },
            "intent": None,
            "note": "baseline",
        }
        text = run_step.format_report(entry, bpf_goal=goal, task="butterworth_bpf5")
        self.assertIn("passband", text.lower())
        self.assertIn("stopband", text.lower())
        self.assertIn("total_cost", text.lower())
        self.assertIn("-0.5", text)
        self.assertIn("-18", text)
        self.assertIn("passband_il_max_db", text)
        self.assertIn("stopband_atten_min_db", text)
        self.assertIn("goal not met", text.lower())

        from jobs.bpf5_agent.skills import DEFAULT_SKILLS_PATH, load_skills_system_prompt

        job_skills = load_skills_system_prompt()
        obs = run_step.format_observation(entry, bpf_goal=goal, task="butterworth_bpf5")
        self.assertIn("=== SYSTEM: BPF tuning skills", obs)
        self.assertIn("Skill A", obs)
        self.assertIn(job_skills, obs)
        self.assertTrue(DEFAULT_SKILLS_PATH.exists())
        self.assertIn(
            "jobs/bpf5_agent/prompts/bpf_tuning_skills_system.md",
            str(DEFAULT_SKILLS_PATH).replace("\\", "/"),
        )
        self.assertIn("L3", obs)
        self.assertIn("C3", obs)
        self.assertIn("bounds", obs.lower())
        # sample L and C bound edges from task tables
        self.assertIn("0.5", obs)  # L lower
        self.assertIn("10000.0", obs)  # L upper
        # I4: units + sweep window + goal thresholds
        self.assertIn("nH", obs)
        self.assertIn("pF", obs)
        self.assertIn("sweep", obs.lower())
        self.assertIn("passband_il_max_db", obs)
        # seed L2 keeps 4-decimal-ish precision in observation
        self.assertIn("6.5577", obs)
        # peak diagnostic line when present on cost
        entry["cost"]["s21_peak_freq_hz"] = 150e6
        obs2 = run_step.format_observation(entry, bpf_goal=goal, task="butterworth_bpf5")
        self.assertIn("S21 peak frequency", obs2)

    def test_format_report_goal_met_true(self):
        from jobs.bpf5_agent.goals import default_goal

        task = get_task("butterworth_bpf5")
        entry = {
            "iteration": 1,
            "params": dict(task.initial_guess),
            "cost": {
                "total_cost": 0.05,
                "passband_min_s21_db": -0.5,
                "stopband_max_s21_db": -25.0,
                "has_passband_samples": True,
                "has_stopband_samples": True,
                "f_low_hz": 135e6,
                "f_high_hz": 165e6,
                "goal_met": True,
            },
            "intent": None,
            "note": "",
        }
        text = run_step.format_report(
            entry, bpf_goal=default_goal(), task="butterworth_bpf5"
        )
        self.assertIn("goal met", text.lower())
        self.assertNotIn("goal not met", text.lower())


class BpfStepDispatchTests(_BpfPluginRegistered):
    def test_step_uses_bpf_variables_and_simulate_kwargs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            calls = []

            def fake_simulate(params, **kwargs):
                calls.append({"params": dict(params), **kwargs})
                return _synthetic_bpf_sim()

            with mock.patch.object(run_step, "RUNS_ROOT", tmp_path):
                with mock.patch(
                    "jobs.bpf5_agent.plugin.simulate", side_effect=fake_simulate
                ):
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
            self.assertIn("goal_met", state.history[0]["cost"])
            self.assertIn("goal_met", state.history[1]["cost"])
            self.assertIsInstance(state.history[1]["cost"]["goal_met"], bool)

    def test_step_uses_relative_step_mode_for_small_L(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with mock.patch.object(run_step, "RUNS_ROOT", tmp_path):
                with mock.patch(
                    "jobs.bpf5_agent.plugin.simulate",
                    return_value=_synthetic_bpf_sim(),
                ):
                    with redirect_stdout(io.StringIO()):
                        run_step.cmd_init(_init_args("bpf_rel"))
                        seed_l2 = RunState(tmp_path / "bpf_rel").params["L2"]
                        run_step.cmd_step(
                            mock.Mock(
                                run="bpf_rel",
                                intent='{"L2":"decrease_slight"}',
                                note="",
                                thinking="",
                                thinking_file="",
                            )
                        )
            state = RunState(tmp_path / "bpf_rel")
            new_l2 = state.params["L2"]
            self.assertGreater(new_l2, 1.0)
            self.assertLess(abs(new_l2 - seed_l2), 20.0)


class BpfGoalValidationInitTests(_BpfPluginRegistered):
    def test_init_rejects_invalid_goal_json(self):
        bad = json.dumps(
            {
                "f_low_hz": 200e6,
                "f_high_hz": 100e6,
                "sweep_hz": [0.0, 300e6],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with mock.patch.object(run_step, "RUNS_ROOT", tmp_path):
                err = io.StringIO()
                with redirect_stderr(err):
                    with self.assertRaises(SystemExit) as cm:
                        run_step.cmd_init(
                            _init_args("bad_goal", goal_json=bad)
                        )
            self.assertEqual(cm.exception.code, 1)
            self.assertRegex(err.getvalue().lower(), r"goal|f_low|invalid")

    def test_init_rejects_goal_missing_keys(self):
        bad = json.dumps({"passband_il_max_db": -1.0})
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with mock.patch.object(run_step, "RUNS_ROOT", tmp_path):
                err = io.StringIO()
                with redirect_stderr(err):
                    with self.assertRaises(SystemExit) as cm:
                        run_step.cmd_init(
                            _init_args("miss_keys", goal_json=bad)
                        )
            self.assertEqual(cm.exception.code, 1)
            msg = err.getvalue().lower()
            self.assertTrue("goal" in msg or "f_low" in msg or "missing" in msg)


if __name__ == "__main__":
    unittest.main()
