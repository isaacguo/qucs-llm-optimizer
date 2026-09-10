"""Tests for corpus gate, index, and coverage."""
from __future__ import annotations

import json
import tempfile
import unittest
import warnings
from pathlib import Path

from cost import s21_db
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


def _mag_for_db(db: float) -> float:
    return 10.0 ** (db / 20.0)


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
        init_db = s21_db(baseline["cost"]["total_cost"])
        mid_db = s21_db(mid["cost"]["total_cost"])
        # Patience gate (eps=0.2): large step updates best_db.
        patience_best = mid_db if mid_db < init_db - 0.2 else init_db
        expected_second = build_multiturn_prompt(
            goal,
            turn_index=1,
            params=mid["params"],
            cost=mid["cost"],
            history=[prior],
            history_window=8,
            initial_db=init_db,
            best_db=patience_best,
        )
        self.assertEqual(records[1]["messages"][1]["content"], expected_second[1]["content"])

    def test_export_best_db_uses_patience_gate_not_strict_min(self):
        """Sub-eps improvement (~0.1 dB) must not update best_db for later turns."""
        init_db = -20.0
        tiny_improve_db = -20.1  # delta 0.1 < patience_eps 0.2
        init_mag = _mag_for_db(init_db)
        tiny_mag = _mag_for_db(tiny_improve_db)
        state = {
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
                    "cost": _cost(init_mag),
                    "thinking": "Baseline.",
                },
                {
                    "iteration": 1,
                    "intent": {"ro": "decrease_slight"},
                    "params": _params(ro=7.5),
                    "cost": _cost(tiny_mag),
                    "thinking": "Tiny step.",
                },
                {
                    "iteration": 2,
                    "intent": {"alpha": "increase_slight"},
                    "params": _params(ro=7.5, alpha=92.0),
                    "cost": _cost(tiny_mag),
                    "thinking": "Hold course.",
                },
            ],
        }
        records = export_multiturn_sft_records(state)
        self.assertEqual(len(records), 2)

        goal = GoalSpec(
            target_freq_hz=5.5e9,
            band_hz=(4.5e9, 6.5e9),
            target_depth_db=-60.0,
        )
        baseline = state["history"][0]
        mid = state["history"][1]
        prior = TurnRecord(
            turn_index=0,
            prompt=[],
            completion_text="",
            valid=True,
            intent={"ro": "decrease_slight"},
            params_before=baseline["params"],
            params_after=mid["params"],
            db_before=init_db,
            db_after=tiny_improve_db,
        )
        patience_tracked = init_db  # 0.1 dB < 0.2 eps → no update
        strict_min = tiny_improve_db
        self.assertLess(strict_min, patience_tracked)
        self.assertLess(patience_tracked - strict_min, 0.2)

        expected_patience = build_multiturn_prompt(
            goal,
            turn_index=1,
            params=mid["params"],
            cost=mid["cost"],
            history=[prior],
            history_window=8,
            initial_db=init_db,
            best_db=patience_tracked,
        )
        expected_strict = build_multiturn_prompt(
            goal,
            turn_index=1,
            params=mid["params"],
            cost=mid["cost"],
            history=[prior],
            history_window=8,
            initial_db=init_db,
            best_db=strict_min,
        )
        user_text = records[1]["messages"][1]["content"]
        self.assertEqual(user_text, expected_patience[1]["content"])
        self.assertNotEqual(user_text, expected_strict[1]["content"])

    def test_export_warns_when_zero_turns(self):
        state = {
            "goal": {
                "target_freq_hz": 5.5e9,
                "band_hz": [4.5e9, 6.5e9],
                "target_depth_db": -60.0,
            },
            "history": [
                {
                    "iteration": 0,
                    "intent": None,
                    "params": _params(),
                    "cost": _cost(0.1),
                }
            ],
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            records = export_multiturn_sft_records(state)
        self.assertEqual(records, [])
        self.assertTrue(any("zero" in str(w.message).lower() for w in caught))

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


class CorpusCliTests(unittest.TestCase):
    """Smoke tests for qucs-corpus CLI (no Qucs)."""

    def setUp(self):
        from training import corpus_cli

        self.corpus_cli = corpus_cli
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        self.runs_root = self.root / "runs"
        self.goal_config = (
            Path(__file__).resolve().parents[1] / "configs" / "goal_distribution.yaml"
        )

    def tearDown(self):
        self._td.cleanup()

    def test_assign_writes_state_skeleton_without_history(self):
        code = self.corpus_cli.main(
            [
                "assign",
                "--run",
                "r001",
                "--goal-config",
                str(self.goal_config),
                "--seed",
                "7",
                "--runs-root",
                str(self.runs_root),
            ]
        )
        self.assertEqual(code, 0)
        state_path = self.runs_root / "r001" / "state.json"
        self.assertTrue(state_path.is_file())
        state = json.loads(state_path.read_text())
        self.assertEqual(state["iteration"], -1)
        self.assertEqual(state["history"], [])
        self.assertEqual(state["start_seed"], 7)
        self.assertIn("target_freq_hz", state["goal"])
        self.assertIn("target_depth_db", state["goal"])
        self.assertIn("band_hz", state["goal"])
        self.assertIn("ri", state["initial_params"])
        # assign must not simulate: no baseline history entry yet
        self.assertFalse(state["history"])

    def test_gate_appends_eligible_run_to_index(self):
        run_dir = self.runs_root / "r_ok"
        run_dir.mkdir(parents=True)
        state = _state_with_goal(-70.0, 2.5e-4)
        state["history"][0]["params"] = _params()
        state["history"][1]["params"] = _params(ro=5.0)
        (run_dir / "state.json").write_text(json.dumps(state))
        index_path = self.root / "corpus" / "index.jsonl"

        code = self.corpus_cli.main(
            [
                "gate",
                "--run",
                "r_ok",
                "--index",
                str(index_path),
                "--runs-root",
                str(self.runs_root),
            ]
        )
        self.assertEqual(code, 0)
        rows = load_index(index_path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["run_id"], "r_ok")
        saved = json.loads((run_dir / "state.json").read_text())
        self.assertTrue(saved["corpus"]["eligible"])

    def test_gate_skips_index_when_ineligible(self):
        run_dir = self.runs_root / "r_bad"
        run_dir.mkdir(parents=True)
        state = _state_with_goal(-70.0, 1.0e-3)
        (run_dir / "state.json").write_text(json.dumps(state))
        index_path = self.root / "corpus" / "index.jsonl"

        code = self.corpus_cli.main(
            [
                "gate",
                "--run",
                "r_bad",
                "--index",
                str(index_path),
                "--runs-root",
                str(self.runs_root),
            ]
        )
        self.assertEqual(code, 0)
        self.assertEqual(load_index(index_path), [])
        saved = json.loads((run_dir / "state.json").read_text())
        self.assertFalse(saved["corpus"]["eligible"])

    def test_coverage_prints_bin_counts(self):
        index_path = self.root / "index.jsonl"
        append_index(
            index_path,
            {
                "run_id": "a",
                "goal": {
                    "target_freq_hz": 5.2e9,
                    "target_depth_db": -62.0,
                    "band_hz": [4.2e9, 6.2e9],
                },
            },
        )
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            code = self.corpus_cli.main(
                [
                    "coverage",
                    "--index",
                    str(index_path),
                    "--goal-config",
                    str(self.goal_config),
                ]
            )
        self.assertEqual(code, 0)
        out = buf.getvalue()
        self.assertRegex(out, r"total\s*[:=]\s*1")
        self.assertIn("bins", out.lower())

    def test_export_writes_jsonl(self):
        state = _multiturn_fixture_state()
        run_dir = self.runs_root / "run_a"
        run_dir.mkdir(parents=True)
        (run_dir / "state.json").write_text(json.dumps(state))
        index_path = self.root / "index.jsonl"
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
        out_path = self.root / "out.jsonl"
        code = self.corpus_cli.main(
            [
                "export",
                "--index",
                str(index_path),
                "--out",
                str(out_path),
                "--runs-root",
                str(self.runs_root),
                "--history-window",
                "8",
            ]
        )
        self.assertEqual(code, 0)
        lines = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
        self.assertEqual(len(lines), 2)
        self.assertIn("messages", lines[0])

    def test_import_run_attaches_llm1_goal_gates_and_indexes(self):
        """Legacy llm1-shaped run: no goal → attach 5.5 GHz / −70 dB, gate, index."""
        state = _state_with_goal(-70.0, 2.5e-4)
        del state["goal"]
        state["history"][0]["params"] = _params()
        state["history"][1]["params"] = _params(ro=5.0)
        run_dir = self.runs_root / "llm1"
        run_dir.mkdir(parents=True)
        (run_dir / "state.json").write_text(json.dumps(state))
        index_path = self.root / "corpus" / "index.jsonl"

        code = self.corpus_cli.main(
            [
                "import-run",
                "--run",
                "llm1",
                "--index",
                str(index_path),
                "--runs-root",
                str(self.runs_root),
            ]
        )
        self.assertEqual(code, 0)
        saved = json.loads((run_dir / "state.json").read_text())
        self.assertEqual(saved["goal"]["target_freq_hz"], 5.5e9)
        self.assertEqual(saved["goal"]["target_depth_db"], -70.0)
        self.assertEqual(saved["goal"]["band_hz"], [4.5e9, 6.5e9])
        self.assertTrue(saved["corpus"]["eligible"])
        rows = load_index(index_path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["run_id"], "llm1")
        self.assertEqual(rows[0]["goal"]["target_freq_hz"], 5.5e9)

    def test_import_run_does_not_index_when_ineligible(self):
        state = _state_with_goal(-70.0, 1.0e-3)
        del state["goal"]
        run_dir = self.runs_root / "llm1_bad"
        run_dir.mkdir(parents=True)
        (run_dir / "state.json").write_text(json.dumps(state))
        index_path = self.root / "corpus" / "index.jsonl"

        code = self.corpus_cli.main(
            [
                "import-run",
                "--run",
                "llm1_bad",
                "--index",
                str(index_path),
                "--runs-root",
                str(self.runs_root),
            ]
        )
        self.assertEqual(code, 0)
        saved = json.loads((run_dir / "state.json").read_text())
        self.assertEqual(saved["goal"]["target_freq_hz"], 5.5e9)
        self.assertFalse(saved["corpus"]["eligible"])
        self.assertEqual(load_index(index_path), [])


if __name__ == "__main__":
    unittest.main()
