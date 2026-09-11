"""Tests for optional SFT examples (legacy state + jsonl index)."""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from training.common.jsonl_index import append_index
from training.common.modeling import ModelConfig
from training.sft.data import load_sft_records
from training.sft.train import build_parser, main as sft_main


def _mag_for_db(db: float) -> float:
    return 10.0 ** (db / 20.0)


def _params(**overrides) -> dict:
    base = {"ri": 0.3, "ro": 8.0, "alpha": 90.0, "Wf": 0.6, "Lc": 3.0}
    base.update(overrides)
    return base


def _cost(total_cost: float) -> dict:
    return {
        "total_cost": total_cost,
        "best_s21_mag": _mag_for_db(-20.0),
        "best_freq_hz": 3.3e9,
    }


def _multiturn_fixture_state() -> dict:
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
                "thinking": "Notch is too low. Shrink ro hard.",
            },
            {
                "iteration": 2,
                "intent": {"alpha": "increase_slight"},
                "params": _params(ro=5.0, alpha=92.0),
                "cost": _cost(0.04),
                "thinking": "Fine tune alpha.",
            },
        ],
    }


class SftDataTests(unittest.TestCase):
    def test_extracts_only_entries_with_observation_and_intent(self):
        state = {
            "history": [
                {
                    "iteration": 0,
                    "observation": None,
                    "thinking": "baseline",
                    "intent": None,
                },
                {
                    "iteration": 1,
                    "observation": {"report_text": "measured state"},
                    "thinking": "notch is too low",
                    "intent": {"ro": "decrease_strong"},
                },
            ]
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "state.json"
            path.write_text(json.dumps(state))
            records = load_sft_records(path)

        self.assertEqual(len(records), 1)
        messages = records[0]["messages"]
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant"])
        self.assertIn("measured state", messages[1]["content"])
        self.assertIn("<reasoning>notch is too low</reasoning>", messages[2]["content"])
        self.assertIn('"ro": "decrease_strong"', messages[2]["content"])


class SftIndexCliTests(unittest.TestCase):
    def test_parser_requires_explicit_index_path(self):
        args = build_parser().parse_args([])
        self.assertIsNone(args.index)
        self.assertEqual(args.runs_root, "runs")
        self.assertEqual(args.history_window, 8)
        self.assertEqual(args.max_length, 2048)
        self.assertIsNone(args.state)
        self.assertFalse(args.dry_run)

    def test_missing_index_and_state_raises(self):
        with self.assertRaises(RuntimeError) as ctx:
            sft_main(["--dry-run"])
        self.assertIn("--index", str(ctx.exception).lower())

    def test_dry_run_counts_export_from_temp_index_without_model(self):
        state = _multiturn_fixture_state()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_dir = root / "run_a"
            run_dir.mkdir()
            (run_dir / "state.json").write_text(json.dumps(state))
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
            buf = io.StringIO()
            with redirect_stdout(buf), patch("training.sft.train.load_policy") as load_policy:
                sft_main(
                    [
                        "--index",
                        str(index_path),
                        "--runs-root",
                        str(root),
                        "--dry-run",
                    ]
                )
            load_policy.assert_not_called()
            out = buf.getvalue()
            self.assertRegex(out, r"\b2\b")

    def test_dry_run_empty_index_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as td:
            index_path = Path(td) / "empty.jsonl"
            index_path.write_text("")
            with self.assertRaises(RuntimeError) as ctx:
                sft_main(
                    [
                        "--index",
                        str(index_path),
                        "--runs-root",
                        str(td),
                        "--dry-run",
                    ]
                )
            self.assertIn("index", str(ctx.exception).lower())

    def test_non_dry_run_passes_max_seq_length_from_max_length(self):
        state = _multiturn_fixture_state()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_dir = root / "run_a"
            run_dir.mkdir()
            (run_dir / "state.json").write_text(json.dumps(state))
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
            buf = io.StringIO()
            with (
                redirect_stdout(buf),
                patch("training.sft.train.verify_runtime", return_value={}),
                patch(
                    "training.sft.train.load_policy",
                    side_effect=RuntimeError("stop-after-load_policy"),
                ) as load_policy,
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    sft_main(
                        [
                            "--index",
                            str(index_path),
                            "--runs-root",
                            str(root),
                            "--output-dir",
                            str(root / "out"),
                            "--max-length",
                            "2048",
                        ]
                    )
            self.assertIn("stop-after-load_policy", str(ctx.exception))
            load_policy.assert_called_once()
            config = load_policy.call_args.args[0]
            self.assertIsInstance(config, ModelConfig)
            self.assertEqual(config.max_seq_length, 2048)


if __name__ == "__main__":
    unittest.main()
