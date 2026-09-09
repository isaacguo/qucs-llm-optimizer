"""Tests for agent run_step writing completions.jsonl."""
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import run_step


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
    worst_freq_hz: float = 4e9
    target_freq_hz: float = 5.5e9
    target_s21_mag: float = 0.1


class RunStepDecisionLogTests(unittest.TestCase):
    def test_step_appends_completions_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            with mock.patch.object(run_step, "RUNS_ROOT", tmp_path):
                with mock.patch.object(run_step, "simulate", return_value={"ok": True}):
                    with mock.patch.object(
                        run_step,
                        "evaluate",
                        side_effect=[
                            FakeCost(0.2, 0.2, 5.5e9),
                            FakeCost(0.05, 0.05, 5.5e9),
                        ],
                    ):
                        run_step.cmd_init(
                            mock.Mock(
                                run="agentlog",
                                note="baseline",
                                thinking="",
                                thinking_file="",
                            )
                        )
                        run_step.cmd_step(
                            mock.Mock(
                                run="agentlog",
                                intent='{"ro":"decrease_strong"}',
                                note="move",
                                thinking="Shrink outer radius.",
                                thinking_file="",
                            )
                        )

            path = tmp_path / "agentlog" / "completions.jsonl"
            self.assertTrue(path.exists())
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            self.assertEqual(len(rows), 1)
            last = rows[0]
            self.assertEqual(last["source"], "agent")
            self.assertTrue(last["prompt"])
            self.assertIn("<reasoning>Shrink outer radius.</reasoning>", last["completion"])
            self.assertIn("decrease_strong", last["completion"])
            self.assertEqual(last["intent"]["ro"], "decrease_strong")


if __name__ == "__main__":
    unittest.main()
