"""Tests for GRPO task datasets and optional SFT examples."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.data import build_grpo_records, load_sft_records


class GrpoDataTests(unittest.TestCase):
    def test_builds_seeded_records_with_reward_columns(self):
        def fake_task(seed, iteration, freq_range=None, **_kwargs):
            return {
                "prompt": [{"role": "user", "content": f"seed {seed}"}],
                "params_json": "{}",
                "baseline_cost_json": '{"total_cost": 0.1}',
                "goal_json": '{"target_freq_hz": 5.5e9, "band_hz": [4e9, 6e9], "target_depth_db": -70.0}',
                "iteration": iteration,
                "seed": seed,
            }

        records = build_grpo_records(
            count=3,
            start_seed=10,
            max_iteration=4,
            task_factory=fake_task,
        )

        self.assertEqual([row["seed"] for row in records], [10, 11, 12])
        self.assertEqual(
            set(records[0]),
            {
                "prompt",
                "params_json",
                "baseline_cost_json",
                "goal_json",
                "iteration",
                "seed",
            },
        )
        self.assertTrue(all(0 <= row["iteration"] <= 4 for row in records))


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


if __name__ == "__main__":
    unittest.main()
