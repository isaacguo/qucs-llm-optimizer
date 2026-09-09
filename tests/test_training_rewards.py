"""Tests for TRL-compatible GRPO reward functions."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from training.rewards import (
    completion_text,
    format_reward,
    make_simulation_reward,
    valid_intent_reward,
)


VALID = (
    "<reasoning>The notch is below target.</reasoning>\n"
    '<intent>{"ro":"decrease"}</intent>'
)


class CompletionTextTests(unittest.TestCase):
    def test_reads_plain_completion(self):
        self.assertEqual(completion_text(VALID), VALID)

    def test_reads_conversational_completion(self):
        completion = [{"role": "assistant", "content": VALID}]
        self.assertEqual(completion_text(completion), VALID)


class StaticRewardTests(unittest.TestCase):
    def test_format_reward_prefers_exact_tags_but_accepts_valid_json(self):
        rewards = format_reward(
            completions=[VALID, '{"ro":"decrease"}', "not json"]
        )
        self.assertEqual(rewards, [1.0, 0.5, 0.0])

    def test_format_reward_accepts_qwen_empty_think_prefix(self):
        completion = "<think>\n\n</think>\n\n" + VALID
        self.assertEqual(format_reward(completions=[completion]), [1.0])

    def test_valid_intent_reward_rejects_hacks(self):
        rewards = valid_intent_reward(
            completions=[
                VALID,
                '{"ro":"hold"}',
                '{"ro":"decrease","ri":"increase","alpha":"decrease"}',
            ]
        )
        self.assertEqual(rewards, [1.0, 0.0, 0.0])


class SimulationRewardTests(unittest.TestCase):
    def _goal_json(self) -> str:
        return json.dumps(
            {
                "target_freq_hz": 5.5e9,
                "band_hz": [4.0e9, 6.0e9],
                "target_depth_db": -70.0,
            },
            sort_keys=True,
        )

    def test_returns_real_delta_and_skips_invalid_completion(self):
        calls = []

        def fake_transition(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                params={"ro": 7.0},
                old_db=-20.0,
                new_db=-29.0,
                delta_db=9.0,
                total_cost=0.035,
            )

        reward = make_simulation_reward(transition=fake_transition, max_workers=1)
        params = json.dumps(
            {"ri": 0.3, "ro": 8.0, "alpha": 90.0, "Wf": 0.6, "Lc": 3.0}
        )
        baseline = json.dumps({"total_cost": 0.1})
        goal = self._goal_json()
        scores = reward(
            completions=[VALID, "broken"],
            params_json=[params, params],
            baseline_cost_json=[baseline, baseline],
            goal_json=[goal, goal],
            iteration=[1, 1],
            seed=[10, 10],
        )

        self.assertEqual(scores, [9.0, -5.0])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["intent"], {"ro": "decrease"})

    def test_clips_extreme_delta(self):
        transition = lambda **_kwargs: SimpleNamespace(
            params={},
            old_db=-1.0,
            new_db=-101.0,
            delta_db=100.0,
            total_cost=1e-5,
        )
        reward = make_simulation_reward(transition=transition, max_workers=1)
        params = '{"ri":0.3,"ro":8,"alpha":90,"Wf":0.6,"Lc":3}'
        baseline = '{"total_cost":0.1}'
        self.assertEqual(
            reward(
                completions=[VALID],
                params_json=[params],
                baseline_cost_json=[baseline],
                goal_json=[self._goal_json()],
                iteration=[1],
                seed=[1],
            ),
            [20.0],
        )

    def test_writes_structured_reward_log(self):
        transition = lambda **_kwargs: SimpleNamespace(
            params={"ro": 7.0},
            old_db=-20.0,
            new_db=-25.0,
            delta_db=5.0,
            total_cost=0.05,
        )
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "rewards.jsonl"
            reward = make_simulation_reward(
                transition=transition,
                max_workers=1,
                log_path=log_path,
            )
            reward(
                completions=[VALID],
                params_json=['{"ri":0.3,"ro":8,"alpha":90,"Wf":0.6,"Lc":3}'],
                baseline_cost_json=['{"total_cost":0.1}'],
                goal_json=[self._goal_json()],
                iteration=[1],
                seed=[7],
            )
            record = json.loads(log_path.read_text().strip())
        self.assertEqual(record["seed"], 7)
        self.assertEqual(record["intent"], {"ro": "decrease"})
        self.assertEqual(record["delta_db"], 5.0)


if __name__ == "__main__":
    unittest.main()
