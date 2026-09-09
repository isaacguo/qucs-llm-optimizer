"""Tests for randomized real-Qucs GRPO tasks and one-step transitions."""
from __future__ import annotations

import json
import math
import unittest
from types import SimpleNamespace

from training.environment import (
    build_prompt,
    generate_task,
    run_transition,
    sample_params,
)
from training.goals import GoalSpec


class RandomTaskTests(unittest.TestCase):
    def test_seeded_parameter_sampling_is_deterministic_and_bounded(self):
        first = sample_params(42)
        second = sample_params(42)
        self.assertEqual(first, second)
        bounds = {
            "ri": (0.10, 1.50),
            "ro": (3.00, 14.00),
            "alpha": (30.0, 150.0),
            "Wf": (0.20, 2.50),
            "Lc": (0.20, 10.00),
        }
        for name, value in first.items():
            self.assertGreaterEqual(value, bounds[name][0])
            self.assertLessEqual(value, bounds[name][1])

    def test_prompt_describes_state_bounds_and_output_contract(self):
        cost = {
            "total_cost": 0.1,
            "target_freq_hz": 5.5e9,
            "best_freq_hz": 4.8e9,
            "best_s21_mag": 0.002,
        }
        prompt = build_prompt(
            {"ri": 0.3, "ro": 8.0, "alpha": 90.0, "Wf": 0.6, "Lc": 3.0},
            cost,
            iteration=2,
            goal=GoalSpec(5.5e9, (4.0e9, 6.0e9)),
        )
        joined = "\n".join(message["content"] for message in prompt)
        self.assertIn("5.500 GHz", joined)
        self.assertIn("4.800 GHz", joined)
        self.assertIn("decrease_strong", joined)
        self.assertIn("<intent>", joined)
        self.assertIn("iteration 2", joined)
        self.assertIn("/no_think", joined)
        self.assertIn('{"ro":"decrease_strong"}', joined)
        self.assertIn("BAD", joined)

    def test_generate_task_serializes_reward_inputs(self):
        fake_cost = SimpleNamespace(
            total_cost=0.2,
            target_freq_hz=5.5e9,
            best_freq_hz=5.0e9,
            best_s21_mag=0.01,
        )

        task = generate_task(
            seed=7,
            iteration=3,
            simulator=lambda *_args, **_kwargs: object(),
            evaluator=lambda *_args, **_kwargs: fake_cost,
        )

        self.assertEqual(task["seed"], 7)
        self.assertEqual(task["iteration"], 3)
        self.assertEqual(set(json.loads(task["params_json"])), {"ri", "ro", "alpha", "Wf", "Lc"})
        self.assertAlmostEqual(json.loads(task["baseline_cost_json"])["total_cost"], 0.2)


class TransitionTests(unittest.TestCase):
    def test_transition_applies_intent_and_skips_layout(self):
        calls = []

        def fake_simulator(params, **kwargs):
            calls.append((params, kwargs))
            return object()

        new_cost = SimpleNamespace(total_cost=0.05)
        result = run_transition(
            params={"ri": 0.3, "ro": 8.0, "alpha": 90.0, "Wf": 0.6, "Lc": 3.0},
            iteration=1,
            baseline_total_cost=0.1,
            intent={"ro": "decrease"},
            goal=GoalSpec(5.5e9, (4.0e9, 6.0e9)),
            simulator=fake_simulator,
            evaluator=lambda *_args, **_kwargs: new_cost,
        )

        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0][1]["export_layout"])
        self.assertLess(calls[0][0]["ro"], 8.0)
        self.assertAlmostEqual(result.old_db, -20.0)
        self.assertAlmostEqual(result.new_db, 20 * math.log10(0.05))
        self.assertAlmostEqual(result.delta_db, result.old_db - result.new_db)


if __name__ == "__main__":
    unittest.main()
