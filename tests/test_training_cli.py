"""Tests for training command configuration without loading a model."""
from __future__ import annotations

import unittest

from training.grpo import build_parser as build_grpo_parser
from training.grpo import grpo_config_kwargs
from training.multiturn_train import build_parser as build_multiturn_parser
from training.probe import build_parser as build_probe_parser
from training.sft import build_parser as build_sft_parser


class GrpoCliTests(unittest.TestCase):
    def test_defaults_use_real_qwen_and_eight_generations(self):
        args = build_grpo_parser().parse_args([])
        self.assertEqual(args.model_name, "unsloth/Qwen3-1.7B-bnb-4bit")
        self.assertEqual(args.generations, 8)
        self.assertEqual(args.tasks, 32)
        self.assertEqual(args.steps, 100)
        self.assertFalse(args.use_vllm)

    def test_grpo_config_is_conservative_for_eight_gb(self):
        args = build_grpo_parser().parse_args(
            ["--steps", "3", "--generations", "4", "--output-dir", "custom"]
        )
        config = grpo_config_kwargs(args)
        self.assertEqual(config["max_steps"], 3)
        self.assertEqual(config["num_generations"], 4)
        self.assertFalse(config["use_vllm"])
        self.assertEqual(config["output_dir"], "custom")
        self.assertEqual(config["reward_weights"], [0.2, 0.2, 1.0])

    def test_accepts_resume_and_dry_run(self):
        args = build_grpo_parser().parse_args(
            ["--resume-from-checkpoint", "outputs/checkpoint-2", "--dry-run"]
        )
        self.assertTrue(args.dry_run)
        self.assertEqual(args.resume_from_checkpoint, "outputs/checkpoint-2")


class OtherCliTests(unittest.TestCase):
    def test_probe_defaults_are_small_but_real(self):
        args = build_probe_parser().parse_args([])
        self.assertEqual(args.tasks, 2)
        self.assertEqual(args.generations, 4)
        self.assertFalse(args.skip_simulation)

    def test_sft_is_explicit_and_shallow(self):
        args = build_sft_parser().parse_args([])
        self.assertEqual(args.epochs, 1)
        self.assertIn("runs/llm1/state.json", args.state)


class MultiturnCliTests(unittest.TestCase):
    def test_defaults_match_enhanced_training_setup(self):
        args = build_multiturn_parser().parse_args([])
        self.assertEqual(args.max_turns, 8)
        self.assertEqual(args.patience, 3)
        self.assertEqual(args.target_depth_db, -30.0)
        self.assertEqual(args.generations, 4)
        self.assertAlmostEqual(args.param_spread, 0.35)


if __name__ == "__main__":
    unittest.main()
