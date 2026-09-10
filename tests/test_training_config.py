"""Tests for training YAML config load/merge (no model load)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from training.config import (
    GrpoConfig,
    MultiturnConfig,
    from_mapping,
    load_yaml,
    resolve_grpo_config,
    resolve_multiturn_config,
)
from training.grpo import build_parser, grpo_config_kwargs
from training.multiturn_train import build_parser as build_multiturn_parser


class LoadYamlTests(unittest.TestCase):
    def test_load_yaml_reads_nested_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.yaml"
            path.write_text("model:\n  name: demo-model\n", encoding="utf-8")
            data = load_yaml(path)
        self.assertEqual(data["model"]["name"], "demo-model")

    def test_missing_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_yaml("/no/such/config.yaml")


class GrpoFromMappingTests(unittest.TestCase):
    def test_empty_mapping_uses_current_defaults(self):
        cfg = from_mapping(GrpoConfig, {})
        self.assertEqual(cfg.model.name, "unsloth/Qwen3-1.7B-bnb-4bit")
        self.assertEqual(cfg.train.steps, 100)
        self.assertEqual(cfg.train.generations, 8)
        self.assertEqual(cfg.train.learning_rate, 5e-6)
        self.assertEqual(cfg.reward.beta, 0.01)
        self.assertEqual(cfg.reward.weights, [0.2, 0.2, 1.0])
        self.assertEqual(cfg.data.tasks, 32)
        self.assertEqual(cfg.runtime.output_dir, "outputs/grpo-qwen3-1.7b")

    def test_partial_yaml_overrides_nested_fields(self):
        cfg = from_mapping(
            GrpoConfig,
            {"model": {"name": "unsloth/Qwen3-4B"}, "train": {"learning_rate": 1e-5}},
        )
        self.assertEqual(cfg.model.name, "unsloth/Qwen3-4B")
        self.assertEqual(cfg.train.learning_rate, 1e-5)
        self.assertEqual(cfg.train.steps, 100)

    def test_unknown_key_fails(self):
        with self.assertRaises(ValueError) as ctx:
            from_mapping(GrpoConfig, {"train": {"learnning_rate": 1e-5}})
        self.assertIn("learnning_rate", str(ctx.exception))


class GrpoMergeTests(unittest.TestCase):
    def test_cli_overrides_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "g.yaml"
            path.write_text(
                "train:\n  steps: 50\n  learning_rate: 1.0e-5\n",
                encoding="utf-8",
            )
            args = build_parser().parse_args(["--config", str(path), "--steps", "3"])
            cfg = resolve_grpo_config(args)
        self.assertEqual(cfg.train.steps, 3)
        self.assertEqual(cfg.train.learning_rate, 1e-5)

    def test_no_config_matches_legacy_defaults(self):
        args = build_parser().parse_args([])
        cfg = resolve_grpo_config(args)
        self.assertEqual(cfg.model.name, "unsloth/Qwen3-1.7B-bnb-4bit")
        self.assertEqual(cfg.train.generations, 8)
        kwargs = grpo_config_kwargs(cfg)
        self.assertEqual(kwargs["learning_rate"], 5e-6)
        self.assertEqual(kwargs["beta"], 0.01)
        self.assertEqual(kwargs["reward_weights"], [0.2, 0.2, 1.0])
        self.assertEqual(kwargs["num_generations"], 8)


class MultiturnConfigTests(unittest.TestCase):
    def test_defaults_match_enhanced_training_setup(self):
        cfg = from_mapping(MultiturnConfig, {})
        self.assertEqual(cfg.train.max_turns, 15)
        self.assertEqual(cfg.train.patience, 5)
        self.assertEqual(cfg.data.target_depth_db, -30.0)
        self.assertEqual(cfg.train.generations, 4)
        self.assertEqual(cfg.model.max_seq_length, 2048)
        self.assertEqual(cfg.train.beta, 0.0)
        self.assertEqual(cfg.train.kl_ref, "start")

    def test_resolve_cli_overrides_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.yaml"
            path.write_text("train:\n  steps: 9\n  beta: 0.01\n", encoding="utf-8")
            args = build_multiturn_parser().parse_args(
                ["--config", str(path), "--steps", "2", "--kl-ref", "base"]
            )
            cfg = resolve_multiturn_config(args)
        self.assertEqual(cfg.train.steps, 2)
        self.assertEqual(cfg.train.beta, 0.01)
        self.assertEqual(cfg.train.kl_ref, "base")


if __name__ == "__main__":
    unittest.main()
