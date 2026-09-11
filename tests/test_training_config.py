"""Tests for training YAML config load/merge (no model load)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from training.config import (
    MultiturnConfig,
    from_mapping,
    load_yaml,
    resolve_multiturn_config,
)
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


class MultiturnConfigTests(unittest.TestCase):
    def test_defaults_match_enhanced_training_setup(self):
        cfg = from_mapping(MultiturnConfig, {})
        self.assertEqual(cfg.train.max_turns, 15)
        self.assertEqual(cfg.train.patience, 5)
        self.assertEqual(cfg.data.goal_config, "configs/goal_distribution.yaml")
        self.assertEqual(cfg.train.generations, 4)
        self.assertEqual(cfg.model.max_seq_length, 2048)
        self.assertEqual(cfg.train.beta, 0.0)
        self.assertEqual(cfg.train.kl_ref, "start")

    def test_partial_model_yaml_keeps_multiturn_max_seq_length(self):
        cfg = from_mapping(MultiturnConfig, {"model": {"name": "unsloth/Qwen3-4B"}})
        self.assertEqual(cfg.model.name, "unsloth/Qwen3-4B")
        self.assertEqual(cfg.model.max_seq_length, 2048)

    def test_unknown_key_fails(self):
        with self.assertRaises(ValueError) as ctx:
            from_mapping(MultiturnConfig, {"train": {"learnning_rate": 1e-5}})
        self.assertIn("learnning_rate", str(ctx.exception))

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

    def test_train_accepts_config_parameter(self):
        import inspect
        from training import multiturn_train

        params = inspect.signature(multiturn_train.train).parameters
        self.assertIn("config", params)


ROOT = Path(__file__).resolve().parents[1]


class ExampleRecipeTests(unittest.TestCase):
    def test_multiturn_example_loads(self):
        data = load_yaml(ROOT / "configs/multiturn_qwen3_1_7b.yaml")
        cfg = from_mapping(MultiturnConfig, data)
        self.assertEqual(cfg.train.max_turns, 15)
        self.assertEqual(cfg.model.max_seq_length, 2048)


if __name__ == "__main__":
    unittest.main()
