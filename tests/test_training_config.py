"""Tests for training YAML config load/merge (no model load)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from training.config import GrpoConfig, from_mapping, load_yaml


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


if __name__ == "__main__":
    unittest.main()
