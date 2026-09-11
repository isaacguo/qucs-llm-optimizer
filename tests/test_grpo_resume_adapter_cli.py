"""Multiturn: shared goal distribution + SFT resume gate (no GPU)."""
from __future__ import annotations

import unittest
from pathlib import Path

from training.grpo.config import MultiturnConfig, from_mapping, load_yaml
from training.grpo.train import (
    build_parser,
    sample_multiturn_goal,
    train,
)

ROOT = Path(__file__).resolve().parents[1]
SHIPPED_GOAL_CONFIG = ROOT / "configs" / "goal_distribution.yaml"


class MultiturnResumeAdapterCliTests(unittest.TestCase):
    def test_train_exits_without_resume_adapter(self):
        cfg = MultiturnConfig()
        self.assertEqual(cfg.runtime.resume_adapter, "")
        self.assertFalse(cfg.runtime.allow_raw_base)
        with self.assertRaises(SystemExit) as ctx:
            train(cfg)
        self.assertIn("resume-adapter", str(ctx.exception).lower())

    def test_allow_raw_base_flag_defaults_false(self):
        args = build_parser().parse_args([])
        from training.grpo.config import resolve_multiturn_config

        cfg = resolve_multiturn_config(args)
        self.assertIs(cfg.runtime.allow_raw_base, False)

    def test_sampling_uses_shipped_goal_distribution_1_to_10_ghz(self):
        self.assertTrue(SHIPPED_GOAL_CONFIG.is_file())
        data = load_yaml(ROOT / "configs" / "multiturn_qwen3_1_7b.yaml")
        cfg = from_mapping(MultiturnConfig, data)
        self.assertEqual(cfg.data.goal_config, "configs/goal_distribution.yaml")
        for seed in range(40):
            goal = sample_multiturn_goal(seed, cfg.data)
            self.assertGreaterEqual(goal.target_freq_hz, 1.0e9)
            self.assertLessEqual(goal.target_freq_hz, 10.0e9)
            self.assertGreaterEqual(goal.target_depth_db, -80.0)
            self.assertLessEqual(goal.target_depth_db, -55.0)


if __name__ == "__main__":
    unittest.main()
