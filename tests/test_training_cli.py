"""Tests for training command configuration without loading a model."""
from __future__ import annotations

import unittest

from training.config import resolve_multiturn_config
from training.multiturn_train import build_parser as build_multiturn_parser
from training.sft import build_parser as build_sft_parser


class OtherCliTests(unittest.TestCase):
    def test_sft_is_explicit_and_shallow(self):
        args = build_sft_parser().parse_args([])
        self.assertEqual(args.epochs, 1)
        self.assertEqual(args.index, "corpus/index.jsonl")
        self.assertIsNone(args.state)
        self.assertEqual(args.max_length, 2048)


class MultiturnCliTests(unittest.TestCase):
    def test_defaults_match_enhanced_training_setup(self):
        args = build_multiturn_parser().parse_args([])
        cfg = resolve_multiturn_config(args)
        self.assertEqual(cfg.train.max_turns, 15)
        self.assertEqual(cfg.train.patience, 5)
        self.assertEqual(cfg.data.goal_config, "configs/goal_distribution.yaml")
        self.assertEqual(cfg.train.generations, 4)
        self.assertAlmostEqual(cfg.data.param_spread, 0.35)
        self.assertAlmostEqual(cfg.data.min_start_headroom_db, 5.0)
        self.assertEqual(cfg.runtime.resume_adapter, "")
        self.assertEqual(cfg.train.beta, 0.0)
        self.assertEqual(cfg.train.kl_ref, "start")

    def test_accepts_resume_adapter_path(self):
        args = build_multiturn_parser().parse_args(
            ["--resume-adapter", "outputs/run/checkpoint-60", "--start-seed", "5240"]
        )
        cfg = resolve_multiturn_config(args)
        self.assertEqual(cfg.runtime.resume_adapter, "outputs/run/checkpoint-60")
        self.assertEqual(cfg.data.start_seed, 5240)

    def test_accepts_kl_beta_and_ref(self):
        args = build_multiturn_parser().parse_args(["--beta", "0.01", "--kl-ref", "base"])
        cfg = resolve_multiturn_config(args)
        self.assertAlmostEqual(cfg.train.beta, 0.01)
        self.assertEqual(cfg.train.kl_ref, "base")

    def test_accepts_dry_run(self):
        args = build_multiturn_parser().parse_args(["--dry-run"])
        self.assertTrue(args.dry_run)


if __name__ == "__main__":
    unittest.main()
