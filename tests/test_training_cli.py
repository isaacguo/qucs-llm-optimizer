"""Tests for training command configuration without loading a model."""
from __future__ import annotations

import unittest

from training.sft import build_parser as build_sft_parser


class OtherCliTests(unittest.TestCase):
    def test_sft_is_explicit_and_shallow(self):
        args = build_sft_parser().parse_args([])
        self.assertEqual(args.epochs, 1)
        self.assertEqual(args.index, "corpus/index.jsonl")
        self.assertIsNone(args.state)
        self.assertEqual(args.max_length, 2048)


if __name__ == "__main__":
    unittest.main()
