"""Import smoke for training.sft package layout."""
from __future__ import annotations

import unittest


class SftPackageImportTests(unittest.TestCase):
    def test_train_importable(self):
        from training.sft.train import build_parser

        self.assertTrue(callable(build_parser))


if __name__ == "__main__":
    unittest.main()
