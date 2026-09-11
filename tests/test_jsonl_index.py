"""Tests for generic JSONL index helpers (no corpus package)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class JsonlIndexTests(unittest.TestCase):
    def test_append_and_load_roundtrip(self):
        from training.common.jsonl_index import append_index, load_index

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "nested" / "index.jsonl"
            append_index(path, {"run_id": "a", "goal": {"target_depth_db": -60}})
            append_index(path, {"run_id": "b", "goal": {"target_depth_db": -65}})
            rows = load_index(path)
        self.assertEqual([r["run_id"] for r in rows], ["a", "b"])

    def test_load_missing_returns_empty(self):
        from training.common.jsonl_index import load_index

        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(load_index(Path(td) / "missing.jsonl"), [])

    def test_module_has_no_corpus_or_gate_imports(self):
        from pathlib import Path

        src = Path("training/common/jsonl_index.py").read_text(encoding="utf-8")
        self.assertNotIn("corpus", src)
        self.assertNotIn("gate_run", src)
        self.assertNotIn("GoalSpec", src)


if __name__ == "__main__":
    unittest.main()
