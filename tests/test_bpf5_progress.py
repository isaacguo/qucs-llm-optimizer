# tests/test_bpf5_progress.py
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jobs.bpf5_agent.buckets import N_BUCKETS, SUCCESS_PER_BUCKET
from jobs.bpf5_agent.progress import ProgressStore


class TestBpf5Progress(unittest.TestCase):
    def test_load_creates_default_when_missing(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "progress.json"
            store = ProgressStore(path)
            data = store.load()
            self.assertEqual(data["version"], 1)
            self.assertEqual(data["success_per_bucket"], SUCCESS_PER_BUCKET)
            self.assertEqual(data["counts"], [0] * N_BUCKETS)
            self.assertFalse(store.is_complete())

    def test_record_success_persists_and_completes(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "progress.json"
            store = ProgressStore(path)
            for b in range(N_BUCKETS):
                for _ in range(SUCCESS_PER_BUCKET):
                    store.record_success(b)
            self.assertTrue(store.is_complete())
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(raw["counts"], [SUCCESS_PER_BUCKET] * N_BUCKETS)

    def test_record_success_increments_one_bucket(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "progress.json"
            store = ProgressStore(path)
            store.record_success(5)
            store.record_success(5)
            data = store.load()
            self.assertEqual(data["counts"][5], 2)
            self.assertEqual(sum(data["counts"]), 2)
            self.assertFalse(store.is_complete())


if __name__ == "__main__":
    unittest.main()
