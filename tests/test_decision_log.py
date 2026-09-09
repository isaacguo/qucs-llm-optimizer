"""Tests for shared decision JSONL logging."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decision_log import append_record, format_agent_completion


class DecisionLogTests(unittest.TestCase):
    def test_format_agent_completion_wraps_thinking_and_intent(self):
        text = format_agent_completion("Shrink ro.", {"ro": "decrease_strong"})
        self.assertIn("<reasoning>Shrink ro.</reasoning>", text)
        self.assertIn('<intent>{"ro": "decrease_strong"}</intent>', text)

    def test_append_record_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "completions.jsonl"
            append_record(
                path,
                {
                    "source": "agent",
                    "prompt": "obs",
                    "completion": "out",
                    "valid": True,
                    "intent": {"ro": "decrease"},
                },
            )
            append_record(path, {"source": "multiturn", "prompt": "p2", "completion": "c2"})
            lines = path.read_text().strip().splitlines()
            self.assertEqual(len(lines), 2)
            first = json.loads(lines[0])
            self.assertEqual(first["source"], "agent")
            self.assertIn("timestamp", first)
            self.assertEqual(first["prompt"], "obs")


if __name__ == "__main__":
    unittest.main()
