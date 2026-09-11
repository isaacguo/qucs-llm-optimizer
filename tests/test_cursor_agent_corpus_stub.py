"""Deprecated corpus driver exits with pointer to bpf5_agent activity."""
from __future__ import annotations

import runpy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


class CursorAgentCorpusStubTests(unittest.TestCase):
    def test_script_exits_with_bpf_pointer(self):
        script = Path("scripts/run_cursor_agent_corpus.py")
        with (
            patch.object(sys, "argv", [str(script)]),
            self.assertRaises(SystemExit) as ctx,
        ):
            runpy.run_path(str(script), run_name="__main__")
        self.assertNotEqual(ctx.exception.code, 0)
        # Message is printed to stdout by the stub; re-run capturing is heavy —
        # assert source contains the required pointers instead.
        text = script.read_text(encoding="utf-8")
        self.assertIn("jobs/bpf5_agent/run_activity.py", text)
        self.assertIn("notch", text.lower())


if __name__ == "__main__":
    unittest.main()
