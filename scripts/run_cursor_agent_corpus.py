#!/usr/bin/env python3
"""Deprecated: old Cursor-agent corpus driver (notch + mixed corpus package).

BPF teacher collection lives under ``jobs/bpf5_agent/run_activity.py``.
Notch collection is deferred to a separate job.
"""
from __future__ import annotations

import sys

_MSG = (
    "scripts/run_cursor_agent_corpus.py is retired.\n"
    "Notch teacher collection is deferred to a separate job.\n"
    "For BPF in this repo, use: jobs/bpf5_agent/run_activity.py\n"
)


def main() -> None:
    sys.stderr.write(_MSG)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
