#!/usr/bin/env python3
"""Deprecated: rule/heuristic teacher that filled the old corpus package.

BPF teacher collection lives under ``jobs/bpf5_agent/run_activity.py``.
Notch collection is deferred to a separate job.
"""
from __future__ import annotations

import sys

_MSG = (
    "scripts/generate_teacher_trajectories.py is retired (depended on deleted corpus/).\n"
    "Notch teacher collection is deferred to a separate job.\n"
    "For BPF in this repo, use: jobs/bpf5_agent/run_activity.py\n"
)


def main() -> None:
    sys.stderr.write(_MSG)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
