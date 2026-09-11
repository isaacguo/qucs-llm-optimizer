# jobs/bpf5_agent/gate.py
from __future__ import annotations

import json
from pathlib import Path

from jobs.bpf5_agent.assign import TASK_NAME
from jobs.bpf5_agent.buckets import MAX_STEPS
from jobs.bpf5_agent.progress import ProgressStore


def _goals_equal(a: dict, b: dict) -> bool:
    """Loose float-aware equality for BPF goal dicts."""
    keys = set(a) | set(b)
    if keys != set(a) or keys != set(b):
        return False
    for k in keys:
        av, bv = a[k], b[k]
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            if abs(float(av) - float(bv)) > 1e-6:
                return False
        elif av != bv:
            return False
    return True


def gate_rollout(
    rollout_dir: Path | str,
    *,
    max_steps: int = MAX_STEPS,
    expected_goal: dict | None = None,
) -> bool:
    """True iff post-assign BPF work met the goal within ``max_steps``.

    Requires:
    - ``state.json`` with ``task == butterworth_bpf5`` and a present goal
    - optional ``expected_goal`` must match state goal when provided
    - ``completions.jsonl`` exists (evidence of harness step work)
    - history has ``goal_met`` at iteration ``>= 1`` and ``<= max_steps``
      (iteration 0 is baseline-only and does not count)
    """
    rollout_dir = Path(rollout_dir)
    state_path = rollout_dir / "state.json"
    if not state_path.is_file():
        return False
    if not (rollout_dir / "completions.jsonl").is_file():
        return False
    data = json.loads(state_path.read_text(encoding="utf-8"))
    if data.get("task") != TASK_NAME:
        return False
    goal = data.get("goal")
    if not isinstance(goal, dict) or not goal:
        return False
    if expected_goal is not None and not _goals_equal(goal, expected_goal):
        return False
    history = data.get("history") or []
    for entry in history:
        cost = entry.get("cost") or {}
        if not cost.get("goal_met"):
            continue
        iteration = int(entry["iteration"])
        if iteration < 1:
            continue
        return iteration <= max_steps
    return False


def on_success(progress: ProgressStore, bucket: int) -> None:
    progress.record_success(bucket)
