# jobs/bpf5_agent/gate.py
from __future__ import annotations

import json
from pathlib import Path

from jobs.bpf5_agent.buckets import MAX_STEPS
from jobs.bpf5_agent.progress import ProgressStore


def gate_rollout(rollout_dir: Path | str, *, max_steps: int = MAX_STEPS) -> bool:
    """True iff history has goal_met at an iteration <= max_steps."""
    state_path = Path(rollout_dir) / "state.json"
    if not state_path.is_file():
        return False
    data = json.loads(state_path.read_text(encoding="utf-8"))
    history = data.get("history") or []
    for entry in history:
        cost = entry.get("cost") or {}
        if not cost.get("goal_met"):
            continue
        iteration = int(entry["iteration"])
        return iteration <= max_steps
    return False


def on_success(progress: ProgressStore, bucket: int) -> None:
    progress.record_success(bucket)
