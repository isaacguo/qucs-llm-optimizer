# jobs/bpf5_agent/assign.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from state import RunState

TASK_NAME = "butterworth_bpf5"


def make_activity_id(now: datetime) -> str:
    return now.strftime("bpf_agent_%Y%m%d_%H%M%S")


def _mhz_token(value: float) -> str:
    """Filesystem-safe MHz token: 112.5 → 112p5, 5.0 → 5."""
    text = f"{float(value):g}"
    return text.replace(".", "p").replace("-", "m")


def make_rollout_id(bucket: int, cf_mhz: float, bw_mhz: float) -> str:
    return f"rollout_b{int(bucket):02d}_cf{_mhz_token(cf_mhz)}_bw{_mhz_token(bw_mhz)}"


def assign_rollout(
    runs_root: Path | str,
    activity_id: str,
    bucket: int,
    goal_dict: dict,
    *,
    cf_mhz: float,
    bw_mhz: float,
) -> Path:
    """Create nested rollout dir and write task/goal meta for run_step init."""
    rollout_id = make_rollout_id(bucket, cf_mhz, bw_mhz)
    rollout_dir = Path(runs_root) / activity_id / rollout_id
    state = RunState(rollout_dir)
    state.set_run_meta(task=TASK_NAME, goal=goal_dict)
    return rollout_dir
