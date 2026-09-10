"""Hard-success corpus gate, JSONL index, and coverage bins."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cost import s21_db  # noqa: E402

from training.goals import (  # noqa: E402
    GoalDistributionConfig,
    GoalSpec,
    is_goal_met,
)


def _goal_from_state(raw: dict | GoalSpec) -> GoalSpec:
    if isinstance(raw, GoalSpec):
        return raw
    band = raw["band_hz"]
    return GoalSpec(
        target_freq_hz=float(raw["target_freq_hz"]),
        band_hz=(float(band[0]), float(band[1])),
        target_depth_db=float(raw["target_depth_db"]),
    )


def gate_run(state: dict) -> dict:
    """Gate a teacher run for corpus eligibility against ``state["goal"]``."""
    if "goal" not in state:
        raise KeyError("state must include 'goal'")

    goal = _goal_from_state(state["goal"])
    history = state.get("history") or []

    best_cost: float | None = None
    goal_met_iteration: int | None = None
    has_non_null_intent = False

    for entry in history:
        if entry.get("intent") is not None:
            has_non_null_intent = True
        cost = entry.get("cost")
        if not isinstance(cost, dict) or "total_cost" not in cost:
            continue
        total = float(cost["total_cost"])
        if best_cost is None or total < best_cost:
            best_cost = total
        if goal_met_iteration is None and is_goal_met(cost, goal):
            goal_met_iteration = int(entry["iteration"])

    best_db = None if best_cost is None else s21_db(best_cost)

    # Null-only (or empty) history never qualifies as hard-success.
    if not has_non_null_intent:
        return {
            "eligible": False,
            "reason": "null_only_history",
            "best_db": best_db,
            "goal_met_iteration": None,
        }

    if goal_met_iteration is not None:
        return {
            "eligible": True,
            "reason": "goal_met",
            "best_db": best_db,
            "goal_met_iteration": goal_met_iteration,
        }
    return {
        "eligible": False,
        "reason": "goal_not_met",
        "best_db": best_db,
        "goal_met_iteration": None,
    }


def append_index(index_path: Path, record: dict) -> None:
    """Append one JSON line to ``index_path`` (create parents if needed)."""
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True) + "\n"
    with open(index_path, "a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()


def load_index(index_path: Path) -> list[dict]:
    """Load all JSONL records from ``index_path`` (empty list if missing)."""
    index_path = Path(index_path)
    if not index_path.is_file():
        return []
    rows: list[dict] = []
    with open(index_path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def coverage_counts(
    index: list[dict],
    dist: GoalDistributionConfig,
) -> dict[tuple[int, int], int]:
    """Count index goals by ``(freq_bin_index, depth_bin_index)``."""
    counts: dict[tuple[int, int], int] = {}
    for record in index:
        goal = record["goal"]
        freq = float(goal["target_freq_hz"])
        depth = float(goal["target_depth_db"])
        freq_bin = int((freq - dist.freq_min_hz) // dist.freq_bin_hz)
        depth_bin = int((depth - dist.depth_db_min) // dist.depth_bin_db)
        key = (freq_bin, depth_bin)
        counts[key] = counts.get(key, 0) + 1
    return counts


def index_record_from_run(
    run_id: str,
    run_dir: str,
    state: dict,
    corpus: dict,
) -> dict:
    """Build one corpus index record from a gated hard-success run."""
    history = state.get("history") or []
    n_steps = sum(1 for entry in history if entry.get("intent") is not None)
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "goal": state["goal"],
        "best_db": corpus["best_db"],
        "n_steps": n_steps,
        "goal_met_iteration": corpus.get("goal_met_iteration"),
        "added_at": datetime.now(timezone.utc).isoformat(),
    }
