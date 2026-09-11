"""Hard-success corpus gate, JSONL index, coverage bins, and multiturn SFT export."""
from __future__ import annotations

import json
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cost import s21_db  # noqa: E402

from training.common.goals import (  # noqa: E402
    GoalDistributionConfig,
    GoalSpec,
    is_goal_met,
)
from training.common.prompts import TurnRecord, build_multiturn_prompt  # noqa: E402


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
    """Append one JSON line to ``index_path`` (create parents if needed).

    Uses an exclusive flock so parallel corpus workers cannot interleave lines.
    """
    import fcntl

    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True) + "\n"
    with open(index_path, "a", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.write(line)
            fh.flush()
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


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


def _compress_thinking(text: str, *, max_sentences: int = 3) -> str:
    """Keep at most ``max_sentences`` sentences (split on ``. `` / newlines)."""
    raw = (text or "").strip()
    if not raw:
        return ""
    parts = re.split(r"(?:\n|\. )+", raw)
    sentences = [p.strip().rstrip(".") for p in parts if p.strip()]
    kept = sentences[:max_sentences]
    if not kept:
        return ""
    return ". ".join(kept) + ("." if kept else "")


def export_multiturn_sft_records(
    state: dict,
    *,
    history_window: int = 8,
    run_id: str | None = None,
    patience_eps: float = 0.2,
) -> list[dict]:
    """Rebuild step-wise SFT examples via ``build_multiturn_prompt`` (not report_text).

    ``best_db`` passed into later-turn prompts follows ``run_trajectory``: update
    only when ``new_db < best_db - patience_eps`` (default 0.2), not strict min.
    """
    goal = _goal_from_state(state["goal"])
    history = state.get("history") or []
    if not history:
        warnings.warn(
            "export_multiturn_sft_records: zero turns exported (empty history)",
            stacklevel=2,
        )
        return []

    records: list[dict] = []
    prior_turns: list[TurnRecord] = []
    initial_db: float | None = None
    best_db: float | None = None
    turn_index = 0

    for idx, entry in enumerate(history):
        intent = entry.get("intent")
        cost = entry.get("cost")
        if intent is None:
            if isinstance(cost, dict) and "total_cost" in cost and initial_db is None:
                initial_db = s21_db(float(cost["total_cost"]))
                best_db = initial_db
            continue

        if idx == 0:
            continue

        before = history[idx - 1]
        before_params = before.get("params")
        before_cost = before.get("cost")
        if not isinstance(before_params, dict) or not isinstance(before_cost, dict):
            continue
        if "total_cost" not in before_cost:
            continue

        db_before = s21_db(float(before_cost["total_cost"]))
        if initial_db is None:
            initial_db = db_before
            best_db = db_before

        # First tuning turn: match rollout by letting build_multiturn_prompt
        # derive init/best from the current cost when no prior turns exist.
        if not prior_turns:
            prompt_init: float | None = None
            prompt_best: float | None = None
        else:
            prompt_init = initial_db
            prompt_best = best_db

        prompt = build_multiturn_prompt(
            goal,
            turn_index,
            before_params,
            before_cost,
            prior_turns,
            history_window=history_window,
            initial_db=prompt_init,
            best_db=prompt_best,
        )
        reasoning = _compress_thinking(entry.get("thinking") or entry.get("note") or "")
        completion = (
            f"<reasoning>{reasoning}</reasoning>\n"
            f"<intent>{json.dumps(intent, sort_keys=True)}</intent>"
        )
        meta: dict = {
            "turn_index": turn_index,
            "iteration": entry.get("iteration"),
            "goal": state["goal"],
        }
        if run_id is not None:
            meta["run_id"] = run_id

        records.append(
            {
                "messages": [
                    prompt[0],
                    prompt[1],
                    {"role": "assistant", "content": completion},
                ],
                "meta": meta,
            }
        )

        after_cost = entry.get("cost") or {}
        db_after = (
            s21_db(float(after_cost["total_cost"]))
            if isinstance(after_cost, dict) and "total_cost" in after_cost
            else db_before
        )
        # Same patience gate as run_trajectory (not strict historical min).
        if best_db is None or db_after < best_db - patience_eps:
            best_db = db_after
        after_params = entry.get("params") or before_params
        prior_turns.append(
            TurnRecord(
                turn_index=turn_index,
                prompt=prompt,
                completion_text=completion,
                valid=True,
                intent=dict(intent),
                params_before=dict(before_params),
                params_after=dict(after_params),
                db_before=db_before,
                db_after=db_after,
            )
        )
        turn_index += 1

    if not records:
        warnings.warn(
            "export_multiturn_sft_records: zero turns exported for this run",
            stacklevel=2,
        )
    return records


def export_from_index(
    index_path: Path,
    runs_root: Path,
    history_window: int = 8,
) -> list[dict]:
    """Export multiturn SFT records for runs listed in the corpus index only."""
    runs_root = Path(runs_root)
    rows = load_index(Path(index_path))
    out: list[dict] = []
    for record in rows:
        run_id = record["run_id"]
        run_dir = Path(record.get("run_dir") or run_id)
        if not run_dir.is_absolute():
            run_dir = runs_root / run_dir
        state_path = run_dir / "state.json"
        if not state_path.is_file():
            raise FileNotFoundError(f"missing state for indexed run {run_id}: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        out.extend(
            export_multiturn_sft_records(
                state,
                history_window=history_window,
                run_id=run_id,
            )
        )
    return out
