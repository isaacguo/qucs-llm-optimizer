"""Export multiturn SFT records from a JSONL run index (legacy notch trajectories)."""
from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cost import s21_db  # noqa: E402

from training.common.goals import GoalSpec  # noqa: E402
from training.common.jsonl_index import load_index  # noqa: E402
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
    """Export multiturn SFT records for runs listed in the index only."""
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
