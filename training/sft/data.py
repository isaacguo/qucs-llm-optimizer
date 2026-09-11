"""Dataset builders for optional trajectory SFT (legacy state + jsonl index)."""
from __future__ import annotations

import json
from pathlib import Path

from training.common.prompts import SYSTEM_PROMPT


def load_sft_records(state_path: Path) -> list[dict]:
    state = json.loads(state_path.read_text())
    records = []
    for entry in state.get("history", []):
        observation = entry.get("observation") or {}
        intent = entry.get("intent")
        if not observation.get("report_text") or not intent:
            continue
        reasoning = (entry.get("thinking") or entry.get("note") or "").strip()
        completion = (
            f"<reasoning>{reasoning}</reasoning>\n"
            f"<intent>{json.dumps(intent, sort_keys=True)}</intent>"
        )
        records.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": observation["report_text"]},
                    {"role": "assistant", "content": completion},
                ]
            }
        )
    return records


def load_multiturn_sft_from_index(
    index_path: Path,
    runs_root: Path,
    *,
    history_window: int = 8,
) -> list[dict]:
    """Thin wrapper: jsonl index → multiturn SFT records isomorphic to rollout prompts."""
    from training.sft.export import export_from_index

    return export_from_index(index_path, runs_root, history_window=history_window)
