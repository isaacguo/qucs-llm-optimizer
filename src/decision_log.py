"""Shared decision JSONL writer for multiturn training and agent runs."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def format_agent_completion(thinking: str, intent: dict[str, Any] | None) -> str:
    reasoning = (thinking or "").strip()
    intent_json = json.dumps(intent or {}, sort_keys=True)
    return f"<reasoning>{reasoning}</reasoning>\n<intent>{intent_json}</intent>"


def append_record(path: Path, record: dict[str, Any]) -> None:
    """Append one decision record; adds UTC timestamp if missing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(record)
    payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
