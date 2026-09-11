"""Generic JSONL index append / load helpers (no task-specific gate logic)."""
from __future__ import annotations

import json
from pathlib import Path


def append_index(index_path: Path, record: dict) -> None:
    """Append one JSON line to ``index_path`` (create parents if needed).

    Uses an exclusive flock so parallel writers cannot interleave lines.
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
