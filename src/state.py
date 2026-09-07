"""
state.py — JSON-backed run history for the butterfly-stub optimization loop.
"""
from __future__ import annotations

import json
from pathlib import Path

MAX_ITERATIONS = 20


class RunState:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.state_path = run_dir / "state.json"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            self._data = json.loads(self.state_path.read_text())
        else:
            self._data = {"iteration": -1, "params": None, "history": []}

    @property
    def iteration(self) -> int:
        return self._data["iteration"]

    @property
    def params(self) -> dict | None:
        return self._data["params"]

    @property
    def history(self) -> list[dict]:
        return self._data["history"]

    def record(self, params: dict, cost: dict, intent: dict | None, note: str) -> int:
        if self._data["iteration"] + 1 > MAX_ITERATIONS - 1:
            raise RuntimeError(f"iteration cap reached ({MAX_ITERATIONS}); refusing further steps")
        self._data["iteration"] += 1
        self._data["params"] = params
        entry = {
            "iteration": self._data["iteration"],
            "params": params,
            "cost": cost,
            "intent": intent,
            "note": note,
        }
        self._data["history"].append(entry)
        self._save()
        return self._data["iteration"]

    def best(self) -> dict | None:
        if not self._data["history"]:
            return None
        return min(self._data["history"], key=lambda e: e["cost"]["total_cost"])

    def _save(self) -> None:
        self.state_path.write_text(json.dumps(self._data, indent=2))
