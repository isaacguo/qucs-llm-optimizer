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

    @property
    def conclusion(self) -> str:
        return self._data.get("conclusion", "")

    @property
    def goal(self) -> dict | None:
        return self._data.get("goal")

    @property
    def start_seed(self) -> int | None:
        return self._data.get("start_seed")

    @property
    def initial_params(self) -> dict | None:
        return self._data.get("initial_params")

    @property
    def corpus(self) -> dict | None:
        return self._data.get("corpus")

    def set_run_meta(
        self,
        *,
        goal: dict | None = None,
        start_seed: int | None = None,
        initial_params: dict | None = None,
        corpus: dict | None = None,
    ) -> None:
        """Persist run-level metadata without touching history/iteration."""
        if goal is not None:
            self._data["goal"] = goal
        if start_seed is not None:
            self._data["start_seed"] = start_seed
        if initial_params is not None:
            self._data["initial_params"] = initial_params
        if corpus is not None:
            self._data["corpus"] = corpus
        self._save()

    def conclude(self, text: str) -> None:
        """Record why the run stopped. Closes the decision log without simulating."""
        self._data["conclusion"] = text
        self._save()

    def record(
        self,
        params: dict,
        cost: dict,
        intent: dict | None,
        note: str,
        thinking: str = "",
        observation: dict | None = None,
    ) -> int:
        """
        thinking:    free-form reasoning the strategy layer produced *before*
                     choosing `intent`. Recorded verbatim so the decision can
                     be audited (and later used as training data).
        observation: what the strategy layer was shown when it reasoned —
                     {"from_iteration": int, "report_text": str}.
        """
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
            "thinking": thinking,
            "observation": observation,
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
