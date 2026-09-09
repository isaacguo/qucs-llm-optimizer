"""Trainable starting-circuit sampling with notch-depth headroom."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cost import evaluate, s21_db  # noqa: E402
from qucs_sim import simulate  # noqa: E402

from training.environment import sample_params  # noqa: E402
from training.goals import GoalSpec  # noqa: E402


def is_start_too_deep(
    initial_db: float,
    *,
    target_depth_db: float,
    headroom_db: float = 5.0,
) -> bool:
    """True when the start is already within ``headroom_db`` of the goal depth."""
    return initial_db <= target_depth_db + headroom_db


def sample_params_with_headroom(
    *,
    start_seed: int,
    goal: GoalSpec,
    spread: float = 0.35,
    headroom_db: float = 5.0,
    max_tries: int = 20,
    cost_fn: Callable[[dict[str, float]], dict] | None = None,
    simulator: Callable = simulate,
) -> tuple[dict[str, float], float, int, int]:
    """Sample until the start notch is shallow enough to train on.

    Returns ``(params, initial_db, seed_used, n_resamples)``.
    ``n_resamples`` counts rejected draws (not including the accepted one).
    """

    def get_cost(params: dict[str, float]) -> dict:
        if cost_fn is not None:
            return cost_fn(params)
        from dataclasses import asdict, is_dataclass

        result = simulator(params, export_layout=False)
        cost = evaluate(result, goal.band_hz, target_hz=goal.target_freq_hz)
        return asdict(cost) if is_dataclass(cost) else dict(vars(cost))

    seed = start_seed
    n_resamples = 0
    last_params: dict[str, float] | None = None
    last_db = 0.0
    for attempt in range(max_tries):
        params = sample_params(seed, spread=spread)
        cost = get_cost(params)
        initial_db = s21_db(float(cost["total_cost"]))
        last_params, last_db = params, initial_db
        if not is_start_too_deep(
            initial_db, target_depth_db=goal.target_depth_db, headroom_db=headroom_db
        ):
            return params, initial_db, seed, n_resamples
        n_resamples += 1
        seed = start_seed + attempt + 1
    assert last_params is not None
    return last_params, last_db, seed, n_resamples
