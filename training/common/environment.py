"""Shared circuit sampling helpers for multi-turn rollout and starts."""
from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from intent import BOUNDS, INITIAL_GUESS, VARIABLES  # noqa: E402


def sample_params(seed: int, spread: float = 1.0) -> dict[str, float]:
    """Draw a random circuit.

    ``spread=1`` is uniform over the full legal box. Values in ``(0, 1)`` shrink
    the box around ``INITIAL_GUESS`` so multi-turn rollouts do not start on
    already-excellent or pathological notches.
    """
    rng = random.Random(seed)
    params: dict[str, float] = {}
    for name in VARIABLES:
        lo, hi = BOUNDS[name]
        if spread >= 1.0:
            value = rng.uniform(lo, hi)
        else:
            span = max(0.0, min(spread, 1.0))
            half = 0.5 * span * (hi - lo)
            center = INITIAL_GUESS[name]
            value = rng.uniform(max(lo, center - half), min(hi, center + half))
        params[name] = round(value, 4)
    return params
