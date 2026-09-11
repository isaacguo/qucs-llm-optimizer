"""
intent.py — deterministic translator: qualitative "intent" -> precise numeric
parameter update.

This is the layer that keeps the LLM (or, for now, the human "cursor agent"
playing that role) out of raw-number arithmetic. The LLM only ever emits a
structured, qualitative *direction* per variable; this module owns all
arithmetic: step-size magnitude, the iteration-based decay schedule (coarse
exploration early, fine refinement late — same idea as simulated-annealing
cooling), and bound clamping.

Intent schema (per free variable): one of
    "increase_strong" | "increase" | "increase_slight" | "hold" |
    "decrease_slight" | "decrease" | "decrease_strong"
"""
from __future__ import annotations

VARIABLES = ("ri", "ro", "alpha", "Wf", "Lc")

BOUNDS = {
    "ri": (0.10, 1.50),      # mm
    "ro": (3.00, 14.00),     # mm
    "alpha": (30.0, 150.0),  # deg
    "Wf": (0.20, 2.50),      # mm
    "Lc": (0.20, 10.00),     # mm
}

INITIAL_GUESS = {
    "ri": 0.30,
    "ro": 8.00,
    "alpha": 90.0,
    "Wf": 0.60,
    "Lc": 3.00,
}

_MAGNITUDE_FRAC = {
    "strong": 0.20,
    "normal": 0.10,
    "slight": 0.04,
}

_DECAY_RATE = 0.93  # step size shrinks each iteration: frac * DECAY_RATE**iteration
_MIN_SCALE = 0.15   # never decay below 15% of the base step, to avoid stalling


def _parse_intent_token(token: str) -> tuple[int, str]:
    """Return (sign, magnitude_key) for a token like 'increase_strong'."""
    if token == "hold":
        return 0, "normal"
    sign = 1 if token.startswith("increase") else -1 if token.startswith("decrease") else None
    if sign is None:
        raise ValueError(f"unrecognised intent token: {token!r}")
    if "_" in token:
        _, mag = token.split("_", 1)
    else:
        mag = "normal"
    if mag not in _MAGNITUDE_FRAC:
        raise ValueError(f"unrecognised magnitude in token: {token!r}")
    return sign, mag


def apply_intent(
    params: dict,
    intent: dict,
    iteration: int,
    *,
    variables: tuple[str, ...] | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> dict:
    """
    params:    current {var: value}
    intent:    {var: token} for a subset (or all) of VARIABLES; missing vars
               are treated as "hold"
    iteration: 0-based iteration count, used for the step-decay schedule
    variables / bounds: optional overrides (defaults to butterfly module tables)
    """
    vars_ = variables if variables is not None else VARIABLES
    bounds_ = bounds if bounds is not None else BOUNDS
    unknown = set(intent) - set(vars_)
    if unknown:
        raise ValueError(f"unknown intent keys for task: {sorted(unknown)}")
    decay = max(_DECAY_RATE ** iteration, _MIN_SCALE)
    new_params = dict(params)
    for var in vars_:
        token = intent.get(var, "hold")
        sign, mag_key = _parse_intent_token(token)
        if sign == 0:
            continue
        lo, hi = bounds_[var]
        step = sign * _MAGNITUDE_FRAC[mag_key] * decay * (hi - lo)
        new_val = params[var] + step
        new_val = min(max(new_val, lo), hi)
        new_params[var] = round(new_val, 4)
    return new_params
