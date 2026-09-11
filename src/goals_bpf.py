# src/goals_bpf.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BpfGoalSpec:
    f_low_hz: float
    f_high_hz: float
    passband_il_max_db: float = -1.0
    stopband_atten_min_db: float = -20.0
    stopband_guard_hz: float = 10e6
    sweep_hz: tuple[float, float] = (0.0, 300e6)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sweep_hz"] = list(self.sweep_hz)
        return d


def default_goal() -> BpfGoalSpec:
    return BpfGoalSpec(f_low_hz=135e6, f_high_hz=165e6)


def validate_goal(goal: BpfGoalSpec) -> None:
    lo, hi = goal.sweep_hz
    if goal.f_low_hz >= goal.f_high_hz:
        raise ValueError("f_low_hz must be < f_high_hz")
    if not (lo <= goal.f_low_hz <= hi and lo <= goal.f_high_hz <= hi):
        raise ValueError("passband edges must lie inside sweep_hz")


def split_bands(
    goal: BpfGoalSpec,
) -> tuple[tuple[float, float], tuple[float, float] | None, tuple[float, float] | None]:
    validate_goal(goal)
    sweep_lo, sweep_hi = goal.sweep_hz
    pb = (goal.f_low_hz, goal.f_high_hz)
    lower_hi = goal.f_low_hz - goal.stopband_guard_hz
    upper_lo = goal.f_high_hz + goal.stopband_guard_hz
    s_lo = (sweep_lo, lower_hi) if lower_hi > sweep_lo else None
    s_hi = (upper_lo, sweep_hi) if upper_lo < sweep_hi else None
    return pb, s_lo, s_hi


def goal_from_dict(data: dict) -> BpfGoalSpec:
    sweep = data.get("sweep_hz", (0.0, 300e6))
    return BpfGoalSpec(
        f_low_hz=float(data["f_low_hz"]),
        f_high_hz=float(data["f_high_hz"]),
        passband_il_max_db=float(data.get("passband_il_max_db", -1.0)),
        stopband_atten_min_db=float(data.get("stopband_atten_min_db", -20.0)),
        stopband_guard_hz=float(data.get("stopband_guard_hz", 10e6)),
        sweep_hz=(float(sweep[0]), float(sweep[1])),
    )


def goal_from_json(raw: str) -> BpfGoalSpec:
    return goal_from_dict(json.loads(raw))


def is_goal_met(cost: dict, goal: BpfGoalSpec) -> bool:
    if not cost.get("has_passband_samples") or not cost.get("has_stopband_samples"):
        return False
    return (
        float(cost["passband_min_s21_db"]) >= goal.passband_il_max_db
        and float(cost["stopband_max_s21_db"]) <= goal.stopband_atten_min_db
    )
