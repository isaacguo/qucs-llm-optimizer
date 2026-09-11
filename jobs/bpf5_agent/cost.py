# jobs/bpf5_agent/cost.py
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from jobs.bpf5_agent.goals import BpfGoalSpec, split_bands
from qucs_sim import SimResult


def s21_db(mag: float) -> float:
    if mag <= 0.0:
        return float("-inf")
    return 20.0 * math.log10(mag)


@dataclass
class CostReport:
    total_cost: float
    passband_min_s21_db: float
    stopband_max_s21_db: float
    passband_mean_s21_db: float
    s11_passband_max_db: float
    has_passband_samples: bool
    has_stopband_samples: bool
    f_low_hz: float
    f_high_hz: float
    s21_peak_freq_hz: float  # freq of max |S21| over the full sweep

    def to_dict(self) -> dict:
        return asdict(self)


def _indices_in(freq: list[float], band: tuple[float, float]) -> list[int]:
    lo, hi = band
    return [i for i, f in enumerate(freq) if lo <= f <= hi]


def evaluate(res: SimResult, goal: BpfGoalSpec, *, lam: float = 1.0) -> CostReport:
    if res.s21 is None or len(res.s21) != len(res.freq_hz):
        raise ValueError("SimResult.s21 required and must match freq_hz")
    pb, s_lo, s_hi = split_bands(goal)
    s21_mags = [abs(s) for s in res.s21]
    s11_mags = [abs(s) for s in res.s11]

    pb_idx = _indices_in(res.freq_hz, pb)
    if not pb_idx:
        raise ValueError(f"no simulated points in passband {pb}")

    stop_idx: list[int] = []
    if s_lo is not None:
        stop_idx.extend(_indices_in(res.freq_hz, s_lo))
    if s_hi is not None:
        stop_idx.extend(_indices_in(res.freq_hz, s_hi))
    if not stop_idx:
        raise ValueError("no simulated points in either stopband; adjust guard/sweep")

    pb_mags = [s21_mags[i] for i in pb_idx]
    stop_mags = [s21_mags[i] for i in stop_idx]
    pass_term = max(1.0 - m for m in pb_mags)
    stop_term = max(stop_mags)
    pb_dbs = [s21_db(m) for m in pb_mags]
    s11_pb_dbs = [s21_db(s11_mags[i]) for i in pb_idx]
    peak_i = max(range(len(s21_mags)), key=lambda i: s21_mags[i])

    return CostReport(
        total_cost=pass_term + lam * stop_term,
        passband_min_s21_db=min(pb_dbs),
        stopband_max_s21_db=s21_db(stop_term),
        passband_mean_s21_db=sum(pb_dbs) / len(pb_dbs),
        s11_passband_max_db=max(s11_pb_dbs),
        has_passband_samples=True,
        has_stopband_samples=True,
        f_low_hz=goal.f_low_hz,
        f_high_hz=goal.f_high_hz,
        s21_peak_freq_hz=float(res.freq_hz[peak_i]),
    )
