"""
cost.py — notch-at-target objective for the shunt butterfly on a 2-port line.

Topology (templates/butterfly_stub.sch.tpl):
  P1 -- MLin -- MCROSS -- MLout -- P2
                    |
              butterfly MRSTUBs

Primary goal: place a deep transmission notch at TARGET_NOTCH_HZ (5.5 GHz).
total_cost = |S21| at that frequency (minimize). Success threshold for a
"-70 dB" notch is |S21| <= 10**(-70/20) ≈ 3.16e-4.

Stopband edge / passband stats remain diagnostic only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from qucs_sim import SimResult

Z0 = 50.0
TARGET_NOTCH_HZ = 5.5e9
TARGET_DEPTH_DB = -70.0
TARGET_S21_MAG = 10 ** (TARGET_DEPTH_DB / 20.0)  # ≈ 3.162e-4


def to_zin(s11: complex, z0: float = Z0) -> complex:
    return z0 * (1 + s11) / (1 - s11)


def s21_db(mag: float) -> float:
    if mag <= 0.0:
        return float("-inf")
    return 20.0 * math.log10(mag)


@dataclass
class CostReport:
    total_cost: float                 # |S21| at target_freq_hz (minimize)
    target_freq_hz: float             # requested notch frequency
    target_s21_mag: float             # same as total_cost; explicit alias
    mean_cost: float                  # mean |S21| in stopband (aux)
    low_edge_cost: float              # |S21| at stopband low edge
    center_cost: float                # |S21| at stopband geometric center
    high_edge_cost: float             # |S21| at stopband high edge
    stopband_max_s21: float           # max |S21| in stopband (old minimax)
    worst_freq_hz: float              # in-stopband freq with largest |S21|
    best_freq_hz: float               # full-sweep freq with smallest |S21|
    best_s21_mag: float               # |S21| at best_freq_hz
    passband_low_mean: float
    passband_high_mean: float
    zin_norm_at_center: float
    band_hz: tuple = field(default=(0.0, 0.0))


def evaluate(
    res: SimResult,
    band_hz: tuple[float, float] = (4e9, 6e9),
    target_hz: float = TARGET_NOTCH_HZ,
) -> CostReport:
    if res.s21 is None or len(res.s21) != len(res.freq_hz):
        raise ValueError("SimResult.s21 required for band-stop cost (same length as freq_hz)")
    if len(res.s11) != len(res.freq_hz):
        raise ValueError("SimResult.s11 length must match freq_hz")

    lo, hi = band_hz
    s21_mags = [abs(s) for s in res.s21]
    zin_norms = [abs(to_zin(s)) / Z0 for s in res.s11]

    stop_idx = [i for i, f in enumerate(res.freq_hz) if lo <= f <= hi]
    if not stop_idx:
        raise ValueError(f"no simulated points fall inside stopband {band_hz}")

    stop_mags = [s21_mags[i] for i in stop_idx]
    worst_i = max(stop_idx, key=lambda i: s21_mags[i])
    mean_cost = sum(stop_mags) / len(stop_mags)

    def nearest(target: float) -> int:
        return min(range(len(res.freq_hz)), key=lambda i: abs(res.freq_hz[i] - target))

    low_i, center_i, high_i = nearest(lo), nearest((lo + hi) / 2), nearest(hi)
    target_i = nearest(target_hz)
    best_i = min(range(len(s21_mags)), key=lambda i: s21_mags[i])
    target_mag = s21_mags[target_i]

    low_pass = [s21_mags[i] for i, f in enumerate(res.freq_hz) if f < lo]
    high_pass = [s21_mags[i] for i, f in enumerate(res.freq_hz) if f > hi]
    passband_low_mean = sum(low_pass) / len(low_pass) if low_pass else float("nan")
    passband_high_mean = sum(high_pass) / len(high_pass) if high_pass else float("nan")

    return CostReport(
        total_cost=target_mag,
        target_freq_hz=res.freq_hz[target_i],
        target_s21_mag=target_mag,
        mean_cost=mean_cost,
        low_edge_cost=s21_mags[low_i],
        center_cost=s21_mags[center_i],
        high_edge_cost=s21_mags[high_i],
        stopband_max_s21=s21_mags[worst_i],
        worst_freq_hz=res.freq_hz[worst_i],
        best_freq_hz=res.freq_hz[best_i],
        best_s21_mag=s21_mags[best_i],
        passband_low_mean=passband_low_mean,
        passband_high_mean=passband_high_mean,
        zin_norm_at_center=zin_norms[center_i],
        band_hz=band_hz,
    )
