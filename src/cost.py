"""
cost.py — objective/metric computation for the butterfly radial stub.

Design intent: the stub is a shunt bias-decoupling element. From the
junction node it should look like a broadband RF *short* (|Zin| -> 0)
across the target band, so RF energy is diverted away from the bias
line while DC continuity elsewhere is unaffected.

We report both a single scalar `total_cost` (the worst-case |Zin|/Z0 in
the target band — a minimax-style metric, appropriate because a bias
choke that fails badly at one edge frequency is a real design failure
even if the average is fine) and a breakdown across low-edge / center /
high-edge sub-bands so the strategy layer (LLM or human) has enough
signal to reason about *which direction* to move each parameter.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from qucs_sim import SimResult

Z0 = 50.0


def to_zin(s11: complex, z0: float = Z0) -> complex:
    return z0 * (1 + s11) / (1 - s11)


@dataclass
class CostReport:
    total_cost: float                 # max |Zin|/Z0 over the full target band (minimize)
    mean_cost: float                  # mean |Zin|/Z0 over the target band
    low_edge_cost: float              # |Zin|/Z0 at band low edge
    center_cost: float                # |Zin|/Z0 at band center
    high_edge_cost: float             # |Zin|/Z0 at band high edge
    worst_freq_hz: float              # frequency where |Zin| is largest in-band
    best_freq_hz: float                # frequency of the global |Zin| minimum (in full sweep)
    best_zin_mag: float
    band_hz: tuple = field(default=(0.0, 0.0))


def evaluate(res: SimResult, band_hz: tuple[float, float] = (4e9, 6e9)) -> CostReport:
    lo, hi = band_hz
    zins = [to_zin(s) for s in res.s11]
    mags = [abs(z) / Z0 for z in zins]

    band_idx = [i for i, f in enumerate(res.freq_hz) if lo <= f <= hi]
    if not band_idx:
        raise ValueError(f"no simulated points fall inside target band {band_hz}")
    band_mags = [mags[i] for i in band_idx]

    worst_i = max(band_idx, key=lambda i: mags[i])
    total_cost = mags[worst_i]
    mean_cost = sum(band_mags) / len(band_mags)

    def nearest(target_hz: float) -> int:
        return min(range(len(res.freq_hz)), key=lambda i: abs(res.freq_hz[i] - target_hz))

    low_i, center_i, high_i = nearest(lo), nearest((lo + hi) / 2), nearest(hi)

    best_i = min(range(len(mags)), key=lambda i: mags[i])

    return CostReport(
        total_cost=total_cost,
        mean_cost=mean_cost,
        low_edge_cost=mags[low_i],
        center_cost=mags[center_i],
        high_edge_cost=mags[high_i],
        worst_freq_hz=res.freq_hz[worst_i],
        best_freq_hz=res.freq_hz[best_i],
        best_zin_mag=mags[best_i] * Z0,
        band_hz=band_hz,
    )
