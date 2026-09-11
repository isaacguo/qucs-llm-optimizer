# jobs/bpf5_agent/buckets.py
from __future__ import annotations

import random

N_BUCKETS = 20
BUCKET_WIDTH_MHZ = 15
SWEEP_MHZ = (0.0, 300.0)
BW_CHOICES_MHZ = (5, 10, 15)
SUCCESS_PER_BUCKET = 10
MAX_STEPS = 20


def _bucket_cf_range(bucket: int) -> tuple[float, float]:
    if not 0 <= bucket < N_BUCKETS:
        raise ValueError(f"bucket must be in 0..{N_BUCKETS - 1}, got {bucket}")
    lo = bucket * BUCKET_WIDTH_MHZ
    hi = (bucket + 1) * BUCKET_WIDTH_MHZ
    if bucket == N_BUCKETS - 1:
        return lo, float(SWEEP_MHZ[1])
    return float(lo), float(hi)


def sample_goal(bucket: int, rng: random.Random) -> tuple[dict, float, float]:
    """Sample (goal_dict, cf_mhz, bw_mhz) with passband inside SWEEP_MHZ."""
    sweep_lo, sweep_hi = SWEEP_MHZ
    cf_lo, cf_hi = _bucket_cf_range(bucket)
    while True:
        bw = float(rng.choice(BW_CHOICES_MHZ))
        half = bw / 2.0
        # Buckets 0..18 are half-open [lo, hi); last bucket includes sweep upper edge.
        cf = rng.uniform(cf_lo, cf_hi)
        if bucket < N_BUCKETS - 1 and cf >= cf_hi:
            continue
        f_low = cf - half
        f_high = cf + half
        if f_low < sweep_lo or f_high > sweep_hi:
            continue
        goal = {
            "f_low_hz": f_low * 1e6,
            "f_high_hz": f_high * 1e6,
        }
        return goal, cf, bw


def pick_bucket(counts: list[int], rng: random.Random) -> int:
    """Prefer buckets with counts[b] < SUCCESS_PER_BUCKET; else any bucket."""
    if len(counts) != N_BUCKETS:
        raise ValueError(f"counts must have length {N_BUCKETS}")
    under = [i for i, c in enumerate(counts) if c < SUCCESS_PER_BUCKET]
    pool = under if under else list(range(N_BUCKETS))
    return int(rng.choice(pool))
