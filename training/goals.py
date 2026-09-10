"""Randomized design-goal specs for the butterfly-stub notch task.

Stage A of goal-conditioned GRPO training: only the notch target frequency
varies. Depth threshold and goal "type" (notch) stay fixed so the policy has
one axis of variation to generalize over before we add more (Stage B/C).
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

TRAIN_FREQ_RANGE_HZ = (4.0e9, 6.0e9)
BAND_HALF_WIDTH_HZ = 1.0e9
DEFAULT_TARGET_DEPTH_DB = -70.0

# Fixed frequencies withheld from the training distribution. Never sampled by
# sample_goal(); only used by evaluate_generalization.py to check whether the
# policy generalizes to goals it never saw during training, as opposed to
# having memorized the training distribution.
HELDOUT_FREQS_HZ = (4.2e9, 4.6e9, 5.0e9, 5.4e9, 5.8e9)
_HELDOUT_EXCLUSION_HZ = 0.05e9  # keep train sampling >= 50 MHz from any held-out point


@dataclass(frozen=True)
class GoalSpec:
    target_freq_hz: float
    band_hz: tuple[float, float]
    target_depth_db: float = DEFAULT_TARGET_DEPTH_DB

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    def describe(self) -> str:
        return (
            f"{self.target_freq_hz / 1e9:.3f} GHz "
            f"(goal: |S21| <= {self.target_depth_db:.0f} dB)"
        )


@dataclass(frozen=True)
class GoalDistributionConfig:
    freq_min_hz: float
    freq_max_hz: float
    depth_db_min: float
    depth_db_max: float
    band_half_width_hz: float
    heldout_enabled: bool
    freq_bin_hz: float
    depth_bin_db: float


def load_goal_distribution(path: str | Path) -> GoalDistributionConfig:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return GoalDistributionConfig(
        freq_min_hz=float(data["freq_min_hz"]),
        freq_max_hz=float(data["freq_max_hz"]),
        depth_db_min=float(data["depth_db_min"]),
        depth_db_max=float(data["depth_db_max"]),
        band_half_width_hz=float(data["band_half_width_hz"]),
        heldout_enabled=bool(data["heldout_enabled"]),
        freq_bin_hz=float(data["freq_bin_hz"]),
        depth_bin_db=float(data["depth_bin_db"]),
    )


def band_for(
    target_freq_hz: float,
    half_width_hz: float | None = None,
) -> tuple[float, float]:
    hw = BAND_HALF_WIDTH_HZ if half_width_hz is None else half_width_hz
    return (target_freq_hz - hw, target_freq_hz + hw)


def sample_goal(
    seed: int,
    freq_range: tuple[float, float] = TRAIN_FREQ_RANGE_HZ,
    target_depth_db: float = DEFAULT_TARGET_DEPTH_DB,
) -> GoalSpec:
    """Sample a training-distribution goal, staying clear of the held-out grid."""
    rng = random.Random(seed ^ 0x60A1)
    lo, hi = freq_range
    for _ in range(50):
        freq = rng.uniform(lo, hi)
        if all(abs(freq - h) >= _HELDOUT_EXCLUSION_HZ for h in HELDOUT_FREQS_HZ):
            return GoalSpec(
                target_freq_hz=freq,
                band_hz=band_for(freq),
                target_depth_db=target_depth_db,
            )
    # Astronomically unlikely fallback: nudge away from the nearest held-out point
    # rather than raising, so a pathological freq_range can't crash a training run.
    nearest = min(HELDOUT_FREQS_HZ, key=lambda h: abs(h - rng.uniform(lo, hi)))
    freq = min(max(nearest - _HELDOUT_EXCLUSION_HZ, lo), hi)
    return GoalSpec(
        target_freq_hz=freq,
        band_hz=band_for(freq),
        target_depth_db=target_depth_db,
    )


def sample_goal_from_distribution(seed: int, dist: GoalDistributionConfig) -> GoalSpec:
    """Sample freq and depth from ``dist``; apply held-out exclusion only if enabled."""
    rng = random.Random(seed ^ 0x60A1)
    depth = rng.uniform(dist.depth_db_min, dist.depth_db_max)
    lo, hi = dist.freq_min_hz, dist.freq_max_hz
    half = dist.band_half_width_hz

    if not dist.heldout_enabled:
        freq = rng.uniform(lo, hi)
        return GoalSpec(
            target_freq_hz=freq,
            band_hz=band_for(freq, half),
            target_depth_db=depth,
        )

    for _ in range(50):
        freq = rng.uniform(lo, hi)
        if all(abs(freq - h) >= _HELDOUT_EXCLUSION_HZ for h in HELDOUT_FREQS_HZ):
            return GoalSpec(
                target_freq_hz=freq,
                band_hz=band_for(freq, half),
                target_depth_db=depth,
            )
    nearest = min(HELDOUT_FREQS_HZ, key=lambda h: abs(h - rng.uniform(lo, hi)))
    freq = min(max(nearest - _HELDOUT_EXCLUSION_HZ, lo), hi)
    return GoalSpec(
        target_freq_hz=freq,
        band_hz=band_for(freq, half),
        target_depth_db=depth,
    )


def is_goal_met(cost: dict, goal: GoalSpec) -> bool:
    threshold = 10 ** (goal.target_depth_db / 20.0)
    return cost["total_cost"] <= threshold


def heldout_goals() -> list[GoalSpec]:
    """The fixed, never-trained-on goals used to test generalization."""
    return [GoalSpec(target_freq_hz=f, band_hz=band_for(f)) for f in HELDOUT_FREQS_HZ]


def goal_from_json(raw: str) -> GoalSpec:
    data = json.loads(raw)
    band = data["band_hz"]
    return GoalSpec(
        target_freq_hz=float(data["target_freq_hz"]),
        band_hz=(float(band[0]), float(band[1])),
        target_depth_db=float(data.get("target_depth_db", DEFAULT_TARGET_DEPTH_DB)),
    )
