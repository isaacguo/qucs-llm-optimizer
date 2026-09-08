"""Randomized design-goal specs for the butterfly-stub notch task.

Stage A of goal-conditioned GRPO training: only the notch target frequency
varies. Depth threshold and goal "type" (notch) stay fixed so the policy has
one axis of variation to generalize over before we add more (Stage B/C).
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass

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


def band_for(target_freq_hz: float) -> tuple[float, float]:
    return (target_freq_hz - BAND_HALF_WIDTH_HZ, target_freq_hz + BAND_HALF_WIDTH_HZ)


def sample_goal(
    seed: int,
    freq_range: tuple[float, float] = TRAIN_FREQ_RANGE_HZ,
) -> GoalSpec:
    """Sample a training-distribution goal, staying clear of the held-out grid."""
    rng = random.Random(seed ^ 0x60A1)
    lo, hi = freq_range
    for _ in range(50):
        freq = rng.uniform(lo, hi)
        if all(abs(freq - h) >= _HELDOUT_EXCLUSION_HZ for h in HELDOUT_FREQS_HZ):
            return GoalSpec(target_freq_hz=freq, band_hz=band_for(freq))
    # Astronomically unlikely fallback: nudge away from the nearest held-out point
    # rather than raising, so a pathological freq_range can't crash a training run.
    nearest = min(HELDOUT_FREQS_HZ, key=lambda h: abs(h - rng.uniform(lo, hi)))
    freq = min(max(nearest - _HELDOUT_EXCLUSION_HZ, lo), hi)
    return GoalSpec(target_freq_hz=freq, band_hz=band_for(freq))


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
