"""Shared depth + notch-frequency terms for one-step and multi-turn rewards.

``best_freq_hz`` is the sweep frequency of the deepest |S21|. Alignment is
linear in |f_notch - f_target| and hits 0 at one stopband half-width (1 GHz).
"""
from __future__ import annotations

from training.goals import BAND_HALF_WIDTH_HZ

REWARD_CLIP = 20.0
BEST_WEIGHT = 0.7
FINAL_WEIGHT = 0.3
FREQ_WEIGHT = 0.5


def clip_reward(value: float, clip: float = REWARD_CLIP) -> float:
    return max(-clip, min(clip, float(value)))


def frequency_alignment(
    notch_hz: float | None,
    target_hz: float | None,
    *,
    scale_hz: float = BAND_HALF_WIDTH_HZ,
) -> float:
    """Return 1.0 when the notch sits on the goal, 0.0 at |Δf| >= scale_hz."""
    if notch_hz is None or target_hz is None or scale_hz <= 0.0:
        return 0.0
    err = abs(float(notch_hz) - float(target_hz))
    return max(0.0, 1.0 - err / scale_hz)


def frequency_improvement_reward(
    old_hz: float | None,
    new_hz: float | None,
    target_hz: float | None,
    *,
    clip: float = REWARD_CLIP,
) -> float:
    delta = frequency_alignment(new_hz, target_hz) - frequency_alignment(old_hz, target_hz)
    return clip_reward(clip * delta, clip)


def combined_step_reward(
    delta_db: float,
    old_freq_hz: float | None,
    new_freq_hz: float | None,
    target_freq_hz: float | None,
    *,
    clip: float = REWARD_CLIP,
    freq_weight: float = FREQ_WEIGHT,
) -> float:
    depth = clip_reward(delta_db, clip)
    freq = frequency_improvement_reward(
        old_freq_hz, new_freq_hz, target_freq_hz, clip=clip
    )
    return depth + freq_weight * freq


def mixed_terminal_reward(
    initial_db: float,
    best_db: float,
    final_db: float,
    *,
    clip: float = REWARD_CLIP,
    best_weight: float = BEST_WEIGHT,
    final_weight: float = FINAL_WEIGHT,
    initial_freq_hz: float | None = None,
    best_freq_hz: float | None = None,
    final_freq_hz: float | None = None,
    target_freq_hz: float | None = None,
    freq_weight: float = FREQ_WEIGHT,
) -> float:
    depth = best_weight * clip_reward(initial_db - best_db, clip) + final_weight * clip_reward(
        initial_db - final_db, clip
    )
    if target_freq_hz is None:
        return depth
    freq = best_weight * frequency_improvement_reward(
        initial_freq_hz, best_freq_hz, target_freq_hz, clip=clip
    ) + final_weight * frequency_improvement_reward(
        initial_freq_hz, final_freq_hz, target_freq_hz, clip=clip
    )
    return depth + freq_weight * freq
