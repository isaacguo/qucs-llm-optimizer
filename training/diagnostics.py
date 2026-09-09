"""Aggregate per-step diagnostics for multiturn GRPO logging."""
from __future__ import annotations

import math
import statistics
from typing import Any

from training.rollout import Trajectory


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * pct / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[f]
    return xs[f] * (c - k) + xs[c] * (k - f)


def build_step_stats(
    step_goal_groups: list[list[Trajectory]],
    turn_advantages: dict[int, list[float]],
    n_resampled_starts: int = 0,
    *,
    pos_eps: float = 0.05,
    zero_adv_eps: float = 1e-6,
) -> dict[str, Any]:
    trajs = [t for group in step_goal_groups for t in group]
    rewards = [t.reward for t in trajs]
    reasons = [t.terminated_reason for t in trajs]
    n = max(1, len(trajs))

    wg_stds: list[float] = []
    wg_ranges: list[float] = []
    group_bests: list[float] = []
    for group in step_goal_groups:
        rs = [t.reward for t in group]
        if not rs:
            continue
        wg_stds.append(statistics.pstdev(rs) if len(rs) > 1 else 0.0)
        wg_ranges.append(max(rs) - min(rs))
        group_bests.append(max(rs))

    turns = [turn for t in trajs for turn in t.turns]
    n_turns = len(turns)
    valid_rate = sum(1 for turn in turns if turn.valid) / n_turns if n_turns else 0.0
    pos_rate = (
        sum(1 for turn in turns if turn.delta_db > pos_eps) / n_turns if n_turns else 0.0
    )
    best_improves = [t.initial_db - t.best_db for t in trajs]

    all_advs = [a for traj in trajs for a in turn_advantages.get(id(traj), [])]
    zero_adv_frac = (
        sum(1 for a in all_advs if abs(a) < zero_adv_eps) / len(all_advs)
        if all_advs
        else 0.0
    )

    def pct_reason(name: str) -> float:
        return 100.0 * sum(1 for r in reasons if r == name) / n

    return {
        "reward_mean": statistics.fmean(rewards) if rewards else 0.0,
        "reward_p25": _percentile(rewards, 25),
        "reward_p50": _percentile(rewards, 50),
        "reward_p75": _percentile(rewards, 75),
        "wg_std_mean": statistics.fmean(wg_stds) if wg_stds else 0.0,
        "wg_range_mean": statistics.fmean(wg_ranges) if wg_ranges else 0.0,
        "group_best_mean": statistics.fmean(group_bests) if group_bests else 0.0,
        "valid_turn_rate": valid_rate,
        "pos_delta_rate": pos_rate,
        "best_improve_mean": statistics.fmean(best_improves) if best_improves else 0.0,
        "best_improve_p50": _percentile(best_improves, 50),
        "goal_met_pct": pct_reason("goal_met"),
        "stop_pct": pct_reason("stop"),
        "patience_pct": pct_reason("patience"),
        "max_turns_pct": pct_reason("max_turns"),
        "zero_adv_frac": zero_adv_frac,
        "n_resampled_starts": n_resampled_starts,
        "n_trajectories": len(trajs),
        "n_turns": n_turns,
    }
