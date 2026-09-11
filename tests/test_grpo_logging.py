"""Tests for multiturn step_stats aggregation helpers."""
from __future__ import annotations

import unittest

from training.grpo.diagnostics import build_step_stats
from training.common.goals import GoalSpec
from training.grpo.rollout import Trajectory, TurnRecord


def _traj(reward, reason, deltas, seed=1):
    turns = []
    db = -10.0
    for i, delta in enumerate(deltas):
        after = db - delta
        turns.append(
            TurnRecord(
                turn_index=i,
                prompt=[{"role": "user", "content": "p"}],
                completion_text="<reasoning>r</reasoning><intent>{\"ro\":\"decrease\"}</intent>",
                valid=True,
                intent={"ro": "decrease"},
                params_before={},
                params_after={},
                db_before=db,
                db_after=after,
            )
        )
        db = after
    traj = Trajectory(
        goal=GoalSpec(5e9, (4e9, 6e9), -30.0),
        seed=seed,
        turns=turns,
        initial_db=-10.0,
        best_db=min(-10.0, min(t.db_after for t in turns)),
    )
    traj.reward = reward
    traj.terminated_reason = reason
    return traj


class StepStatsTests(unittest.TestCase):
    def test_build_step_stats_has_required_keys(self):
        group = [
            _traj(4.0, "patience", [1.0, -0.5], seed=1),
            _traj(1.0, "patience", [0.2], seed=1),
            _traj(8.0, "goal_met", [5.0], seed=1),
            _traj(-1.0, "stop", [0.0], seed=1),
        ]
        # second group
        group2 = [
            _traj(2.0, "patience", [0.5], seed=2),
            _traj(2.1, "patience", [0.4], seed=2),
            _traj(2.2, "max_turns", [0.3, 0.1], seed=2),
            _traj(2.3, "patience", [0.2], seed=2),
        ]
        stats = build_step_stats(
            [group, group2],
            turn_advantages={id(t): [0.1] * len(t.turns) for g in (group, group2) for t in g},
            n_resampled_starts=3,
        )
        for key in (
            "reward_mean",
            "reward_p50",
            "reward_p25",
            "reward_p75",
            "wg_std_mean",
            "wg_range_mean",
            "group_best_mean",
            "valid_turn_rate",
            "pos_delta_rate",
            "best_improve_mean",
            "best_improve_p50",
            "goal_met_pct",
            "stop_pct",
            "patience_pct",
            "max_turns_pct",
            "zero_adv_frac",
            "n_resampled_starts",
        ):
            self.assertIn(key, stats)
        self.assertEqual(stats["n_resampled_starts"], 3)
        self.assertGreater(stats["wg_range_mean"], 0.0)


class LoraSnapshotTests(unittest.TestCase):
    def test_snapshot_restores_current_lora_after_context(self):
        import torch
        from torch import nn

        from training.grpo.train import _snapshot_lora, _use_lora_snapshot

        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.lora_A = nn.Parameter(torch.tensor([1.0, 2.0]))

        model = Toy()
        snap = _snapshot_lora(model)
        model.lora_A.data.add_(10.0)
        with _use_lora_snapshot(model, snap):
            self.assertEqual(model.lora_A.tolist(), [1.0, 2.0])
        self.assertEqual(model.lora_A.tolist(), [11.0, 12.0])


if __name__ == "__main__":
    unittest.main()
