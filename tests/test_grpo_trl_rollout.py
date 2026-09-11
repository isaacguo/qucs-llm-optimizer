"""Tests for packing multi-turn trajectories into TRL rollout_func batches."""
from __future__ import annotations

import unittest

from training.common.goals import GoalSpec
from training.grpo.rollout import Trajectory, TurnRecord
from training.grpo.trl_rollout import pack_rollout_from_trajectories


def _turn(text: str, turn_index: int = 0) -> TurnRecord:
    return TurnRecord(
        turn_index=turn_index,
        prompt=[{"role": "user", "content": f"turn-{turn_index}"}],
        completion_text=text,
        valid=True,
        intent={"ro": "decrease"},
        params_before={"ro": 1.0},
        params_after={"ro": 0.9},
        db_before=-10.0,
        db_after=-12.0,
        freq_before_hz=5e9,
        freq_after_hz=5e9,
    )


class PackRolloutTests(unittest.TestCase):
    def test_flattens_turns_into_trl_rollout_fields(self):
        goal = GoalSpec(
            target_freq_hz=5.5e9,
            band_hz=(5.4e9, 5.6e9),
            target_depth_db=-70.0,
        )
        traj = Trajectory(goal=goal, seed=1, initial_db=-10.0, best_db=-12.0, reward=1.5)
        traj.turns = [_turn("a", 0), _turn("b", 1)]

        class Tok:
            def encode(self, text, add_special_tokens=False):
                return [ord(c) % 17 + 1 for c in text] or [1]

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
                return "|".join(m["content"] for m in messages)

        batch = pack_rollout_from_trajectories([traj], Tok(), advantages=[[0.5, -0.2]])
        self.assertEqual(len(batch["prompt_ids"]), 2)
        self.assertEqual(len(batch["completion_ids"]), 2)
        self.assertEqual(len(batch["logprobs"]), 2)
        self.assertEqual(batch["trajectory_rewards"], [1.5, 1.5])
        self.assertEqual(batch["advantages"], [0.5, -0.2])
        self.assertTrue(all(len(lp) == len(cid) for lp, cid in zip(batch["logprobs"], batch["completion_ids"])))


if __name__ == "__main__":
    unittest.main()
