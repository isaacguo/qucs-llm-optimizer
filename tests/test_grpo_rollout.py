"""Tests for multi-turn rollout stop, reward, patience, and GRPO grouping helpers."""
from __future__ import annotations

import unittest

from training.common.environment import sample_params
from training.common.goals import DEFAULT_TARGET_DEPTH_DB, GoalSpec, sample_goal
from training.grpo.reward_math import finalize_trajectory_reward
from training.grpo.rollout import (
    REWARD_CLIP,
    mixed_terminal_reward,
    run_trajectory,
    shaped_turn_advantages,
    trajectory_to_json,
)


def _mixed_from_traj(traj) -> float:
    return mixed_terminal_reward(
        traj.initial_db,
        traj.best_db,
        traj.turns[-1].db_after,
        initial_freq_hz=traj.initial_freq_hz,
        best_freq_hz=traj.best_freq_hz,
        final_freq_hz=traj.final_freq_hz,
        target_freq_hz=traj.goal.target_freq_hz,
        target_depth_db=traj.goal.target_depth_db,
    )


def _mag(db: float) -> float:
    return 10 ** (db / 20.0)


def _cost_from_db(db: float, freq_hz: float) -> dict:
    mag = _mag(db)
    return {
        "total_cost": mag,
        "best_s21_mag": mag,
        "best_freq_hz": freq_hz,
    }


class GoalAndParamSamplingTests(unittest.TestCase):
    def test_sample_goal_accepts_training_depth(self):
        goal = sample_goal(7, target_depth_db=-30.0)
        self.assertEqual(goal.target_depth_db, -30.0)
        self.assertIn("-30 dB", goal.describe())

    def test_heldout_default_depth_stays_minus_seventy(self):
        self.assertEqual(DEFAULT_TARGET_DEPTH_DB, -70.0)
        goal = sample_goal(3)
        self.assertEqual(goal.target_depth_db, -70.0)

    def test_narrow_spread_stays_near_initial_guess(self):
        from intent import BOUNDS, INITIAL_GUESS, VARIABLES

        params = sample_params(42, spread=0.35)
        for name in VARIABLES:
            lo, hi = BOUNDS[name]
            half = 0.5 * 0.35 * (hi - lo)
            self.assertGreaterEqual(params[name], max(lo, INITIAL_GUESS[name] - half) - 1e-6)
            self.assertLessEqual(params[name], min(hi, INITIAL_GUESS[name] + half) + 1e-6)


class RewardMathTests(unittest.TestCase):
    def test_default_reward_clip_is_twenty(self):
        self.assertEqual(REWARD_CLIP, 20.0)

    def test_mixed_reward_prefers_best_over_collapsed_final(self):
        # init -10, best -60, final -2: old terminal reward is negative; mixed stays positive
        mixed = mixed_terminal_reward(-10.0, -60.0, -2.0, clip=40.0)
        terminal_only = max(-40.0, min(40.0, -10.0 - (-2.0)))
        self.assertGreater(mixed, 0.0)
        self.assertLess(terminal_only, 0.0)
        self.assertAlmostEqual(mixed, 0.7 * 40.0 + 0.3 * (-8.0))

    def test_default_clip_caps_at_twenty(self):
        mixed = mixed_terminal_reward(-10.0, -60.0, -60.0)
        self.assertAlmostEqual(mixed, 20.0)

    def test_mixed_reward_higher_when_notch_moves_onto_target(self):
        kwargs = dict(
            initial_db=-10.0,
            best_db=-20.0,
            final_db=-20.0,
            initial_freq_hz=3.3e9,
            target_freq_hz=5.5e9,
        )
        off = mixed_terminal_reward(
            **kwargs, best_freq_hz=3.3e9, final_freq_hz=3.3e9
        )
        on = mixed_terminal_reward(
            **kwargs, best_freq_hz=5.5e9, final_freq_hz=5.5e9
        )
        self.assertGreater(on, off)

    def test_spec_term_separates_clipped_near_miss_from_goal_depth(self):
        near_miss = mixed_terminal_reward(
            -7.3, -30.0, -25.0, clip=20.0, target_depth_db=-50.0
        )
        hit_spec = mixed_terminal_reward(
            -7.3, -56.0, -56.0, clip=20.0, target_depth_db=-50.0
        )
        self.assertGreater(hit_spec - near_miss, 8.0)

    def test_finalize_scales_non_success_and_bonuses_goal_met(self):
        mixed = 20.0
        self.assertAlmostEqual(finalize_trajectory_reward(mixed, "goal_met"), 25.0)
        self.assertAlmostEqual(finalize_trajectory_reward(mixed, "patience"), 2.0)
        self.assertAlmostEqual(finalize_trajectory_reward(mixed, "max_turns"), 2.0)
        self.assertAlmostEqual(finalize_trajectory_reward(mixed, "stop"), 2.0)


class RolloutBehaviourTests(unittest.TestCase):
    def setUp(self):
        self.goal = GoalSpec(
            target_freq_hz=5.0e9,
            band_hz=(4.0e9, 6.0e9),
            target_depth_db=-30.0,
        )
        self.start = {"ri": 0.3, "ro": 8.0, "alpha": 90.0, "Wf": 0.6, "Lc": 3.0}

    def _run(self, generate_fn, **kwargs):
        db_by_ro = kwargs.pop("db_by_ro", {8.0: -10.0, 7.0: -32.0})

        def cost_fn(params):
            closest = min(db_by_ro, key=lambda k: abs(k - params["ro"]))
            return _cost_from_db(db_by_ro[closest], self.goal.target_freq_hz)

        return run_trajectory(
            generate_fn,
            goal=self.goal,
            seed=0,
            initial_params=dict(self.start),
            cost_fn=cost_fn,
            **kwargs,
        )

    def test_goal_met_at_training_depth_stops_early(self):
        def generate(_messages):
            return (
                "<reasoning>Shrink outer radius to move the notch.</reasoning>"
                '<intent>{"ro":"decrease_strong"}</intent>'
            )

        traj = self._run(generate, max_turns=8, patience=10)
        self.assertEqual(traj.terminated_reason, "goal_met")
        self.assertEqual(traj.num_turns, 1)
        self.assertLess(traj.turns[0].db_after, -30.0)

    def test_goal_met_reward_beats_clipped_near_miss(self):
        self.goal = GoalSpec(
            target_freq_hz=5.0e9,
            band_hz=(4.0e9, 6.0e9),
            target_depth_db=-50.0,
        )

        def generate(_messages):
            return '<intent>{"ro":"decrease_strong"}</intent>'

        miss = self._run(
            generate, max_turns=4, patience=10, db_by_ro={8.0: -7.3, 7.0: -30.0}
        )
        hit = self._run(
            generate, max_turns=4, patience=10, db_by_ro={8.0: -7.3, 7.0: -56.0}
        )
        self.assertEqual(hit.terminated_reason, "goal_met")
        self.assertNotEqual(miss.terminated_reason, "goal_met")
        self.assertGreater(hit.reward - miss.reward, 8.0)
        self.assertLess(abs(miss.reward), 5.0)
        self.assertGreater(hit.reward, 20.0)

    def test_first_turn_stop_is_ignored_when_goal_not_met(self):
        n = {"i": 0}

        def generate(messages):
            n["i"] += 1
            joined = "\n".join(m["content"] for m in messages)
            self.assertNotIn('{"action":"stop"}', joined)
            return '<intent>{"action":"stop"}</intent>'

        traj = self._run(generate, max_turns=3, patience=10, db_by_ro={8.0: -10.0})
        self.assertNotEqual(traj.terminated_reason, "stop")
        self.assertGreaterEqual(traj.num_turns, 2)
        self.assertFalse(traj.turns[0].stopped)
        self.assertFalse(traj.turns[0].valid)

    def test_stop_after_improve_then_worsen_is_allowed(self):
        completions = [
            '<intent>{"ro":"decrease_strong"}</intent>',
            '<intent>{"ro":"increase_strong"}</intent>',
            '<intent>{"action":"stop"}</intent>',
        ]

        def generate(messages):
            joined = "\n".join(m["content"] for m in messages)
            if len(completions) == 1:
                self.assertIn('{"action":"stop"}', joined)
            return completions.pop(0)

        traj = self._run(
            generate,
            max_turns=8,
            patience=10,
            db_by_ro={8.0: -10.0, 6.0: -20.0, 5.0: -22.0, 9.0: -8.0},
        )
        self.assertEqual(traj.terminated_reason, "stop")
        self.assertTrue(traj.turns[-1].stopped)
        self.assertGreater(traj.num_turns, 1)

    def test_legal_stop_with_improvement_gets_bonus(self):
        completions = [
            '<intent>{"ro":"decrease_strong"}</intent>',
            '<intent>{"ro":"increase_strong"}</intent>',
            '<intent>{"action":"stop"}</intent>',
        ]

        def generate(_messages):
            return completions.pop(0)

        traj = self._run(
            generate,
            max_turns=8,
            patience=10,
            db_by_ro={8.0: -10.0, 6.0: -20.0, 5.0: -22.0, 9.0: -8.0},
        )
        self.assertEqual(traj.terminated_reason, "stop")
        base = _mixed_from_traj(traj)
        self.assertAlmostEqual(traj.reward, 0.1 * (base + 1.0), places=3)

    def test_stop_without_real_gain_gets_no_bonus(self):
        def generate(_messages):
            return '<intent>{"action":"stop"}</intent>'

        traj = self._run(generate, max_turns=4, patience=10, db_by_ro={8.0: -32.0})
        self.assertEqual(traj.terminated_reason, "stop")
        base = _mixed_from_traj(traj)
        self.assertAlmostEqual(traj.reward, 0.1 * base, places=3)

    def test_first_turn_stop_ok_if_already_at_goal(self):
        def generate(_messages):
            return '<intent>{"action":"stop"}</intent>'

        traj = self._run(generate, max_turns=4, patience=10, db_by_ro={8.0: -32.0})
        self.assertEqual(traj.terminated_reason, "stop")
        self.assertEqual(traj.num_turns, 1)

    def test_patience_stops_after_stale_turns(self):
        def generate(_messages):
            return '<intent>{"Lc":"increase_slight"}</intent>'

        traj = self._run(
            generate,
            max_turns=8,
            patience=3,
            patience_eps=0.2,
            db_by_ro={8.0: -10.0},  # Lc tweaks do not change cost
        )
        self.assertEqual(traj.terminated_reason, "patience")
        self.assertEqual(traj.num_turns, 3)
        self.assertLess(abs(traj.reward), 5.0)

    def test_max_turns_near_miss_reward_stays_near_zero(self):
        def generate(_messages):
            return '<intent>{"Lc":"increase_slight"}</intent>'

        traj = self._run(
            generate,
            max_turns=3,
            patience=0,
            db_by_ro={8.0: -10.0},
        )
        self.assertEqual(traj.terminated_reason, "max_turns")
        self.assertLess(abs(traj.reward), 5.0)

    def test_injected_params_ignore_seed_sampling(self):
        seen = []

        def generate(_messages):
            seen.append(1)
            return '<intent>{"ro":"decrease_slight"}</intent>'

        a = self._run(generate, max_turns=2, patience=10)
        b = self._run(generate, max_turns=2, patience=10)
        self.assertEqual(a.turns[0].params_before, b.turns[0].params_before)
        self.assertEqual(a.turns[0].params_before["ro"], 8.0)

    def test_prompt_mentions_best_so_far(self):
        def generate(messages):
            joined = "\n".join(m["content"] for m in messages)
            self.assertIn("Best |S21| this run", joined)
            return '<intent>{"ro":"decrease_slight"}</intent>'

        traj = self._run(generate, max_turns=1, patience=10, db_by_ro={8.0: -10.0})
        self.assertGreaterEqual(traj.num_turns, 1)

    def test_reward_higher_when_notch_frequency_moves_toward_target(self):
        def generate(_messages):
            return '<intent>{"ro":"decrease_strong"}</intent>'

        def run_with_freq(end_freq_hz: float):
            def cost_fn(params):
                at_start = abs(params["ro"] - 8.0) < 1e-6
                db = -10.0 if at_start else -20.0
                freq = 3.3e9 if at_start else end_freq_hz
                return _cost_from_db(db, freq)

            return run_trajectory(
                generate,
                goal=self.goal,
                seed=0,
                initial_params=dict(self.start),
                cost_fn=cost_fn,
                max_turns=2,
                patience=10,
            )

        closer = run_with_freq(self.goal.target_freq_hz)
        farther = run_with_freq(3.3e9)
        self.assertAlmostEqual(closer.initial_db, farther.initial_db)
        self.assertAlmostEqual(closer.best_db, farther.best_db)
        self.assertGreater(closer.reward, farther.reward)

    def test_json_includes_best_and_reason(self):
        def generate(_messages):
            return '<intent>{"ro":"decrease_slight"}</intent>'

        payload = trajectory_to_json(
            self._run(generate, max_turns=1, patience=10, db_by_ro={8.0: -10.0})
        )
        self.assertIn("best_db", payload)
        self.assertIn("initial_db", payload)
        self.assertIn("terminated_reason", payload)


class ShapedAdvantageTests(unittest.TestCase):
    def test_destroying_turn_gets_lower_advantage_than_improving_turn(self):
        from training.grpo.rollout import Trajectory, TurnRecord

        def fake_traj(deltas, reward):
            turns = []
            db = -10.0
            for i, delta in enumerate(deltas):
                after = db - delta
                turns.append(
                    TurnRecord(
                        turn_index=i,
                        prompt=[],
                        completion_text="x",
                        valid=True,
                        intent={"ro": "decrease"},
                        params_before={},
                        params_after={},
                        db_before=db,
                        db_after=after,
                    )
                )
                db = after
            traj = Trajectory(goal=GoalSpec(5e9, (4e9, 6e9), -30.0), seed=1, turns=turns)
            traj.reward = reward
            traj.terminated_reason = "max_turns"
            return traj

        winner = fake_traj([5.0, -8.0], reward=4.0)
        loser = fake_traj([-1.0, -1.0], reward=-2.0)
        adv = shaped_turn_advantages([winner, loser], gamma=1.0)
        win_turns = adv[id(winner)]
        self.assertGreater(win_turns[0], win_turns[1])


if __name__ == "__main__":
    unittest.main()
