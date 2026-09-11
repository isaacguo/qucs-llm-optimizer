"""Multi-turn on-policy rollout collection for the butterfly-stub optimizer.

Unlike ``training/common/environment.py`` (one isolated random state per training
example), this module runs a *sequential* trajectory: the model sees its own
past decisions and their real simulated outcomes, decides the next intent,
and that intent is applied to the *same* running parameter set.

A trajectory ends on the first of: goal met, an explicit stop / all-hold
intent, ``patience`` consecutive turns without a new best dB, or
``max_turns``.

Reward mixes best-so-far |S21| improvement at the target frequency with
the final state, plus a term that scores whether the deepest-notch
frequency moved toward the goal frequency.
"""
from __future__ import annotations

import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cost import evaluate, s21_db  # noqa: E402
from intent import apply_intent  # noqa: E402
from qucs_sim import simulate  # noqa: E402

from training.common.contracts import IntentParseError, parse_intent_completion  # noqa: E402
from training.common.environment import sample_params  # noqa: E402
from training.common.goals import GoalSpec, is_goal_met as _is_goal_met  # noqa: E402
from training.common.prompts import (  # noqa: E402
    TurnRecord,
    build_multiturn_prompt,
    stop_is_allowed,
)
from training.grpo.reward_math import (  # noqa: E402
    FREQ_WEIGHT,
    REWARD_CLIP,
    finalize_trajectory_reward,
    frequency_alignment,
    mixed_terminal_reward,
)

STOP_BONUS = 1.0
STOP_BONUS_IMPROVE_EPS = 0.2


@dataclass
class Trajectory:
    goal: GoalSpec
    seed: int
    turns: list[TurnRecord] = field(default_factory=list)
    terminated_reason: str = ""
    reward: float = 0.0
    initial_db: float = 0.0
    best_db: float = 0.0
    initial_freq_hz: float | None = None
    best_freq_hz: float | None = None
    final_freq_hz: float | None = None

    @property
    def num_turns(self) -> int:
        return len(self.turns)


def _cost_dict(res, goal: GoalSpec) -> dict:
    from dataclasses import asdict, is_dataclass

    cost = evaluate(res, goal.band_hz, target_hz=goal.target_freq_hz)
    return asdict(cost) if is_dataclass(cost) else dict(vars(cost))


def run_trajectory(
    generate_fn: Callable[[list[dict[str, str]]], str],
    goal: GoalSpec,
    seed: int,
    max_turns: int = 8,
    history_window: int = 8,
    simulator: Callable = simulate,
    on_turn: Callable[[TurnRecord], None] | None = None,
    *,
    initial_params: dict[str, float] | None = None,
    cost_fn: Callable[[dict[str, float]], dict] | None = None,
    patience: int = 3,
    patience_eps: float = 0.2,
    param_spread: float = 1.0,
    reward_clip: float = REWARD_CLIP,
) -> Trajectory:
    """Run one full sequential rollout with the current policy.

    ``generate_fn`` takes chat-formatted messages and returns the raw
    completion text for the current model weights. Injected so the caller
    controls sampling params/tokenization and this module stays a pure
    environment-and-bookkeeping layer.

    Pass the same ``initial_params`` to every generation in a GRPO group so
    the group compares sampling noise, not different random circuits.
    """

    def get_cost(params: dict[str, float]) -> dict:
        if cost_fn is not None:
            return cost_fn(params)
        result = simulator(params, export_layout=False)
        return _cost_dict(result, goal)

    params = dict(initial_params) if initial_params is not None else sample_params(seed, spread=param_spread)
    cost = get_cost(params)
    initial_db = s21_db(cost["total_cost"])
    initial_freq = cost.get("best_freq_hz")
    best_db = initial_db
    current_freq = initial_freq

    traj = Trajectory(
        goal=goal,
        seed=seed,
        initial_db=initial_db,
        best_db=best_db,
        initial_freq_hz=initial_freq,
        best_freq_hz=initial_freq,
        final_freq_hz=initial_freq,
    )
    current_db = initial_db
    stale = 0

    for turn_index in range(max_turns):
        prompt = build_multiturn_prompt(
            goal,
            turn_index,
            params,
            cost,
            traj.turns,
            history_window=history_window,
            initial_db=initial_db,
            best_db=best_db,
        )
        completion_text = generate_fn(prompt)

        try:
            parsed = parse_intent_completion(completion_text, allow_stop=True)
        except (IntentParseError, TypeError):
            turn = TurnRecord(
                turn_index=turn_index,
                prompt=prompt,
                completion_text=completion_text,
                valid=False,
                intent=None,
                params_before=dict(params),
                params_after=dict(params),
                db_before=current_db,
                db_after=current_db,
                freq_before_hz=current_freq,
                freq_after_hz=current_freq,
            )
            traj.turns.append(turn)
            if on_turn is not None:
                on_turn(turn)
            stale += 1
            if patience > 0 and stale >= patience:
                traj.terminated_reason = "patience"
                break
            continue

        if parsed.stop:
            if not stop_is_allowed(
                traj.turns, current_db, goal, best_db, initial_db
            ):
                turn = TurnRecord(
                    turn_index=turn_index,
                    prompt=prompt,
                    completion_text=completion_text,
                    valid=False,
                    intent={"action": "stop"},
                    params_before=dict(params),
                    params_after=dict(params),
                    db_before=current_db,
                    db_after=current_db,
                    freq_before_hz=current_freq,
                    freq_after_hz=current_freq,
                    stopped=False,
                )
                traj.turns.append(turn)
                if on_turn is not None:
                    on_turn(turn)
                stale += 1
                if patience > 0 and stale >= patience:
                    traj.terminated_reason = "patience"
                    break
                continue
            turn = TurnRecord(
                turn_index=turn_index,
                prompt=prompt,
                completion_text=completion_text,
                valid=True,
                intent={"action": "stop"},
                params_before=dict(params),
                params_after=dict(params),
                db_before=current_db,
                db_after=current_db,
                freq_before_hz=current_freq,
                freq_after_hz=current_freq,
                stopped=True,
            )
            traj.turns.append(turn)
            if on_turn is not None:
                on_turn(turn)
            traj.terminated_reason = "stop"
            break

        new_params = apply_intent(params, parsed.intent, iteration=turn_index + 1)
        new_cost = get_cost(new_params)
        new_db = s21_db(new_cost["total_cost"])
        new_freq = new_cost.get("best_freq_hz")

        turn = TurnRecord(
            turn_index=turn_index,
            prompt=prompt,
            completion_text=completion_text,
            valid=True,
            intent=parsed.intent,
            params_before=dict(params),
            params_after=dict(new_params),
            db_before=current_db,
            db_after=new_db,
            freq_before_hz=current_freq,
            freq_after_hz=new_freq,
        )
        traj.turns.append(turn)
        if on_turn is not None:
            on_turn(turn)

        params, cost, current_db, current_freq = new_params, new_cost, new_db, new_freq
        if new_db < best_db - patience_eps:
            best_db = new_db
            stale = 0
        else:
            stale += 1

        if _is_goal_met(cost, goal):
            traj.terminated_reason = "goal_met"
            break
        if patience > 0 and stale >= patience:
            traj.terminated_reason = "patience"
            break
    else:
        traj.terminated_reason = "max_turns"

    if not traj.terminated_reason:
        traj.terminated_reason = "max_turns"

    snapshots = [(initial_db, initial_freq)]
    snapshots.extend((t.db_after, t.freq_after_hz) for t in traj.turns)
    traj.best_db, traj.best_freq_hz = min(snapshots, key=lambda item: item[0])
    traj.final_freq_hz = current_freq
    reward = mixed_terminal_reward(
        initial_db,
        traj.best_db,
        current_db,
        clip=reward_clip,
        initial_freq_hz=initial_freq,
        best_freq_hz=traj.best_freq_hz,
        final_freq_hz=current_freq,
        target_freq_hz=goal.target_freq_hz,
        target_depth_db=goal.target_depth_db,
    )
    if (
        traj.terminated_reason == "stop"
        and traj.best_db < initial_db - STOP_BONUS_IMPROVE_EPS
    ):
        reward += STOP_BONUS
    traj.reward = finalize_trajectory_reward(reward, traj.terminated_reason)
    return traj


def shaped_turn_advantages(
    group: list[Trajectory],
    *,
    gamma: float = 0.95,
) -> dict[int, list[float]]:
    """Per-turn advantages for one GRPO group.

    Trajectory-level mixed rewards are group-normalized (GRPO). Each turn then
    adds a centered reward-to-go term so a destroying step inside a winning
    trajectory is not reinforced as strongly as the improving steps.
    """
    rewards = [t.reward for t in group]
    mean_r = statistics.fmean(rewards) if rewards else 0.0
    std_r = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    out: dict[int, list[float]] = {}
    for traj in group:
        traj_adv = (traj.reward - mean_r) / (std_r + 1e-4)
        target_hz = traj.goal.target_freq_hz
        deltas = []
        for turn in traj.turns:
            freq_delta = frequency_alignment(turn.freq_after_hz, target_hz) - frequency_alignment(
                turn.freq_before_hz, target_hz
            )
            deltas.append(turn.delta_db + FREQ_WEIGHT * REWARD_CLIP * freq_delta)
        returns: list[float] = [0.0] * len(deltas)
        running = 0.0
        for i in range(len(deltas) - 1, -1, -1):
            running = deltas[i] + gamma * running
            returns[i] = running
        if returns:
            mean_g = statistics.fmean(returns)
            std_g = statistics.pstdev(returns) if len(returns) > 1 else 0.0
            local = [(g - mean_g) / (std_g + 1e-4) for g in returns]
        else:
            local = []
        out[id(traj)] = [traj_adv + 0.5 * loc for loc in local]
    return out


def trajectory_to_json(traj: Trajectory) -> dict:
    return {
        "seed": traj.seed,
        "goal": json.loads(traj.goal.to_json()),
        "num_turns": traj.num_turns,
        "terminated_reason": traj.terminated_reason,
        "reward": traj.reward,
        "initial_db": traj.initial_db,
        "best_db": traj.best_db,
        "initial_freq_hz": traj.initial_freq_hz,
        "best_freq_hz": traj.best_freq_hz,
        "final_freq_hz": traj.final_freq_hz,
        "turns": [
            {
                "turn_index": t.turn_index,
                "valid": t.valid,
                "intent": t.intent,
                "db_before": t.db_before,
                "db_after": t.db_after,
                "freq_before_hz": t.freq_before_hz,
                "freq_after_hz": t.freq_after_hz,
                "stopped": t.stopped,
            }
            for t in traj.turns
        ],
    }
