"""Bridge multi-turn Qucs trajectories into TRL ``rollout_func`` batch dicts.

TRL's ``GRPOTrainer(rollout_func=...)`` expects parallel lists of
``prompt_ids``, ``completion_ids``, and ``logprobs``. This module flattens
already-collected ``Trajectory`` turns into that shape so a TRL trainer can
consume the same environment loop as ``training.grpo.rollout.run_trajectory``.

Placeholder logprobs (zeros) are fine when the trainer recomputes log-probs
from ``completion_ids`` under the current policy; callers that already have
token logprobs can pass ``token_logprobs`` instead.
"""
from __future__ import annotations

from typing import Sequence

from training.grpo.rollout import Trajectory


def _prompt_text(tokenizer, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def pack_rollout_from_trajectories(
    trajectories: Sequence[Trajectory],
    tokenizer,
    *,
    advantages: Sequence[Sequence[float]] | None = None,
    token_logprobs: Sequence[Sequence[Sequence[float]]] | None = None,
) -> dict[str, list]:
    """Flatten trajectory turns into a TRL rollout_func-compatible batch."""
    prompt_ids: list[list[int]] = []
    completion_ids: list[list[int]] = []
    logprobs: list[list[float]] = []
    trajectory_rewards: list[float] = []
    flat_advantages: list[float] = []

    for t_idx, traj in enumerate(trajectories):
        turn_advs = (
            list(advantages[t_idx]) if advantages is not None else [0.0] * len(traj.turns)
        )
        if len(turn_advs) != len(traj.turns):
            raise ValueError(
                f"advantages[{t_idx}] length {len(turn_advs)} != turns {len(traj.turns)}"
            )
        for turn_i, turn in enumerate(traj.turns):
            p_text = _prompt_text(tokenizer, turn.prompt)
            c_ids = list(tokenizer.encode(turn.completion_text, add_special_tokens=False))
            p_ids = list(tokenizer.encode(p_text, add_special_tokens=False))
            if token_logprobs is not None:
                lp = list(token_logprobs[t_idx][turn_i])
            else:
                lp = [0.0] * len(c_ids)
            if len(lp) != len(c_ids):
                raise ValueError("token_logprobs length must match completion token count")
            prompt_ids.append(p_ids)
            completion_ids.append(c_ids)
            logprobs.append(lp)
            trajectory_rewards.append(float(traj.reward))
            flat_advantages.append(float(turn_advs[turn_i]))

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "trajectory_rewards": trajectory_rewards,
        "advantages": flat_advantages,
    }
