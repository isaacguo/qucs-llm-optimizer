"""Multi-turn on-policy rollout collection for the butterfly-stub optimizer.

Unlike ``training/environment.py`` (one isolated random state per training
example), this module runs a *sequential* trajectory: the model sees its own
past decisions and their real simulated outcomes, decides the next intent,
and that intent is applied to the *same* running parameter set. A trajectory
ends when the goal is met or ``max_turns`` is reached (fixed rule, no
learned "stop" behaviour yet).

Reward is computed once per trajectory (terminal-only): total dB
improvement from the first simulated state to the last. All turns in the
same trajectory share that single scalar reward — this is standard
whole-episode REINFORCE/policy-gradient credit assignment, not a per-turn
signal.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cost import evaluate, s21_db  # noqa: E402
from intent import BOUNDS, VARIABLES, apply_intent  # noqa: E402
from qucs_sim import simulate  # noqa: E402

from training.contracts import IntentParseError, parse_intent_completion  # noqa: E402
from training.environment import sample_params  # noqa: E402
from training.goals import GoalSpec  # noqa: E402

SYSTEM_PROMPT = """You control a butterfly radial-stub optimizer across several turns.
This is one continuous optimization run: your own past decisions and their
real measured outcomes are shown below, and you keep tuning the SAME circuit
turn after turn. The notch target changes every run - read the Target line
before choosing an intent; do not assume a fixed frequency.
Use only qualitative intent; never output raw parameter values.
Return exactly:
<reasoning>At most three short evidence-based sentences.</reasoning>
<intent>{"variable":"token"}</intent>

Variables: ri, ro, alpha, Wf, Lc.
Tokens: increase_strong, increase, increase_slight, hold,
decrease_slight, decrease, decrease_strong.
Change one variable normally; at most two active variables are allowed.

GOOD: <intent>{"ro":"decrease_strong"}</intent>
GOOD: <intent>{"alpha":"increase_slight","Lc":"decrease_slight"}</intent>
BAD: <intent>decrease</intent>
BAD: <intent>{"ro":5.3}</intent>
/no_think"""


@dataclass
class TurnRecord:
    turn_index: int
    prompt: list[dict[str, str]]
    completion_text: str
    valid: bool
    intent: dict[str, str] | None
    params_before: dict[str, float]
    params_after: dict[str, float]
    db_before: float
    db_after: float


@dataclass
class Trajectory:
    goal: GoalSpec
    seed: int
    turns: list[TurnRecord] = field(default_factory=list)
    terminated_reason: str = ""
    reward: float = 0.0

    @property
    def num_turns(self) -> int:
        return len(self.turns)


def _cost_dict(res, goal: GoalSpec) -> dict:
    from dataclasses import asdict, is_dataclass

    cost = evaluate(res, goal.band_hz, target_hz=goal.target_freq_hz)
    return asdict(cost) if is_dataclass(cost) else dict(vars(cost))


def _is_goal_met(cost: dict, goal: GoalSpec) -> bool:
    threshold = 10 ** (goal.target_depth_db / 20.0)
    return cost["total_cost"] <= threshold


def build_multiturn_prompt(
    goal: GoalSpec,
    turn_index: int,
    params: dict[str, float],
    cost: dict,
    history: list[TurnRecord],
    history_window: int = 8,
) -> list[dict[str, str]]:
    param_lines = "\n".join(
        f"- {name}={params[name]:.4f}, bounds={BOUNDS[name]}" for name in VARIABLES
    )
    recent = history[-history_window:]
    if recent:
        history_lines = "\n".join(
            f"turn {t.turn_index + 1}: intent={t.intent} -> "
            f"|S21| {t.db_before:.2f}dB -> {t.db_after:.2f}dB "
            f"({'improved' if t.db_after < t.db_before else 'worse/flat'})"
            for t in recent
        )
        history_block = f"\nYour last {len(recent)} turns on this same circuit:\n{history_lines}\n"
    else:
        history_block = "\nThis is the first turn of this run; no history yet.\n"

    user = f"""Turn {turn_index + 1} of this optimization run.

Target: minimize |S21| at {goal.describe()}.
Current |S21| at target: {cost['total_cost']:.8f} ({s21_db(cost['total_cost']):.3f} dB).
Deepest notch: {cost['best_s21_mag']:.8f} ({s21_db(cost['best_s21_mag']):.3f} dB)
at {cost['best_freq_hz'] / 1e9:.3f} GHz.
{history_block}
Current parameters:
{param_lines}

Select an intent that is likely to reduce |S21| at the target on the next simulation."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def run_trajectory(
    generate_fn: Callable[[list[dict[str, str]]], str],
    goal: GoalSpec,
    seed: int,
    max_turns: int = 20,
    history_window: int = 8,
    simulator: Callable = simulate,
    on_turn: Callable[[TurnRecord], None] | None = None,
) -> Trajectory:
    """Run one full sequential rollout with the current policy.

    ``generate_fn`` takes chat-formatted messages and returns the raw
    completion text for the current model weights. Injected so the caller
    controls sampling params/tokenization and this module stays a pure
    environment-and-bookkeeping layer.
    """
    params = sample_params(seed)
    result = simulator(params, export_layout=False)
    cost = _cost_dict(result, goal)
    initial_db = s21_db(cost["total_cost"])

    traj = Trajectory(goal=goal, seed=seed)
    current_db = initial_db

    for turn_index in range(max_turns):
        prompt = build_multiturn_prompt(
            goal, turn_index, params, cost, traj.turns, history_window=history_window
        )
        completion_text = generate_fn(prompt)

        try:
            parsed = parse_intent_completion(completion_text)
        except (IntentParseError, TypeError):
            # Invalid completion: no parameter change this turn, but the turn
            # still counts (the model must learn that malformed output stalls
            # progress) and the trajectory continues.
            traj.turns.append(
                TurnRecord(
                    turn_index=turn_index,
                    prompt=prompt,
                    completion_text=completion_text,
                    valid=False,
                    intent=None,
                    params_before=dict(params),
                    params_after=dict(params),
                    db_before=current_db,
                    db_after=current_db,
                )
            )
            if on_turn is not None:
                on_turn(traj.turns[-1])
            continue

        new_params = apply_intent(params, parsed.intent, iteration=turn_index + 1)
        new_result = simulator(new_params, export_layout=False)
        new_cost = _cost_dict(new_result, goal)
        new_db = s21_db(new_cost["total_cost"])

        traj.turns.append(
            TurnRecord(
                turn_index=turn_index,
                prompt=prompt,
                completion_text=completion_text,
                valid=True,
                intent=parsed.intent,
                params_before=dict(params),
                params_after=dict(new_params),
                db_before=current_db,
                db_after=new_db,
            )
        )
        if on_turn is not None:
            on_turn(traj.turns[-1])

        params, cost, current_db = new_params, new_cost, new_db

        if _is_goal_met(cost, goal):
            traj.terminated_reason = "goal_met"
            break
    else:
        traj.terminated_reason = "max_turns"

    if not traj.terminated_reason:
        traj.terminated_reason = "max_turns"

    # Terminal-only reward: total dB improvement across the whole run.
    # Clipped to match the single-step reward's scale for stable advantages.
    traj.reward = max(-20.0, min(20.0, initial_db - current_db))
    return traj


def trajectory_to_json(traj: Trajectory) -> dict:
    return {
        "seed": traj.seed,
        "goal": json.loads(traj.goal.to_json()),
        "num_turns": traj.num_turns,
        "terminated_reason": traj.terminated_reason,
        "reward": traj.reward,
        "turns": [
            {
                "turn_index": t.turn_index,
                "valid": t.valid,
                "intent": t.intent,
                "db_before": t.db_before,
                "db_after": t.db_after,
            }
            for t in traj.turns
        ],
    }
