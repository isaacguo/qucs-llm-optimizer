"""Shared multi-turn prompt construction for SFT, GRPO, and corpus."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cost import s21_db  # noqa: E402
from intent import BOUNDS, VARIABLES  # noqa: E402

from training.common.goals import GoalSpec  # noqa: E402

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
Do not emit a stop action unless the user message explicitly allows it.

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
    freq_before_hz: float | None = None
    freq_after_hz: float | None = None
    stopped: bool = False

    @property
    def delta_db(self) -> float:
        return self.db_before - self.db_after


def stop_is_allowed(
    history: list[TurnRecord],
    current_db: float,
    goal: GoalSpec,
    best_db: float,
    initial_db: float,
    *,
    improve_eps: float = 0.2,
    worsen_eps: float = 0.05,
) -> bool:
    """Stop is only a legal lock-in after a real gain, or if the goal is met.

    Advertising stop on turn 1 made an untrained Qwen3 copy the shortest
    valid JSON and abort every rollout with reward 0.
    """
    if current_db <= goal.target_depth_db:
        return True
    if not history:
        return False
    found_gain = best_db < initial_db - improve_eps
    last = history[-1]
    last_worsened = last.db_after > last.db_before + worsen_eps
    return found_gain and last_worsened


def _last_move_hint(history: list[TurnRecord], *, allow_stop: bool) -> str:
    if not history:
        return "This is the first turn of this run; no history yet. You must pick a tuning intent."
    last = history[-1]
    if last.db_after < last.db_before - 0.05:
        return "The last move improved |S21| at the target. Keep tuning."
    if last.db_after > last.db_before + 0.05:
        if allow_stop:
            return (
                "The last move made |S21| worse. Reverse that variable, try another, "
                'or emit {"action":"stop"} to keep the best notch of this run.'
            )
        return "The last move made |S21| worse. Reverse that variable or try another."
    return "The last move was flat. Try a different variable."


def build_multiturn_prompt(
    goal: GoalSpec,
    turn_index: int,
    params: dict[str, float],
    cost: dict,
    history: list[TurnRecord],
    history_window: int = 8,
    *,
    initial_db: float | None = None,
    best_db: float | None = None,
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

    current_db = s21_db(cost["total_cost"])
    init = initial_db if initial_db is not None else current_db
    best = best_db if best_db is not None else min(init, current_db)
    db_to_goal = current_db - goal.target_depth_db
    allow_stop = stop_is_allowed(history, current_db, goal, best, init)
    hint = _last_move_hint(history, allow_stop=allow_stop)
    if allow_stop:
        closer = (
            'You may lock in the best notch with <intent>{"action":"stop"}</intent> '
            "(all-hold is also stop). Otherwise pick a tuning intent."
        )
    else:
        closer = (
            "Stop is not allowed on this turn. Select a tuning intent that is "
            "likely to reduce |S21| at the target on the next simulation."
        )

    user = f"""Turn {turn_index + 1} of this optimization run.

Target: minimize |S21| at {goal.describe()}.
Current |S21| at target: {cost['total_cost']:.8f} ({current_db:.3f} dB).
Best |S21| this run: {best:.3f} dB (start was {init:.3f} dB).
dB still needed to reach the goal: {db_to_goal:.3f} (negative means the goal is already met).
Deepest notch anywhere in the sweep: {cost['best_s21_mag']:.8f} ({s21_db(cost['best_s21_mag']):.3f} dB)
at {cost['best_freq_hz'] / 1e9:.3f} GHz.
{history_block}
{hint}

Current parameters:
{param_lines}

Select an intent that is likely to reduce |S21| at the target.
{closer}"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
