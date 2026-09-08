"""Random one-step optimization tasks backed by the real Qucs simulator."""
from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cost import TARGET_NOTCH_HZ, evaluate, s21_db  # noqa: E402
from intent import BOUNDS, VARIABLES, apply_intent  # noqa: E402
from qucs_sim import simulate  # noqa: E402

TARGET_BAND_HZ = (4e9, 6e9)

SYSTEM_PROMPT = """You control one step of a butterfly radial-stub optimizer.
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


@dataclass(frozen=True)
class TransitionResult:
    params: dict[str, float]
    old_db: float
    new_db: float
    delta_db: float
    total_cost: float


def _cost_dict(cost: object) -> dict:
    if is_dataclass(cost):
        return asdict(cost)
    return dict(vars(cost))


def sample_params(seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    return {
        name: round(rng.uniform(*BOUNDS[name]), 4)
        for name in VARIABLES
    }


def build_prompt(
    params: dict[str, float],
    cost: dict,
    iteration: int,
) -> list[dict[str, str]]:
    param_lines = "\n".join(
        f"- {name}={params[name]:.4f}, bounds={BOUNDS[name]}"
        for name in VARIABLES
    )
    user = f"""Choose the intent for iteration {iteration + 1} from the measured state after iteration {iteration}.

Target: minimize |S21| at {cost.get('target_freq_hz', TARGET_NOTCH_HZ) / 1e9:.3f} GHz.
Current |S21| at target: {cost['total_cost']:.8f} ({s21_db(cost['total_cost']):.3f} dB).
Deepest notch: {cost['best_s21_mag']:.8f} ({s21_db(cost['best_s21_mag']):.3f} dB)
at {cost['best_freq_hz'] / 1e9:.3f} GHz.

Current parameters:
{param_lines}

Select an intent that is likely to reduce |S21| at the target on the next simulation."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def generate_task(
    seed: int,
    iteration: int = 0,
    simulator: Callable = simulate,
    evaluator: Callable = evaluate,
) -> dict:
    params = sample_params(seed)
    result = simulator(params, export_layout=False)
    cost = evaluator(result, TARGET_BAND_HZ, target_hz=TARGET_NOTCH_HZ)
    cost_data = _cost_dict(cost)
    return {
        "prompt": build_prompt(params, cost_data, iteration),
        "params_json": json.dumps(params, sort_keys=True),
        "baseline_cost_json": json.dumps(cost_data, sort_keys=True),
        "iteration": iteration,
        "seed": seed,
    }


def run_transition(
    params: dict[str, float],
    iteration: int,
    baseline_total_cost: float,
    intent: dict[str, str],
    simulator: Callable = simulate,
    evaluator: Callable = evaluate,
) -> TransitionResult:
    new_params = apply_intent(params, intent, iteration=iteration)
    result = simulator(new_params, export_layout=False)
    cost = evaluator(result, TARGET_BAND_HZ, target_hz=TARGET_NOTCH_HZ)
    old_db = s21_db(baseline_total_cost)
    new_db = s21_db(cost.total_cost)
    return TransitionResult(
        params=new_params,
        old_db=old_db,
        new_db=new_db,
        delta_db=old_db - new_db,
        total_cost=cost.total_cost,
    )

