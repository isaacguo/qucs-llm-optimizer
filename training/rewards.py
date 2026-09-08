"""TRL-compatible rewards that execute one real Qucs optimization step."""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from training.contracts import IntentParseError, parse_intent_completion
from training.environment import run_transition


def completion_text(completion: object) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict) and isinstance(last.get("content"), str):
            return last["content"]
    raise TypeError(f"unsupported completion shape: {type(completion).__name__}")


def format_reward(completions: list, **_kwargs) -> list[float]:
    exact = re.compile(
        r"^\s*(?:<think>\s*</think>\s*)?"
        r"<reasoning>.*?</reasoning>\s*"
        r"<intent>\s*\{.*\}\s*</intent>\s*$",
        re.S,
    )
    rewards = []
    for completion in completions:
        try:
            text = completion_text(completion)
            parse_intent_completion(text)
        except (IntentParseError, TypeError):
            rewards.append(0.0)
            continue
        rewards.append(1.0 if exact.match(text) else 0.5)
    return rewards


def valid_intent_reward(completions: list, **_kwargs) -> list[float]:
    rewards = []
    for completion in completions:
        try:
            parse_intent_completion(completion_text(completion))
        except (IntentParseError, TypeError):
            rewards.append(0.0)
        else:
            rewards.append(1.0)
    return rewards


def _column_value(column: object, index: int) -> object:
    if isinstance(column, (list, tuple)):
        return column[index]
    return column


def make_simulation_reward(
    *,
    transition: Callable = run_transition,
    max_workers: int = 2,
    log_path: Path | None = None,
) -> Callable:
    """Create a reward callable; dependency injection keeps unit tests fast."""

    def score_one(index: int, completion: object, kwargs: dict) -> tuple[float, dict]:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seed": _column_value(kwargs.get("seed"), index),
            "iteration": _column_value(kwargs["iteration"], index),
            "completion": completion_text(completion),
        }
        try:
            parsed = parse_intent_completion(record["completion"])
            params = json.loads(_column_value(kwargs["params_json"], index))
            baseline = json.loads(
                _column_value(kwargs["baseline_cost_json"], index)
            )
            result = transition(
                params=params,
                iteration=int(record["iteration"]) + 1,
                baseline_total_cost=float(baseline["total_cost"]),
                intent=parsed.intent,
            )
            score = max(-20.0, min(20.0, float(result.delta_db)))
            record.update(
                {
                    "valid": True,
                    "intent": parsed.intent,
                    "reasoning": parsed.reasoning,
                    "old_db": result.old_db,
                    "new_db": result.new_db,
                    "delta_db": result.delta_db,
                    "reward": score,
                    "new_params": result.params,
                }
            )
            return score, record
        except (IntentParseError, TypeError, ValueError, json.JSONDecodeError) as exc:
            record.update(
                {
                    "valid": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "reward": -5.0,
                }
            )
            return -5.0, record
        except Exception as exc:
            record.update(
                {
                    "valid": False,
                    "environment_error": f"{type(exc).__name__}: {exc}",
                    "reward": -20.0,
                }
            )
            return -20.0, record

    def simulation_reward(completions: list, **kwargs) -> list[float]:
        worker_count = max(1, min(max_workers, len(completions)))
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            results = list(
                pool.map(
                    lambda item: score_one(item[0], item[1], kwargs),
                    enumerate(completions),
                )
            )
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a") as stream:
                for _, record in results:
                    stream.write(json.dumps(record, sort_keys=True) + "\n")
        return [score for score, _ in results]

    simulation_reward.__name__ = "real_qucs_delta_db_reward"
    return simulation_reward

