"""Dataset builders for real-Qucs GRPO and optional trajectory SFT."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable

from training.environment import SYSTEM_PROMPT, generate_task


def build_grpo_records(
    *,
    count: int,
    start_seed: int,
    max_iteration: int,
    task_factory: Callable = generate_task,
) -> list[dict]:
    records = []
    for seed in range(start_seed, start_seed + count):
        iteration = random.Random(seed ^ 0x51A7).randint(0, max_iteration)
        records.append(task_factory(seed, iteration))
    return records


def build_grpo_dataset(
    *,
    count: int,
    start_seed: int = 1000,
    max_iteration: int = 12,
):
    from datasets import Dataset

    return Dataset.from_list(
        build_grpo_records(
            count=count,
            start_seed=start_seed,
            max_iteration=max_iteration,
        )
    )


def load_sft_records(state_path: Path) -> list[dict]:
    state = json.loads(state_path.read_text())
    records = []
    for entry in state.get("history", []):
        observation = entry.get("observation") or {}
        intent = entry.get("intent")
        if not observation.get("report_text") or not intent:
            continue
        reasoning = (entry.get("thinking") or entry.get("note") or "").strip()
        completion = (
            f"<reasoning>{reasoning}</reasoning>\n"
            f"<intent>{json.dumps(intent, sort_keys=True)}</intent>"
        )
        records.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": observation["report_text"]},
                    {"role": "assistant", "content": completion},
                ]
            }
        )
    return records


def build_sft_dataset(state_path: Path):
    from datasets import Dataset

    return Dataset.from_list(load_sft_records(state_path))

