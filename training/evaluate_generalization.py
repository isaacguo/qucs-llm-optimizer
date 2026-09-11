"""Check whether a trained policy generalizes to notch targets it never saw.

Usage:
    uv run python -m training.evaluate_generalization \\
        --adapter outputs/2026-09-08-1608-grpo-qwen3-1.7b/final_lora \\
        --model-name unsloth/Qwen3-1.7B-bnb-4bit \\
        --samples-per-goal 3

For each frequency in training.goals.HELDOUT_FREQS_HZ (never sampled during
training - see training/goals.py), this draws a few random starting circuits,
builds the same prompt format used in training, generates a completion from
the policy, and reports whether the model produced a syntactically valid
intent and what real delta_db that intent achieved. A policy that only
memorized the pre-Stage-A single-frequency task should perform noticeably
worse here than on training-distribution goals; a policy that learned to
condition on the stated target should perform comparably.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.contracts import IntentParseError, parse_intent_completion
from training.environment import generate_task, run_transition
from training.goals import heldout_goals
from training.modeling import ModelConfig, load_policy


def _generate(model, tokenizer, prompt: list[dict], max_new_tokens: int = 256) -> str:
    text = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.0,
    )
    completion_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        default=None,
        help=(
            "path to a saved LoRA adapter; omit to evaluate the raw, untrained "
            "base model as a baseline"
        ),
    )
    parser.add_argument("--model-name", default="unsloth/Qwen3-4B-Instruct-2507-bnb-4bit")
    parser.add_argument("--samples-per-goal", type=int, default=3)
    parser.add_argument("--start-seed", type=int, default=9000)
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)

    model, tokenizer = load_policy(ModelConfig(model_name=args.model_name))
    if args.adapter:
        model.load_adapter(args.adapter, adapter_name="default")

    results = []
    seed = args.start_seed
    for goal in heldout_goals():
        for _ in range(args.samples_per_goal):
            task = generate_task(seed=seed, iteration=0, goal=goal)
            completion = _generate(model, tokenizer, task["prompt"])
            record = {
                "seed": seed,
                "target_freq_hz": goal.target_freq_hz,
                "completion": completion,
            }
            try:
                parsed = parse_intent_completion(completion)
                baseline = json.loads(task["baseline_cost_json"])
                params = json.loads(task["params_json"])
                transition = run_transition(
                    params=params,
                    iteration=1,
                    baseline_total_cost=float(baseline["total_cost"]),
                    intent=parsed.intent,
                    goal=goal,
                )
                record.update(
                    {
                        "valid": True,
                        "intent": parsed.intent,
                        "delta_db": transition.delta_db,
                    }
                )
            except (IntentParseError, TypeError, ValueError, json.JSONDecodeError) as exc:
                record.update({"valid": False, "error": f"{type(exc).__name__}: {exc}"})
            results.append(record)
            seed += 1

    valid = [r for r in results if r["valid"]]
    summary = {
        "adapter": args.adapter or "(none - raw base model baseline)",
        "goals_tested": len(heldout_goals()),
        "samples_per_goal": args.samples_per_goal,
        "total_samples": len(results),
        "valid_intent_rate": len(valid) / len(results) if results else 0.0,
        "mean_delta_db": (
            sum(r["delta_db"] for r in valid) / len(valid) if valid else None
        ),
    }
    print(json.dumps({"summary": summary, "results": results}, indent=2, sort_keys=True))

    if args.output:
        Path(args.output).write_text(
            json.dumps({"summary": summary, "results": results}, indent=2, sort_keys=True)
        )


if __name__ == "__main__":
    main()
