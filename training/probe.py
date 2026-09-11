"""Probe the untrained policy with real Qucs rewards, without updating weights."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from training.environment import generate_task
from training.modeling import ModelConfig, load_policy
from training.preflight import verify_runtime
from training.rewards import (
    completion_text,
    format_reward,
    make_simulation_reward,
    valid_intent_reward,
)

DEFAULT_MODEL = "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--tasks", type=int, default=2)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--start-seed", type=int, default=2000)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-dir", default="outputs/probe-qwen3-4b")
    parser.add_argument(
        "--skip-simulation",
        action="store_true",
        help="score only the output contract; real Qucs scoring is the default",
    )
    return parser


def _generate(model, tokenizer, prompt: list[dict], count: int, max_tokens: int):
    import torch
    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)
    input_ids = tokenizer.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            num_return_sequences=count,
            pad_token_id=tokenizer.eos_token_id,
        )
    prompt_length = input_ids.shape[-1]
    return tokenizer.batch_decode(
        outputs[:, prompt_length:],
        skip_special_tokens=True,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print("preflight:", json.dumps(verify_runtime(), sort_keys=True))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_policy(ModelConfig(model_name=args.model_name))

    group_summaries = []
    for offset in range(args.tasks):
        task = generate_task(seed=args.start_seed + offset, iteration=0)
        completions = _generate(
            model,
            tokenizer,
            task["prompt"],
            args.generations,
            args.max_new_tokens,
        )
        format_scores = format_reward(completions=completions)
        valid_scores = valid_intent_reward(completions=completions)
        if args.skip_simulation:
            physics_scores = [0.0] * len(completions)
        else:
            reward = make_simulation_reward(
                max_workers=2,
                log_path=output_dir / "rewards.jsonl",
            )
            physics_scores = reward(
                completions=completions,
                params_json=[task["params_json"]] * len(completions),
                baseline_cost_json=[task["baseline_cost_json"]] * len(completions),
                iteration=[task["iteration"]] * len(completions),
                seed=[task["seed"]] * len(completions),
            )
        summary = {
            "seed": task["seed"],
            "format_rate": sum(score > 0 for score in format_scores) / len(completions),
            "valid_rate": sum(valid_scores) / len(completions),
            "reward_mean": statistics.fmean(physics_scores),
            "reward_std": statistics.pstdev(physics_scores),
        }
        group_summaries.append(summary)
        print(json.dumps(summary, sort_keys=True))
        for completion, reward_value in zip(completions, physics_scores):
            print(f"reward={reward_value:+.3f}\n{completion_text(completion)}\n---")

    report_path = output_dir / "summary.json"
    report_path.write_text(json.dumps(group_summaries, indent=2))
    print(f"wrote probe summary to {report_path}")


if __name__ == "__main__":
    main()

