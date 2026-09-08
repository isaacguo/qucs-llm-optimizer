"""Train the one-step intent policy with Unsloth GRPO and real Qucs rewards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.data import build_grpo_dataset
from training.environment import generate_task
from training.goals import sample_goal
from training.modeling import ModelConfig, load_policy
from training.preflight import verify_runtime
from training.rewards import (
    format_reward,
    make_simulation_reward,
    valid_intent_reward,
)

DEFAULT_MODEL = "unsloth/Qwen3-1.7B-bnb-4bit"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--tasks", type=int, default=32)
    parser.add_argument("--start-seed", type=int, default=1000)
    parser.add_argument("--max-iteration", type=int, default=12)
    parser.add_argument(
        "--goal-freq-min-ghz",
        type=float,
        default=4.0,
        help="lower bound of the randomized notch-target training distribution",
    )
    parser.add_argument(
        "--goal-freq-max-ghz",
        type=float,
        default=6.0,
        help="upper bound of the randomized notch-target training distribution",
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--sim-workers", type=int, default=2)
    parser.add_argument("--output-dir", default="outputs/grpo-qwen3-1.7b")
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run one real Qucs reward without loading the model",
    )
    return parser


def grpo_config_kwargs(args: argparse.Namespace) -> dict:
    return {
        "output_dir": args.output_dir,
        "learning_rate": 5e-6,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "optim": "adamw_8bit",
        "logging_steps": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "generation_batch_size": args.generations,
        "num_generations": args.generations,
        "max_prompt_length": 768,
        "max_completion_length": 256,
        "max_steps": args.steps,
        "save_steps": max(1, min(25, args.steps)),
        "max_grad_norm": 0.1,
        "report_to": "none",
        "bf16": True,
        "fp16": False,
        "temperature": 1.0,
        "beta": 0.01,
        "loss_type": "dr_grpo",
        "reward_weights": [0.2, 0.2, 1.0],
        "use_vllm": args.use_vllm,
        "remove_unused_columns": False,
    }


def _dry_run(args: argparse.Namespace) -> None:
    freq_range = (args.goal_freq_min_ghz * 1e9, args.goal_freq_max_ghz * 1e9)
    goal = sample_goal(args.start_seed, freq_range=freq_range)
    task = generate_task(seed=args.start_seed, iteration=0, goal=goal)
    reward = make_simulation_reward(
        max_workers=1,
        log_path=Path(args.output_dir) / "rewards.jsonl",
    )
    completion = (
        "<reasoning>Probe the dominant radius with a conservative step.</reasoning>"
        '<intent>{"ro":"decrease_slight"}</intent>'
    )
    score = reward(
        completions=[completion],
        params_json=[task["params_json"]],
        baseline_cost_json=[task["baseline_cost_json"]],
        goal_json=[task["goal_json"]],
        iteration=[task["iteration"]],
        seed=[task["seed"]],
    )[0]
    print(json.dumps({"real_qucs_reward": score, "seed": task["seed"], "goal": goal.describe()}))


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    info = verify_runtime(require_cuda=not args.dry_run)
    print("preflight:", json.dumps(info, sort_keys=True))
    if args.dry_run:
        _dry_run(args)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_grpo_dataset(
        count=args.tasks,
        start_seed=args.start_seed,
        max_iteration=args.max_iteration,
        freq_range=(args.goal_freq_min_ghz * 1e9, args.goal_freq_max_ghz * 1e9),
    )
    print(f"prepared {len(dataset)} real-Qucs task states")

    model, tokenizer = load_policy(
        ModelConfig(
            model_name=args.model_name,
            fast_inference=args.use_vllm,
        )
    )
    from trl import GRPOConfig, GRPOTrainer

    simulation_reward = make_simulation_reward(
        max_workers=args.sim_workers,
        log_path=output_dir / "rewards.jsonl",
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward, valid_intent_reward, simulation_reward],
        args=GRPOConfig(**grpo_config_kwargs(args)),
        train_dataset=dataset,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    final_dir = output_dir / "final_lora"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"saved LoRA adapter to {final_dir}")


if __name__ == "__main__":
    main()

