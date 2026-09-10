"""Train the one-step intent policy with Unsloth GRPO and real Qucs rewards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.config import (
    GrpoConfig,
    config_to_dict,
    resolve_grpo_config,
    to_model_config,
)
from training.data import build_grpo_dataset
from training.environment import generate_task
from training.goals import sample_goal
from training.modeling import load_policy, mixed_precision_config
from training.preflight import verify_runtime
from training.rewards import (
    format_reward,
    make_simulation_reward,
    valid_intent_reward,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="YAML hyperparameter config path")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--tasks", type=int, default=None)
    parser.add_argument("--start-seed", type=int, default=None)
    parser.add_argument("--max-iteration", type=int, default=None)
    parser.add_argument(
        "--goal-freq-min-ghz",
        type=float,
        default=None,
        help="lower bound of the randomized notch-target training distribution",
    )
    parser.add_argument(
        "--goal-freq-max-ghz",
        type=float,
        default=None,
        help="upper bound of the randomized notch-target training distribution",
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--sim-workers", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--use-vllm", action="store_true", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run one real Qucs reward without loading the model",
    )
    return parser


def grpo_config_kwargs(config: GrpoConfig) -> dict:
    t, r = config.train, config.runtime
    return {
        "output_dir": r.output_dir,
        "learning_rate": t.learning_rate,
        "weight_decay": t.weight_decay,
        "warmup_ratio": t.warmup_ratio,
        "lr_scheduler_type": t.lr_scheduler_type,
        "optim": t.optim,
        "logging_steps": 1,
        "per_device_train_batch_size": t.per_device_train_batch_size,
        "gradient_accumulation_steps": t.gradient_accumulation_steps,
        "generation_batch_size": t.generations,
        "num_generations": t.generations,
        "max_prompt_length": t.max_prompt_length,
        "max_completion_length": t.max_completion_length,
        "max_steps": t.steps,
        "save_steps": max(1, min(25, t.steps)),
        "max_grad_norm": t.max_grad_norm,
        "report_to": "none",
        **mixed_precision_config(),
        "temperature": t.temperature,
        "beta": config.reward.beta,
        "loss_type": t.loss_type,
        "reward_weights": list(config.reward.weights),
        "use_vllm": r.use_vllm,
        "remove_unused_columns": False,
    }


def _dry_run(cfg: GrpoConfig) -> None:
    freq_range = (cfg.data.goal_freq_min_ghz * 1e9, cfg.data.goal_freq_max_ghz * 1e9)
    goal = sample_goal(cfg.data.start_seed, freq_range=freq_range)
    task = generate_task(seed=cfg.data.start_seed, iteration=0, goal=goal)
    reward = make_simulation_reward(
        max_workers=1,
        log_path=Path(cfg.runtime.output_dir) / "rewards.jsonl",
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
    cfg = resolve_grpo_config(args)
    print("config:", json.dumps(config_to_dict(cfg), sort_keys=True))
    info = verify_runtime(require_cuda=not args.dry_run)
    print("preflight:", json.dumps(info, sort_keys=True))
    if args.dry_run:
        _dry_run(cfg)
        return

    output_dir = Path(cfg.runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_grpo_dataset(
        count=cfg.data.tasks,
        start_seed=cfg.data.start_seed,
        max_iteration=cfg.data.max_iteration,
        freq_range=(cfg.data.goal_freq_min_ghz * 1e9, cfg.data.goal_freq_max_ghz * 1e9),
    )
    print(f"prepared {len(dataset)} real-Qucs task states")

    model, tokenizer = load_policy(
        to_model_config(cfg.model, fast_inference=cfg.runtime.use_vllm)
    )
    from trl import GRPOConfig, GRPOTrainer

    simulation_reward = make_simulation_reward(
        max_workers=cfg.runtime.sim_workers,
        log_path=output_dir / "rewards.jsonl",
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[format_reward, valid_intent_reward, simulation_reward],
        args=GRPOConfig(**grpo_config_kwargs(cfg)),
        train_dataset=dataset,
    )
    trainer.train(resume_from_checkpoint=cfg.runtime.resume_from_checkpoint or None)
    final_dir = output_dir / "final_lora"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"saved LoRA adapter to {final_dir}")


if __name__ == "__main__":
    main()
