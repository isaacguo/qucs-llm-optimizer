"""Check whether a multi-turn-trained policy generalizes on held-out targets.

Training (``multiturn_train.py``) optimizes the policy over *full multi-turn
rollouts* (predict -> simulate -> decide whether to keep going), so a fair
generalization check has to run the same kind of rollout, not a single-shot
intent prediction. ``evaluate_generalization.py`` only tests one Stage-A-style
step and cannot tell whether the multi-turn behaviour (best-so-far tracking,
patience, self-stop timing) generalizes to goals the policy never trained on.

This script runs full ``run_trajectory`` rollouts (the same environment loop
used during training) against ``training.goals.heldout_goals()``, for both a
baseline (no adapter -> equivalent to the untrained base model, since a fresh
LoRA's B matrix is zero-initialized) and a trained adapter, so the two are
directly comparable.

Usage:
    uv run python -m training.evaluate_multiturn_generalization \\
        --adapter outputs/2026-09-09-1810-multiturn-nightly-4x4x60/final_lora \\
        --episodes-per-goal 3 --target-depth-db -70
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import random

from training.goals import GoalSpec, heldout_goals, sample_goal
from training.modeling import ModelConfig, load_policy
from training.multiturn_train import _make_generate_fn
from training.rollout import run_trajectory, trajectory_to_json


def _random_goals(
    num_goals: int,
    *,
    seed_base: int,
    freq_range: tuple[float, float],
    depth_min: float,
    depth_max: float,
) -> list[tuple[int, GoalSpec]]:
    """``num_goals`` goals with both frequency and target depth varied.

    Uses seeds far outside any range multiturn_train.py has used so there is
    no accidental overlap with the training seed stream. Depth is sampled
    independently of ``sample_goal``'s own RNG stream (which only varies
    frequency) so this does not have to touch training/goals.py.
    """
    depth_rng = random.Random(seed_base ^ 0xD3B7)
    out = []
    for i in range(num_goals):
        seed = seed_base + i
        base = sample_goal(seed, freq_range=freq_range)
        depth = depth_rng.uniform(depth_min, depth_max)
        out.append((seed, GoalSpec(target_freq_hz=base.target_freq_hz, band_hz=base.band_hz, target_depth_db=depth)))
    return out


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
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--episodes-per-goal", type=int, default=3)
    parser.add_argument("--start-seed", type=int, default=9000)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--patience-eps", type=float, default=0.2)
    parser.add_argument("--history-window", type=int, default=8)
    parser.add_argument("--param-spread", type=float, default=0.35)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--target-depth-db",
        type=float,
        default=None,
        help="override heldout_goals()'s target_depth_db (default: keep as-is, -70)",
    )
    parser.add_argument(
        "--num-random-goals",
        type=int,
        default=0,
        help=(
            "if > 0, ignore heldout_goals() and instead sample this many goals "
            "with both frequency and target depth varied (see --depth-min/--depth-max)"
        ),
    )
    parser.add_argument("--depth-min", type=float, default=-60.0)
    parser.add_argument("--depth-max", type=float, default=-15.0)
    parser.add_argument("--goal-freq-min-ghz", type=float, default=4.0)
    parser.add_argument("--goal-freq-max-ghz", type=float, default=6.0)
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)

    model, tokenizer = load_policy(
        ModelConfig(model_name=args.model_name, max_seq_length=args.max_seq_length)
    )
    if args.adapter:
        model.load_adapter(args.adapter, adapter_name="default")

    # ``_make_generate_fn`` expects a ``MultiturnConfig``-shaped object with a
    # nested ``.train`` (max_new_tokens/temperature), not this script's flat
    # argparse Namespace -- wrap the two fields it actually reads.
    generate_fn = _make_generate_fn(
        model,
        tokenizer,
        SimpleNamespace(
            train=SimpleNamespace(
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
        ),
    )

    if args.num_random_goals > 0:
        freq_range = (args.goal_freq_min_ghz * 1e9, args.goal_freq_max_ghz * 1e9)
        seeded_goals = _random_goals(
            args.num_random_goals,
            seed_base=args.start_seed,
            freq_range=freq_range,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
        )
    else:
        goals = heldout_goals()
        if args.target_depth_db is not None:
            goals = [
                GoalSpec(
                    target_freq_hz=g.target_freq_hz,
                    band_hz=g.band_hz,
                    target_depth_db=args.target_depth_db,
                )
                for g in goals
            ]
        # One (seed, goal) pair per requested episode.
        seeded_goals = []
        seed = args.start_seed
        for g in goals:
            for _ in range(args.episodes_per_goal):
                seeded_goals.append((seed, g))
                seed += 1

    trajectories = []
    for seed, goal in seeded_goals:
        traj = run_trajectory(
            generate_fn,
            goal=goal,
            seed=seed,
            max_turns=args.max_turns,
            history_window=args.history_window,
            patience=args.patience,
            patience_eps=args.patience_eps,
            param_spread=args.param_spread,
        )
        trajectories.append(trajectory_to_json(traj))
        print(
            f"goal={goal.target_freq_hz/1e9:.2f}GHz depth={goal.target_depth_db:.1f}dB "
            f"seed={seed} turns={traj.num_turns} reason={traj.terminated_reason} "
            f"reward={traj.reward:.3f} initial_db={traj.initial_db:.2f} "
            f"best_db={traj.best_db:.2f}",
            flush=True,
        )

    rewards = [t["reward"] for t in trajectories]
    improvements = [t["initial_db"] - t["best_db"] for t in trajectories]
    reasons = defaultdict(int)
    for t in trajectories:
        reasons[t["terminated_reason"]] += 1
    goal_met = reasons.get("goal_met", 0)

    summary = {
        "adapter": args.adapter or "(none - raw base model baseline)",
        "goals_tested": len(seeded_goals),
        "episodes_per_goal": args.episodes_per_goal,
        "total_trajectories": len(trajectories),
        "mean_reward": statistics.fmean(rewards) if rewards else None,
        "std_reward": statistics.pstdev(rewards) if len(rewards) > 1 else 0.0,
        "mean_best_improvement_db": statistics.fmean(improvements) if improvements else None,
        "goal_met_rate": goal_met / len(trajectories) if trajectories else 0.0,
        "terminated_reason_counts": dict(reasons),
        "mean_num_turns": statistics.fmean(t["num_turns"] for t in trajectories)
        if trajectories
        else None,
    }
    print(json.dumps({"summary": summary}, indent=2, sort_keys=True))

    if args.output:
        Path(args.output).write_text(
            json.dumps(
                {"summary": summary, "trajectories": trajectories},
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
