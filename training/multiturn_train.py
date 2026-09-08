"""Custom on-policy multi-turn GRPO-style trainer.

Why this file exists instead of reusing ``trl.GRPOTrainer`` directly: TRL's
``GRPOTrainer._generate_and_score_completions`` always calls its own internal
``_generate`` to produce completions for a batch of prompts - it has no way
to accept externally pre-generated completions. Our multi-turn rollout
*requires* pre-generated completions, because turn N's prompt depends on the
real simulated outcome of turn N-1's completion; TRL cannot regenerate a
turn in isolation and get a consistent trajectory.

So this module implements the same on-policy update rule GRPO uses -
generate a fresh batch with the *current* weights, score it, take one
gradient step, throw the batch away - by hand:

1. For each of ``tasks_per_step`` goals, roll out ``generations`` independent
   trajectories with the current policy (``training.rollout.run_trajectory``).
2. Compute a GRPO-style group-relative advantage per goal: reward normalized
   against the mean/std of that goal's own ``generations`` trajectories.
3. Flatten every turn of every trajectory into one training example, sharing
   its trajectory's single advantage value (terminal-only reward, broadcast
   across turns - whole-episode policy gradient credit assignment).
4. Teacher-force a forward pass over (prompt + completion) to get the
   completion's token log-probs under the *same* weights that generated it,
   and take ``loss = -mean(advantage * sum(log_probs))``.

Because the batch is generated and consumed in the same step (used for
exactly one gradient update, then discarded), this is on-policy with no
importance-sampling ratio needed - policy-gradient/REINFORCE with a
group-normalized baseline, same spirit as GRPO's advantage estimator.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from training.goals import TRAIN_FREQ_RANGE_HZ, sample_goal
from training.modeling import ModelConfig, load_policy
from training.preflight import verify_runtime
from training.rollout import Trajectory, run_trajectory, trajectory_to_json

DEFAULT_MODEL = "unsloth/Qwen3-1.7B-bnb-4bit"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="model context window; multi-turn history can exceed the 1024 "
        "default used by Stage A, so this trainer defaults higher",
    )
    parser.add_argument("--tasks-per-step", type=int, default=2)
    parser.add_argument("--generations", type=int, default=4, help="trajectories per goal (GRPO group size)")
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--history-window", type=int, default=8)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--start-seed", type=int, default=5000)
    parser.add_argument("--goal-freq-min-ghz", type=float, default=4.0)
    parser.add_argument("--goal-freq-max-ghz", type=float, default=6.0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--output-dir", default="outputs/multiturn-grpo-demo")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run one trajectory with a canned completion, no model load",
    )
    return parser


def _canned_generate(_messages: list[dict[str, str]]) -> str:
    return (
        "<reasoning>Probe the dominant radius with a conservative step.</reasoning>"
        '<intent>{"ro":"decrease_slight"}</intent>'
    )


def _dry_run(args: argparse.Namespace) -> None:
    freq_range = (args.goal_freq_min_ghz * 1e9, args.goal_freq_max_ghz * 1e9)
    goal = sample_goal(args.start_seed, freq_range=freq_range)
    traj = run_trajectory(
        _canned_generate,
        goal=goal,
        seed=args.start_seed,
        max_turns=args.max_turns,
        history_window=args.history_window,
    )
    print(json.dumps(trajectory_to_json(traj), sort_keys=True))


def _make_generate_fn(model, tokenizer, args: argparse.Namespace):
    device = next(model.parameters()).device

    def generate(messages: list[dict[str, str]]) -> str:
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            model.eval()
            out = model.generate(
                input_ids=prompt_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        completion_ids = out[0, prompt_ids.shape[1]:]
        return tokenizer.decode(completion_ids, skip_special_tokens=True)

    return generate


def _turn_logprob_sum(model, tokenizer, prompt_messages, completion_text, device) -> torch.Tensor:
    """Teacher-forced sum of log-probs the *current* weights assign to
    ``completion_text`` given ``prompt_messages``. Differentiable."""
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(device)
    completion_ids = tokenizer(
        completion_text, return_tensors="pt", add_special_tokens=False
    )["input_ids"].to(device)
    if completion_ids.shape[1] == 0:
        return torch.tensor(0.0, device=device)

    full_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.ones_like(full_ids)
    model.train()
    logits = model(input_ids=full_ids, attention_mask=attention_mask).logits
    # Predict token t from logits at position t-1; only score completion tokens.
    prompt_len = prompt_ids.shape[1]
    pred_logits = logits[:, prompt_len - 1 : -1, :]
    target_ids = full_ids[:, prompt_len:]
    # Defensive guard: some backends (e.g. unsloth) silently truncate the
    # forward pass when full_ids exceeds the model's configured
    # max_seq_length, returning fewer logit positions than input tokens.
    # Rather than let torch.gather crash the whole run, align to the
    # shorter length (keep the *last* N completion tokens, since a left
    # truncation drops the oldest context first) and skip the turn's
    # gradient contribution if nothing usable is left.
    n_pred, n_target = pred_logits.shape[1], target_ids.shape[1]
    if n_pred != n_target:
        n = min(n_pred, n_target)
        if n == 0:
            return torch.tensor(0.0, device=device)
        pred_logits = pred_logits[:, -n:, :]
        target_ids = target_ids[:, -n:]
    log_probs = torch.log_softmax(pred_logits.float(), dim=-1)
    token_logps = torch.gather(log_probs, 2, target_ids.unsqueeze(-1)).squeeze(-1)
    return token_logps.sum()


def train(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    trajectories_path = output_dir / "trajectories.jsonl"

    model, tokenizer = load_policy(
        ModelConfig(model_name=args.model_name, max_seq_length=args.max_seq_length)
    )
    device = next(model.parameters()).device
    generate_fn = _make_generate_fn(model, tokenizer, args)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    freq_range = (args.goal_freq_min_ghz * 1e9, args.goal_freq_max_ghz * 1e9)

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as fh:
            fh.write(line + "\n")

    seed_cursor = args.start_seed
    for step in range(1, args.steps + 1):
        step_trajectories: list[Trajectory] = []
        step_goal_groups: list[list[Trajectory]] = []

    seed_cursor = args.start_seed
    total_trajectories = args.steps * args.tasks_per_step * args.generations
    trajectories_done = 0
    for step in range(1, args.steps + 1):
        step_trajectories: list[Trajectory] = []
        step_goal_groups: list[list[Trajectory]] = []

        for task_idx in range(args.tasks_per_step):
            goal = sample_goal(seed_cursor, freq_range=freq_range)
            group: list[Trajectory] = []
            for gen_idx in range(args.generations):
                turn_counter = {"n": 0}

                def _on_turn(turn, _task_idx=task_idx, _gen_idx=gen_idx, _counter=turn_counter):
                    _counter["n"] += 1
                    log(
                        "  rollout step %d task %d/%d gen %d/%d turn %d/%d "
                        "valid=%s db %.2f->%.2f"
                        % (
                            step,
                            _task_idx + 1,
                            args.tasks_per_step,
                            _gen_idx + 1,
                            args.generations,
                            _counter["n"],
                            args.max_turns,
                            turn.valid,
                            turn.db_before,
                            turn.db_after,
                        )
                    )

                traj = run_trajectory(
                    generate_fn,
                    goal=goal,
                    seed=seed_cursor,
                    max_turns=args.max_turns,
                    history_window=args.history_window,
                    on_turn=_on_turn,
                )
                group.append(traj)
                seed_cursor += 1
                trajectories_done += 1
                log(
                    "trajectory done: step %d task %d/%d gen %d/%d turns=%d "
                    "reason=%s reward=%.3f  [%d/%d trajectories overall]"
                    % (
                        step,
                        task_idx + 1,
                        args.tasks_per_step,
                        gen_idx + 1,
                        args.generations,
                        traj.num_turns,
                        traj.terminated_reason,
                        traj.reward,
                        trajectories_done,
                        total_trajectories,
                    )
                )
            step_goal_groups.append(group)
            step_trajectories.extend(group)

        with trajectories_path.open("a") as fh:
            for traj in step_trajectories:
                fh.write(json.dumps(trajectory_to_json(traj), sort_keys=True) + "\n")

        # Group-relative advantage: normalize each trajectory's reward against
        # the mean/std of the other trajectories sampled for the *same* goal.
        advantages: dict[int, float] = {}
        for group in step_goal_groups:
            rewards = [t.reward for t in group]
            mean_r = statistics.fmean(rewards)
            std_r = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
            for traj in group:
                advantages[id(traj)] = (traj.reward - mean_r) / (std_r + 1e-4)

        # Count turns first so each turn's loss can be pre-scaled by 1/total_turns,
        # then backward() one turn at a time. Accumulating one computation graph
        # across all ~hundreds of turns (single backward() at the end) blew past
        # 8GB VRAM; backward-per-turn keeps only one turn's activations alive at
        # a time and gradients simply accumulate in .grad across calls - the sum
        # of many small backwards is mathematically identical to one big one.
        total_turns = sum(
            len(traj.turns) for traj in step_trajectories if advantages[id(traj)] != 0.0
        )

        optimizer.zero_grad()
        loss_sum = 0.0
        if total_turns > 0:
            grad_turn_idx = 0
            for traj in step_trajectories:
                adv = advantages[id(traj)]
                if adv == 0.0:
                    continue
                for turn in traj.turns:
                    logp = _turn_logprob_sum(model, tokenizer, turn.prompt, turn.completion_text, device)
                    turn_loss = (-adv * logp) / total_turns
                    turn_loss.backward()
                    loss_sum += turn_loss.detach().item()
                    del logp, turn_loss
                    grad_turn_idx += 1
                    if grad_turn_idx % 20 == 0 or grad_turn_idx == total_turns:
                        log(f"  gradient pass {grad_turn_idx}/{total_turns} turns backpropagated")
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
        loss_value = loss_sum

        rewards_all = [t.reward for t in step_trajectories]
        turns_all = [t.num_turns for t in step_trajectories]
        met = sum(1 for t in step_trajectories if t.terminated_reason == "goal_met")
        log(
            "step %d/%d loss=%.5f reward_mean=%.3f reward_std=%.3f turns_mean=%.1f goal_met=%d/%d"
            % (
                step,
                args.steps,
                loss_value,
                statistics.fmean(rewards_all),
                statistics.pstdev(rewards_all) if len(rewards_all) > 1 else 0.0,
                statistics.fmean(turns_all),
                met,
                len(step_trajectories),
            )
        )

        if step % args.save_every == 0 or step == args.steps:
            ckpt_dir = output_dir / f"checkpoint-{step}"
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            log(f"saved checkpoint to {ckpt_dir}")

    final_dir = output_dir / "final_lora"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    log(f"saved final LoRA adapter to {final_dir}")


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    info = verify_runtime(require_cuda=not args.dry_run)
    print("preflight:", json.dumps(info, sort_keys=True))
    if args.dry_run:
        _dry_run(args)
        return
    train(args)


if __name__ == "__main__":
    main()
