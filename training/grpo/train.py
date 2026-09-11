"""Custom on-policy multi-turn GRPO-style trainer.

Owns Qucs-shaped trajectory advantages end-to-end. ``training.grpo.trl_rollout`` can
pack the same trajectories for TRL ``rollout_func``; this module remains the
default trainer until that path is wired as the primary loop.

Update rule (on-policy GRPO-style):
1. For each of ``tasks_per_step`` goals, draw one shared initial circuit and
   roll out ``generations`` independent trajectories from that same start
   (``training.grpo.rollout.run_trajectory``).
2. Score each trajectory with a mixed best-so-far / final dB reward, then
   form GRPO group-relative advantages and add a per-turn reward-to-go term
   so destroying steps are not reinforced as strongly as improving ones.
3. Flatten every turn into one training example and take
   ``loss = -mean(advantage * sum(log_probs))``.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from decision_log import append_record  # noqa: E402

from training.grpo.config import (  # noqa: E402
    MultiturnConfig,
    MultiturnDataSection,
    config_to_dict,
    resolve_multiturn_config,
    to_model_config,
)
from training.grpo.diagnostics import build_step_stats  # noqa: E402
from training.common.goals import (  # noqa: E402
    GoalDistributionConfig,
    GoalSpec,
    load_goal_distribution,
    sample_goal_from_distribution,
)
from training.common.modeling import load_policy  # noqa: E402
from training.common.preflight import verify_runtime  # noqa: E402
from training.grpo.rollout import (  # noqa: E402
    Trajectory,
    run_trajectory,
    shaped_turn_advantages,
    trajectory_to_json,
)
from training.grpo.starts import sample_params_with_headroom  # noqa: E402


def resolve_multiturn_goal_distribution(data: MultiturnDataSection) -> GoalDistributionConfig:
    """Load shared goal YAML; optional CLI freq bounds mutate a copy."""
    dist = load_goal_distribution(data.goal_config)
    overrides: dict[str, float] = {}
    if data.goal_freq_min_ghz is not None:
        overrides["freq_min_hz"] = data.goal_freq_min_ghz * 1e9
    if data.goal_freq_max_ghz is not None:
        overrides["freq_max_hz"] = data.goal_freq_max_ghz * 1e9
    if overrides:
        return replace(dist, **overrides)
    return dist


def sample_multiturn_goal(seed: int, data: MultiturnDataSection) -> GoalSpec:
    return sample_goal_from_distribution(seed, resolve_multiturn_goal_distribution(data))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="YAML hyperparameter config path")
    parser.add_argument("--model-name", default=None)
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="model context window; multi-turn history needs more than a short "
        "single-prompt window, so this trainer defaults higher",
    )
    parser.add_argument("--tasks-per-step", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None, help="trajectories per goal (GRPO group size)")
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="stop a trajectory after this many turns without a new best dB; 0 disables",
    )
    parser.add_argument(
        "--patience-eps",
        type=float,
        default=None,
        help="dB improvement required to reset patience",
    )
    parser.add_argument(
        "--goal-config",
        default=None,
        help="path to shared goal distribution YAML (default: configs/goal_distribution.yaml)",
    )
    parser.add_argument(
        "--param-spread",
        type=float,
        default=None,
        help="fraction of each bound range sampled around INITIAL_GUESS",
    )
    parser.add_argument(
        "--min-start-headroom-db",
        type=float,
        default=None,
        help="reject starts with initial_db <= target_depth_db + this margin",
    )
    parser.add_argument("--history-window", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--start-seed", type=int, default=None)
    parser.add_argument("--goal-freq-min-ghz", type=float, default=None)
    parser.add_argument("--goal-freq-max-ghz", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--resume-adapter",
        default=None,
        help=(
            "load a previously saved LoRA (checkpoint-* or final_lora) before "
            "the first rollout, then keep training; required unless "
            "--allow-raw-base"
        ),
    )
    parser.add_argument(
        "--allow-raw-base",
        action="store_true",
        default=None,
        help=(
            "escape hatch: allow training from a fresh random LoRA without "
            "--resume-adapter (otherwise an SFT adapter is required)"
        ),
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help=(
            "KL coefficient in turn_loss = (-A * logp + beta * (logp - logp_ref)) "
            "/ N; 0 disables the extra ref forward"
        ),
    )
    parser.add_argument(
        "--kl-ref",
        choices=("start", "base"),
        default=None,
        help=(
            "KL reference policy: 'start' freezes LoRA weights from the beginning "
            "of this run (the resume adapter, or the fresh LoRA if not resuming); "
            "'base' uses the 4-bit backbone with adapters disabled"
        ),
    )
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


def _dry_run(config: MultiturnConfig) -> None:
    d = config.data
    t = config.train
    goal = sample_multiturn_goal(d.start_seed, d)
    traj = run_trajectory(
        _canned_generate,
        goal=goal,
        seed=d.start_seed,
        max_turns=t.max_turns,
        history_window=t.history_window,
        patience=t.patience,
        patience_eps=t.patience_eps,
        param_spread=d.param_spread,
    )
    print(json.dumps(trajectory_to_json(traj), sort_keys=True))


def _make_generate_fn(model, tokenizer, config: MultiturnConfig):
    import torch

    device = next(model.parameters()).device
    t = config.train

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
                max_new_tokens=t.max_new_tokens,
                do_sample=True,
                temperature=t.temperature,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        completion_ids = out[0, prompt_ids.shape[1]:]
        return tokenizer.decode(completion_ids, skip_special_tokens=True)

    return generate


def _load_trainable_adapter(model, adapter_path: str) -> None:
    """Replace the freshly initialized LoRA with a saved adapter and keep it trainable.

    ``load_policy`` always calls ``get_peft_model``, which starts from random B=0
    LoRA. Eval already uses ``load_adapter`` for inference; continued GRPO needs
    the same weights with ``requires_grad`` left on. Saved ``adapter_config.json``
    often has ``inference_mode: true``, so we pass ``is_trainable=True`` when the
    PEFT API accepts it.
    """
    path = Path(adapter_path)
    weights = path / "adapter_model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"no adapter_model.safetensors under {path}")
    try:
        model.load_adapter(str(path), adapter_name="default", is_trainable=True)
    except (TypeError, ValueError):
        from safetensors.torch import load_file

        state = load_file(str(weights))
        model.load_state_dict(state, strict=False)
    if hasattr(model, "set_adapter"):
        model.set_adapter("default")
    model.train()
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True


def _snapshot_lora(model) -> dict:
    return {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}


@contextmanager
def _use_lora_snapshot(model, snapshot: dict):
    if not snapshot:
        yield
        return
    backup = {n: p.detach().clone() for n, p in model.named_parameters() if n in snapshot}
    try:
        for n, p in model.named_parameters():
            if n in snapshot:
                p.data.copy_(snapshot[n])
        yield
    finally:
        for n, p in model.named_parameters():
            if n in backup:
                p.data.copy_(backup[n])


@contextmanager
def _kl_ref_context(model, kl_ref: str, start_lora: dict):
    if kl_ref == "base" and hasattr(model, "disable_adapter"):
        with model.disable_adapter():
            yield
        return
    with _use_lora_snapshot(model, start_lora):
        yield


def _turn_logprob_sum(model, tokenizer, prompt_messages, completion_text, device):
    """Teacher-forced sum of log-probs the *current* weights assign to
    ``completion_text`` given ``prompt_messages``. Differentiable."""
    import torch

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
    # Defensive guard: if the forward pass returns fewer logit positions than
    # input tokens (e.g. silent truncation past max_seq_length), align to the
    # shorter length (keep the *last* N completion tokens) and skip the turn's
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


def train(config: MultiturnConfig) -> None:
    t = config.train
    d = config.data
    r = config.runtime

    if not r.resume_adapter and not r.allow_raw_base:
        raise SystemExit(
            "multiturn training requires --resume-adapter / runtime.resume_adapter "
            "(or pass --allow-raw-base)"
        )

    import torch

    output_dir = Path(r.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    trajectories_path = output_dir / "trajectories.jsonl"
    completions_path = output_dir / "completions.jsonl"

    model, tokenizer = load_policy(to_model_config(config.model))
    if r.resume_adapter:
        _load_trainable_adapter(model, r.resume_adapter)
    device = next(model.parameters()).device
    generate_fn = _make_generate_fn(model, tokenizer, config)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=t.lr)

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a") as fh:
            fh.write(line + "\n")

    seed_cursor = d.start_seed
    total_trajectories = t.steps * t.tasks_per_step * t.generations
    trajectories_done = 0
    start_lora = _snapshot_lora(model) if t.beta > 0 and t.kl_ref == "start" else {}
    if r.resume_adapter:
        log(
            f"resuming LoRA from {r.resume_adapter} start_seed={d.start_seed} "
            f"beta={t.beta} kl_ref={t.kl_ref}"
        )
    elif t.beta > 0:
        log(f"KL enabled beta={t.beta} kl_ref={t.kl_ref}")
    for step in range(1, t.steps + 1):
        step_trajectories: list[Trajectory] = []
        step_goal_groups: list[list[Trajectory]] = []
        step_resamples = 0

        for task_idx in range(t.tasks_per_step):
            goal = sample_multiturn_goal(seed_cursor, d)
            shared_params, initial_db, seed_used, n_resamples = sample_params_with_headroom(
                start_seed=seed_cursor,
                goal=goal,
                spread=d.param_spread,
                headroom_db=d.min_start_headroom_db,
            )
            step_resamples += n_resamples
            if n_resamples:
                log(
                    "  resampled start task %d/%d times=%d seed %d->%d initial_db=%.2f"
                    % (
                        task_idx + 1,
                        t.tasks_per_step,
                        n_resamples,
                        seed_cursor,
                        seed_used,
                        initial_db,
                    )
                )
            group: list[Trajectory] = []
            for gen_idx in range(t.generations):
                turn_counter = {"n": 0}

                def _on_turn(
                    turn,
                    _task_idx=task_idx,
                    _gen_idx=gen_idx,
                    _counter=turn_counter,
                    _goal=goal,
                    _seed=seed_used,
                ):
                    _counter["n"] += 1
                    log(
                        "  rollout step %d task %d/%d gen %d/%d turn %d/%d "
                        "valid=%s db %.2f->%.2f"
                        % (
                            step,
                            _task_idx + 1,
                            t.tasks_per_step,
                            _gen_idx + 1,
                            t.generations,
                            _counter["n"],
                            t.max_turns,
                            turn.valid,
                            turn.db_before,
                            turn.db_after,
                        )
                    )
                    append_record(
                        completions_path,
                        {
                            "source": "multiturn",
                            "step": step,
                            "task": _task_idx + 1,
                            "gen": _gen_idx + 1,
                            "turn": turn.turn_index,
                            "seed": _seed,
                            "target_freq_hz": _goal.target_freq_hz,
                            "prompt": turn.prompt,
                            "completion": turn.completion_text,
                            "valid": turn.valid,
                            "intent": turn.intent,
                            "stopped": turn.stopped,
                            "db_before": turn.db_before,
                            "db_after": turn.db_after,
                        },
                    )

                traj = run_trajectory(
                    generate_fn,
                    goal=goal,
                    seed=seed_used,
                    max_turns=t.max_turns,
                    history_window=t.history_window,
                    on_turn=_on_turn,
                    initial_params=shared_params,
                    patience=t.patience,
                    patience_eps=t.patience_eps,
                    param_spread=d.param_spread,
                )
                if traj.turns:
                    append_record(
                        completions_path,
                        {
                            "source": "multiturn",
                            "event": "trajectory_end",
                            "step": step,
                            "task": task_idx + 1,
                            "gen": gen_idx + 1,
                            "seed": seed_used,
                            "target_freq_hz": goal.target_freq_hz,
                            "terminated_reason": traj.terminated_reason,
                            "trajectory_reward": traj.reward,
                            "initial_db": traj.initial_db,
                            "best_db": traj.best_db,
                            "num_turns": traj.num_turns,
                        },
                    )
                group.append(traj)
                trajectories_done += 1
                log(
                    "trajectory done: step %d task %d/%d gen %d/%d turns=%d "
                    "reason=%s reward=%.3f  [%d/%d trajectories overall]"
                    % (
                        step,
                        task_idx + 1,
                        t.tasks_per_step,
                        gen_idx + 1,
                        t.generations,
                        traj.num_turns,
                        traj.terminated_reason,
                        traj.reward,
                        trajectories_done,
                        total_trajectories,
                    )
                )
            seed_cursor = max(seed_cursor + 1, seed_used + 1)
            step_goal_groups.append(group)
            step_trajectories.extend(group)

        with trajectories_path.open("a") as fh:
            for traj in step_trajectories:
                fh.write(json.dumps(trajectory_to_json(traj), sort_keys=True) + "\n")

        turn_advantages: dict[int, list[float]] = {}
        for group in step_goal_groups:
            turn_advantages.update(shaped_turn_advantages(group))

        stats = build_step_stats(
            step_goal_groups, turn_advantages, n_resampled_starts=step_resamples
        )
        log("step_stats=" + json.dumps(stats, sort_keys=True))

        scored = [
            (traj, turn, adv)
            for traj in step_trajectories
            for turn, adv in zip(traj.turns, turn_advantages[id(traj)])
            if adv != 0.0
        ]
        total_turns = len(scored)

        optimizer.zero_grad()
        loss_sum = 0.0
        kl_sum = 0.0
        if total_turns > 0:
            for grad_turn_idx, (traj, turn, adv) in enumerate(scored, start=1):
                logp = _turn_logprob_sum(model, tokenizer, turn.prompt, turn.completion_text, device)
                if t.beta > 0:
                    with torch.no_grad():
                        with _kl_ref_context(model, t.kl_ref, start_lora):
                            logp_ref = _turn_logprob_sum(
                                model, tokenizer, turn.prompt, turn.completion_text, device
                            )
                    kl = logp - logp_ref.detach()
                    turn_loss = (-adv * logp + t.beta * kl) / total_turns
                    kl_sum += kl.detach().item()
                    del logp_ref, kl
                else:
                    turn_loss = (-adv * logp) / total_turns
                turn_loss.backward()
                loss_sum += turn_loss.detach().item()
                del logp, turn_loss
                if grad_turn_idx % 20 == 0 or grad_turn_idx == total_turns:
                    log(f"  gradient pass {grad_turn_idx}/{total_turns} turns backpropagated")
            torch.nn.utils.clip_grad_norm_(trainable_params, t.max_grad_norm)
            optimizer.step()
        loss_value = loss_sum

        rewards_all = [traj.reward for traj in step_trajectories]
        turns_all = [traj.num_turns for traj in step_trajectories]
        met = sum(1 for traj in step_trajectories if traj.terminated_reason == "goal_met")
        stopped = sum(1 for traj in step_trajectories if traj.terminated_reason == "stop")
        patience_n = sum(1 for traj in step_trajectories if traj.terminated_reason == "patience")
        kl_mean = (kl_sum / total_turns) if total_turns and t.beta > 0 else 0.0
        log(
            "step %d/%d loss=%.5f reward_mean=%.3f reward_std=%.3f "
            "turns_mean=%.1f goal_met=%d/%d stop=%d patience=%d kl_mean=%.5f"
            % (
                step,
                t.steps,
                loss_value,
                statistics.fmean(rewards_all),
                statistics.pstdev(rewards_all) if len(rewards_all) > 1 else 0.0,
                statistics.fmean(turns_all),
                met,
                len(step_trajectories),
                stopped,
                patience_n,
                kl_mean,
            )
        )

        if step % t.save_every == 0 or step == t.steps:
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
    cfg = resolve_multiturn_config(args)
    print("config:", json.dumps(config_to_dict(cfg), sort_keys=True))
    info = verify_runtime(require_cuda=not args.dry_run)
    print("preflight:", json.dumps(info, sort_keys=True))
    if args.dry_run:
        _dry_run(cfg)
        return
    train(cfg)


if __name__ == "__main__":
    main()
