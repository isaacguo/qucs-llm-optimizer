"""Optional one-epoch Unsloth SFT warm-up from corpus index (or legacy state)."""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

from training.data import load_multiturn_sft_from_index, load_sft_records
from training.modeling import ModelConfig, load_policy, mixed_precision_config
from training.preflight import verify_runtime

DEFAULT_MODEL = "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit"
DEFAULT_INDEX = "corpus/index.jsonl"
DEFAULT_RUNS_ROOT = "runs"
DEFAULT_HISTORY_WINDOW = 8
DEFAULT_MAX_LENGTH = 2048


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument(
        "--index",
        default=DEFAULT_INDEX,
        help="corpus index JSONL (default path for multiturn cold-start)",
    )
    parser.add_argument(
        "--runs-root",
        default=DEFAULT_RUNS_ROOT,
        help="root directory for relative run_dir entries in the index",
    )
    parser.add_argument(
        "--history-window",
        type=int,
        default=DEFAULT_HISTORY_WINDOW,
        help="prior turns included in each exported multiturn prompt",
    )
    parser.add_argument(
        "--state",
        default=None,
        help="legacy single-file trajectory; prefer --index (prints a warning)",
    )
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--output-dir", default="outputs/sft-qwen3-4b")
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="SFTTrainer max sequence length",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and count examples without loading the model",
    )
    return parser


def _load_records(args: argparse.Namespace) -> list[dict]:
    if args.state is not None:
        warnings.warn(
            "--state is legacy single-file SFT; prefer --index for multiturn cold-start",
            stacklevel=2,
        )
        records = load_sft_records(Path(args.state))
        if not records:
            raise RuntimeError(f"no SFT records found in {args.state}")
        return records

    records = load_multiturn_sft_from_index(
        Path(args.index),
        Path(args.runs_root),
        history_window=args.history_window,
    )
    if not records:
        raise RuntimeError(
            f"no SFT records exported from index {args.index} "
            f"(empty index or no eligible turns under {args.runs_root})"
        )
    return records


def _to_prompt_completion(record: dict) -> dict:
    """Split a {"messages": [...]} SFT record into trl's prompt-completion format.

    Every exported record is exactly [system, user, assistant] (one rollout-
    aligned turn; multiturn history lives inside the user text itself, per
    ``build_multiturn_prompt`` — see corpus.py). Keeping the natural message
    list (instead of pre-flattening via apply_chat_template into one "text"
    string) lets SFTTrainer auto-detect a conversational prompt-completion
    dataset and mask the loss so only the assistant's
    <reasoning>/<intent> tokens are supervised; the system+user tokens are
    still visible to the model as context but contribute zero gradient.
    """
    messages = record["messages"]
    if len(messages) < 2 or messages[-1]["role"] != "assistant":
        raise ValueError(
            f"expected [..., assistant] messages, got roles={[m['role'] for m in messages]}"
        )
    return {"prompt": messages[:-1], "completion": [messages[-1]]}


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    records = _load_records(args)
    print(f"loaded {len(records)} optional SFT examples")
    if args.dry_run:
        return

    print("preflight:", json.dumps(verify_runtime(), sort_keys=True))
    model, tokenizer = load_policy(
        ModelConfig(model_name=args.model_name, max_seq_length=args.max_length)
    )
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    dataset = Dataset.from_list([_to_prompt_completion(record) for record in records])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_8bit",
        logging_steps=1,
        save_strategy="epoch",
        max_length=args.max_length,
        completion_only_loss=True,
        report_to="none",
        **mixed_precision_config(),
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=dataset,
    )
    trainer.train()
    final_dir = output_dir / "final_lora"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"saved optional SFT adapter to {final_dir}")


if __name__ == "__main__":
    main()
