"""Optional one-epoch Unsloth SFT warm-up from the recorded llm1 trajectory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.data import load_sft_records
from training.modeling import ModelConfig, load_policy, mixed_precision_config
from training.preflight import verify_runtime

DEFAULT_MODEL = "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--state", default="runs/llm1/state.json")
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--output-dir", default="outputs/sft-qwen3-4b")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and count examples without loading the model",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    records = load_sft_records(Path(args.state))
    if not records:
        raise RuntimeError(f"no SFT records found in {args.state}")
    print(f"loaded {len(records)} optional SFT examples")
    if args.dry_run:
        return

    print("preflight:", json.dumps(verify_runtime(), sort_keys=True))
    model, tokenizer = load_policy(ModelConfig(model_name=args.model_name))
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    texts = [
        {
            "text": tokenizer.apply_chat_template(
                record["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }
        for record in records
    ]
    dataset = Dataset.from_list(texts)
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
        max_length=1024,
        dataset_text_field="text",
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

