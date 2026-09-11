# Multiturn Qwen3-4B-Instruct YAML recipe

**Date:** 2026-09-11  
**Status:** Approved (approach C)

## Goal

Add a git-tracked hyperparameter recipe for multiturn GRPO on
`unsloth/Qwen3-4B-Instruct-2507-bnb-4bit`, so later runs use `--config`
instead of a long CLI, without baking a specific LoRA checkpoint into git.

## Non-goals

- One-step `qucs-grpo` 4B recipe (same pattern later if needed)
- Changing `training/config.py` schema
- Changing `configs/goal_distribution.yaml` depth/freq sampling
- Committing `outputs/` adapters or dated per-run YAML copies

## Decision: C — stable recipe + CLI for run identity

| Layer | Lives in | Fields |
|---|---|---|
| Stable | `configs/multiturn_qwen3_4b_instruct.yaml` | model, train hardness, data hardness, empty `resume_adapter` |
| Per run | CLI (or an untracked dated YAML copy) | `--resume-adapter`, `--output-dir`, `--start-seed` |

Precedence already implemented: dataclass defaults → YAML → CLI.

When the checkpoint changes, the operator does **not** have to edit the
stable file. They pass a new `--resume-adapter`. Optionally they copy the
stable YAML, fill `runtime.resume_adapter` / `output_dir` / `data.start_seed`,
and point `--config` at the copy without committing it.

## Recipe values (snapshot of the hard 4B run)

- `model.name`: `unsloth/Qwen3-4B-Instruct-2507-bnb-4bit`
- `max_seq_length`: 2048, `lora_rank`: 16
- `steps`: 50, `tasks_per_step`: 2, `generations`: 6
- `max_turns`: 15, `patience`: 5, `patience_eps`: 0.2, `history_window`: 8
- `max_new_tokens`: 256, `temperature`: 1.0, `lr`: 5e-6, `max_grad_norm`: 0.1
- `save_every`: 5, `beta`: 0.01, `kl_ref`: start
- `start_seed`: 5000 (override on continue)
- `goal_config`: `configs/goal_distribution.yaml` (depth −80…−55; no `target_depth_db` field)
- `param_spread`: 0.35, `min_start_headroom_db`: 40.0
- `output_dir`: `outputs/multiturn-qwen3-4b-instruct` (override on continue)
- `resume_adapter`: `""`, `allow_raw_base`: false

GRPO `checkpoint-*` is a valid `--resume-adapter`. Cold-start gate only
requires a non-empty adapter path; `--allow-raw-base` is not required when
resuming `outputs/multiturn-50step-hard-beta001/checkpoint-20`.

## Continue-training example

```bash
python -m training.multiturn_train \
  --config configs/multiturn_qwen3_4b_instruct.yaml \
  --resume-adapter outputs/multiturn-50step-hard-beta001/checkpoint-20 \
  --output-dir outputs/multiturn-from-ckpt20 \
  --start-seed 5286
```

## Test

`tests/test_training_config.py` loads the new YAML and asserts model name,
`steps`/`generations`/`beta`/`min_start_headroom_db`, and empty
`resume_adapter`.
