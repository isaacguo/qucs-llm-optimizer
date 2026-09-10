# Training hyperparameter YAML config

**Date:** 2026-09-10  
**Status:** Approved (user confirmed approach + design sections)

## Goal

Make training hyperparameters maintainable and reproducible when switching base models (e.g. Qwen3-1.7B → Qwen3-4B Instruct) by introducing optional nested YAML recipe files for the main trainers, without breaking existing CLI-only invocations.

## Non-goals

- Config for `sft`, `probe`, or `evaluate_*` (deferred; same pattern later)
- Hydra / OmegaConf / multi-file overlay / named profiles inside one file
- Simulator / Qucs path configuration inside training recipes
- Forcing `--config` on every run

## Decisions (locked)

| Topic | Choice |
|-------|--------|
| Scope | `training/grpo.py` + `training/multiturn_train.py` only |
| Layout | One YAML file per experiment/model; invoke with `--config path` |
| Precedence | Dataclass defaults → YAML (if provided) → CLI overrides for explicitly set flags |
| Optional config | No `--config` keeps today’s behavior |
| Format | YAML (add `PyYAML` to the `train` extra) |
| Schema shape | Nested sections: `model` / `train` / `data` / `reward` / `runtime` |
| Implementation | Typed nested dataclasses + load/merge helpers (`training/config.py`) |

## Design

### Architecture

Add a thin config layer that owns schema, YAML loading, and CLI merge. Train entrypoints consume one merged typed config instead of splitting knobs between `argparse` defaults and hard-coded dicts (notably `grpo_config_kwargs`).

```text
configs/
  grpo_qwen3_1_7b.yaml         # reference recipe = current defaults
  multiturn_qwen3_1_7b.yaml
  # copy/rename for new models, e.g. grpo_qwen3_4b_instruct.yaml
```

### Schema (conceptual)

**GRPO** (illustrative; field set must cover today’s CLI + values currently hard-coded in `grpo_config_kwargs`):

```yaml
model:
  name: unsloth/Qwen3-1.7B-bnb-4bit
  max_seq_length: 1024
  lora_rank: 16
train:
  steps: 100
  generations: 8
  learning_rate: 5.0e-6
  weight_decay: 0.01
  warmup_ratio: 0.1
  lr_scheduler_type: cosine
  optim: adamw_8bit
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 4
  max_prompt_length: 768
  max_completion_length: 256
  max_grad_norm: 0.1
  temperature: 1.0
  loss_type: dr_grpo
data:
  tasks: 32
  start_seed: 1000
  max_iteration: 12
  goal_freq_min_ghz: 4.0
  goal_freq_max_ghz: 6.0
reward:
  weights: [0.2, 0.2, 1.0]
  beta: 0.01
runtime:
  output_dir: outputs/grpo-qwen3-1.7b
  sim_workers: 2
  use_vllm: false
  resume_from_checkpoint: ""
```

**Multiturn** (same section style; no GRPO-Trainer-only keys):

```yaml
model:
  name: unsloth/Qwen3-1.7B-bnb-4bit
  max_seq_length: 2048
  lora_rank: 16
train:
  steps: 5
  tasks_per_step: 2
  generations: 4
  max_turns: 15
  patience: 5
  patience_eps: 0.2
  history_window: 8
  max_new_tokens: 256
  temperature: 1.0
  lr: 5.0e-6
  max_grad_norm: 0.1
  save_every: 5
  beta: 0.0
  kl_ref: start
data:
  start_seed: 5000
  goal_freq_min_ghz: 4.0
  goal_freq_max_ghz: 6.0
  target_depth_db: -30.0
  param_spread: 0.35
  min_start_headroom_db: 5.0
reward: {}
runtime:
  output_dir: outputs/multiturn-grpo-demo
  resume_adapter: ""
```

Shared mapping: `model.name` → `ModelConfig.model_name`; `model.lora_rank` / `max_seq_length` → `ModelConfig`.

**CLI-only flags (not recipe fields):** `--dry-run` (and any future one-shot debug switches). Derived Trainer fields such as GRPO `save_steps` may stay computed from `train.steps` rather than duplicated in YAML.

### Components

1. **`training/config.py`**
   - Nested dataclasses: `GrpoConfig`, `MultiturnConfig` (+ section types)
   - `load_yaml(path) -> dict`
   - `from_mapping(...)` — unknown keys fail; missing keys use dataclass defaults
   - `merge_cli(config, args)` — override only fields the user explicitly set on the CLI
   - Helpers as needed for tests / debug dumps of the final config

2. **Entrypoint changes (`grpo`, `multiturn_train` only)**
   - Add `--config`
   - Keep existing flags for compatibility; use `default=None` (or equivalent) so “omitted” means “do not override”
   - `main`: parse → optional YAML load → merge → train from config
   - `grpo_config_kwargs(config)` reads the typed config (no second copy of magic numbers)
   - Build `ModelConfig` from `config.model`

3. **Example recipes** under `configs/` matching current defaults so copying a file is the swap-model workflow

4. **Dependency:** `PyYAML` in `[project.optional-dependencies] train`

### Data flow

```text
CLI argv → argparse (most defaults None)
              ├─ --config? → YAML → dataclass (file + defaults)
              └─ explicit flags → merge_cli
                                   ↓
                         train(config) / Trainer kwargs
```

Priority: **dataclass defaults < YAML < explicit CLI**.

### Error handling

| Case | Behavior |
|------|----------|
| Missing `--config` path | Fail fast with path in the message |
| YAML syntax error | Fail with file context |
| Unknown key (any nesting level) | Fail (no silent typos) |
| Wrong type for a field | Fail with field path |
| Partial YAML (missing section) | Allowed; fill from dataclass defaults |
| No `--config` | Do not read a file; full dataclass defaults (= today’s numbers) |
| YAML vs CLI conflict | CLI wins; log the final resolved config at startup for reproducibility |

### Testing

Unit tests without loading a model (extend CLI tests and/or add `tests/test_training_config.py`):

1. No `--config` → values match today’s defaults (regression lock)
2. YAML-only overrides (e.g. `model.name`, `train.learning_rate`) apply
3. YAML + CLI → explicit `--steps` (etc.) overrides the file
4. Unknown key / bad path / bad type → errors
5. `grpo_config_kwargs(config)` sources `learning_rate`, `beta`, `reward_weights` from config
6. Shipped example YAMLs load into complete configs

### Success criteria

- Switching models is “copy YAML, edit `model.name` (+ knobs), pass `--config`”
- Existing no-`--config` commands keep working
- Single source of truth for hyperparameter defaults: dataclass schema (+ recipes), not duplicated magic numbers in trainer helpers

## Out of scope follow-ups

- Migrate `sft` / `probe` / evaluators to the same loader
- Optional config composition (base + model overlay)
- Dumping resolved config next to `output_dir` as `resolved_config.yaml` (nice-to-have; startup log is enough for v1)
