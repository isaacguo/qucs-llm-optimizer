# Training Hyperparameter YAML Config Implementation Plan

> **ARCHIVED (2026-09-12):** Single-step `GrpoConfig` / `training/grpo.py` were removed.
> Current training entrypoints are `qucs-multiturn` and `qucs-sft` only.
> Keep this file as historical context; do not implement the one-step GRPO sections.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional nested YAML recipes for `grpo` and `multiturn_train` so hyperparameters live in typed dataclasses + files, with CLI overrides, without breaking no-`--config` runs.

**Architecture:** New `training/config.py` owns nested dataclasses, YAML load, unknown-key validation, and CLI merge. Entrypoints resolve a `GrpoConfig` / `MultiturnConfig`, log it, and train from that object. Magic numbers move out of `grpo_config_kwargs`. Example recipes ship under `configs/`.

**Tech Stack:** Python 3.12, stdlib `dataclasses`, `PyYAML` (train extra), existing `unittest` suite (`python -m unittest …`).

## Global Constraints

- Scope only `training/grpo.py` and `training/multiturn_train.py` (no sft/probe/evaluate).
- Precedence: dataclass defaults < YAML < explicit CLI (`default=None` means “do not override”).
- No Hydra / overlays / profiles.
- `--dry-run` stays CLI-only (not a YAML field).
- GRPO `save_steps` stays derived: `max(1, min(25, steps))`.
- Unknown YAML keys must fail; missing sections use defaults.
- Do not load models in unit tests.

## File map

| Path | Responsibility |
|------|----------------|
| `training/config.py` | Schemas, `load_yaml`, `from_mapping`, CLI maps, `resolve_*`, `to_model_config`, `config_to_dict` |
| `training/grpo.py` | `--config`, resolve config, `grpo_config_kwargs(GrpoConfig)`, train/dry-run from config |
| `training/multiturn_train.py` | `--config`, resolve config, `train`/`_dry_run` from `MultiturnConfig` |
| `training/modeling.py` | Unchanged API; callers pass fields from `config.model` |
| `configs/grpo_qwen3_1_7b.yaml` | Reference GRPO recipe (= current defaults) |
| `configs/multiturn_qwen3_1_7b.yaml` | Reference multiturn recipe |
| `tests/test_training_config.py` | Load/merge/validation/recipe tests |
| `tests/test_training_cli.py` | Update for `None` defaults + resolve-config defaults |
| `pyproject.toml` | Add `PyYAML` to `train` extra |

---

### Task 1: Core YAML loader + `GrpoConfig` schema

**Files:**
- Create: `training/config.py`
- Create: `tests/test_training_config.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces:
  - `load_yaml(path: str | Path) -> dict`
  - `from_mapping(cls: type[T], data: dict \| None, path: str = "") -> T`
  - dataclasses: `ModelSection`, `GrpoTrainSection`, `GrpoDataSection`, `GrpoRewardSection`, `GrpoRuntimeSection`, `GrpoConfig`
- Consumes: none

- [ ] **Step 1: Add PyYAML to the train extra**

In `pyproject.toml` under `[project.optional-dependencies] train`, add `"PyYAML",` (keep alphabetical-ish near other deps).

- [ ] **Step 2: Write failing tests for load + Grpo defaults + unknown keys**

```python
"""Tests for training YAML config load/merge (no model load)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from training.config import GrpoConfig, from_mapping, load_yaml


class LoadYamlTests(unittest.TestCase):
    def test_load_yaml_reads_nested_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.yaml"
            path.write_text("model:\n  name: demo-model\n", encoding="utf-8")
            data = load_yaml(path)
        self.assertEqual(data["model"]["name"], "demo-model")

    def test_missing_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_yaml("/no/such/config.yaml")


class GrpoFromMappingTests(unittest.TestCase):
    def test_empty_mapping_uses_current_defaults(self):
        cfg = from_mapping(GrpoConfig, {})
        self.assertEqual(cfg.model.name, "unsloth/Qwen3-1.7B-bnb-4bit")
        self.assertEqual(cfg.train.steps, 100)
        self.assertEqual(cfg.train.generations, 8)
        self.assertEqual(cfg.train.learning_rate, 5e-6)
        self.assertEqual(cfg.reward.beta, 0.01)
        self.assertEqual(cfg.reward.weights, [0.2, 0.2, 1.0])
        self.assertEqual(cfg.data.tasks, 32)
        self.assertEqual(cfg.runtime.output_dir, "outputs/grpo-qwen3-1.7b")

    def test_partial_yaml_overrides_nested_fields(self):
        cfg = from_mapping(
            GrpoConfig,
            {"model": {"name": "unsloth/Qwen3-4B"}, "train": {"learning_rate": 1e-5}},
        )
        self.assertEqual(cfg.model.name, "unsloth/Qwen3-4B")
        self.assertEqual(cfg.train.learning_rate, 1e-5)
        self.assertEqual(cfg.train.steps, 100)

    def test_unknown_key_fails(self):
        with self.assertRaises(ValueError) as ctx:
            from_mapping(GrpoConfig, {"train": {"learnning_rate": 1e-5}})
        self.assertIn("learnning_rate", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run tests — expect FAIL (module missing)**

Run: `python -m unittest tests.test_training_config -v`  
Expected: `ModuleNotFoundError: No module named 'training.config'` (or import errors).

- [ ] **Step 4: Implement `training/config.py` (Grpo schema + load/from_mapping only)**

```python
"""Typed training recipes: YAML load + dataclass schemas + CLI merge."""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


@dataclass
class ModelSection:
    name: str = "unsloth/Qwen3-1.7B-bnb-4bit"
    max_seq_length: int = 1024
    lora_rank: int = 16


@dataclass
class GrpoTrainSection:
    steps: int = 100
    generations: int = 8
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_8bit"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_prompt_length: int = 768
    max_completion_length: int = 256
    max_grad_norm: float = 0.1
    temperature: float = 1.0
    loss_type: str = "dr_grpo"


@dataclass
class GrpoDataSection:
    tasks: int = 32
    start_seed: int = 1000
    max_iteration: int = 12
    goal_freq_min_ghz: float = 4.0
    goal_freq_max_ghz: float = 6.0


@dataclass
class GrpoRewardSection:
    weights: list[float] = field(default_factory=lambda: [0.2, 0.2, 1.0])
    beta: float = 0.01


@dataclass
class GrpoRuntimeSection:
    output_dir: str = "outputs/grpo-qwen3-1.7b"
    sim_workers: int = 2
    use_vllm: bool = False
    resume_from_checkpoint: str = ""


@dataclass
class GrpoConfig:
    model: ModelSection = field(default_factory=ModelSection)
    train: GrpoTrainSection = field(default_factory=GrpoTrainSection)
    data: GrpoDataSection = field(default_factory=GrpoDataSection)
    reward: GrpoRewardSection = field(default_factory=GrpoRewardSection)
    runtime: GrpoRuntimeSection = field(default_factory=GrpoRuntimeSection)


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    try:
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return data


def from_mapping(cls: type[T], data: dict | None, path: str = "") -> T:
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"{path or 'root'} must be a mapping, got {type(data).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        where = path or cls.__name__
        raise ValueError(f"unknown key(s) in {where}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, f in known.items():
        if name not in data:
            continue
        value = data[name]
        if is_dataclass(f.type):
            child_path = f"{path}.{name}" if path else name
            kwargs[name] = from_mapping(f.type, value, path=child_path)
        else:
            kwargs[name] = value
    return cls(**kwargs)
```

Note: `is_dataclass(f.type)` may fail under `from __future__ import annotations` because `f.type` is a string. Implement a small `_resolve_type(f)` using `typing.get_type_hints(cls)[f.name]` instead of `f.type` directly.

- [ ] **Step 5: Run tests — expect PASS**

Run: `python -m unittest tests.test_training_config -v`  
Expected: all PASS (install PyYAML in the env if import fails: `uv pip install PyYAML` or equivalent).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml training/config.py tests/test_training_config.py
git commit -m "Add GrpoConfig schema and YAML loader for training recipes."
```

---

### Task 2: GRPO CLI merge + `resolve_grpo_config` + `grpo_config_kwargs`

**Files:**
- Modify: `training/config.py`
- Modify: `training/grpo.py`
- Modify: `tests/test_training_config.py`
- Modify: `tests/test_training_cli.py`

**Interfaces:**
- Consumes: `GrpoConfig`, `from_mapping`, `load_yaml` from Task 1
- Produces:
  - `GRPO_CLI_FIELD_MAP: dict[str, tuple[str, str]]` mapping argparse dest → `(section, field)`
  - `merge_grpo_cli(config: GrpoConfig, args: Namespace) -> GrpoConfig`
  - `resolve_grpo_config(args: Namespace) -> GrpoConfig`
  - `to_model_config(model: ModelSection, *, fast_inference: bool = False) -> ModelConfig`
  - `config_to_dict(config) -> dict` (for logging)
  - `grpo_config_kwargs(config: GrpoConfig) -> dict` in `grpo.py`

CLI map (argparse `dest` → nested path):

```text
model_name → model.name
max_seq_length → model.max_seq_length   # optional on GRPO parser if useful; else omit until multiturn
tasks → data.tasks
start_seed → data.start_seed
max_iteration → data.max_iteration
goal_freq_min_ghz → data.goal_freq_min_ghz
goal_freq_max_ghz → data.goal_freq_max_ghz
steps → train.steps
generations → train.generations
sim_workers → runtime.sim_workers
output_dir → runtime.output_dir
resume_from_checkpoint → runtime.resume_from_checkpoint
use_vllm → runtime.use_vllm
```

- [ ] **Step 1: Write failing merge / kwargs tests**

Append to `tests/test_training_config.py`:

```python
import argparse
from training.config import merge_grpo_cli, resolve_grpo_config
from training.grpo import build_parser, grpo_config_kwargs


class GrpoMergeTests(unittest.TestCase):
    def test_cli_overrides_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "g.yaml"
            path.write_text(
                "train:\n  steps: 50\n  learning_rate: 1.0e-5\n",
                encoding="utf-8",
            )
            args = build_parser().parse_args(["--config", str(path), "--steps", "3"])
            cfg = resolve_grpo_config(args)
        self.assertEqual(cfg.train.steps, 3)
        self.assertEqual(cfg.train.learning_rate, 1e-5)

    def test_no_config_matches_legacy_defaults(self):
        args = build_parser().parse_args([])
        cfg = resolve_grpo_config(args)
        self.assertEqual(cfg.model.name, "unsloth/Qwen3-1.7B-bnb-4bit")
        self.assertEqual(cfg.train.generations, 8)
        kwargs = grpo_config_kwargs(cfg)
        self.assertEqual(kwargs["learning_rate"], 5e-6)
        self.assertEqual(kwargs["beta"], 0.01)
        self.assertEqual(kwargs["reward_weights"], [0.2, 0.2, 1.0])
        self.assertEqual(kwargs["num_generations"], 8)
```

Update `tests/test_training_cli.py` `GrpoCliTests`:

- `parse_args([])` then `resolve_grpo_config(args)` for default assertions (because argparse defaults become `None`).
- `test_grpo_config_is_conservative_for_eight_gb`: pass resolved config into `grpo_config_kwargs`.

- [ ] **Step 2: Run tests — expect FAIL**

Run: `python -m unittest tests.test_training_config tests.test_training_cli -v`  
Expected: FAIL on missing `resolve_grpo_config` / signature change.

- [ ] **Step 3: Implement merge + resolve in `training/config.py`**

```python
import copy
from training.modeling import ModelConfig

GRPO_CLI_FIELD_MAP: dict[str, tuple[str, str]] = {
    "model_name": ("model", "name"),
    "tasks": ("data", "tasks"),
    "start_seed": ("data", "start_seed"),
    "max_iteration": ("data", "max_iteration"),
    "goal_freq_min_ghz": ("data", "goal_freq_min_ghz"),
    "goal_freq_max_ghz": ("data", "goal_freq_max_ghz"),
    "steps": ("train", "steps"),
    "generations": ("train", "generations"),
    "sim_workers": ("runtime", "sim_workers"),
    "output_dir": ("runtime", "output_dir"),
    "resume_from_checkpoint": ("runtime", "resume_from_checkpoint"),
    "use_vllm": ("runtime", "use_vllm"),
}


def merge_grpo_cli(config: GrpoConfig, args: argparse.Namespace) -> GrpoConfig:
    cfg = copy.deepcopy(config)
    for dest, (section, name) in GRPO_CLI_FIELD_MAP.items():
        value = getattr(args, dest, None)
        if value is None:
            continue
        setattr(getattr(cfg, section), name, value)
    return cfg


def resolve_grpo_config(args: argparse.Namespace) -> GrpoConfig:
    config_path = getattr(args, "config", None)
    if config_path:
        cfg = from_mapping(GrpoConfig, load_yaml(config_path))
    else:
        cfg = GrpoConfig()
    return merge_grpo_cli(cfg, args)


def to_model_config(model: ModelSection, *, fast_inference: bool = False) -> ModelConfig:
    return ModelConfig(
        model_name=model.name,
        max_seq_length=model.max_seq_length,
        lora_rank=model.lora_rank,
        fast_inference=fast_inference,
    )


def config_to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: config_to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, list):
        return [config_to_dict(x) for x in obj]
    return obj
```

- [ ] **Step 4: Wire `training/grpo.py`**

1. Add `--config` (default `None`).
2. Change hyperparameter flags to `default=None` (keep `type=` / `action=`). For `use_vllm`, use `action="store_true", default=None`.
3. Keep `--dry-run` as `store_true` (always bool; not in YAML).
4. Replace `grpo_config_kwargs`:

```python
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
```

5. In `main`: `args = parse…`; `cfg = resolve_grpo_config(args)`; `print("config:", json.dumps(config_to_dict(cfg), sort_keys=True))`; dry-run and train use `cfg` (+ `args.dry_run` only from args).

`_dry_run(cfg: GrpoConfig)` uses `cfg.data.*` and `cfg.runtime.output_dir`.

Train path:

```python
ModelConfig via to_model_config(cfg.model, fast_inference=cfg.runtime.use_vllm)
GRPOConfig(**grpo_config_kwargs(cfg))
resume_from_checkpoint=cfg.runtime.resume_from_checkpoint or None
```

- [ ] **Step 5: Run tests — expect PASS**

Run: `python -m unittest tests.test_training_config tests.test_training_cli -v`

- [ ] **Step 6: Commit**

```bash
git add training/config.py training/grpo.py tests/test_training_config.py tests/test_training_cli.py
git commit -m "Wire GRPO trainer to resolved YAML/CLI hyperparameter config."
```

---

### Task 3: `MultiturnConfig` schema + resolve/merge

**Files:**
- Modify: `training/config.py`
- Modify: `tests/test_training_config.py`

**Interfaces:**
- Produces:
  - `MultiturnTrainSection`, `MultiturnDataSection`, `MultiturnRuntimeSection`, `MultiturnConfig`
  - `MULTITURN_CLI_FIELD_MAP`
  - `merge_multiturn_cli`, `resolve_multiturn_config`
- Note: multiturn `model.max_seq_length` default **2048** (not 1024). Implement via `MultiturnConfig` defaulting `model=ModelSection(max_seq_length=2048)` or a dedicated factory — do not change GRPO’s 1024 default on shared `ModelSection` defaults incorrectly.
  Recommended:

```python
@dataclass
class MultiturnConfig:
    model: ModelSection = field(
        default_factory=lambda: ModelSection(max_seq_length=2048)
    )
    train: MultiturnTrainSection = field(default_factory=MultiturnTrainSection)
    data: MultiturnDataSection = field(default_factory=MultiturnDataSection)
    reward: dict = field(default_factory=dict)  # unused placeholder OR empty dataclass
    runtime: MultiturnRuntimeSection = field(default_factory=MultiturnRuntimeSection)
```

Prefer an empty `@dataclass class MultiturnRewardSection: pass` (or no fields) so `from_mapping` stays uniform — **not** a raw `dict` field if that breaks unknown-key handling. Spec shows `reward: {}`; empty dataclass accepting `{}` is fine.

Field defaults must match today’s multiturn CLI (see spec YAML).

CLI map highlights:

```text
model_name → model.name
max_seq_length → model.max_seq_length
tasks_per_step, generations, max_turns, patience, patience_eps,
history_window, max_new_tokens, temperature, lr, max_grad_norm,
save_every, beta, kl_ref → train.*
start_seed, goal_freq_*, target_depth_db, param_spread,
min_start_headroom_db → data.*
output_dir, resume_adapter → runtime.*
```

- [ ] **Step 1: Failing tests for multiturn defaults + CLI override**

```python
from training.config import MultiturnConfig, resolve_multiturn_config
from training.multiturn_train import build_parser as build_multiturn_parser

class MultiturnConfigTests(unittest.TestCase):
    def test_defaults_match_enhanced_training_setup(self):
        cfg = from_mapping(MultiturnConfig, {})
        self.assertEqual(cfg.train.max_turns, 15)
        self.assertEqual(cfg.train.patience, 5)
        self.assertEqual(cfg.data.target_depth_db, -30.0)
        self.assertEqual(cfg.train.generations, 4)
        self.assertEqual(cfg.model.max_seq_length, 2048)
        self.assertEqual(cfg.train.beta, 0.0)
        self.assertEqual(cfg.train.kl_ref, "start")

    def test_resolve_cli_overrides_yaml(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.yaml"
            path.write_text("train:\n  steps: 9\n  beta: 0.01\n", encoding="utf-8")
            args = build_multiturn_parser().parse_args(
                ["--config", str(path), "--steps", "2", "--kl-ref", "base"]
            )
            cfg = resolve_multiturn_config(args)
        self.assertEqual(cfg.train.steps, 2)
        self.assertEqual(cfg.train.beta, 0.01)
        self.assertEqual(cfg.train.kl_ref, "base")
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement Multiturn dataclasses + merge/resolve** (mirror GRPO helpers; reuse `ModelSection`).

- [ ] **Step 4: Run — expect PASS** for these tests (multiturn entrypoint may still use old defaults until Task 4).

For Step 4 to pass `build_multiturn_parser().parse_args(["--config", …])`, Task 3 must either:
- add a stub `--config` argument on the multiturn parser in this task, or
- construct a Namespace manually in the test until Task 4.

**Prefer:** add `--config` + set hyperparameter defaults to `None` on the multiturn parser in Task 3 Step 3/4 together with schema — but leave `train()` reading `args` until Task 4. Then `resolve_multiturn_config` works; `main` still uses old path briefly.

Minimal parser change in Task 3:

```python
parser.add_argument("--config", default=None)
# change hyperparameter defaults to None; keep --dry-run as store_true
```

Update `tests/test_training_cli.py` `MultiturnCliTests` to assert via `resolve_multiturn_config(parse_args([]))` instead of raw argparse defaults.

- [ ] **Step 5: Commit**

```bash
git add training/config.py training/multiturn_train.py tests/test_training_config.py tests/test_training_cli.py
git commit -m "Add MultiturnConfig schema and CLI/YAML resolve helpers."
```

---

### Task 4: Wire `multiturn_train.py` to `MultiturnConfig`

**Files:**
- Modify: `training/multiturn_train.py`
- Modify: `tests/test_training_cli.py` (if any remain)

**Interfaces:**
- Consumes: `resolve_multiturn_config`, `to_model_config`, `config_to_dict`
- Produces: `train(config: MultiturnConfig)`, `_dry_run(config: MultiturnConfig)`, `_make_generate_fn(..., config)`

- [ ] **Step 1: Write failing test that `train` takes `config`**

```python
def test_train_accepts_config_parameter(self):
    import inspect
    from training import multiturn_train

    params = inspect.signature(multiturn_train.train).parameters
    self.assertIn("config", params)
```

Do not add Qucs dry-run integration tests here.

- [ ] **Step 2: Refactor multiturn to consume `MultiturnConfig`**

In `main`:

```python
args = build_parser().parse_args(argv)
cfg = resolve_multiturn_config(args)
print("config:", json.dumps(config_to_dict(cfg), sort_keys=True))
...
if args.dry_run:
    _dry_run(cfg)
    return
train(cfg)
```

Replace every `args.<hyper>` in `train` / `_dry_run` / `_make_generate_fn` with `config.train|data|runtime|model` fields. Keep using local names for readability if helpful (`t = config.train`).

`load_policy(to_model_config(config.model))`.

- [ ] **Step 3: Run unit tests**

Run: `python -m unittest tests.test_training_config tests.test_training_cli -v`  
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add training/multiturn_train.py tests/test_training_config.py tests/test_training_cli.py
git commit -m "Drive multiturn GRPO training from MultiturnConfig recipes."
```

---

### Task 5: Ship example YAML recipes + load regression tests

**Files:**
- Create: `configs/grpo_qwen3_1_7b.yaml`
- Create: `configs/multiturn_qwen3_1_7b.yaml`
- Modify: `tests/test_training_config.py`

**Interfaces:**
- Consumes: `from_mapping`, `load_yaml`, `GrpoConfig`, `MultiturnConfig`
- Produces: checked-in recipes matching dataclass defaults

- [ ] **Step 1: Failing test that recipe files load**

```python
ROOT = Path(__file__).resolve().parents[1]

class ExampleRecipeTests(unittest.TestCase):
    def test_grpo_example_loads(self):
        data = load_yaml(ROOT / "configs/grpo_qwen3_1_7b.yaml")
        cfg = from_mapping(GrpoConfig, data)
        self.assertEqual(cfg.model.name, "unsloth/Qwen3-1.7B-bnb-4bit")
        self.assertEqual(cfg.train.steps, 100)

    def test_multiturn_example_loads(self):
        data = load_yaml(ROOT / "configs/multiturn_qwen3_1_7b.yaml")
        cfg = from_mapping(MultiturnConfig, data)
        self.assertEqual(cfg.train.max_turns, 15)
        self.assertEqual(cfg.model.max_seq_length, 2048)
```

- [ ] **Step 2: Run — expect FAIL (missing files)**

- [ ] **Step 3: Write YAML files** matching the approved spec skeletons exactly (GRPO + multiturn blocks in `docs/superpowers/specs/2026-09-10-training-hyperparam-config-design.md`).

- [ ] **Step 4: Run full relevant suite — expect PASS**

Run: `python -m unittest tests.test_training_config tests.test_training_cli -v`

- [ ] **Step 5: Commit**

```bash
git add configs/grpo_qwen3_1_7b.yaml configs/multiturn_qwen3_1_7b.yaml tests/test_training_config.py
git commit -m "Add reference Qwen3-1.7B YAML training recipes."
```

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| Optional `--config`, no-config = today’s defaults | 2, 3, 4 |
| Nested YAML sections | 1, 3, 5 |
| Dataclass defaults < YAML < CLI | 2, 3 |
| Unknown keys / missing path / bad YAML errors | 1 |
| `grpo_config_kwargs` from config (no magic numbers) | 2 |
| `ModelConfig` from `config.model` | 2, 4 |
| Example recipes under `configs/` | 5 |
| PyYAML in train extra | 1 |
| Log resolved config at startup | 2, 4 |
| `--dry-run` CLI-only | 2, 4 |
| Derived `save_steps` | 2 |
| No sft/probe/evaluate migration | (explicit non-goal) |

## Out of scope (do not implement in this plan)

- Hydra, overlays, profiles
- Writing `resolved_config.yaml` to `output_dir`
- Config for other entrypoints
