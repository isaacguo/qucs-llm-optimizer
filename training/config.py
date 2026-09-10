"""Typed training recipes: YAML load + dataclass schemas + CLI merge."""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import yaml

from training.modeling import ModelConfig

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
    # ``from __future__ import annotations`` makes ``f.type`` a string; resolve hints.
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name, f in known.items():
        if name not in data:
            continue
        value = data[name]
        resolved = hints[name]
        if is_dataclass(resolved):
            child_path = f"{path}.{name}" if path else name
            kwargs[name] = from_mapping(resolved, value, path=child_path)
        else:
            kwargs[name] = value
    return cls(**kwargs)


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
