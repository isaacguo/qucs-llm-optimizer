"""Typed training recipes: YAML load + dataclass schemas + CLI merge."""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import yaml

from training.common.modeling import ModelConfig

T = TypeVar("T")


@dataclass
class MultiturnModelSection:
    name: str = "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit"
    max_seq_length: int = 2048
    lora_rank: int = 16


@dataclass
class MultiturnTrainSection:
    steps: int = 5
    tasks_per_step: int = 2
    generations: int = 4
    max_turns: int = 15
    patience: int = 5
    patience_eps: float = 0.2
    history_window: int = 8
    max_new_tokens: int = 256
    temperature: float = 1.0
    lr: float = 5e-6
    max_grad_norm: float = 0.1
    save_every: int = 5
    beta: float = 0.0
    kl_ref: str = "start"


@dataclass
class MultiturnDataSection:
    start_seed: int = 5000
    goal_config: str = "configs/goal_distribution.yaml"
    # Optional CLI overrides that mutate a loaded distribution copy (None = use YAML).
    goal_freq_min_ghz: float | None = None
    goal_freq_max_ghz: float | None = None
    param_spread: float = 0.35
    min_start_headroom_db: float = 5.0


@dataclass
class MultiturnRewardSection:
    """Reserved YAML section (`reward: {}`); knobs live in ``reward_math`` today."""

    pass


@dataclass
class MultiturnRuntimeSection:
    output_dir: str = "outputs/multiturn-grpo-demo"
    resume_adapter: str = ""
    allow_raw_base: bool = False


@dataclass
class MultiturnConfig:
    model: MultiturnModelSection = field(default_factory=MultiturnModelSection)
    train: MultiturnTrainSection = field(default_factory=MultiturnTrainSection)
    data: MultiturnDataSection = field(default_factory=MultiturnDataSection)
    reward: MultiturnRewardSection = field(default_factory=MultiturnRewardSection)
    runtime: MultiturnRuntimeSection = field(default_factory=MultiturnRuntimeSection)


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


MULTITURN_CLI_FIELD_MAP: dict[str, tuple[str, str]] = {
    "model_name": ("model", "name"),
    "max_seq_length": ("model", "max_seq_length"),
    "tasks_per_step": ("train", "tasks_per_step"),
    "generations": ("train", "generations"),
    "max_turns": ("train", "max_turns"),
    "patience": ("train", "patience"),
    "patience_eps": ("train", "patience_eps"),
    "history_window": ("train", "history_window"),
    "max_new_tokens": ("train", "max_new_tokens"),
    "temperature": ("train", "temperature"),
    "lr": ("train", "lr"),
    "max_grad_norm": ("train", "max_grad_norm"),
    "save_every": ("train", "save_every"),
    "beta": ("train", "beta"),
    "kl_ref": ("train", "kl_ref"),
    "steps": ("train", "steps"),
    "start_seed": ("data", "start_seed"),
    "goal_config": ("data", "goal_config"),
    "goal_freq_min_ghz": ("data", "goal_freq_min_ghz"),
    "goal_freq_max_ghz": ("data", "goal_freq_max_ghz"),
    "param_spread": ("data", "param_spread"),
    "min_start_headroom_db": ("data", "min_start_headroom_db"),
    "output_dir": ("runtime", "output_dir"),
    "resume_adapter": ("runtime", "resume_adapter"),
    "allow_raw_base": ("runtime", "allow_raw_base"),
}


def merge_multiturn_cli(config: MultiturnConfig, args: argparse.Namespace) -> MultiturnConfig:
    cfg = copy.deepcopy(config)
    for dest, (section, name) in MULTITURN_CLI_FIELD_MAP.items():
        value = getattr(args, dest, None)
        if value is None:
            continue
        setattr(getattr(cfg, section), name, value)
    return cfg


def resolve_multiturn_config(args: argparse.Namespace) -> MultiturnConfig:
    config_path = getattr(args, "config", None)
    if config_path:
        cfg = from_mapping(MultiturnConfig, load_yaml(config_path))
    else:
        cfg = MultiturnConfig()
    return merge_multiturn_cli(cfg, args)


def to_model_config(model: MultiturnModelSection) -> ModelConfig:
    return ModelConfig(
        model_name=model.name,
        max_seq_length=model.max_seq_length,
        lora_rank=model.lora_rank,
    )


def config_to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: config_to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, list):
        return [config_to_dict(x) for x in obj]
    return obj
