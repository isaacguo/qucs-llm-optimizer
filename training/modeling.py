"""Unsloth Qwen3 policy setup, isolated from the simulator package."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit"
    max_seq_length: int = 2048
    lora_rank: int = 16
    fast_inference: bool = False
    gpu_memory_utilization: float = 0.45
    random_state: int = 3407


def mixed_precision_config(torch_module=None) -> dict[str, bool]:
    if torch_module is None:
        import torch as torch_module

    use_bf16 = bool(
        torch_module.cuda.is_available()
        and torch_module.cuda.is_bf16_supported()
    )
    return {"bf16": use_bf16, "fp16": not use_bf16}


def load_policy(config: ModelConfig, backend=None):
    if backend is None:
        from unsloth import FastLanguageModel

        backend = FastLanguageModel

    model, tokenizer = backend.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=True,
        fast_inference=config.fast_inference,
        max_lora_rank=config.lora_rank,
        gpu_memory_utilization=config.gpu_memory_utilization,
    )
    model = backend.get_peft_model(
        model,
        r=config.lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=config.lora_rank,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.random_state,
        use_rslora=False,
    )
    return model, tokenizer

