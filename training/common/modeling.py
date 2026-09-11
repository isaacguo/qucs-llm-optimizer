"""HF + PEFT + bitsandbytes policy setup, isolated from the simulator package."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit"
    max_seq_length: int = 2048
    lora_rank: int = 16
    load_in_4bit: bool = True


@dataclass(frozen=True)
class LoadHooks:
    load_tokenizer: Callable[[ModelConfig], Any]
    load_model: Callable[[ModelConfig], Any]
    prepare_model: Callable[[Any, ModelConfig], Any]
    apply_lora: Callable[[Any, ModelConfig], Any]


def mixed_precision_config(torch_module=None) -> dict[str, bool]:
    if torch_module is None:
        import torch as torch_module

    use_bf16 = bool(
        torch_module.cuda.is_available()
        and torch_module.cuda.is_bf16_supported()
    )
    return {"bf16": use_bf16, "fp16": not use_bf16}


def _default_compute_dtype():
    import torch

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _default_hooks() -> LoadHooks:
    def load_tokenizer(config: ModelConfig):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            use_fast=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = config.max_seq_length
        return tokenizer

    def load_model(config: ModelConfig):
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        quant_config = None
        if config.load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=_default_compute_dtype(),
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=quant_config,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=_default_compute_dtype(),
            trust_remote_code=True,
        )
        if hasattr(model, "config"):
            model.config.use_cache = False
        return model

    def prepare_model(model, config: ModelConfig):
        from peft import prepare_model_for_kbit_training

        if config.load_in_4bit:
            return prepare_model_for_kbit_training(model)
        return model

    def apply_lora(model, config: ModelConfig):
        from peft import LoraConfig, get_peft_model, TaskType

        lora = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_rank,
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(model, lora)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        return model

    return LoadHooks(
        load_tokenizer=load_tokenizer,
        load_model=load_model,
        prepare_model=prepare_model,
        apply_lora=apply_lora,
    )


def load_policy(config: ModelConfig, hooks: LoadHooks | None = None):
    """Load a 4-bit causal LM and attach LoRA via Hugging Face + PEFT."""
    active = hooks or _default_hooks()
    tokenizer = active.load_tokenizer(config)
    model = active.load_model(config)
    model = active.prepare_model(model, config)
    model = active.apply_lora(model, config)
    return model, tokenizer
