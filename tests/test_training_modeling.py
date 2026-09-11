"""Tests for HF + PEFT + bitsandbytes policy loading."""
from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any

from training.common.modeling import (
    ModelConfig,
    LoadHooks,
    load_policy,
    mixed_precision_config,
)


class FakeCuda:
    def __init__(self, supports_bf16: bool):
        self.supports_bf16 = supports_bf16

    def is_available(self):
        return True

    def is_bf16_supported(self):
        return self.supports_bf16


class FakeTorch:
    def __init__(self, supports_bf16: bool):
        self.cuda = FakeCuda(supports_bf16)


@dataclass
class RecordingHooks:
    calls: dict[str, Any]

    def __post_init__(self):
        self.calls = {}

    def as_hooks(self) -> LoadHooks:
        def load_tokenizer(config: ModelConfig):
            self.calls["tokenizer_config"] = config
            return "tokenizer"

        def load_model(config: ModelConfig):
            self.calls["model_config"] = config
            return "base-model"

        def prepare_model(model, config: ModelConfig):
            self.calls["prepare"] = (model, config.lora_rank)
            return f"prepared:{model}"

        def apply_lora(model, config: ModelConfig):
            self.calls["lora"] = (model, config.lora_rank, config.max_seq_length)
            return f"peft:{model}"

        return LoadHooks(
            load_tokenizer=load_tokenizer,
            load_model=load_model,
            prepare_model=prepare_model,
            apply_lora=apply_lora,
        )


class ModelSetupTests(unittest.TestCase):
    def test_mixed_precision_tracks_gpu_bf16_support(self):
        self.assertEqual(
            mixed_precision_config(FakeTorch(supports_bf16=True)),
            {"bf16": True, "fp16": False},
        )
        self.assertEqual(
            mixed_precision_config(FakeTorch(supports_bf16=False)),
            {"bf16": False, "fp16": True},
        )

    def test_defaults_target_qwen3_1_7b_on_eight_gb_profile(self):
        config = ModelConfig()
        self.assertEqual(config.model_name, "unsloth/Qwen3-1.7B-bnb-4bit")
        self.assertEqual(config.max_seq_length, 1024)
        self.assertEqual(config.lora_rank, 16)
        self.assertTrue(config.load_in_4bit)

    def test_load_policy_uses_injected_hf_hooks(self):
        recorder = RecordingHooks(calls={})
        model, tokenizer = load_policy(ModelConfig(), hooks=recorder.as_hooks())
        self.assertEqual(tokenizer, "tokenizer")
        self.assertEqual(model, "peft:prepared:base-model")
        self.assertEqual(recorder.calls["lora"][1], 16)
        self.assertEqual(recorder.calls["prepare"][0], "base-model")
        self.assertIsInstance(recorder.as_hooks(), LoadHooks)


if __name__ == "__main__":
    unittest.main()
