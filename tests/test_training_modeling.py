"""Tests for conservative 8 GB Unsloth model setup."""
from __future__ import annotations

import unittest

from training.modeling import ModelConfig, load_policy, mixed_precision_config


class FakeBackend:
    pretrained_kwargs = None
    peft_kwargs = None

    @classmethod
    def from_pretrained(cls, **kwargs):
        cls.pretrained_kwargs = kwargs
        return "model", "tokenizer"

    @classmethod
    def get_peft_model(cls, model, **kwargs):
        cls.peft_kwargs = kwargs
        return f"peft:{model}"


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

    def test_defaults_target_qwen3_4b_instruct_on_eight_gb_profile(self):
        config = ModelConfig()
        self.assertEqual(config.model_name, "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit")
        self.assertEqual(config.max_seq_length, 2048)
        self.assertEqual(config.lora_rank, 16)
        self.assertFalse(config.fast_inference)

    def test_loads_four_bit_model_and_attaches_lora(self):
        model, tokenizer = load_policy(ModelConfig(), backend=FakeBackend)
        self.assertEqual(model, "peft:model")
        self.assertEqual(tokenizer, "tokenizer")
        self.assertTrue(FakeBackend.pretrained_kwargs["load_in_4bit"])
        self.assertFalse(FakeBackend.pretrained_kwargs["fast_inference"])
        self.assertEqual(FakeBackend.peft_kwargs["r"], 16)
        self.assertIn("q_proj", FakeBackend.peft_kwargs["target_modules"])
        self.assertEqual(
            FakeBackend.peft_kwargs["use_gradient_checkpointing"],
            "unsloth",
        )


if __name__ == "__main__":
    unittest.main()
