# Unsloth GRPO Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a working Unsloth training framework that trains `unsloth/Qwen3-1.7B-bnb-4bit` to emit one-step qualitative intent JSON, scored by the real Qucs simulator.

**Architecture:** Keep Qucs simulation and intent arithmetic independent of ML dependencies. A new `training` package builds randomized one-step states, parses model completions, and exposes TRL-compatible reward functions that execute the real `apply_intent -> simulate -> evaluate` transition. Unsloth owns model loading, QLoRA, generation, SFT, and GRPO optimization.

**Tech Stack:** Python 3.12, uv, Unsloth, TRL, Hugging Face datasets, Qwen3-1.7B 4-bit QLoRA, Qucs-S/qucsator_rf.

---

### Task 1: Project and training environment

**Files:**
- Create: `pyproject.toml`
- Modify: `.gitignore`

- [ ] Define a lightweight project and a `train` dependency group containing `unsloth`, `trl`, `datasets`, and `accelerate`.
- [ ] Generate `uv.lock` with `uv sync --extra train`.
- [ ] Verify the uv environment can import `torch`, `unsloth`, `trl`, and detect CUDA.

### Task 2: Simulation mode for training

**Files:**
- Modify: `src/qucs_sim.py`
- Modify: `tests/test_qucs_sim.py`

- [ ] Write a failing test showing `simulate(..., export_layout=False)` skips `export_layout_svg` but still returns parsed S-parameters.
- [ ] Add the optional `export_layout` argument, defaulting to `True` so existing CLI behavior remains unchanged.
- [ ] Run the simulation tests and one real no-layout Qucs transition.

### Task 3: Intent completion contract

**Files:**
- Create: `training/__init__.py`
- Create: `training/contracts.py`
- Create: `tests/test_training_contracts.py`

- [ ] Write failing tests for fenced JSON, reasoning followed by JSON, invalid variables, invalid tokens, raw numeric values, and empty/all-hold actions.
- [ ] Implement completion extraction and strict intent validation against `VARIABLES` and the seven allowed tokens.
- [ ] Verify all contract tests pass.

### Task 4: Randomized real-Qucs tasks and one-step transition

**Files:**
- Create: `training/environment.py`
- Create: `tests/test_training_environment.py`

- [ ] Write failing tests for deterministic seeded task generation, bounds, prompt contents, and dependency-injected transitions.
- [ ] Implement task records carrying prompt, parameters, iteration, baseline cost, and seed.
- [ ] Implement one-step transition using `apply_intent`, `simulate(export_layout=False)`, and `evaluate`.
- [ ] Verify a real generated task produces a finite baseline and action reward.

### Task 5: GRPO reward functions

**Files:**
- Create: `training/rewards.py`
- Create: `tests/test_training_rewards.py`

- [ ] Write failing tests for conversational/flat completions, format reward, anti-hack reward, simulation improvement reward, invalid completion handling, and per-generation parameter alignment.
- [ ] Implement TRL-compatible reward callables returning one float per completion.
- [ ] Add structured JSONL reward logging without mutating the checked-in run history.
- [ ] Verify reward functions against the real simulator for a small completion group.

### Task 6: Dataset and model setup

**Files:**
- Create: `training/data.py`
- Create: `training/modeling.py`
- Create: `tests/test_training_data.py`

- [ ] Write failing tests for GRPO dataset columns and SFT extraction from `runs/llm1/state.json`.
- [ ] Build randomized prompts without answer labels for GRPO.
- [ ] Build optional SFT examples from recorded observation/thinking/intent entries.
- [ ] Configure Qwen3-1.7B 4-bit LoRA with a conservative 8 GB profile.

### Task 7: Executable entry points

**Files:**
- Create: `training/grpo.py`
- Create: `training/sft.py`
- Create: `training/probe.py`
- Create: `tests/test_training_cli.py`
- Modify: `pyproject.toml`

- [ ] Write failing CLI/config tests for dry-run, model overrides, generation count, output directory, and simulator preflight.
- [ ] Implement `qucs-grpo`, `qucs-sft`, and `qucs-probe` entry points.
- [ ] Make GRPO the default workflow; keep SFT optional.
- [ ] Ensure every long-running entry point logs continuously under `outputs/logs/`.

### Task 8: End-to-end verification and documentation

**Files:**
- Modify: `README.md`

- [ ] Run the complete unit and integration test suite.
- [ ] Run `qucs-probe` with the real model and real Qucs simulator.
- [ ] Run a minimal real Unsloth GRPO smoke job and confirm a LoRA checkpoint plus nonzero simulator rewards.
- [ ] Document install, preflight, probe, optional SFT, GRPO, resume, and inference commands.
- [ ] Inspect `git diff` and verify no model weights, caches, or temporary simulator artifacts are tracked.
