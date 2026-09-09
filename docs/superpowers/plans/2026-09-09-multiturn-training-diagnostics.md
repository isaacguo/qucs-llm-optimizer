# Multiturn Diagnostics + Reward/Start Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add shared decision JSONL logging (multiturn + agent), richer step_stats, start headroom filtering, patience/max-turns defaults, reward clip 20, and +1 stop bonus.

**Architecture:** Put JSONL schema in `src/decision_log.py` so `training/multiturn_train.py` and `run_step.py` share one writer. Reward/start logic stays in `training/rollout.py` / a small start helper; trainer only orchestrates.

**Tech Stack:** Python 3.12, existing unittest suite, no new dependencies.

## Global Constraints

- Do not add one-step warm-up wiring or offline summarize scripts.
- Do not add destroy-without-stop penalty.
- Keep `train.log` free of full prompt/completion dumps.
- Agent writes `runs/<run>/completions.jsonl`; multiturn writes under `--output-dir`.

---

### Task 1: Shared decision_log

**Files:**
- Create: `src/decision_log.py`
- Create: `tests/test_decision_log.py`

- [ ] **Step 1:** Write failing tests for `format_agent_completion` and `append_record`.
- [ ] **Step 2:** Implement `decision_log.py`.
- [ ] **Step 3:** Run tests; confirm pass.

### Task 2: Reward clip + stop bonus

**Files:**
- Modify: `training/rollout.py`
- Modify: `tests/test_training_rollout.py`

- [ ] **Step 1:** Failing tests for default clip 20 and stop bonus (+1 when legal stop with improvement).
- [ ] **Step 2:** Implement in `mixed_terminal_reward` / `run_trajectory` end.
- [ ] **Step 3:** Run rollout tests.

### Task 3: Start headroom filter

**Files:**
- Create or modify: `training/starts.py` (or `environment.py`)
- Modify: `tests/test_training_rollout.py` or new `tests/test_training_starts.py`
- Modify: `training/multiturn_train.py`

- [ ] **Step 1:** Failing tests for `is_start_too_deep` and resample counting.
- [ ] **Step 2:** Implement helper; wire into multiturn group sampling with CLI `--min-start-headroom-db`.
- [ ] **Step 3:** Defaults: patience=5, max_turns=15.

### Task 4: Multiturn step_stats + completions.jsonl

**Files:**
- Modify: `training/multiturn_train.py`
- Create: `tests/test_multiturn_logging.py`

- [ ] **Step 1:** Failing test that a dry helper builds step_stats keys and writes completion records.
- [ ] **Step 2:** Log `step_stats=` JSON; append per-turn completions via `decision_log`.
- [ ] **Step 3:** Run tests.

### Task 5: Agent run_step parity

**Files:**
- Modify: `run_step.py`
- Create/modify: `tests/test_run_step_decision_log.py`

- [ ] **Step 1:** Failing test: `step` appends `runs/<run>/completions.jsonl` with prompt/completion.
- [ ] **Step 2:** Wire `append_record` in `cmd_step` (and `cmd_init` when thinking present).
- [ ] **Step 3:** Run full relevant unittest suite.
