# Agent Trajectory SFT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a corpus-ops pipeline so Cursor-agent hard-successful multiturn runs (shared 1–10 GHz / depth goals) export rollout-aligned SFT data, then multiturn GRPO resumes from that LoRA.

**Architecture:** Shared `GoalDistributionConfig` feeds `assign-run`, hard-success gating, and multiturn GRPO sampling. Corpus index is the only SFT entrypoint. Export rebuilds each turn with `build_multiturn_prompt`. `qucs-sft` trains the LoRA; multiturn GRPO from an SFT adapter requires `resume_adapter`.

**Tech Stack:** Python 3.12, existing `unittest`, PyYAML, `training/rollout.py`, HF/PEFT/TRL SFT (train extra), root `run_step.py` + `src/state.py` / `src/cost.py`.

## Global Constraints

- Multiturn-only primary path (do not revive single-turn GRPO as the primary path).
- Teacher = Cursor agent + `run_step.py` (no external API teacher, no synthetic main corpus).
- Hard success = meet **that run’s** `target_depth_db`; depth sampling algebraic range `[depth_db_min, depth_db_max]` with `depth_db_max == -55` floor (shallower bound).
- Frequency range Hz `[1e9, 10e9]`; `heldout_enabled: false` for this pipeline’s shared config.
- SFT user text must call `training.rollout.build_multiturn_prompt` (same function object), not a copied template.
- SFT reads **only** `corpus/index.jsonl`, never a raw scan of all `runs/`.
- Multiturn GRPO errors if `resume_adapter` is empty unless `--allow-raw-base`.
- TDD: failing test → implement → pass → commit per task.
- Do not load GPU models in unit tests; use `--dry-run` for SFT count checks.
- Spec: `docs/superpowers/specs/2026-09-10-agent-trajectory-sft-qwen3-1.7b-design.md`.

## File map

| Path | Responsibility |
|------|----------------|
| `configs/goal_distribution.yaml` | Shared freq/depth/heldout/coverage bins |
| `training/goals.py` | `GoalDistributionConfig`, load/sample, public `is_goal_met`, depth sampling |
| `training/corpus.py` | Index I/O, gate, coverage, `export_multiturn_sft_records` |
| `training/corpus_cli.py` | `assign` / `gate` / `coverage` / `export` CLI |
| `training/data.py` | Keep legacy `load_sft_records`; add thin wrapper or deprecate path for multiturn export |
| `training/sft.py` | Train from index export; multiturn prompt length |
| `training/rollout.py` | Delegate goal-met to `goals.is_goal_met` |
| `training/multiturn_train.py` | Sample from shared distribution; enforce resume |
| `training/config.py` | Multiturn data fields for depth range + `goal_config` path |
| `configs/multiturn_qwen3_1_7b.yaml` | Point at shared goal config / updated ranges |
| `src/state.py` | Persist `goal`, `corpus`, `start_seed`, `initial_params` |
| `run_step.py` | Goal-conditioned evaluate/observe/init from `state.goal` |
| `pyproject.toml` | Entry points `qucs-corpus`, keep `qucs-sft` |
| `tests/test_goal_distribution.py` | Config load + sampling + `is_goal_met` |
| `tests/test_corpus.py` | Gate, index, coverage, export isomorphism |
| `tests/test_run_step_goal.py` | Goal-conditioned cost path (mock sim if needed) |
| `tests/test_training_data.py` | Update/extend for multiturn export |
| `tests/test_multiturn_resume_adapter_cli.py` | Resume-adapter enforcement |
| `corpus/.gitkeep` | Ensure `corpus/` exists; `index.jsonl` gitignored or empty |

---

### Task 1: Shared goal distribution + public `is_goal_met`

**Files:**
- Create: `configs/goal_distribution.yaml`
- Create: `tests/test_goal_distribution.py`
- Modify: `training/goals.py`
- Modify: `training/rollout.py` (use shared `is_goal_met`)

**Interfaces:**
- Produces:
  - `@dataclass GoalDistributionConfig` with fields: `freq_min_hz: float`, `freq_max_hz: float`, `depth_db_min: float`, `depth_db_max: float`, `band_half_width_hz: float`, `heldout_enabled: bool`, `freq_bin_hz: float`, `depth_bin_db: float`
  - `load_goal_distribution(path: str | Path) -> GoalDistributionConfig`
  - `sample_goal_from_distribution(seed: int, dist: GoalDistributionConfig) -> GoalSpec` (samples freq in range; samples `target_depth_db` uniform in `[depth_db_min, depth_db_max]`; if `heldout_enabled` keep current exclusion else ignore held-out list)
  - `is_goal_met(cost: dict, goal: GoalSpec) -> bool`
- Consumes: existing `GoalSpec`, `band_for` (update `band_for` to accept optional half-width from dist)

- [ ] **Step 1: Write failing tests**

```python
"""Tests for shared goal distribution and is_goal_met."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from training.goals import (
    GoalDistributionConfig,
    GoalSpec,
    is_goal_met,
    load_goal_distribution,
    sample_goal_from_distribution,
)


class LoadGoalDistributionTests(unittest.TestCase):
    def test_loads_default_goal_distribution(self):
        path = Path("configs/goal_distribution.yaml")
        dist = load_goal_distribution(path)
        self.assertEqual(dist.freq_min_hz, 1e9)
        self.assertEqual(dist.freq_max_hz, 10e9)
        self.assertEqual(dist.depth_db_max, -55.0)
        self.assertLess(dist.depth_db_min, dist.depth_db_max)
        self.assertFalse(dist.heldout_enabled)


class SampleFromDistributionTests(unittest.TestCase):
    def test_samples_inside_freq_and_depth_ranges(self):
        dist = GoalDistributionConfig(
            freq_min_hz=1e9,
            freq_max_hz=10e9,
            depth_db_min=-80.0,
            depth_db_max=-55.0,
            band_half_width_hz=1e9,
            heldout_enabled=False,
            freq_bin_hz=1e9,
            depth_bin_db=5.0,
        )
        for seed in range(20):
            goal = sample_goal_from_distribution(seed, dist)
            self.assertGreaterEqual(goal.target_freq_hz, 1e9)
            self.assertLessEqual(goal.target_freq_hz, 10e9)
            self.assertGreaterEqual(goal.target_depth_db, -80.0)
            self.assertLessEqual(goal.target_depth_db, -55.0)


class IsGoalMetTests(unittest.TestCase):
    def test_meets_own_target_depth(self):
        goal = GoalSpec(5.5e9, (4.5e9, 6.5e9), target_depth_db=-70.0)
        # |S21| for -70 dB ≈ 3.162e-4
        self.assertTrue(is_goal_met({"total_cost": 3.0e-4}, goal))
        self.assertFalse(is_goal_met({"total_cost": 1.0e-3}, goal))

    def test_minus_60_fails_minus_70_target(self):
        goal = GoalSpec(5.5e9, (4.5e9, 6.5e9), target_depth_db=-70.0)
        # -60 dB ≈ 1e-3
        self.assertFalse(is_goal_met({"total_cost": 1.0e-3}, goal))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `python -m unittest tests.test_goal_distribution -v`  
Expected: import/attribute errors for new APIs / missing YAML.

- [ ] **Step 3: Add `configs/goal_distribution.yaml`**

```yaml
freq_min_hz: 1.0e9
freq_max_hz: 10.0e9
depth_db_min: -80.0
depth_db_max: -55.0
band_half_width_hz: 1.0e9
heldout_enabled: false
freq_bin_hz: 1.0e9
depth_bin_db: 5.0
```

- [ ] **Step 4: Implement APIs in `training/goals.py`**

- Add `GoalDistributionConfig` + `load_goal_distribution` (PyYAML).
- Implement `sample_goal_from_distribution`.
- Implement `is_goal_met` (same formula as current `rollout._is_goal_met`).
- Keep legacy `sample_goal` working for old callers; when `heldout_enabled` is false, new sampler must not apply held-out exclusion.

- [ ] **Step 5: Point `rollout._is_goal_met` at `goals.is_goal_met`**

```python
from training.goals import is_goal_met as _is_goal_met  # or call goals.is_goal_met directly
```

Remove duplicated threshold logic from rollout (thin wrapper OK).

- [ ] **Step 6: Run tests — expect PASS**

Run: `python -m unittest tests.test_goal_distribution -v`  
Also: `python -m unittest tests.test_training_rollout -v`  
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add configs/goal_distribution.yaml training/goals.py training/rollout.py tests/test_goal_distribution.py
git commit -m "Add shared goal distribution config and is_goal_met."
```

---

### Task 2: Corpus gate + index + coverage

**Files:**
- Create: `training/corpus.py`
- Create: `tests/test_corpus.py`
- Create: `corpus/.gitkeep`
- Modify: `.gitignore` (ignore `corpus/index.jsonl` if desired; keep directory)

**Interfaces:**
- Produces:
  - `gate_run(state: dict) -> dict` → updates/returns `corpus` dict `{eligible, reason, best_db, goal_met_iteration}`
  - `append_index(index_path: Path, record: dict) -> None` (atomic append one JSON line)
  - `load_index(index_path: Path) -> list[dict]`
  - `coverage_counts(index: list[dict], dist: GoalDistributionConfig) -> dict[tuple[int, int], int]` keyed by `(freq_bin_index, depth_bin_index)`
  - `index_record_from_run(run_id: str, run_dir: str, state: dict, corpus: dict) -> dict`
- Consumes: `GoalSpec` / `goal_from_json` or dict goal, `is_goal_met`, `s21_db` for best_db reporting

- [ ] **Step 1: Write failing tests**

```python
"""Tests for corpus gate, index, and coverage."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.corpus import (
    append_index,
    coverage_counts,
    gate_run,
    load_index,
)
from training.goals import GoalDistributionConfig


def _state_with_goal(depth_db: float, best_total_cost: float) -> dict:
    return {
        "goal": {
            "target_freq_hz": 5.5e9,
            "band_hz": [4.5e9, 6.5e9],
            "target_depth_db": depth_db,
        },
        "history": [
            {
                "iteration": 0,
                "intent": None,
                "cost": {"total_cost": 0.1},
            },
            {
                "iteration": 1,
                "intent": {"ro": "decrease_strong"},
                "cost": {"total_cost": best_total_cost},
            },
        ],
    }


class GateRunTests(unittest.TestCase):
    def test_eligible_when_own_target_met(self):
        # -72 dB-ish vs -70 target
        state = _state_with_goal(-70.0, 2.5e-4)
        corpus = gate_run(state)
        self.assertTrue(corpus["eligible"])
        self.assertEqual(corpus["goal_met_iteration"], 1)

    def test_reject_when_only_minus_60_vs_minus_70(self):
        state = _state_with_goal(-70.0, 1.0e-3)
        corpus = gate_run(state)
        self.assertFalse(corpus["eligible"])


class IndexTests(unittest.TestCase):
    def test_append_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "index.jsonl"
            append_index(path, {"run_id": "a", "goal": {"target_depth_db": -60}})
            append_index(path, {"run_id": "b", "goal": {"target_depth_db": -65}})
            rows = load_index(path)
        self.assertEqual([r["run_id"] for r in rows], ["a", "b"])


class CoverageTests(unittest.TestCase):
    def test_bins_count_hard_successes(self):
        dist = GoalDistributionConfig(
            freq_min_hz=1e9,
            freq_max_hz=10e9,
            depth_db_min=-80.0,
            depth_db_max=-55.0,
            band_half_width_hz=1e9,
            heldout_enabled=False,
            freq_bin_hz=1e9,
            depth_bin_db=5.0,
        )
        index = [
            {
                "goal": {
                    "target_freq_hz": 5.2e9,
                    "target_depth_db": -62.0,
                    "band_hz": [4.2e9, 6.2e9],
                }
            }
        ]
        counts = coverage_counts(index, dist)
        self.assertEqual(sum(counts.values()), 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run — expect FAIL**

Run: `python -m unittest tests.test_corpus -v`

- [ ] **Step 3: Implement `training/corpus.py` gate/index/coverage**

`gate_run` must:
- Require `state["goal"]`.
- Scan history costs; `eligible` if any step’s cost satisfies `is_goal_met`.
- Set `goal_met_iteration` to first meeting iteration; `best_db` from best `total_cost`.
- Never mark eligible on null-only history.

`append_index`: open with append + write one `json.dumps(..., sort_keys=True) + "\n"`; flush.

- [ ] **Step 4: Run — expect PASS**

Run: `python -m unittest tests.test_corpus -v`

- [ ] **Step 5: Commit**

```bash
git add training/corpus.py tests/test_corpus.py corpus/.gitkeep .gitignore
git commit -m "Add corpus hard-success gate, index, and coverage bins."
```

---

### Task 3: Multiturn SFT export isomorphic to `build_multiturn_prompt`

**Files:**
- Modify: `training/corpus.py` (add export)
- Modify: `tests/test_corpus.py`
- Modify: `training/data.py` (optional thin `load_multiturn_sft_from_index`)

**Interfaces:**
- Produces:
  - `export_multiturn_sft_records(state: dict, *, history_window: int = 8) -> list[dict]` each `{"messages": [...], "meta": {...}}`
  - `export_from_index(index_path: Path, runs_root: Path, history_window: int = 8) -> list[dict]`
- Consumes: `build_multiturn_prompt`, multiturn `SYSTEM_PROMPT`, `GoalSpec`, skip `intent is None`

- [ ] **Step 1: Write failing isomorphism test**

Rebuild a minimal state with two tuning steps. For turn 0 after baseline, call `build_multiturn_prompt` with synthetic `TurnRecord` history matching prior intents/dbs, and assert exported `messages[1]["content"]` equals that user string. Assert `messages[0]["content"]` is rollout `SYSTEM_PROMPT`. Assert assistant wraps reasoning/intent tags.

Also: `export_from_index` must ignore a run_dir that is not listed in the index even if present on disk (test with two run dirs, index lists one).

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement export**

Algorithm per run:
1. Parse `goal` → `GoalSpec`.
2. Walk history in order; maintain list of prior `TurnRecord`-like facts for prompt history (intent, db_before/after from consecutive costs).
3. For each entry with non-null `intent`, build prompt via `build_multiturn_prompt(...)`, compress `thinking` to at most 3 sentences (split on `. ` / newlines; join first 3) for `<reasoning>`.
4. Skip baseline null intents.

Do not use `observation.report_text` as the training user content.

- [ ] **Step 4: Run — expect PASS**

Run: `python -m unittest tests.test_corpus -v`

- [ ] **Step 5: Commit**

```bash
git add training/corpus.py training/data.py tests/test_corpus.py
git commit -m "Export multiturn SFT records isomorphic to rollout prompts."
```

---

### Task 4: Goal-conditioned `RunState` + `run_step.py`

**Files:**
- Modify: `src/state.py`
- Modify: `run_step.py`
- Create: `tests/test_run_step_goal.py` (or extend `tests/test_run_step_decision_log.py`)

**Interfaces:**
- Produces:
  - `RunState` stores/loads `goal`, `start_seed`, `initial_params`, `corpus`
  - `run_step init/observe/step` use `state.goal` for `evaluate(..., target_hz=..., band_hz=...)` and print Target from goal depth/freq
- Consumes: `GoalSpec` dict shape from assign-run

- [ ] **Step 1: Write failing tests**

- `RunState` round-trip persists `goal`.
- With a temp run dir whose `state.json` has `goal.target_freq_hz = 7e9` and `target_depth_db = -60`, a unit test that mocks `simulate` / injects cost should show `format_report` / observation text containing `7.00 GHz` and goal `-60 dB` (prefer testing pure helpers if cheaper: extract `_goal_from_state(state) -> GoalSpec` and `_evaluate_for_run(res, goal)`).

Keep backward compatibility: missing `goal` falls back to today’s `TARGET_NOTCH_HZ` / `TARGET_DEPTH_DB` so old demos still run.

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement state + run_step goal paths**

- On `init`, if `--goal-json` provided, store it; else default legacy constants wrapped as goal.
- `cmd_observe` / reports must show this run’s target depth, not a global constant only.

- [ ] **Step 4: Run — expect PASS**

Run: `python -m unittest tests.test_run_step_goal tests.test_run_step_decision_log -v`

- [ ] **Step 5: Commit**

```bash
git add src/state.py run_step.py tests/test_run_step_goal.py
git commit -m "Make run_step and RunState goal-conditioned."
```

---

### Task 5: Corpus CLI — assign, gate, coverage, export

**Files:**
- Create: `training/corpus_cli.py`
- Modify: `pyproject.toml` (`qucs-corpus = "training.corpus_cli:main"`)
- Modify: `tests/test_corpus.py` (CLI smoke with tempfile, no Qucs)

**Interfaces:**
- Produces CLI subcommands:
  - `qucs-corpus assign --run ID --goal-config configs/goal_distribution.yaml [--seed N] [--prefer-coverage]`
  - `qucs-corpus gate --run ID [--index corpus/index.jsonl]`
  - `qucs-corpus coverage --index corpus/index.jsonl --goal-config ...`
  - `qucs-corpus export --index corpus/index.jsonl --out path.jsonl [--history-window 8]`
- `assign` creates `runs/<id>/`, writes `state.json` with sampled goal + `initial_params` from `training.environment.sample_params`, iteration `-1` / empty history ready for `run_step.py init` **or** performs init-equivalent without simulating if `--no-simulate` (prefer: assign writes goal metadata; document that user still runs `run_step.py init` **after** assign only if init overwrites goal — **better:** change `init` to preserve existing `state.goal` when present).

**Decision locked for implementers:** `assign` writes `runs/<id>/state.json` skeleton with `goal`/`start_seed`/`initial_params`/`history: []`/`iteration: -1`. `run_step.py init` must **preserve** pre-existing `goal` and use `initial_params` if set; only simulate baseline and append history[0].

- [ ] **Step 1: Write failing CLI tests** using `corpus_cli.main([...])` argv lists on temp dirs.

- [ ] **Step 2: Implement CLI + wire entry point**

- [ ] **Step 3: Tests PASS**

- [ ] **Step 4: Commit**

```bash
git add training/corpus_cli.py pyproject.toml tests/test_corpus.py run_step.py src/state.py
git commit -m "Add qucs-corpus CLI for assign, gate, coverage, export."
```

---

### Task 6: Upgrade `qucs-sft` to multiturn index training

**Files:**
- Modify: `training/sft.py`
- Modify: `tests/test_training_data.py` (dry-run path / argparse)
- Modify: `README.md` (SFT section: index-based multiturn GRPO from an SFT adapter)

**Interfaces:**
- CLI:
  - `--index` default `corpus/index.jsonl`
  - `--runs-root` default `runs`
  - `--history-window` default `8`
  - `--state` retained for legacy single-file path **or** removed with clear error pointing to `--index` (prefer: if `--state` set, use legacy `load_sft_records` and print warning; default path is `--index`)
  - `--dry-run` prints example count from export
  - `max_length` / `max_seq_length` default **2048**

- [ ] **Step 1: Failing test** — `build_parser` defaults; dry-run main with temp index counts export size without loading model.

- [ ] **Step 2: Implement**

- [ ] **Step 3: `uv run qucs-sft --dry-run` against empty index → clear error; against fixture index → count > 0**

- [ ] **Step 4: Commit**

```bash
git add training/sft.py tests/test_training_data.py README.md
git commit -m "Train multiturn SFT from corpus index with rollout-aligned data."
```

---

### Task 7: Multiturn GRPO reads shared distribution + enforces SFT resume

**Files:**
- Modify: `training/config.py` (`MultiturnDataSection`: add `goal_config: str = "configs/goal_distribution.yaml"`; add `depth_db_min`/`depth_db_max` optional overrides OR drop fixed `target_depth_db` in favor of distribution)
- Modify: `configs/multiturn_qwen3_1_7b.yaml`
- Modify: `training/multiturn_train.py`
- Create: `tests/test_multiturn_resume_adapter_cli.py`

**Interfaces:**
- When resolving tasks, call `sample_goal_from_distribution(seed, load_goal_distribution(path))` instead of fixed `target_depth_db`.
- Add CLI `--allow-raw-base` (default False).
- At start of `train()`: if not `runtime.resume_adapter` and not `allow_raw_base`: `raise SystemExit("multiturn training requires --resume-adapter / runtime.resume_adapter")`.

Locked field mapping:
- Prefer `data.goal_config` path to YAML distribution.
- Keep `goal_freq_min_ghz` / `goal_freq_max_ghz` only as CLI overrides that mutate a loaded dist copy (optional). If YAML `goal_config` present, it wins for defaults.

- [ ] **Step 1: Failing tests** for exit without adapter; sampling uses 1–10 GHz when loading shipped `goal_distribution.yaml`.

- [ ] **Step 2: Implement**

- [ ] **Step 3: unittest PASS** (no GPU)

- [ ] **Step 4: Commit**

```bash
git add training/config.py training/multiturn_train.py configs/multiturn_qwen3_1_7b.yaml tests/test_multiturn_resume_adapter_cli.py
git commit -m "Wire multiturn GRPO to shared goals and require SFT resume."
```

---

### Task 8: Docs + milestone checklist + optional `llm1` import helper

**Files:**
- Modify: `README.md` (corpus workflow, milestones 50/150/300/500)
- Modify: `docs/superpowers/specs/2026-09-10-agent-trajectory-sft-qwen3-1.7b-design.md` status → Approved
- Optional: `qucs-corpus import-run --run llm1` that attaches a goal matching llm1’s 5.5 GHz / −70 dB, gates, and indexes if eligible

- [ ] **Step 1: Document end-to-end operator flow**

```bash
uv run qucs-corpus assign --run r001
# Cursor agent loop: run_step observe/step until success
uv run qucs-corpus gate --run r001
uv run qucs-corpus coverage
# milestones: export + SFT dry-run / short SFT
uv run qucs-corpus export --out /tmp/sft.jsonl
uv run qucs-sft --index corpus/index.jsonl --dry-run
uv run qucs-sft --index corpus/index.jsonl --output-dir outputs/sft-qwen3-1.7b
# then multiturn with resume_adapter: outputs/sft-qwen3-1.7b/final_lora
```

- [ ] **Step 2: If implementing import-run, TDD gate for llm1 fixture path**

- [ ] **Step 3: Commit**

```bash
git add README.md docs/superpowers/specs/2026-09-10-agent-trajectory-sft-qwen3-1.7b-design.md training/corpus_cli.py tests/
git commit -m "Document corpus collection operator workflow and milestones."
```

---

## Spec coverage self-review

| Spec requirement | Task |
|------------------|------|
| Shared Goal Config 1–10 GHz, depth from −55 deeper, heldout off | Task 1 |
| Hard success = own `target_depth_db` | Tasks 1–2 |
| Corpus index only for SFT | Tasks 2–3, 6 |
| Cursor agent + `run_step` teacher | Tasks 4–5 |
| Step-wise export ≡ `build_multiturn_prompt` | Task 3 |
| Coverage report | Tasks 2, 5 |
| `assign-run` / gate / export CLI | Task 5 |
| Multiturn SFT upgrade | Task 6 |
| GRPO resume + same distribution | Task 7 |
| Milestones / docs / non-goals respected | Task 8 |
| 500+ scale | Ops via coverage; no code hard-cap (by design) |

## Placeholder scan

No TBD steps. `depth_db_min` default **−80.0** is explicit in shipped YAML (configurable). Coverage bin defaults **1 GHz / 5 dB** are explicit.

## Type consistency

- `GoalDistributionConfig` and `is_goal_met(cost: dict, goal: GoalSpec) -> bool` are the shared names across Tasks 1–7.
- Index records always include `run_id`, `run_dir`, `goal`.
- Export returns `list[dict]` with `messages` (+ optional `meta`).

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-10-agent-trajectory-sft-qwen3-1.7b.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks  
2. **Inline Execution** — execute tasks in this session with executing-plans and checkpoints  

Which approach?
