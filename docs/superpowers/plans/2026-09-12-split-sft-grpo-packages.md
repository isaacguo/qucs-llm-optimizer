# Split SFT / GRPO / corpus Packages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move SFT and multi-turn GRPO into separate folders under `training/`, and move corpus code to a top-level `corpus/` package beside `training/`, with shared pieces in `training/common/`.

**Architecture:** Four functional areas — `training/common` (shared), `training/sft`, `training/grpo`, and top-level `corpus` (code + `index.jsonl`). No import shims. No algorithm changes. CLI command names stay `qucs-sft` / `qucs-multiturn` / `qucs-corpus`.

**Tech Stack:** Python 3.12, `unittest`, `uv`, setuptools package discovery via `pyproject.toml`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-12-split-sft-grpo-packages-design.md` (Approved).
- Dependency rules: `common` → nothing under sft/grpo/corpus; `corpus` → only `training.common`; `sft` → `common` + `corpus`; `grpo` → only `common`. Never `sft` ↔ `grpo`.
- Do not write `cold-start` / `coldstart` anywhere.
- Do not change reward math, YAML field names, or training loop behavior.
- Prefer `git mv` when relocating files; update imports in the same task.
- Tests use `unittest` (`uv run python -m unittest …`), not pytest.
- Commit after each task.

---

### Task 1: Create `training/common` and extract prompts

**Files:**
- Create: `training/common/__init__.py`
- Create: `training/common/prompts.py` (from `training/rollout.py` extract)
- Create: `training/common/modeling.py` (`git mv` from `training/modeling.py`)
- Create: `training/common/goals.py` (`git mv` from `training/goals.py`)
- Create: `training/common/environment.py` (`git mv` from `training/environment.py`)
- Create: `training/common/contracts.py` (`git mv` from `training/contracts.py`)
- Create: `training/common/preflight.py` (`git mv` from `training/preflight.py`)
- Modify: `training/rollout.py` — import prompts/helpers from `training.common.prompts` (temporary; Task 2 moves this file)
- Modify: any in-tree imports of the moved modules (tests + `training/*` still flat)
- Test: `tests/test_common_prompts.py` (new); update `tests/test_training_modeling.py`, `tests/test_goal_distribution.py`, `tests/test_training_contracts.py`, `tests/test_training_environment.py` imports

**Interfaces:**
- Consumes: existing `SYSTEM_PROMPT`, `TurnRecord`, `stop_is_allowed`, `_last_move_hint`, `build_multiturn_prompt` bodies from `training/rollout.py`; `GoalSpec` from goals; `BOUNDS` / `VARIABLES` / `s21_db` as today.
- Produces:
  - `training.common.prompts.SYSTEM_PROMPT: str`
  - `training.common.prompts.TurnRecord` (dataclass, same fields)
  - `training.common.prompts.stop_is_allowed(...)` (same signature as today)
  - `training.common.prompts.build_multiturn_prompt(...)` (same signature as today)
  - `training.common.modeling.ModelConfig`, `load_policy`, `mixed_precision_config`, `LoadHooks`
  - `training.common.goals.*` (unchanged public API)
  - `training.common.environment.sample_params`
  - `training.common.contracts.parse_intent_completion`, `IntentParseError`
  - `training.common.preflight.verify_runtime`

- [ ] **Step 1: Write the failing import test**

```python
# tests/test_common_prompts.py
from __future__ import annotations

import unittest

from training.common.goals import GoalSpec
from training.common.prompts import SYSTEM_PROMPT, TurnRecord, build_multiturn_prompt


class CommonPromptsTests(unittest.TestCase):
    def test_system_prompt_mentions_butterfly(self):
        self.assertIn("butterfly", SYSTEM_PROMPT.lower())

    def test_build_multiturn_prompt_returns_system_and_user(self):
        goal = GoalSpec(target_freq_hz=5.5e9, band_hz=(4.5e9, 6.5e9), target_depth_db=-70.0)
        messages = build_multiturn_prompt(
            goal,
            turn_index=0,
            params={"ri": 0.5, "ro": 0.5, "alpha": 0.5},  # use real BOUNDS keys from src
            cost={
                "total_cost": 1e-3,
                "best_s21_mag": 1e-3,
                "best_freq_hz": 5.5e9,
            },
            history=[],
        )
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIsInstance(TurnRecord, type)


if __name__ == "__main__":
    unittest.main()
```

Adjust `params` keys to match real `VARIABLES` from this repo when implementing.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_common_prompts -q`  
Expected: FAIL (no `training.common.prompts`)

- [ ] **Step 3: Create package + move common modules + extract prompts**

```bash
mkdir -p training/common
printf '"""Shared training primitives (model load, goals, prompts, contracts)."\n' > training/common/__init__.py
git mv training/modeling.py training/common/modeling.py
git mv training/goals.py training/common/goals.py
git mv training/environment.py training/common/environment.py
git mv training/contracts.py training/common/contracts.py
git mv training/preflight.py training/common/preflight.py
```

Create `training/common/prompts.py` by moving `SYSTEM_PROMPT`, `TurnRecord`, `stop_is_allowed`, `_last_move_hint` (keep private or rename to `last_move_hint` only if all call sites update — prefer keep `_last_move_hint` private in the same module), and `build_multiturn_prompt` out of `training/rollout.py`.

In `training/rollout.py`, replace local definitions with:

```python
from training.common.prompts import (  # noqa: E402
    SYSTEM_PROMPT,
    TurnRecord,
    build_multiturn_prompt,
    stop_is_allowed,
)
```

Update every `from training.modeling|goals|environment|contracts|preflight` to `training.common.…` in live code and tests for this task.

- [ ] **Step 4: Run tests**

Run:

```bash
uv run python -m unittest \
  tests.test_common_prompts \
  tests.test_training_modeling \
  tests.test_goal_distribution \
  tests.test_training_contracts \
  tests.test_training_environment \
  tests.test_training_rollout \
  -q
```

Expected: OK

- [ ] **Step 5: Commit**

```bash
git add training/common training/rollout.py tests/
git commit -m "Extract training.common and shared multiturn prompts."
```

---

### Task 2: Create `training/grpo` package

**Files:**
- Create: `training/grpo/__init__.py`
- Create via `git mv`:
  - `training/grpo/config.py` ← `training/config.py`
  - `training/grpo/train.py` ← `training/multiturn_train.py`
  - `training/grpo/rollout.py` ← `training/rollout.py`
  - `training/grpo/reward_math.py` ← `training/reward_math.py`
  - `training/grpo/starts.py` ← `training/starts.py`
  - `training/grpo/diagnostics.py` ← `training/diagnostics.py`
  - `training/grpo/trl_rollout.py` ← `training/trl_rollout.py`
  - `training/grpo/evaluate.py` ← `training/evaluate_multiturn_generalization.py`
- Modify: imports inside those files to `training.common.*` and `training.grpo.*`
- Modify: `training/__init__.py` docstring only (no re-exports)
- Rename tests (optional but preferred):
  - `tests/test_training_rollout.py` → `tests/test_grpo_rollout.py`
  - `tests/test_training_config.py` → `tests/test_grpo_config.py`
  - `tests/test_training_cli.py` → split/update GRPO half to `tests/test_grpo_cli.py`
  - `tests/test_multiturn_resume_adapter_cli.py` → `tests/test_grpo_resume_adapter_cli.py`
  - `tests/test_multiturn_logging.py` → `tests/test_grpo_logging.py`
  - `tests/test_trl_rollout.py` → `tests/test_grpo_trl_rollout.py`
  - `tests/test_training_starts.py` → `tests/test_grpo_starts.py`

**Interfaces:**
- Consumes: `training.common.*`
- Produces:
  - `training.grpo.train.main`, `build_parser`, `train`, `sample_multiturn_goal`
  - `training.grpo.rollout.run_trajectory`, `Trajectory`, `shaped_turn_advantages`, …
  - `training.grpo.config.MultiturnConfig`, `resolve_multiturn_config`, …

- [ ] **Step 1: Write a failing entrypoint import test**

```python
# tests/test_grpo_package_import.py
import unittest

class GrpoPackageImportTests(unittest.TestCase):
    def test_train_module_importable(self):
        from training.grpo.train import build_parser
        self.assertTrue(callable(build_parser))
```

- [ ] **Step 2: Run — expect FAIL**

`uv run python -m unittest tests.test_grpo_package_import -q`

- [ ] **Step 3: `git mv` grpo modules and fix imports**

Ensure `training/grpo/train.py` uses:

```python
from training.grpo.config import ...
from training.grpo.rollout import ...
from training.common.modeling import load_policy
from training.common.goals import ...
```

Ensure no `from training.sft` and no `from corpus`.

- [ ] **Step 4: Run GRPO-related tests**

```bash
uv run python -m unittest discover -s tests -p 'test_grpo*.py' -q
uv run python -m unittest tests.test_training_rollout tests.test_training_config tests.test_multiturn_resume_adapter_cli tests.test_trl_rollout tests.test_multiturn_logging tests.test_training_starts -q
```

(If renamed, only the new names.) Expected: OK

- [ ] **Step 5: Commit**

```bash
git commit -m "Move multiturn GRPO modules into training.grpo."
```

---

### Task 3: Create top-level `corpus` package

**Files:**
- Create: `corpus/__init__.py` (re-export public API from `corpus.index` for short imports)
- Create: `corpus/index.py` (`git mv` from `training/corpus.py`)
- Create: `corpus/cli.py` (`git mv` from `training/corpus_cli.py`)
- Keep: `corpus/index.jsonl`, `corpus/.gitkeep`
- Modify: `corpus/index.py` imports → `training.common.prompts`, `training.common.goals` (never `training.grpo`)
- Modify: `corpus/cli.py` → `from corpus.index import …`, `training.common.*`
- Modify: `tests/test_corpus.py` imports
- Modify: scripts that import `training.corpus`

**Interfaces:**
- Consumes: `training.common.goals`, `training.common.prompts`, `training.common.environment` (if CLI still samples params)
- Produces: same functions as today’s `training.corpus` (`gate_run`, `append_index`, `export_from_index`, `load_index`, `coverage_counts`, `index_record_from_run`, …) living in `corpus.index`; `corpus.cli:main`

- [ ] **Step 1: Failing import test**

```python
# tests/test_corpus_package_import.py
import unittest

class CorpusPackageImportTests(unittest.TestCase):
    def test_gate_run_importable(self):
        from corpus.index import gate_run
        self.assertTrue(callable(gate_run))
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Move files and fix imports**

`corpus/__init__.py` example:

```python
"""Teacher-trajectory corpus: gate, index, coverage, export."""
from corpus.index import (
    append_index,
    coverage_counts,
    export_from_index,
    gate_run,
    index_record_from_run,
    load_index,
)

__all__ = [
    "append_index",
    "coverage_counts",
    "export_from_index",
    "gate_run",
    "index_record_from_run",
    "load_index",
]
```

Fix `ROOT = parents[…]` in `cli.py` / `index.py` after the move (was `parents[1]` under `training/`; still repo root for `corpus/cli.py`).

- [ ] **Step 4: Run**

```bash
uv run python -m unittest tests.test_corpus tests.test_corpus_package_import -q
```

Expected: OK

- [ ] **Step 5: Commit**

```bash
git commit -m "Move corpus library and CLI to top-level corpus package."
```

---

### Task 4: Create `training/sft` package

**Files:**
- Create: `training/sft/__init__.py`
- Create: `training/sft/data.py` (`git mv` from `training/data.py`)
- Create: `training/sft/train.py` (`git mv` from `training/sft.py`)
- Modify: imports → `training.common.*`, `corpus.index` / `corpus` (not `training.grpo`)
- Rename: `tests/test_training_data.py` → `tests/test_sft_data.py`; SFT half of CLI tests → `tests/test_sft_cli.py`

**Interfaces:**
- Consumes: `training.common.modeling`, `training.common.preflight`, `training.common.prompts.SYSTEM_PROMPT`, `corpus.index.export_from_index`
- Produces: `training.sft.train.main`, `build_parser`; `training.sft.data.load_sft_records`, `load_multiturn_sft_from_index`

- [ ] **Step 1: Failing import test**

```python
import unittest

class SftPackageImportTests(unittest.TestCase):
    def test_train_importable(self):
        from training.sft.train import build_parser
        self.assertTrue(callable(build_parser))
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Move + fix imports**

`training/sft/data.py` must use:

```python
from training.common.prompts import SYSTEM_PROMPT
from corpus.index import export_from_index  # inside load_multiturn_sft_from_index
```

- [ ] **Step 4: Run**

```bash
uv run python -m unittest tests.test_sft_data tests.test_training_data tests.test_training_cli -q
```

Expected: OK (use new names if renamed)

- [ ] **Step 5: Commit**

```bash
git commit -m "Move SFT train and data modules into training.sft."
```

---

### Task 5: Wire `pyproject.toml` entry points and remaining callers

**Files:**
- Modify: `pyproject.toml`
- Modify: `run_step.py`, `scripts/*.py` that still import old paths
- Modify: `README.md` (package layout sentence; no command renames)
- Modify: `training/__init__.py`
- Delete: any leftover empty flat modules under `training/` that were not moved
- Optional: delete temporary `tests/test_*_package_import.py` after broader tests cover imports, or keep them

**Interfaces:**
- Produces console scripts:

```toml
[project.scripts]
qucs-corpus = "corpus.cli:main"
qucs-multiturn = "training.grpo.train:main"
qucs-sft = "training.sft.train:main"

[tool.setuptools.packages.find]
include = ["training*", "corpus*"]
```

- [ ] **Step 1: Update `pyproject.toml` as above**

- [ ] **Step 2: `uv sync --extra train` and help smoke**

```bash
uv sync --extra train
uv run qucs-sft --help
uv run qucs-multiturn --help
uv run qucs-corpus --help
```

Expected: each prints usage; no ImportError

- [ ] **Step 3: Fix remaining imports**

```bash
rg -n 'from training\.(modeling|goals|environment|contracts|preflight|data|sft|config|multiturn_train|rollout|corpus|corpus_cli|reward_math|starts|diagnostics|trl_rollout|evaluate_multiturn)' \
  --glob '!.venv/**' -g '*.py' -g '*.md' -g '*.sh'
```

Expected: no live hits outside archived docs (historical plans may remain).

- [ ] **Step 4: Full related unittest suite**

```bash
uv run python -m unittest discover -s tests -p 'test_*.py' -q
```

Expected: OK (or only pre-existing unrelated failures — fix any import failures introduced by this split)

- [ ] **Step 5: Dependency sanity check**

```bash
rg -n 'from training\.grpo|import training\.grpo' training/sft corpus -g '*.py'
rg -n 'from training\.sft|import training\.sft' training/grpo corpus -g '*.py'
rg -n 'from corpus|import corpus' training/grpo -g '*.py'
```

Expected: empty

- [ ] **Step 6: Commit**

```bash
git commit -m "Point CLI entry points at sft/grpo/corpus packages and finish import migration."
```

---

## Self-review vs spec

| Spec requirement | Task |
|------------------|------|
| `training/common` + `sft` + `grpo` | 1, 2, 4 |
| Top-level `corpus/` sibling of `training/` | 3 |
| Shared prompts in `common.prompts` | 1 |
| CLI names unchanged; new entry points | 5 |
| No shims; delete old flat modules | 2–5 |
| Tests updated / renamed | 1–5 |
| No algorithm / YAML semantic change | all |
| Ban `cold-start` wording | already done; do not reintroduce |
| README notes layout | 5 |

No TBD placeholders remain in this plan.
