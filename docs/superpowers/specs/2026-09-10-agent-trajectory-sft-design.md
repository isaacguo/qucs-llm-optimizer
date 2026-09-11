# Agent trajectory SFT for multiturn GRPO

**Date:** 2026-09-10  
**Status:** Approved (user confirmed design sections + this file)  
**Approach:** Corpus ops + rollout-aligned multiturn SFT + resume multiturn GRPO (Approach 2)

## Goal

Train a small intent policy (Qwen3-1.7B class) for **multiturn** notch optimization by supervised fine-tuning on **hard-successful** trajectories produced by a **Cursor agent** (auto model) through the existing `run_step.py` loop. After SFT, multiturn GRPO resumes from the SFT LoRA so RL does not start from an unformatted, domain-naive base policy.

## Motivation

Multiturn GRPO on a small model is hard to start: format failures, invalid intents, and near-zero within-group reward variance burn the horizon without a learning signal. The reference run `runs/llm1` shows that agent thinking carries transferable RF reasoning (empirical `f = k/(ro + c)` fits, relative sensitivities of `ri`/`ro`/`alpha`, quantization limits). The repo already has a thin optional SFT path over `runs/llm1`, but it is single-file, roughly eleven step-isolated examples, and uses the **single-turn** prompt — insufficient and distribution-mismatched for multiturn GRPO.

## Non-goals

- Single-turn GRPO as a primary training path
- External LLM API teachers
- Synthetic / templated fake trajectories as the main corpus
- Auxiliary SFT set from failed-but-interesting runs (deferred)
- Held-out goal evaluation in this phase (`heldout.enabled: false`)
- Wiring a trained LoRA back into `run_step.py` as the live strategy backend
- Changing Qucs reward to a learned judge

## Decisions (locked)

| Topic | Choice |
|-------|--------|
| Pipeline | Teacher multiturn trajectories → multiturn SFT → multiturn GRPO |
| Teacher | Cursor agent (auto) + `run_step.py` (no external API) |
| SFT packing | Step-wise examples aligned with `training/rollout.py` (`build_multiturn_prompt`) |
| Goal distribution | **Shared config** for collection and GRPO: freq **1–10 GHz**; `target_depth_db` sampled from **−55 dB toward deeper**; no held-out this phase |
| Hard success | Must meet **that run’s own** `target_depth_db` (not a fixed −55 gate alone) |
| Corpus scale | **500+** hard-successful trajectories |
| Corpus entrypoint | Only runs in the corpus **index** feed SFT (not a raw scan of all `runs/`) |
| Architecture | Approach 2: corpus ops layer + training handoff |

## Design

### 1. Architecture

Four bands with hard boundaries:

```text
Shared Goal Config
        │
        ├──────────────────────────────┐
        v                              v
 Corpus Ops                         Multiturn GRPO
  assign-run → run_step (agent)         resume SFT LoRA
  → hard-success gate → index           same GoalSpec sampling
        │
        v
 Multiturn SFT
  export step-wise (rollout-aligned) → LoRA
```

**Shared Goal Config** is the only source of truth for frequency/depth sampling and success semantics.

**Corpus Ops** assigns goals, stores agent runs, gates hard success, maintains the index and coverage report.

**Multiturn SFT** reads only the index and trains a LoRA.

**Multiturn GRPO** resumes that LoRA and samples goals from the same shared config.

### 2. Data contracts

#### 2.1 Shared Goal Config

Logical fields (YAML under `configs/`, exact filename chosen at implementation):

| Field | Intent |
|-------|--------|
| `freq_range_hz` | `[1e9, 10e9]` |
| `depth_db_range` | `[−55, −D_max]` (`−D_max` configurable, e.g. −80) |
| `band_half_width_hz` | Build `band_hz` around each target frequency |
| `heldout.enabled` | `false` this phase |
| Optional shared knobs | e.g. `history_window` may live here or in multiturn recipe; **freq/depth must not be duplicated as conflicting magic numbers** |

Per-run target remains `GoalSpec`: `target_freq_hz`, `band_hz`, `target_depth_db`.

Hard-success / goal-met predicate must share one implementation with `training.rollout`’s `_is_goal_met` (same threshold from `target_depth_db`).

#### 2.2 Teacher run: `runs/<run_id>/state.json`

Extend existing history format; do not invent a parallel step log.

**Run-level additions:**

- `goal`: serialized `GoalSpec` (written by `assign-run`, immutable for the run)
- `start_seed` / `initial_params` (recommended for reproducibility)
- `corpus`: `{ eligible, reason, best_db, goal_met_iteration }` filled by the gate only

**History entries (unchanged shape):** `observation`, `thinking`, `intent`, `params`, `cost`, `note`.

- Baseline steps with `intent is null` stay for audit; **SFT export skips them**.
- `cost` must be evaluated at **`goal.target_freq_hz`** (goal-conditioned `evaluate` / `run_step`), not a hard-coded 5.5 GHz default.

#### 2.3 Corpus index

Separate manifest (e.g. `corpus/index.jsonl`): one record per **hard-successful** run.

| Field | Intent |
|-------|--------|
| `run_id`, `run_dir` | Pointer into `runs/` |
| `goal` | Full `GoalSpec` |
| `best_db`, `n_steps`, `goal_met_iteration` | Gate summary |
| `added_at` | Admission time |

Rules:

- Failed runs may remain under `runs/` but **must not** appear in the index.
- SFT export **reads only the index**.
- Index appends must be safe under concurrent collectors (atomic jsonl append or file lock).

#### 2.4 SFT example schema

Each supervised example:

```text
messages = [
  {role: system, content: multiturn SYSTEM_PROMPT from training.rollout},
  {role: user, content: rebuild via build_multiturn_prompt(...)},
  {role: assistant, content: "<reasoning>...</reasoning>\n<intent>...</intent>"}
]
```

Rules:

- Do **not** train on raw `observation.report_text` as the final user distribution; rebuild the multiturn user string so SFT matches GRPO rollouts.
- Map `thinking` → `<reasoning>`; if over budget, compress for training while keeping full text in `state.json` for audit.
- One N-step successful trajectory yields up to N examples (skip null intents); include stop turns only when consistent with rollout stop-allowance rules.
- Optional non-token meta: `run_id`, `turn_index`, `goal` for debugging.

#### 2.5 Coverage report

Aggregate the index on a configurable grid (e.g. 1 GHz frequency bins × depth bins such as −55/−60/−65/−70/…). Report counts and holes to steer the next `assign-run`. Target: **≥ 500** hard successes without large empty regions.

### 3. Collection workflow (Cursor agent)

Lifecycle per run:

1. **`assign-run`** — sample `(GoalSpec, start_seed)` from shared config (optionally bias toward coverage holes); create `runs/<run_id>/`; write `state.goal`; print Target line for the agent.
2. **Optimize** — `observe` → Cursor agent writes thinking + qualitative intent → `step`; repeat until goal met, iteration cap, or `conclude`.
3. **Gate** — if best `|S21|` at target meets **this run’s** `target_depth_db`, set `corpus.eligible=true` and append the index; else keep run on disk only.
4. **Coverage** — refresh grid; assign the next run toward holes.

Agent boundaries:

- Strategy layer only (reasoning + intent tokens); numeric deltas stay in `intent.py`.
- Agent must not hand-edit `corpus.*` or the index.

Scale / ops:

- 500+ is expected to take many sessions; progress = index count + coverage, not calendar heroics.
- Milestones (e.g. 50 / 150 / 300 / 500): dry-run export + small SFT smoke so contract bugs surface early.
- `runs/llm1` may be one-shot imported into the index only if goal metadata is completed and the hard-success predicate passes.

### 4. Training handoff

#### 4.1 Multiturn SFT

- Upgrade `qucs-sft` / `training/sft.py` + `training/data.py` from single-file single-turn llm1 loading to **index → rollout-aligned export → SFTTrainer**.
- Align `max_seq_length` with multiturn recipes (e.g. 2048), not the old 1024 single-turn default when prompts include history.
- Output: `final_lora` under `outputs/`.

After-SFT check: after SFT, format / valid-intent rates on a **multiturn** probe improve, and rollouts are no longer systematically dead.

#### 4.2 Multiturn GRPO

- Set `resume_adapter` to the SFT `final_lora` (field already exists on multiturn recipes).
- **Resume-adapter mode:** missing `resume_adapter` must error unless an explicit escape hatch (e.g. `--allow-raw-base`) is set.
- Goal sampling reads **Shared Goal Config** (1–10 GHz; depth in `[−55, −D_max]`), replacing conflicting narrow defaults in multiturn YAML/`goals.py` for this pipeline.
- Keep real Qucs rewards and existing multiturn rollout/reward structure; thresholds must follow `GoalSpec`.

### 5. Error handling

| Case | Behavior |
|------|----------|
| Index points at missing/corrupt `state.json` | Fail export with run ids; no silent under-count |
| Missing `goal` or inconsistent goal | Refuse index admission / refuse export |
| Floating-point goal check | Single shared predicate with rollout |
| Prompt rebuild failure / incomplete history | Drop turn; warn if a run exports zero turns |
| GRPO from an SFT adapter without adapter | Non-zero exit unless explicit bypass |
| Simulator failure | No forged SFT label; GRPO follows existing failure handling |

### 6. Testing

Contract tests first (Qucs not required for most):

1. Load shared config (1–10 GHz, depth floor −55, held-out off).
2. Hard-success gate: meets own `target_depth_db` → eligible; −60 dB vs −70 dB target → reject.
3. Index filter: non-eligible runs absent from export list.
4. SFT export isomorphism: exported user text matches `build_multiturn_prompt` on a fixture trajectory.
5. Skip null-intent baselines; compressed reasoning still well-tagged.
6. GRPO/task sampling respects shared freq/depth ranges.
7. Multiturn entrypoint errors without `resume_adapter`.
8. Optional integration: `assign-run` → fixture history → gate → export → `qucs-sft --dry-run` count.

### 7. Implementation sketch (for planning; not started)

Likely touch points:

- `training/goals.py` / new shared config module — widen ranges, depth sampling, held-out flag
- `src/cost.py`, `run_step.py`, `src/state.py` — goal-conditioned cost and run metadata
- New corpus CLI(s): assign, gate, coverage, export
- `training/data.py`, `training/sft.py` — multiturn index export + train
- `training/multiturn_train.py` / configs — shared goals + enforce resume for resume-adapter
- Tests under `tests/` for contracts above

TDD: write contract tests before production code for gate, export isomorphism, and resume enforcement.

## Spec self-review (2026-09-10)

| Check | Result |
|-------|--------|
| Placeholders | `−D_max` and coverage bin widths left **configurable** (intentional); no TBD sections |
| Consistency | Teacher, SFT, and GRPO all bound to shared goal semantics and hard-success = own target |
| Ambiguity | −55 is sampling **floor** for `target_depth_db`, not a standalone admission shortcut |
| Scope | Single implementation plan is large but one pipeline; corpus ops + goalizing `run_step` + SFT/GRPO handoff belong together |
| Conflict with current code | Today `cost.py` / reports lean on 5.5 GHz and −70 dB; multiturn YAML still shows 4–6 GHz — this spec **requires** those to yield to shared config for this pipeline |

## Open parameters (configurable, not blockers)

- Exact `−D_max` (e.g. −80)
- Coverage bin widths
- SFT epoch count / LR (recipe-level)
- Milestone counts for smoke (defaults: 50 / 150 / 300 / 500)
