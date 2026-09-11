# 5th-order Butterworth LC band-pass filter task (`butterworth_bpf5`)

**Date:** 2026-09-11  
**Status:** Draft (awaiting user review of this file)  
**Approach:** Parallel task package beside the butterfly notch (Approach 1)

## Goal

Add an end-to-end optimization task for a **5th-order Butterworth lumped LC band-pass filter** (series-first ladder), with:

1. A Qucs schematic template that embeds **charts** (like `butterfly_stub.sch.tpl`) so a rendered `circuit.sch` can be opened in Qucs-S, Simulated, and plotted immediately.
2. Simulation render path, **intent** variable table, **goal/cost**, and `run_step.py` wiring so `init` / `observe` / `step` work for this task.
3. Optimization window semantics aligned with **0–300 MHz** sweep; passband `(f_low, f_high)` comes from the **goal**, not hard-coded into the template element formulas.

## Motivation

The repo today is a single-fixture butterfly radial-stub notch optimizer. A second, lumped BPF task exercises the same intent-only loop on a different RF structure (10 free L/C values, passband/stopband objectives) without disturbing existing corpus/training assumptions tied to the notch.

## Non-goals

- Migrating GRPO / SFT / corpus collection onto BPF in this phase
- Lossy L/C parasitics, coupled-resonator / mutual-inductance topologies
- Dual “shunt-first” ladder template
- Analytic re-seed of `INITIAL_GUESS` from each goal’s `f_low`/`f_high`
- Matching an ideal Butterworth `|S21(f)|` curve as the primary cost (curve-fit approach rejected)
- Requiring `qucsrflayout` for this task

## Decisions (locked)

| Topic | Choice |
|-------|--------|
| Response | Band-pass (not LPF/HPF) |
| Order / ladder | 5th order, **series-first**: series–shunt–series–shunt–series resonators |
| Element model | Ideal lumped `L` / `C` only |
| Template params | Ten independent `{L1}`…`{L5}`, `{C1}`…`{C5}` (no in-template synthesis) |
| Param units in state/intent | `L*` in **nH**, `C*` in **pF** |
| Architecture | **Parallel task package** (Approach 1); butterfly remains default |
| Goal / cost | Passband-window driven (Approach A): `f_low`/`f_high` + IL / stopband thresholds |
| Sweep default | 0–300 MHz |
| Layout | Skip `qucsrflayout` for BPF |
| Training | Out of scope this phase |

## Design

### 1. Architecture

**Task id:** `butterworth_bpf5`

```
strategy (intent JSON over L1..C5)
  → task-scoped intent (VARIABLES / BOUNDS / INITIAL_GUESS + shared apply_intent core)
  → render templates/butterworth_bpf5.sch.tpl → circuit.sch
  → qucs-s -n → qucsator_rf → SimResult
  → cost_bpf + BpfGoalSpec → total_cost / is_goal_met
  → state.json (task + goal + history) → report.html
```

| Concern | Convention |
|---------|------------|
| Butterfly path | Unchanged default; corpus/training semantics stay notch-based |
| Task selection | `run_step.py init --task butterworth_bpf5` (default `butterfly_stub`) |
| Persistence | `state.task`; changing task mid-run is an error |
| Cost / goal modules | New BPF modules; do not overload notch `cost.evaluate` / `GoalSpec` |
| Intent arithmetic | Reuse token/decay/`apply_intent` core; inject per-task variable tables |

### 2. Schematic template

**File:** `templates/butterworth_bpf5.sch.tpl`

**Topology:**

```
P1 -- [L1+C1 series] --+-- [L2||C2 shunt] --+-- [L3+C3] --+-- [L4||C4] --+-- [L5+C5] -- P2
                       |                    |             |              |
                      GND                  GND           GND            GND
```

- Ports: `Pac` P1/P2 at **50 Ω**, with GNDs.
- Odd sections (1,3,5): series LC resonators.
- Even sections (2,4): shunt LC resonators to ground.
- No substrate / microstrip components.

**Placeholders:**

| Placeholder | Role |
|-------------|------|
| `{L1}`…`{L5}` | Inductance values (rendered as `"{n} nH"`) |
| `{C1}`…`{C5}` | Capacitance values (rendered as `"{n} pF"`) |
| `{sweep_start_hz}`, `{sweep_stop_hz}`, `{sweep_points}` | `.SP` linear sweep |
| `{f0_hz}` | Pac nominal frequency field (geometric/mid sweep; not used for synthesis) |

**Embedded analysis / plots (parity with butterfly):**

- `.SP` linear sweep
- `Eqn` for `S11_dB`, `S21_dB`, `S11_phase`, `S21_phase` (optional Zin helpers)
- Diagrams: Rect S11/S21 dB; Rect phase; Smith `S[1,1]`; Polar `S[1,1]` and `S[2,1]`
- Short `Paintings/Text` describing the fixture

Implementation must validate property order against Qucs-S exports for `L`/`C`.

### 3. Parameters, bounds, intent

```
VARIABLES = ("L1","C1","L2","C2","L3","C3","L4","C4","L5","C5")
```

| Variable | Bounds (first cut) |
|----------|-------------------|
| `L1`…`L5` | 1 – 2000 nH |
| `C1`…`C5` | 0.1 – 500 pF |

Intent tokens unchanged: `increase_strong` | `increase` | `increase_slight` | `hold` | `decrease_slight` | `decrease` | `decrease_strong`.

Unknown intent keys for the active task → **hard error** (no silent ignore).

**INITIAL_GUESS** (seed only; not the live goal): classical Butterworth LP prototype → BPF transform at `Z0=50 Ω`, `f0=150 MHz`, `BW=30 MHz`, `g=[0.618, 1.618, 2.000, 1.618, 0.618]`:

| Arm | L (nH) | C (pF) |
|-----|--------|--------|
| series 1 | 163.9296 | 6.8675 |
| shunt 2 | 6.5577 | 171.6751 |
| series 3 | 530.5165 | 2.1221 |
| shunt 4 | 6.5577 | 171.6751 |
| series 5 | 163.9296 | 6.8675 |

Store these constants in the BPF task config (document the formula in comments). Do not recompute on every `init` unless tests assert the closed form. Magnitude/decay schedule may share butterfly’s `_MAGNITUDE_FRAC` initially.

### 4. Goal, cost, success

**`BpfGoalSpec` fields:**

| Field | Meaning | Default |
|-------|---------|---------|
| `f_low_hz` | Passband low edge | required (default goal: `135e6`) |
| `f_high_hz` | Passband high edge | required (default goal: `165e6`) |
| `passband_il_max_db` | Worst allowed passband `S21_dB` (floor) | `-1.0` |
| `stopband_atten_min_db` | Worst allowed stopband `S21_dB` (ceiling) | `-20.0` |
| `stopband_guard_hz` | Exclude transition skirts from stopband | `10e6` |
| `sweep_hz` | Simulation window | `(0, 300e6)` |

Require `f_low < f_high` and both edges inside `sweep_hz`.

**Bands:**

- Passband `P = [f_low, f_high]`
- Lower stopband `S_lo = [sweep_start, f_low - guard]` (skip if empty)
- Upper stopband `S_hi = [f_high + guard, sweep_stop]` (skip if empty)
- Transition regions are **not** in primary cost

**Cost fields** (from `SimResult`):

- `passband_min_s21_db`, `stopband_max_s21_db`, passband mean, optional `s11` diagnostics
- Primary scalar (minimize, linear domain):

```
total_cost = max_{f in P}(1 - |S21(f)|) + λ * max_{f in S}(|S21(f)|)
```

with `λ = 1.0`. If both stopbands lack samples → **hard failure** in `evaluate` (force guard/sweep fix). If only one side exists, use that side.

**`is_goal_met`:** all of:

1. `passband_min_s21_db >= passband_il_max_db`
2. `stopband_max_s21_db <= stopband_atten_min_db`
3. Passband and at least one stopband have samples

No BPF `sample_goal` / held-out distribution in this phase; `run_step` consumes explicit goal JSON or the default above.

### 5. Integration, errors, tests

**`run_step`:** `--task` on `init`; dispatch observe/step/report/best/html by `state.task`. BPF HTML: usable cost summary + history (notch-specific chart labels need not be perfect on day one).

**`qucs_sim`:** task-select template and template values; BPF pipeline omits layout export; missing `layout.svg` must not fail the BPF report path. Default `sweep_points = 301`.

**Errors:** invalid goal geometry; empty passband samples; empty stopbands on both sides; missing Qucs binaries (existing behavior); task mismatch on an existing run.

**Tests (TDD):**

1. Template render contains all placeholders / topology markers / diagram eqn names
2. Goal band splitting + validation
3. `cost_bpf.evaluate` / `is_goal_met` on synthetic `SimResult`
4. Intent bounds for ten variables
5. Optional integration test when Qucs is present

**Docs:** README subsection for `butterworth_bpf5` usage and goal JSON example.

## File / module sketch

| Piece | Location (indicative) |
|-------|------------------------|
| Schematic template | `templates/butterworth_bpf5.sch.tpl` |
| BPF cost | `src/cost_bpf.py` |
| BPF goal | `training/goals_bpf.py` (or `src/goals_bpf.py` if kept out of training) |
| Task param tables | e.g. `src/tasks/butterworth_bpf5.py` or extend intent with task registry |
| Sim template switch | `src/qucs_sim.py` |
| CLI dispatch | `run_step.py` |
| Tests | `tests/test_butterworth_bpf5_*.py` |

Exact module paths may shift during planning so long as butterfly and BPF stay isolated.

## Self-review notes

- No TBD placeholders left for required behavior; seed L/C numbers are explicit.
- Cost uses linear `total_cost` while gates use dB — both defined; do not conflate.
- Scope is one implementation plan: template + sim + cost/goal + intent + `run_step`; training deferred.
- Ambiguity resolved: series-first ladder; ten independent params; skip layout; hard-fail dual-empty stopbands; hard-error unknown intent keys.

## Approval

User approved design sections 1–5 in brainstorming (2026-09-11). This file is the written spec gate before `writing-plans`.
