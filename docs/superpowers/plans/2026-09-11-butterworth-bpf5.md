# Butterworth BPF5 Parallel Task Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship an end-to-end `butterworth_bpf5` optimization task (schematic template with plots, render/sim without layout, intent over 10 L/C values, passband-window goal/cost, `run_step` dispatch) beside the existing butterfly notch path.

**Architecture:** Parallel task package. New modules own BPF goal/cost/params/template; butterfly defaults stay unchanged. `intent.apply_intent` becomes table-driven; `qucs_sim.simulate` selects template and skips `qucsrflayout` for BPF; `state.task` + `run_step --task` dispatch evaluate/observe paths.

**Tech Stack:** Python 3, Qucs-S / qucsator_rf, unittest/pytest, existing `run_step.py` JSON run state.

**Spec:** `docs/superpowers/specs/2026-09-11-butterworth-bpf5-design.md`

## Global Constraints

- Task id string is exactly `butterworth_bpf5`; butterfly default remains `butterfly_stub`.
- Param units in state/intent: `L*` in **nH**, `C*` in **pF**.
- Ladder: series–shunt–series–shunt–series (series-first), ideal L/C only.
- Sweep default `(0, 300e6)` Hz; default goal passband `(135e6, 165e6)`.
- `total_cost = max_P(1-|S21|) + 1.0 * max_S(|S21|)`; gates use dB thresholds separately.
- Skip `qucsrflayout` for BPF; missing `layout.svg` must not fail BPF HTML.
- Do not migrate GRPO/corpus/SFT onto BPF in this plan.
- Unknown intent keys for the active task → `ValueError` (hard fail).
- Both stopbands empty of samples → `evaluate` raises (hard fail).
- TDD: failing test → implement → pass → commit per task.

---

## File map

| File | Responsibility |
|------|----------------|
| `src/goals_bpf.py` | `BpfGoalSpec`, validation, band split, JSON helpers, `default_goal()`, `is_goal_met` |
| `src/cost_bpf.py` | `CostReport`, `evaluate(SimResult, BpfGoalSpec)` |
| `src/tasks/__init__.py` | `TASK_BUTTERFLY`, `TASK_BPF5`, `get_task(name) -> TaskConfig` |
| `src/tasks/butterfly_stub.py` | Thin wrapper re-exporting butterfly variables/bounds/initial + template path |
| `src/tasks/butterworth_bpf5.py` | BPF `VARIABLES` / `BOUNDS` / `INITIAL_GUESS` / template path / sim defaults |
| `src/intent.py` | Generalize `apply_intent(..., variables=..., bounds=...)`; reject unknown intent keys |
| `templates/butterworth_bpf5.sch.tpl` | Authoritative Qucs schematic + Diagrams |
| `src/qucs_sim.py` | Task-aware `render_schematic` / `simulate`; BPF skips layout |
| `src/state.py` | Persist/load `task`; `set_run_meta(task=...)` |
| `run_step.py` | `--task`, goal/cost/intent/sim dispatch |
| `src/report_html.py` | BPF-tolerant cost table + missing layout OK; avoid notch-only crash |
| `README.md` | BPF usage subsection |
| `tests/test_goals_bpf.py` | Goal validation / bands / `is_goal_met` |
| `tests/test_cost_bpf.py` | Synthetic S21 cost |
| `tests/test_intent_bpf.py` | 10-var clamp + unknown key |
| `tests/test_butterworth_bpf5_template.py` | Render markers / placeholders |
| `tests/test_run_step_bpf.py` | Init/step dispatch with mocked simulate (or skip if heavy) |

---

### Task 1: `BpfGoalSpec` and band helpers

**Files:**
- Create: `src/goals_bpf.py`
- Test: `tests/test_goals_bpf.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class BpfGoalSpec` with fields `f_low_hz: float`, `f_high_hz: float`, `passband_il_max_db: float = -1.0`, `stopband_atten_min_db: float = -20.0`, `stopband_guard_hz: float = 10e6`, `sweep_hz: tuple[float, float] = (0.0, 300e6)`
  - `def validate_goal(goal: BpfGoalSpec) -> None` — raises `ValueError` if `f_low >= f_high` or edges outside `sweep_hz`
  - `def split_bands(goal: BpfGoalSpec) -> tuple[tuple[float,float], tuple[float,float] | None, tuple[float,float] | None]` — `(passband, s_lo_or_None, s_hi_or_None)`
  - `def default_goal() -> BpfGoalSpec` — 135e6–165e6 with defaults
  - `def goal_from_dict(data: dict) -> BpfGoalSpec`
  - `def is_goal_met(cost: dict, goal: BpfGoalSpec) -> bool`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_goals_bpf.py
from __future__ import annotations
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from goals_bpf import (  # noqa: E402
    BpfGoalSpec,
    default_goal,
    goal_from_dict,
    is_goal_met,
    split_bands,
    validate_goal,
)


class BpfGoalTests(unittest.TestCase):
    def test_default_goal_passband(self):
        g = default_goal()
        self.assertEqual(g.f_low_hz, 135e6)
        self.assertEqual(g.f_high_hz, 165e6)
        self.assertEqual(g.sweep_hz, (0.0, 300e6))

    def test_validate_rejects_inverted_band(self):
        g = BpfGoalSpec(f_low_hz=200e6, f_high_hz=100e6)
        with self.assertRaises(ValueError):
            validate_goal(g)

    def test_split_bands_with_guard(self):
        g = BpfGoalSpec(
            f_low_hz=135e6,
            f_high_hz=165e6,
            stopband_guard_hz=10e6,
            sweep_hz=(0.0, 300e6),
        )
        pb, slo, shi = split_bands(g)
        self.assertEqual(pb, (135e6, 165e6))
        self.assertEqual(slo, (0.0, 125e6))
        self.assertEqual(shi, (175e6, 300e6))

    def test_is_goal_met_requires_both_gates(self):
        g = default_goal()
        cost_ok = {
            "passband_min_s21_db": -0.5,
            "stopband_max_s21_db": -25.0,
            "has_passband_samples": True,
            "has_stopband_samples": True,
        }
        self.assertTrue(is_goal_met(cost_ok, g))
        cost_bad_pb = dict(cost_ok, passband_min_s21_db=-3.0)
        self.assertFalse(is_goal_met(cost_bad_pb, g))

    def test_goal_from_dict_roundtrip_fields(self):
        g = goal_from_dict(
            {
                "f_low_hz": 100e6,
                "f_high_hz": 120e6,
                "passband_il_max_db": -1.5,
                "stopband_atten_min_db": -30.0,
                "stopband_guard_hz": 5e6,
                "sweep_hz": [0.0, 300e6],
            }
        )
        self.assertEqual(g.f_low_hz, 100e6)
        self.assertEqual(g.sweep_hz, (0.0, 300e6))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_goals_bpf.py -v`  
Expected: FAIL with `ModuleNotFoundError: goals_bpf` (or import error)

- [ ] **Step 3: Write minimal implementation**

```python
# src/goals_bpf.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BpfGoalSpec:
    f_low_hz: float
    f_high_hz: float
    passband_il_max_db: float = -1.0
    stopband_atten_min_db: float = -20.0
    stopband_guard_hz: float = 10e6
    sweep_hz: tuple[float, float] = (0.0, 300e6)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sweep_hz"] = list(self.sweep_hz)
        return d


def default_goal() -> BpfGoalSpec:
    return BpfGoalSpec(f_low_hz=135e6, f_high_hz=165e6)


def validate_goal(goal: BpfGoalSpec) -> None:
    lo, hi = goal.sweep_hz
    if goal.f_low_hz >= goal.f_high_hz:
        raise ValueError("f_low_hz must be < f_high_hz")
    if not (lo <= goal.f_low_hz <= hi and lo <= goal.f_high_hz <= hi):
        raise ValueError("passband edges must lie inside sweep_hz")


def split_bands(
    goal: BpfGoalSpec,
) -> tuple[tuple[float, float], tuple[float, float] | None, tuple[float, float] | None]:
    validate_goal(goal)
    sweep_lo, sweep_hi = goal.sweep_hz
    pb = (goal.f_low_hz, goal.f_high_hz)
    lower_hi = goal.f_low_hz - goal.stopband_guard_hz
    upper_lo = goal.f_high_hz + goal.stopband_guard_hz
    s_lo = (sweep_lo, lower_hi) if lower_hi > sweep_lo else None
    s_hi = (upper_lo, sweep_hi) if upper_lo < sweep_hi else None
    return pb, s_lo, s_hi


def goal_from_dict(data: dict) -> BpfGoalSpec:
    sweep = data.get("sweep_hz", (0.0, 300e6))
    return BpfGoalSpec(
        f_low_hz=float(data["f_low_hz"]),
        f_high_hz=float(data["f_high_hz"]),
        passband_il_max_db=float(data.get("passband_il_max_db", -1.0)),
        stopband_atten_min_db=float(data.get("stopband_atten_min_db", -20.0)),
        stopband_guard_hz=float(data.get("stopband_guard_hz", 10e6)),
        sweep_hz=(float(sweep[0]), float(sweep[1])),
    )


def goal_from_json(raw: str) -> BpfGoalSpec:
    return goal_from_dict(json.loads(raw))


def is_goal_met(cost: dict, goal: BpfGoalSpec) -> bool:
    if not cost.get("has_passband_samples") or not cost.get("has_stopband_samples"):
        return False
    return (
        float(cost["passband_min_s21_db"]) >= goal.passband_il_max_db
        and float(cost["stopband_max_s21_db"]) <= goal.stopband_atten_min_db
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_goals_bpf.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/goals_bpf.py tests/test_goals_bpf.py
git commit -m "Add BpfGoalSpec validation and band splitting."
```

---

### Task 2: BPF cost evaluation

**Files:**
- Create: `src/cost_bpf.py`
- Test: `tests/test_cost_bpf.py`

**Interfaces:**
- Consumes: `goals_bpf.BpfGoalSpec`, `goals_bpf.split_bands`; `qucs_sim.SimResult`
- Produces:
  - `@dataclass class CostReport` with at least: `total_cost`, `passband_min_s21_db`, `stopband_max_s21_db`, `passband_mean_s21_db`, `s11_passband_max_db`, `has_passband_samples`, `has_stopband_samples`, `f_low_hz`, `f_high_hz`
  - `def evaluate(res: SimResult, goal: BpfGoalSpec, *, lam: float = 1.0) -> CostReport`
  - Raises `ValueError` if passband has no samples, or both stopbands have no samples

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cost_bpf.py
from __future__ import annotations
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cost_bpf import evaluate  # noqa: E402
from goals_bpf import default_goal  # noqa: E402
from qucs_sim import SimResult  # noqa: E402


def _db(mag: float) -> float:
    return 20.0 * math.log10(mag)


class CostBpfTests(unittest.TestCase):
    def test_total_cost_passband_and_stopband(self):
        # freqs: stop, pass, pass, stop
        freq = [50e6, 140e6, 150e6, 250e6]
        # |S21|: leaky stop, good pass, good pass, leaky stop
        s21 = [0.2 + 0j, 0.95 + 0j, 0.9 + 0j, 0.15 + 0j]
        s11 = [0.1 + 0j] * 4
        res = SimResult(freq_hz=freq, s11=s11, s21=s21)
        report = evaluate(res, default_goal(), lam=1.0)
        # passband worst (1 - 0.9) = 0.1; stopband max |S21| = 0.2
        self.assertAlmostEqual(report.total_cost, 0.1 + 0.2, places=9)
        self.assertAlmostEqual(report.passband_min_s21_db, _db(0.9), places=6)
        self.assertAlmostEqual(report.stopband_max_s21_db, _db(0.2), places=6)
        self.assertTrue(report.has_passband_samples)
        self.assertTrue(report.has_stopband_samples)

    def test_empty_passband_raises(self):
        res = SimResult(freq_hz=[10e6, 20e6], s11=[0j, 0j], s21=[0.1j, 0.1j])
        with self.assertRaises(ValueError):
            evaluate(res, default_goal())

    def test_both_stopbands_empty_raises(self):
        # sweep only inside passband+guard so stop bands empty
        g = default_goal()
        # override via goal with tiny sweep equal to passband
        from goals_bpf import BpfGoalSpec
        tight = BpfGoalSpec(
            f_low_hz=140e6,
            f_high_hz=160e6,
            stopband_guard_hz=50e6,
            sweep_hz=(140e6, 160e6),
        )
        res = SimResult(
            freq_hz=[150e6],
            s11=[0j],
            s21=[0.99 + 0j],
        )
        with self.assertRaises(ValueError):
            evaluate(res, tight)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cost_bpf.py -v`  
Expected: FAIL import `cost_bpf`

- [ ] **Step 3: Write minimal implementation**

```python
# src/cost_bpf.py
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from goals_bpf import BpfGoalSpec, split_bands
from qucs_sim import SimResult


def s21_db(mag: float) -> float:
    if mag <= 0.0:
        return float("-inf")
    return 20.0 * math.log10(mag)


@dataclass
class CostReport:
    total_cost: float
    passband_min_s21_db: float
    stopband_max_s21_db: float
    passband_mean_s21_db: float
    s11_passband_max_db: float
    has_passband_samples: bool
    has_stopband_samples: bool
    f_low_hz: float
    f_high_hz: float

    def to_dict(self) -> dict:
        return asdict(self)


def _indices_in(freq: list[float], band: tuple[float, float]) -> list[int]:
    lo, hi = band
    return [i for i, f in enumerate(freq) if lo <= f <= hi]


def evaluate(res: SimResult, goal: BpfGoalSpec, *, lam: float = 1.0) -> CostReport:
    if res.s21 is None or len(res.s21) != len(res.freq_hz):
        raise ValueError("SimResult.s21 required and must match freq_hz")
    pb, s_lo, s_hi = split_bands(goal)
    s21_mags = [abs(s) for s in res.s21]
    s11_mags = [abs(s) for s in res.s11]

    pb_idx = _indices_in(res.freq_hz, pb)
    if not pb_idx:
        raise ValueError(f"no simulated points in passband {pb}")

    stop_idx: list[int] = []
    if s_lo is not None:
        stop_idx.extend(_indices_in(res.freq_hz, s_lo))
    if s_hi is not None:
        stop_idx.extend(_indices_in(res.freq_hz, s_hi))
    if not stop_idx:
        raise ValueError("no simulated points in either stopband; adjust guard/sweep")

    pb_mags = [s21_mags[i] for i in pb_idx]
    stop_mags = [s21_mags[i] for i in stop_idx]
    pass_term = max(1.0 - m for m in pb_mags)
    stop_term = max(stop_mags)
    pb_dbs = [s21_db(m) for m in pb_mags]
    s11_pb_dbs = [s21_db(s11_mags[i]) for i in pb_idx]

    return CostReport(
        total_cost=pass_term + lam * stop_term,
        passband_min_s21_db=min(pb_dbs),
        stopband_max_s21_db=s21_db(stop_term),
        passband_mean_s21_db=sum(pb_dbs) / len(pb_dbs),
        s11_passband_max_db=max(s11_pb_dbs),
        has_passband_samples=True,
        has_stopband_samples=True,
        f_low_hz=goal.f_low_hz,
        f_high_hz=goal.f_high_hz,
    )
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_cost_bpf.py tests/test_goals_bpf.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/cost_bpf.py tests/test_cost_bpf.py
git commit -m "Add BPF passband/stopband cost evaluation."
```

---

### Task 3: Task registry + table-driven intent

**Files:**
- Create: `src/tasks/__init__.py`, `src/tasks/butterfly_stub.py`, `src/tasks/butterworth_bpf5.py`
- Modify: `src/intent.py`
- Test: `tests/test_intent_bpf.py` (and keep existing intent tests green if any)

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class TaskConfig` with `name: str`, `variables: tuple[str, ...]`, `bounds: dict[str, tuple[float,float]]`, `initial_guess: dict[str, float]`, `template_path: Path`, `export_layout: bool`
  - `get_task(name: str) -> TaskConfig`
  - `apply_intent(params, intent, iteration, *, variables=None, bounds=None)` — defaults to butterfly module-level tables; if `intent` has keys not in `variables`, raise `ValueError`

BPF constants (from spec):

```python
VARIABLES = ("L1", "C1", "L2", "C2", "L3", "C3", "L4", "C4", "L5", "C5")
BOUNDS = {v: ((1.0, 2000.0) if v.startswith("L") else (0.1, 500.0)) for v in VARIABLES}
INITIAL_GUESS = {
    "L1": 163.9296, "C1": 6.8675,
    "L2": 6.5577, "C2": 171.6751,
    "L3": 530.5165, "C3": 2.1221,
    "L4": 6.5577, "C4": 171.6751,
    "L5": 163.9296, "C5": 6.8675,
}
```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_intent_bpf.py
from __future__ import annotations
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intent import apply_intent  # noqa: E402
from tasks import get_task  # noqa: E402
from tasks.butterworth_bpf5 import BOUNDS, INITIAL_GUESS, VARIABLES  # noqa: E402


class IntentBpfTests(unittest.TestCase):
    def test_get_task_bpf(self):
        t = get_task("butterworth_bpf5")
        self.assertEqual(t.variables, VARIABLES)
        self.assertFalse(t.export_layout)

    def test_increase_clamps_to_bound(self):
        params = dict(INITIAL_GUESS)
        params["L1"] = 1990.0
        out = apply_intent(
            params,
            {"L1": "increase_strong"},
            iteration=0,
            variables=VARIABLES,
            bounds=BOUNDS,
        )
        self.assertEqual(out["L1"], 2000.0)

    def test_unknown_intent_key_raises(self):
        with self.assertRaises(ValueError):
            apply_intent(
                dict(INITIAL_GUESS),
                {"L1": "hold", "ro": "increase"},
                iteration=0,
                variables=VARIABLES,
                bounds=BOUNDS,
            )

    def test_butterfly_default_still_works(self):
        from intent import INITIAL_GUESS as BF_INIT
        out = apply_intent(dict(BF_INIT), {"ro": "decrease"}, iteration=0)
        self.assertLess(out["ro"], BF_INIT["ro"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test — expect fail** (`tasks` missing / `apply_intent` missing kwargs)

- [ ] **Step 3: Implement task modules + generalize `apply_intent`**

In `intent.py`, change `apply_intent` to:

```python
def apply_intent(
    params: dict,
    intent: dict,
    iteration: int,
    *,
    variables: tuple[str, ...] | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> dict:
    vars_ = variables if variables is not None else VARIABLES
    bounds_ = bounds if bounds is not None else BOUNDS
    unknown = set(intent) - set(vars_)
    if unknown:
        raise ValueError(f"unknown intent keys for task: {sorted(unknown)}")
    decay = max(_DECAY_RATE ** iteration, _MIN_SCALE)
    new_params = dict(params)
    for var in vars_:
        token = intent.get(var, "hold")
        sign, mag_key = _parse_intent_token(token)
        if sign == 0:
            continue
        lo, hi = bounds_[var]
        step = sign * _MAGNITUDE_FRAC[mag_key] * decay * (hi - lo)
        new_val = params[var] + step
        new_val = min(max(new_val, lo), hi)
        new_params[var] = round(new_val, 4)
    return new_params
```

Create `src/tasks/butterworth_bpf5.py` with the constants above and:

```python
from pathlib import Path
TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "templates" / "butterworth_bpf5.sch.tpl"
TASK_NAME = "butterworth_bpf5"
EXPORT_LAYOUT = False
SWEEP_POINTS = 301
```

Create `src/tasks/butterfly_stub.py` wrapping existing `intent.VARIABLES/BOUNDS/INITIAL_GUESS` and `templates/butterfly_stub.sch.tpl`, `EXPORT_LAYOUT = True`.

Create `src/tasks/__init__.py`:

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class TaskConfig:
    name: str
    variables: tuple[str, ...]
    bounds: dict
    initial_guess: dict
    template_path: Path
    export_layout: bool
    sweep_points: int = 181

def get_task(name: str) -> TaskConfig:
    if name == "butterfly_stub":
        from tasks import butterfly_stub as m
    elif name == "butterworth_bpf5":
        from tasks import butterworth_bpf5 as m
    else:
        raise ValueError(f"unknown task: {name!r}")
    return TaskConfig(
        name=m.TASK_NAME,
        variables=m.VARIABLES,
        bounds=m.BOUNDS,
        initial_guess=dict(m.INITIAL_GUESS),
        template_path=m.TEMPLATE_PATH,
        export_layout=m.EXPORT_LAYOUT,
        sweep_points=getattr(m, "SWEEP_POINTS", 181),
    )
```

- [ ] **Step 4: Run** `python -m pytest tests/test_intent_bpf.py -v` — PASS  
  Also run any existing intent-related tests.

- [ ] **Step 5: Commit**

```bash
git add src/intent.py src/tasks tests/test_intent_bpf.py
git commit -m "Add task registry and table-driven BPF intent."
```

---

### Task 4: Schematic template + render

**Files:**
- Create: `templates/butterworth_bpf5.sch.tpl`
- Modify: `src/qucs_sim.py` (`render_schematic` accepts `template_path` / task kwargs)
- Test: `tests/test_butterworth_bpf5_template.py`

**Interfaces:**
- Produces: `render_schematic(params, ..., template_path: Path | None = None)` — default remains butterfly path for backward compatibility
- BPF template placeholders: `{L1}`…`{L5}`, `{C1}`…`{C5}`, `{sweep_start_hz}`, `{sweep_stop_hz}`, `{sweep_points}`, `{f0_hz}`
- Component property syntax (from Qucs-S examples):  
  `<L … "{L1} nH" 1 "" 0>`  
  `<C … "{C1} pF" 1 "" 0 "neutral" 0>`

- [ ] **Step 1: Write failing render test**

```python
# tests/test_butterworth_bpf5_template.py
from __future__ import annotations
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qucs_sim import render_schematic  # noqa: E402
from tasks.butterworth_bpf5 import INITIAL_GUESS, TEMPLATE_PATH  # noqa: E402


class BpfTemplateTests(unittest.TestCase):
    def test_render_contains_topology_and_plots(self):
        text = render_schematic(
            dict(INITIAL_GUESS),
            f0_hz=150e6,
            sweep_start_hz=0.0,
            sweep_stop_hz=300e6,
            sweep_points=301,
            template_path=TEMPLATE_PATH,
        )
        self.assertIn("<Qucs Schematic", text)
        self.assertIn("Pac P1 ", text)
        self.assertIn("Pac P2 ", text)
        for n in range(1, 6):
            self.assertIn(f"<L L{n} ", text)
            self.assertIn(f"<C C{n} ", text)
        self.assertIn("163.9296 nH", text)
        self.assertIn("6.8675 pF", text)
        self.assertIn("S21_dB=dB(S[2,1])", text)
        self.assertIn("S11_dB=dB(S[1,1])", text)
        self.assertIn("<Rect ", text)
        self.assertIn("<Smith ", text)
        self.assertGreaterEqual(text.count("<Polar "), 2)
        self.assertNotIn("MRSTUB", text)
        self.assertNotIn("SUBST", text)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run — expect fail** (template missing / `template_path` unsupported)

- [ ] **Step 3: Add `templates/butterworth_bpf5.sch.tpl`**

Create a complete Qucs schematic (header like butterfly `25.2.0`) with:

1. `Pac` P1/P2 @ 50 Ω, GNDs  
2. Series arms: `L1-C1`, `L3-C3`, `L5-C5` on the through path (y≈200)  
3. Shunt arms: at intermediate nodes, `L2` and `C2` both to GND (parallel); same for `L4`/`C4`  
4. `.SP` lin `{sweep_start_hz}Hz` … `{sweep_stop_hz}Hz` `{sweep_points}`  
5. `Eqn` with `S11_dB`, `S21_dB`, `S11_phase`, `S21_phase`  
6. Diagrams mirroring butterfly (Rect dB, Rect phase, Smith, two Polars) using sweep placeholders on axes where applicable  
7. Painting text: 5th-order Butterworth BPF, series-first ladder  

Wire coordinates must connect the ladder without floating nodes. Prefer validating with `qucs-s -n` in Task 5.

Extend `render_schematic`:

```python
def render_schematic(..., template_path: Path | None = None) -> str:
    path = template_path or SCH_TEMPLATE_PATH
    tpl = path.read_text()
    if path == SCH_TEMPLATE_PATH or "butterfly" in path.name:
        values = _template_values(...)  # existing
    else:
        values = {
            "f0_hz": f0_hz,
            "sweep_start_hz": sweep_start_hz,
            "sweep_stop_hz": sweep_stop_hz,
            "sweep_points": sweep_points,
            **{k: params[k] for k in ("L1","C1","L2","C2","L3","C3","L4","C4","L5","C5")},
        }
    return tpl.format(**values)
```

(Prefer branching on explicit task name passed from `simulate` rather than filename substring if cleaner.)

- [ ] **Step 4: Run** `python -m pytest tests/test_butterworth_bpf5_template.py -v` — PASS  
  Confirm butterfly `tests/test_qucs_sim.py::test_render_schematic_is_two_port` still passes.

- [ ] **Step 5: Commit**

```bash
git add templates/butterworth_bpf5.sch.tpl src/qucs_sim.py tests/test_butterworth_bpf5_template.py
git commit -m "Add butterworth_bpf5 schematic template and render path."
```

---

### Task 5: `simulate` task wiring + optional Qucs integration

**Files:**
- Modify: `src/qucs_sim.py` (`simulate` takes `template_path`, uses existing `export_layout` flag; BPF callers pass `False`)
- Test: extend `tests/test_butterworth_bpf5_template.py` or add `tests/test_butterworth_bpf5_sim.py`

**Interfaces:**
- `simulate(..., template_path: Path | None = None, export_layout: bool = True)` already has `export_layout`; wire `template_path` through to `render_schematic`
- When `export_layout=False`, do not call `export_layout_svg` and do not require `qucsrflayout`

- [ ] **Step 1: Write test** (skip without tools)

```python
@unittest.skipUnless(_has_qucs_tools(), "qucs-s / qucsator_rf not available")
def test_simulate_bpf_returns_sweep_length(self):
    from qucs_sim import simulate
    from tasks.butterworth_bpf5 import INITIAL_GUESS, TEMPLATE_PATH
    with tempfile.TemporaryDirectory() as td:
        res = simulate(
            dict(INITIAL_GUESS),
            f0_hz=150e6,
            sweep_start_hz=0.0,
            sweep_stop_hz=300e6,
            sweep_points=61,
            workdir=Path(td),
            export_layout=False,
            template_path=TEMPLATE_PATH,
        )
        self.assertEqual(len(res.freq_hz), 61)
        self.assertEqual(len(res.s21), 61)
        self.assertFalse((Path(td) / "layout.svg").exists())
        net = (Path(td) / "circuit.net").read_text()
        self.assertIn("L:L1", net)  # adjust if qucs-s naming differs
```

If netlist naming differs, assert on presence of `L1` / `C1` substrings instead.

- [ ] **Step 2: Run — fail until `simulate` forwards `template_path`**

- [ ] **Step 3: Implement forward of `template_path` in `simulate` → `render_schematic`**

- [ ] **Step 4: Run tests** (integration may PASS or SKIP)

- [ ] **Step 5: Commit**

```bash
git add src/qucs_sim.py tests/test_butterworth_bpf5_template.py
git commit -m "Wire BPF simulate path without RF layout export."
```

---

### Task 6: `state.task` + `run_step` dispatch

**Files:**
- Modify: `src/state.py`, `run_step.py`
- Test: `tests/test_run_step_bpf.py`

**Interfaces:**
- `RunState.task -> str | None`; `set_run_meta(..., task: str | None = None)`
- `init --task butterworth_bpf5` writes task + default BPF goal (or `--goal-json` via `goal_from_dict`)
- If run already has `task` and CLI `--task` differs → exit with error
- Dispatch:
  - butterfly: existing `evaluate` / `GoalSpec` / butterfly variables
  - bpf: `cost_bpf.evaluate` / `goals_bpf` / BPF variables; `simulate(..., export_layout=False, template_path=..., sweep from goal)`
- `format_report` / `format_observation` branch on task (print L/C bounds; print passband/stopband dB + `total_cost`)

- [ ] **Step 1: Failing test** — init BPF run with mocked `simulate` returning synthetic `SimResult` covering stop+pass+stop; assert `state.task == "butterworth_bpf5"` and cost keys include `passband_min_s21_db`. Prefer unittest.mock.patch on `run_step.simulate`.

- [ ] **Step 2: Run — fail**

- [ ] **Step 3: Implement state + CLI dispatch**

Minimal `cmd_init` sketch:

```python
task_name = getattr(args, "task", None) or "butterfly_stub"
if state.task is not None and state.task != task_name:
    print(f"run task mismatch: state={state.task} cli={task_name}", file=sys.stderr)
    sys.exit(1)
state.set_run_meta(task=task_name)
task = get_task(task_name)
# goal: bpf vs butterfly parsers
# params = state.initial_params or task.initial_guess
# res = simulate(..., template_path=task.template_path, export_layout=task.export_layout,
#                sweep_start_hz=goal.sweep_hz[0], sweep_stop_hz=goal.sweep_hz[1],
#                sweep_points=task.sweep_points, f0_hz=sqrt(f_low*f_high) or mid)
# cost = asdict(cost_bpf.evaluate(...)) or existing evaluate
```

Add argparse: `p_init.add_argument("--task", default="butterfly_stub", choices=["butterfly_stub", "butterworth_bpf5"])`.

- [ ] **Step 4: pytest `tests/test_run_step_bpf.py` + smoke butterfly init still works**

- [ ] **Step 5: Commit**

```bash
git add src/state.py run_step.py tests/test_run_step_bpf.py
git commit -m "Dispatch run_step by task for butterworth_bpf5."
```

---

### Task 7: Report HTML + README

**Files:**
- Modify: `src/report_html.py`, `README.md`
- Test: small unit test or extend `tests/test_report_html.py` if present — BPF history without `layout.svg` and with BPF cost keys must not crash; notch goal line can remain butterfly-oriented when cost lacks notch fields, use `passband_min_s21_db` for a simple progress trace when `task` is bpf (pass `task` into `write_report` or detect cost keys)

- [ ] **Step 1: Write/adjust test** — `write_report` on fake BPF history succeeds; HTML contains `passband_min_s21_db` or `total_cost`

- [ ] **Step 2: Fail / implement / pass**

- [ ] **Step 3: README subsection**

```markdown
## Butterworth BPF5 task

```bash
python3 run_step.py init --run bpf_demo --task butterworth_bpf5
python3 run_step.py observe --run bpf_demo
python3 run_step.py step --run bpf_demo --intent '{"L3":"decrease","C3":"increase"}' --note "..."
```

Optional `--goal-json` example:

```json
{"f_low_hz": 135e6, "f_high_hz": 165e6, "passband_il_max_db": -1.0, "stopband_atten_min_db": -20.0, "stopband_guard_hz": 10e6, "sweep_hz": [0, 3e8]}
```

No `qucsrflayout` for this lumped task. Open `runs/<run>/iter_*/circuit.sch` in Qucs-S to view embedded S-parameter plots.
```

- [ ] **Step 4: Commit**

```bash
git add src/report_html.py README.md tests/test_report_html.py
git commit -m "Document BPF task and harden HTML reports without layout."
```

- [ ] **Step 5: Mark spec approved**

Update `docs/superpowers/specs/2026-09-11-butterworth-bpf5-design.md` status line to `Approved`.

```bash
git add docs/superpowers/specs/2026-09-11-butterworth-bpf5-design.md
git commit -m "Mark butterworth_bpf5 design spec approved."
```

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| Template + Diagrams | 4 |
| 10 independent L/C placeholders, nH/pF | 3, 4 |
| Series-first 5th-order ladder | 4 |
| Ideal L/C, no layout | 4, 5 |
| Parallel task package / `--task` | 3, 6 |
| `BpfGoalSpec` + defaults 135–165 / 0–300 | 1 |
| Cost + `is_goal_met` | 1, 2 |
| Intent table + unknown key error | 3 |
| `run_step` end-to-end | 6 |
| README | 7 |
| HTML without layout | 7 |
| Training/corpus migration | Non-goal (omitted) |

## Placeholder / consistency self-check

- No TBD steps; seed L/C numbers match the spec table.
- `TaskConfig.export_layout` / `template_path` / `variables` names consistent across Tasks 3–6.
- `CostReport.to_dict()` field names match `is_goal_met` expectations (`has_passband_samples`, `has_stopband_samples`, dB fields).
- Butterfly path remains default when `--task` omitted.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-09-11-butterworth-bpf5.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — execute tasks in this session with executing-plans checkpoints  

Which approach?
