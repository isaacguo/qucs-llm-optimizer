# Qucs LLM Optimizer — Butterfly Radial Stub

A headless Qucs-S / qucsator_rf pipeline for optimizing a butterfly
(bowtie) microstrip radial stub, driven by an "intent-only" strategy
layer — currently a human/cursor-agent playing the LLM role, designed to
be swapped later for a fine-tuned small model.

## Architecture

```
strategy layer (LLM / human)  -->  qualitative intent per variable
                                    e.g. {"ro": "decrease_strong"}
        |
        v
intent.py (deterministic)     -->  precise numeric delta
                                    - fixed magnitude table (slight/normal/strong)
                                    - iteration-based decay schedule (coarse->fine)
                                    - hard bound clamping
        |
        v
qucs_sim.py                   -->  renders netlist, invokes qucsator_rf.exe
                                    (Windows host, via WSL interop), parses
                                    the resulting S-parameter dataset
        |
        v
cost.py                       -->  scalar total_cost (minimax |Zin|/Z0 in
                                    band) + low/center/high edge breakdown
        |
        v
state.py                      -->  JSON history log, iteration cap (20)
```

The strategy layer **never touches raw numbers** — only directional intent
(`increase` / `decrease` / `hold`, with `_slight` / `_strong` magnitude
qualifiers). This is deliberate: it keeps the numerically fragile part of
optimization (step size, convergence schedule, bounds) in code that is
deterministic and auditable, while leaving room for an LLM (or later, a
reinforcement-trained small model) to contribute what it's actually good
at — high-level reasoning about which direction to move and why, given the
cost breakdown as evidence.

## Circuit and target

Two `MRSTUB` (microstrip radial stub) elements tied to the same node,
fed by a short `MLIN` from the junction — a circuit-level model of a
symmetric butterfly/bowtie fan used as a shunt bias-decoupling element.
Substrate: Rogers RO4003C (er=3.38, h=0.508mm, t=0.035mm, tand=0.0027).

Goal: from the junction, the stub should look like a broadband RF
**short** (`|Zin| -> 0`) across 4-6 GHz (40% fractional bandwidth), so
RF energy is diverted away from a DC bias line. Cost metric is the
worst-case (minimax) `|Zin|/Z0` anywhere in that band — a single bad
point counts as a real design failure, not just the average.

Free variables (5, chosen for room to demonstrate more complex
optimization later): `ri` (feed-to-fan transition radius), `ro` (fan
radius), `alpha` (sector angle), `Wf` (feed line width), `Lc` (length of
the connecting line from the junction to the stub root).

## Toolchain notes (verified empirically this session)

- Qucs-S is installed at `C:\Program Files\Qucs-S`; `qucsator_rf.exe` and
  `qucs-s.exe` run fine when invoked from WSL via the transparent
  `/mnt/c/...` interop — no `wine` needed.
- We author the qucsator **netlist** format directly (see
  `templates/butterfly_stub.net.tpl`) instead of building a `.sch`
  schematic. Positional property ordering in `.sch` files does not
  reliably match the internal C++ struct declaration order for every
  component (confirmed by probing `MRSTUB`: the correct authoring order
  is `Subst, ri, ro, Wf, alpha, EffDimens, Model`, which required
  empirical verification, not just reading the struct dump) — the raw
  `Key="Value"` netlist form sidesteps that whole class of bug.
- Simulation directives use a **leading dot** in the raw netlist
  (`.SP:SP1 ...`), unlike component instances.

## Usage

```bash
python3 run_step.py init --run demo
python3 run_step.py step --run demo --intent '{"ro":"decrease_strong"}' \
    --note "why this move"
python3 run_step.py report --run demo   # full history
python3 run_step.py best   --run demo   # best iteration so far
```

Max 20 iterations per run (enforced in `state.py`).

## Demo run (`runs/demo`)

18 hand-driven iterations (human acting as the strategy layer) converged
from an initial guess (`total_cost=1.54`) to `total_cost≈0.31`
(iteration 14: `ri=0.51mm, ro=3.00mm(bound), alpha=99.1deg, Wf=0.60mm,
Lc=2.78mm`), i.e. worst-case `|Zin|` in-band ≈ 15.3 Ω against a 50 Ω
reference. The run plateaued around iteration 13-17, consistent with a
single symmetric butterfly stub reaching its practical bandwidth limit
for a 40%-fractional-bandwidth target — extending further (asymmetric
wings, multi-stage stubs) is a natural next step for demonstrating a
more capable optimizer.

## Next steps (not yet implemented)

- Swap the human-in-the-loop strategy layer for an actual LLM API call
  (flexible interface — see the `apply_intent` boundary in
  `run_step.py`'s `cmd_step`), then later a fine-tuned small model.
- Relax the wing symmetry constraint (independent `ro`/`alpha` per wing)
  for a harder, higher-dimensional optimization problem.
- Add a proper `MTEE` junction discontinuity model and a through-path
  to a second port if two-port (insertion-loss) behavior matters.
