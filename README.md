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
qucs_sim.py                   -->  render circuit.sch from .sch.tpl
                                    qucs-s -n  -> circuit.net
                                    qucsrflayout -> layout.svg
                                    qucsator_rf -> circuit.dat (S-params)
        |
        v
cost.py                       -->  scalar total_cost = |S21| at 5.5 GHz
                                    (goal ≤ −70 dB); stopband edges aux only
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

**Band-stop / notch** fixture: two-port through path
(`P1`–`MLin`–`MCROSS`–`MLout`–`P2`) with a shunt butterfly (two `MRSTUB`
wings) on the cross. The `MCROSS` is required so [Qucs-RFlayout](https://github.com/thomaslepoix/Qucs-RFlayout)
can export layout (wires may connect only two ports). Substrate: Rogers
RO4003C (er=3.38, h=0.508mm, t=0.035mm, tand=0.0027).

When the stub looks like an RF short in-band, `S21` notches — that is the
band-stop mechanism. Optimizer cost is **|S21| at 5.5 GHz** (minimize), with
success target **−70 dB** (`|S21| ≤ 3.16×10⁻⁴`). Passband / stopband-edge
stats remain diagnostics.

Free variables (5, chosen for room to demonstrate more complex
optimization later): `ri` (feed-to-fan transition radius), `ro` (fan
radius), `alpha` (sector angle), `Wf` (feed line width), `Lc` (length of
the connecting line from the junction to the stub root).

## Toolchain notes

- **Schematic is authoritative.** Each iteration writes `circuit.sch` from
  `templates/butterfly_stub.sch.tpl`, then derives `circuit.net` with
  `qucs-s -n` before calling `qucsator_rf`. Open `circuit.sch` in Qucs-S
  (backend: **QucsatorRF**) to inspect the circuit; after Simulate, the
  embedded charts show `|Zin|/Z0`, `S11`/`S21` (dB and phase), a Smith
  chart of `S[1,1]`, and Polar charts of `S[1,1]` and `S[2,1]`.
- **Layout is Qucs-RFlayout.** Each iteration also runs `qucsrflayout` and
  writes `layout.svg` (official copper geometry, not a hand-drawn preview).
  macOS has no upstream binary — build from source with
  `-DQRFL_MINIMAL=ON` (this repo looks for
  `../third_party/Qucs-RFlayout/build/qucsrflayout`). Override with
  `QUCS_RFLAYOUT`. A local patch teaches the parser Qucs-S `MRSTUB`
  property order (`ri,ro,Wf,alpha`).
- Tool binaries are auto-detected (macOS app bundle or Windows/WSL paths).
  Override with `QUCS_S` / `QUCSATOR_RF` / `QUCS_RFLAYOUT` if needed.
- `MRSTUB` schematic property order must remain
  `Subst, ri, ro, Wf, alpha, EffDimens, Model` (matches Qucs-S). Deriving
  the netlist via `qucs-s -n` keeps that mapping correct.

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

The checked-in `runs/demo/state.json` history was collected under older
cost definitions (1-port `|Zin|`, then briefly 2-port `|Zin|`). After the
band-stop retarget, start a fresh run:

```bash
python3 run_step.py init --run bandstop1
python3 run_step.py step --run bandstop1 --intent '{"ro":"decrease_strong"}'
python3 run_step.py best --run bandstop1
```

Lower `total_cost` means a deeper worst-case stopband notch (smaller max `|S21|`
in 4–6 GHz).

## Next steps (not yet implemented)

- Swap the human-in-the-loop strategy layer for an actual LLM API call
  (flexible interface — see the `apply_intent` boundary in
  `run_step.py`'s `cmd_step`), then later a fine-tuned small model.
- Relax the wing symmetry constraint (independent `ro`/`alpha` per wing)
  for a harder, higher-dimensional optimization problem.
- Multi-stage / asymmetric stubs to widen the 4–6 GHz stopband rejection.
- Explicit passband constraint in `total_cost` (keep out-of-band `|S21|`
  high while deepening the notch).
- Optionally also export `.kicad_pcb` via `qucsrflayout -f .kicad_pcb`.
