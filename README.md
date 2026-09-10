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
                                    per step: observation + thinking + intent
        |
        v
report_html.py                -->  runs/<run>/report.html
                                    one collapsible panel per iteration
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
  `../third_party/Qucs-RFlayout/build/qucsrflayout`, then for a Windows
  `~/Downloads/qucsrflayout/bin/qucsrflayout.exe`). Override with
  `QUCS_RFLAYOUT`.
- **A stock `qucsrflayout` draws the butterfly wrong, and the code works
  around it.** Upstream parses `MRSTUB` properties positionally as
  `ri, ro, alpha, Wf`, while Qucs-S writes `ri, ro, Wf, alpha`. A stock build
  therefore reads the feed width (0.6) as the sector angle in degrees and
  collapses each wing to a sliver — simulation is unaffected, only the
  picture is wrong. `export_layout_svg` detects the collapsed fan via
  `layout_fan_is_degenerate` and retries once against a throwaway schematic
  with the two properties swapped, so stock and patched builds both produce
  correct layouts.
- Tool binaries are auto-detected (macOS app bundle or Windows/WSL paths).
  Override with `QUCS_S` / `QUCSATOR_RF` / `QUCS_RFLAYOUT` if needed.
- `MRSTUB` schematic property order must remain
  `Subst, ri, ro, Wf, alpha, EffDimens, Model` (matches Qucs-S). Deriving
  the netlist via `qucs-s -n` keeps that mapping correct.

## Usage

```bash
python3 run_step.py init    --run demo
python3 run_step.py observe --run demo   # evidence block to reason over next
python3 run_step.py step    --run demo --intent '{"ro":"decrease_strong"}' \
    --note "one-line summary" --thinking-file /tmp/reasoning.md
python3 run_step.py report  --run demo   # full history
python3 run_step.py best    --run demo   # best iteration so far
python3 run_step.py conclude --run demo --thinking-file /tmp/why-i-stopped.md
python3 run_step.py html    --run demo   # -> runs/demo/report.html
```

Max 20 iterations per run (enforced in `state.py`).

## Decision log (`report.html`)

`observe` prints exactly the evidence the strategy layer is allowed to see;
`step` stores that same block on the resulting history entry alongside the
reasoning passed via `--thinking` / `--thinking-file`. Every step therefore
carries its full decision record — input, reasoning, output, measurement —
and not just the numbers.

`run_step.py html` renders that record as a single self-contained page
(`runs/<run>/report.html`; no external assets, layout SVGs inlined). Each
iteration is a `<details>` panel whose summary shows the cost and the intent
chips, and whose body holds four sections:

1. **Input** — the observation block handed to the strategy layer
2. **Thinking** — the reasoning recorded for that move
3. **Intent** — the qualitative tokens emitted, plus the before/after params
4. **Result** — what the simulator returned

A run-level `conclude` note renders at the bottom as "why the run stopped".
Runs recorded before this existed (`runs/demo`) still render; the missing
fields degrade to placeholders.

## Unsloth training

The optional `training/` package replaces the human strategy layer with a
Qwen3-1.7B intent policy trained by Unsloth GRPO. Unsloth owns 4-bit model
loading, LoRA, generation, group-relative advantages and weight updates.
This repository supplies the executable environment and verifiable reward:

```
observation -> Qwen3 completion -> strict intent JSON -> apply_intent
            -> qucsator_rf (no layout export) -> delta dB reward
```

The default model is `unsloth/Qwen3-1.7B-bnb-4bit`, with rank-16 LoRA and
vLLM disabled for the tested 8 GB GPU profile. Training never replaces the
Qucs reward with a mock or learned judge. Layout export is skipped only
during reward evaluation; the electrical simulation is the same one used by
`run_step.py`.

Install and verify the real reward path:

```bash
uv sync --extra train
uv run qucs-grpo --dry-run
```

The dry run does not load a model or update weights. It executes one
baseline simulation and one candidate intent through real Qucs, then prints
the measured dB improvement.

Probe the untrained policy before deciding whether optional SFT is useful:

```bash
mkdir -p outputs/logs
uv run qucs-probe 2>&1 | tee outputs/logs/probe.log
```

`qucs-probe` samples four intents for each of two randomized circuit states,
executes all candidates in Qucs, and writes format rate, valid-intent rate,
and within-group reward standard deviation to
`outputs/probe-qwen3-1.7b/summary.json`.

Start GRPO:

```bash
uv run qucs-grpo 2>&1 | tee outputs/logs/grpo.log
```

The default run builds 32 randomized real-Qucs states, samples eight
completions per prompt group, and trains for 100 optimizer steps. The three
reward components are weighted `0.2 / 0.2 / 1.0`:

1. exact `<reasoning>...<intent>...</intent>` output shape;
2. a valid non-empty intent changing no more than two variables;
3. clipped real-Qucs improvement
   `old_s21_db - new_s21_db` in `[-20, 20]`.

Every simulated completion is written to `outputs/.../rewards.jsonl`.
Checkpoints and final LoRA adapters stay under the ignored `outputs/`
directory. Resume with:

```bash
uv run qucs-grpo \
  --resume-from-checkpoint outputs/grpo-qwen3-1.7b/checkpoint-25
```

### Google Colab from VS Code

VS Code's remote Jupyter connection sends notebook cells to the Colab kernel;
it does not mount the local checkout or copy Linux executables into the Colab
VM. Clone or upload this repository into `/content`, select a GPU runtime, and
run the repository bootstrap from a notebook cell:

```python
!git clone <repo-url> /content/qucs-llm-optimizer
%cd /content/qucs-llm-optimizer
!bash scripts/setup_colab.sh
```

The bootstrap installs the pinned Python 3.12 environment with `uv`, extracts
the pinned Qucs-S AppImage without FUSE, exposes `qucs-s` and `qucsator_rf` in
headless mode, and executes `qucs-grpo --dry-run` against the real simulator.
The training reward path needs both binaries because every baseline and
candidate circuit is netlisted by `qucs-s` and simulated by `qucsator_rf`.
The training path does not need `qucsrflayout`, because reward simulations set
`export_layout=False`.

Start a small smoke run before committing a full Colab session:

```python
!source .env.colab && uv run qucs-grpo --tasks 2 --steps 1 --generations 2 \
    --sim-workers 2 --output-dir outputs/colab-smoke
```

Then start the normal run and stream its log:

```python
!mkdir -p outputs/logs
!source .env.colab && uv run qucs-grpo --output-dir outputs/colab-grpo \
    2>&1 | tee outputs/logs/colab-grpo.log
```

Colab VMs are ephemeral. Store completed adapters elsewhere or use an output
directory under a mounted Google Drive when checkpoint durability matters.
Writing every checkpoint directly to Drive is more durable but usually slower
than local `/content` storage. The training configuration selects BF16 on GPUs
that support it and falls back to FP16 on GPUs such as T4.

SFT cold-start trains from the corpus index: each indexed run is exported to
rollout-aligned multiturn chat records (`history_window` default 8,
`max_length` 2048). Gate successful runs into `corpus/index.jsonl` first
(see `qucs-corpus`), then:

```bash
uv run qucs-sft --dry-run   # count exported examples; no model load
uv run qucs-sft             # one shallow epoch → outputs/sft-qwen3-1.7b
# overrides: --index PATH --runs-root runs --history-window 8 --max-length 2048
```

Legacy single-file mode remains via `--state runs/llm1/state.json` (prints a
warning). Prefer the index path for multiturn cold-start.

Run `qucs-probe` first. Use SFT only if the base policy cannot reliably
produce valid intents or its completion groups have no reward variance.

## Runs

### `runs/llm1` — reasoned run, 12 iterations, **−84.62 dB**

The reference run. Every step carries its observation and reasoning, so
`runs/llm1/report.html` is a complete audit trail. What the strategy layer
worked out along the way, and none of it was known at the start:

- The notch frequency does not follow `1/ro`. Fitting `f = k/(ro + c)`
  against two measurements exposes a fixed electrical offset from the feed,
  inner radius and cross junction, and predicts the next move to inside one
  50 MHz sweep grid step.
- `ri` moves the notch **four times** further per millimetre than `ro`
  (5.68 vs 1.42 GHz/mm), the opposite of the initial guess.
- `alpha` is the fine knob (~4.6 MHz/deg) and also sets how perfectly the
  transmission zero is realised.
- The run stops at iteration 11 not because the goal is met but because the
  optimum sits 0.088 deg away in `alpha`, while the smallest step the intent
  ladder can ever emit is 0.72 deg. The search is quantization-limited, and
  the fix belongs in `intent.py`, not in the strategy layer.

Goal (−70 dB) was passed at iteration 9. Best geometry: `ri=0.203`,
`ro=4.935`, `alpha=83.356`, `Wf=0.600`, `Lc=3.000` (mm/deg).

### `runs/demo` — earlier hand-driven run, 18 iterations, −70.28 dB

Kept for comparison. Predates observation/thinking capture, so its panels
show placeholders where the reasoning would be.

## Next steps

- Wire a trained LoRA adapter into `run_step.py` as an optional policy
  backend. The human/cursor-agent remains the current runtime strategy layer;
  `training/` now covers policy training, not deployment.
- Add a magnitude below `slight`, or lower `_MIN_SCALE`, so late iterations
  can emit sub-0.1 deg steps — this is what currently caps `runs/llm1`.
- Explicit passband constraint in `total_cost`. Nothing penalises passband
  loss today, and `runs/llm1` degraded it steadily while deepening the notch
  (mean `|S21|` 0.75 below 4 GHz at the best iteration).
- Relax the wing symmetry constraint (independent `ro`/`alpha` per wing)
  for a harder, higher-dimensional optimization problem.
- Multi-stage / asymmetric stubs to widen the 4–6 GHz stopband rejection.
- Optionally also export `.kicad_pcb` via `qucsrflayout -f .kicad_pcb`.
