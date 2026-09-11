from __future__ import annotations

from pathlib import Path

VARIABLES = ("L1", "C1", "L2", "C2", "L3", "C3", "L4", "C4", "L5", "C5")
# Wide enough for narrow VHF BPFs (e.g. CF=125 MHz, BW=5 MHz → L3~3 µH, C2~1 nF).
BOUNDS = {v: ((0.5, 10000.0) if v.startswith("L") else (0.05, 5000.0)) for v in VARIABLES}
INITIAL_GUESS = {
    "L1": 163.9296, "C1": 6.8675,
    "L2": 6.5577, "C2": 171.6751,
    "L3": 530.5165, "C3": 2.1221,
    "L4": 6.5577, "C4": 171.6751,
    "L5": 163.9296, "C5": 6.8675,
}

TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "butterworth_bpf5.sch.tpl"
TASK_NAME = "butterworth_bpf5"
EXPORT_LAYOUT = False
SWEEP_POINTS = 301
STEP_MODE = "relative"
