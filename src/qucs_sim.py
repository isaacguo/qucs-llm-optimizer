"""
qucs_sim.py — headless driver for Qucs-S / qucsator_rf.exe on the Windows host,
invoked from WSL via the transparent Windows<->WSL filesystem interop.

Verified empirically in this session (2026-09-07):
  - qucsator_rf.exe accepts the same textual netlist format that
    `qucs-s.exe -n` emits; we author it by hand (see templates/) instead of
    building a .sch schematic, since positional .sch property ordering for
    components such as MRSTUB does not match the internal C++ struct order
    and is a source of silent parameter mis-assignment.
  - MRSTUB required properties (confirmed via `qucsator_rf.exe -l`):
        ri, ro, Wf, alpha, Subst, EffDimens, Model
    ri = inner transition radius, ro = outer/fan radius, Wf = feed line
    width, alpha = subtended sector angle (deg, 0-180).
  - The Windows .exe can read/write files that live on the native WSL
    filesystem directly (no need to stage under /mnt/c).
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

QUCSATOR = "/mnt/c/Program Files/Qucs-S/bin/qucsator_rf.exe"
TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "templates" / "butterfly_stub.net.tpl"

# Fixed substrate: Rogers RO4003C, 20 mil, 1oz copper.
SUBSTRATE = dict(er=3.38, h_mm=0.508, t_mm=0.035, tand=0.0027, rho=1.72e-08, d_m=1.5e-07)


@dataclass
class SimResult:
    freq_hz: list[float]
    s11: list[complex]


def render_netlist(params: dict, f0_hz: float, sweep_start_hz: float, sweep_stop_hz: float,
                    sweep_points: int) -> str:
    tpl = TEMPLATE_PATH.read_text()
    values = {
        "f0_hz": f0_hz,
        "wf_mm": params["Wf"],
        "lc_mm": params["Lc"],
        "ri_mm": params["ri"],
        "ro_mm": params["ro"],
        "alpha_deg": params["alpha"],
        "sweep_start_hz": sweep_start_hz,
        "sweep_stop_hz": sweep_stop_hz,
        "sweep_points": sweep_points,
        **SUBSTRATE,
    }
    return tpl.format(**values)


def run_qucsator(netlist_text: str, workdir: Path) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    net_path = workdir / "circuit.net"
    dat_path = workdir / "circuit.dat"
    net_path.write_text(netlist_text)
    result = subprocess.run(
        [QUCSATOR, "-i", str(net_path), "-o", str(dat_path)],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0 or not dat_path.exists():
        raise RuntimeError(
            f"qucsator_rf.exe failed (exit={result.returncode})\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return dat_path


_COMPLEX_RE = re.compile(
    r"([+-]?\d+\.\d+e[+-]\d+)([+-])j(\d+\.\d+e[+-]\d+)"
)


def parse_dataset(dat_path: Path) -> SimResult:
    text = dat_path.read_text()

    def extract_block(name: str) -> list[str]:
        m = re.search(rf"<(?:indep|dep) {re.escape(name)}[^>]*>\n(.*?)\n</(?:indep|dep)>", text, re.S)
        if not m:
            raise ValueError(f"block '{name}' not found in dataset")
        return [ln.strip() for ln in m.group(1).splitlines() if ln.strip()]

    freq_lines = extract_block("frequency")
    freq = [float(x) for x in freq_lines]

    s11_lines = extract_block("S[1,1]")
    s11: list[complex] = []
    for ln in s11_lines:
        m = _COMPLEX_RE.match(ln)
        if not m:
            raise ValueError(f"cannot parse complex value: {ln!r}")
        re_part = float(m.group(1))
        sign = 1.0 if m.group(2) == "+" else -1.0
        im_part = sign * float(m.group(3))
        s11.append(complex(re_part, im_part))

    return SimResult(freq_hz=freq, s11=s11)


def simulate(params: dict, f0_hz: float = 5e9, sweep_start_hz: float = 1e9,
             sweep_stop_hz: float = 10e9, sweep_points: int = 181,
             workdir: Path | None = None) -> SimResult:
    netlist_text = render_netlist(params, f0_hz, sweep_start_hz, sweep_stop_hz, sweep_points)
    if workdir is None:
        with tempfile.TemporaryDirectory() as td:
            dat_path = run_qucsator(netlist_text, Path(td))
            return parse_dataset(dat_path)
    else:
        dat_path = run_qucsator(netlist_text, workdir)
        return parse_dataset(dat_path)
