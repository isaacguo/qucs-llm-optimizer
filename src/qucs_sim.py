"""
qucs_sim.py — schematic-first headless driver for Qucs-S / qucsator_rf.

Pipeline per iteration:
  1. Render templates/butterfly_stub.sch.tpl  -> circuit.sch
  2. qucs-s -n  (derive netlist from schematic) -> circuit.net
  3. qucsrflayout (official RF layout)         -> layout.svg
  4. qucsator_rf -i circuit.net -o circuit.dat
  5. Parse S[1,1] / S[2,1] dataset

The schematic is the authoritative circuit description. Simulation always
goes through qucs-s netlist export so MRSTUB property ordering stays correct.
Layout uses Qucs-RFlayout (requires MCROSS at multi-port junctions).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
SCH_TEMPLATE_PATH = _REPO_ROOT / "templates" / "butterfly_stub.sch.tpl"

# Fixed substrate: Rogers RO4003C, 20 mil, 1oz copper.
SUBSTRATE = dict(er=3.38, h_mm=0.508, t_mm=0.035, tand=0.0027, rho=1.72e-08, d_m=1.5e-07)


@dataclass
class SimResult:
    freq_hz: list[float]
    s11: list[complex]
    s21: list[complex] | None = None


def resolve_qucs_s() -> str:
    env = os.environ.get("QUCS_S")
    if env:
        return env
    candidates = [
        "/Applications/qucs-s.app/Contents/MacOS/qucs-s",
        "/mnt/c/Program Files/Qucs-S/bin/qucs-s.exe",
        "qucs-s",
    ]
    for c in candidates:
        if c == "qucs-s":
            found = shutil.which(c)
            if found:
                return found
        elif Path(c).exists():
            return c
    raise FileNotFoundError(
        "qucs-s not found; set QUCS_S to the qucs-s binary path"
    )


def resolve_qucsator() -> str:
    env = os.environ.get("QUCSATOR_RF")
    if env:
        return env
    candidates = [
        "/Applications/qucs-s.app/Contents/MacOS/bin/qucsator_rf",
        "/mnt/c/Program Files/Qucs-S/bin/qucsator_rf.exe",
        "qucsator_rf",
    ]
    for c in candidates:
        if c == "qucsator_rf":
            found = shutil.which(c)
            if found:
                return found
        elif Path(c).exists():
            return c
    raise FileNotFoundError(
        "qucsator_rf not found; set QUCSATOR_RF to the simulator binary path"
    )


def resolve_qucsrflayout() -> str:
    env = os.environ.get("QUCS_RFLAYOUT")
    if env:
        return env
    candidates = [
        _REPO_ROOT.parent / "third_party" / "Qucs-RFlayout" / "build" / "qucsrflayout",
        _REPO_ROOT / "third_party" / "Qucs-RFlayout" / "build" / "qucsrflayout",
        Path.home() / "opt" / "qucsrflayout" / "bin" / "qucsrflayout",
        Path("/mnt/c/Users") / os.environ.get("USER", "") / "Downloads" / "qucsrflayout" / "bin" / "qucsrflayout.exe",
        "qucsrflayout",
    ]
    for c in candidates:
        if c == "qucsrflayout":
            found = shutil.which(c)
            if found:
                return found
        elif Path(c).exists():
            return str(Path(c))
    raise FileNotFoundError(
        "qucsrflayout not found; set QUCS_RFLAYOUT to the binary path "
        "(build https://github.com/thomaslepoix/Qucs-RFlayout)"
    )


def _template_values(
    params: dict,
    f0_hz: float,
    sweep_start_hz: float,
    sweep_stop_hz: float,
    sweep_points: int,
) -> dict:
    return {
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


def render_schematic(
    params: dict,
    f0_hz: float,
    sweep_start_hz: float,
    sweep_stop_hz: float,
    sweep_points: int,
) -> str:
    tpl = SCH_TEMPLATE_PATH.read_text()
    return tpl.format(**_template_values(
        params, f0_hz, sweep_start_hz, sweep_stop_hz, sweep_points
    ))


def export_netlist_from_sch(sch_path: Path, net_path: Path) -> Path:
    """Derive a qucsator netlist from a .sch via `qucs-s -n`."""
    qucs_s = resolve_qucs_s()
    result = subprocess.run(
        [qucs_s, "-n", "-i", str(sch_path), "-o", str(net_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0 or not net_path.exists():
        raise RuntimeError(
            f"qucs-s -n failed (exit={result.returncode})\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return net_path


_MRSTUB_PROPS_RE = re.compile(
    r'(<MRSTUB \w+ [^"]*"Subst1" 0 )"([^"]+)" 1 "([^"]+)" 1 "([^"]+)" 1 "([^"]+)" 1 '
)


def swap_mrstub_wf_alpha(sch_text: str) -> str:
    """
    Rewrite MRSTUB property order from Qucs-S's (ri, ro, Wf, alpha) to the
    (ri, ro, alpha, Wf) order upstream Qucs-RFlayout parses positionally.
    """
    def _swap(m: re.Match) -> str:
        return (f'{m.group(1)}"{m.group(2)}" 1 "{m.group(3)}" 1 '
                f'"{m.group(5)}" 1 "{m.group(4)}" 1 ')

    return _MRSTUB_PROPS_RE.sub(_swap, sch_text)


def layout_fan_is_degenerate(svg_text: str) -> bool:
    """
    True when the exported MSW1 wing collapsed to a zero-width line.

    Upstream qucsrflayout reads MRSTUB properties positionally as
    (ri, ro, alpha, Wf), so against a Qucs-S schematic it takes the feed
    width (fractions of a mm) as the sector angle in degrees and draws a
    sliver instead of a fan. Patched builds parse the Qucs-S order and are
    unaffected, so this is a detection, not an unconditional rewrite.
    """
    m = re.search(r'id="MSW1" d="([^"]*)"', svg_text)
    if not m:
        return True
    xs = [float(x) for x in re.findall(r"[ML]\s*(-?\d+(?:\.\d+)?)", m.group(1))]
    return not xs or (max(xs) - min(xs)) < 1e-3


def _run_qucsrflayout(sch_path: Path, net_path: Path, out_dir: Path) -> Path:
    result = subprocess.run(
        [
            resolve_qucsrflayout(),
            "-i", str(sch_path),
            "-n", str(net_path),
            "-q", resolve_qucs_s(),
            "-o", str(out_dir),
            "-f", ".svg",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    produced = out_dir / f"{sch_path.stem}.svg"
    if result.returncode != 0 or not produced.exists():
        raise RuntimeError(
            f"qucsrflayout failed (exit={result.returncode})\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return produced


def export_layout_svg(sch_path: Path, net_path: Path, svg_path: Path) -> Path:
    """
    Export official RF layout SVG via qucsrflayout.

    Uses the already-exported netlist (`-n`) so Qucs is not re-invoked for
    netlisting. Output is renamed to svg_path (tool writes <stem>.svg).

    If the tool draws a degenerate butterfly, the run is retried against a
    layout-only copy of the schematic with MRSTUB properties reordered; see
    layout_fan_is_degenerate for why.
    """
    out_dir = svg_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    produced = _run_qucsrflayout(sch_path, net_path, out_dir)

    if layout_fan_is_degenerate(produced.read_text()):
        shim_path = out_dir / f"{sch_path.stem}_layout_shim.sch"
        shim_path.write_text(swap_mrstub_wf_alpha(sch_path.read_text()))
        try:
            retried = _run_qucsrflayout(shim_path, net_path, out_dir)
            if not layout_fan_is_degenerate(retried.read_text()):
                produced.unlink(missing_ok=True)
                produced = retried
            else:
                retried.unlink(missing_ok=True)
        finally:
            shim_path.unlink(missing_ok=True)

    if produced.resolve() != svg_path.resolve():
        produced.replace(svg_path)
    return svg_path


def run_qucsator(net_path: Path, dat_path: Path) -> Path:
    qucsator = resolve_qucsator()
    result = subprocess.run(
        [qucsator, "-i", str(net_path), "-o", str(dat_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0 or not dat_path.exists():
        raise RuntimeError(
            f"qucsator_rf failed (exit={result.returncode})\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return dat_path


_COMPLEX_RE = re.compile(
    r"([+-]?\d+\.\d+e[+-]\d+)([+-])j(\d+\.\d+e[+-]\d+)"
)


def parse_dataset(dat_path: Path) -> SimResult:
    text = dat_path.read_text()

    def extract_block(name: str) -> list[str]:
        m = re.search(
            rf"<(?:indep|dep) {re.escape(name)}[^>]*>\n(.*?)\n</(?:indep|dep)>",
            text,
            re.S,
        )
        if not m:
            raise ValueError(f"block '{name}' not found in dataset")
        return [ln.strip() for ln in m.group(1).splitlines() if ln.strip()]

    def parse_complex_block(name: str) -> list[complex]:
        values: list[complex] = []
        for ln in extract_block(name):
            m = _COMPLEX_RE.match(ln)
            if not m:
                raise ValueError(f"cannot parse complex value: {ln!r}")
            re_part = float(m.group(1))
            sign = 1.0 if m.group(2) == "+" else -1.0
            im_part = sign * float(m.group(3))
            values.append(complex(re_part, im_part))
        return values

    freq = [float(x) for x in extract_block("frequency")]
    s11 = parse_complex_block("S[1,1]")
    s21 = parse_complex_block("S[2,1]")
    if len(s11) != len(freq) or len(s21) != len(freq):
        raise ValueError("S-parameter block length mismatch vs frequency")
    return SimResult(freq_hz=freq, s11=s11, s21=s21)


def simulate(
    params: dict,
    f0_hz: float = 5e9,
    sweep_start_hz: float = 1e9,
    sweep_stop_hz: float = 10e9,
    sweep_points: int = 181,
    workdir: Path | None = None,
    export_layout: bool = True,
) -> SimResult:
    def _run(wd: Path) -> SimResult:
        wd.mkdir(parents=True, exist_ok=True)
        sch_path = wd / "circuit.sch"
        net_path = wd / "circuit.net"
        dat_path = wd / "circuit.dat"
        svg_path = wd / "layout.svg"

        sch_path.write_text(
            render_schematic(params, f0_hz, sweep_start_hz, sweep_stop_hz, sweep_points)
        )
        export_netlist_from_sch(sch_path, net_path)
        if export_layout:
            export_layout_svg(sch_path, net_path, svg_path)
        run_qucsator(net_path, dat_path)
        return parse_dataset(dat_path)

    if workdir is None:
        with tempfile.TemporaryDirectory() as td:
            return _run(Path(td))
    return _run(workdir)
