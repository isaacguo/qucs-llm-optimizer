#!/usr/bin/env python3
"""Generate hard-successful teacher trajectories via assign → step loop → gate.

Plays the strategy layer with RF heuristics + written <thinking> on each step,
using the real Qucs path through run_step / qucs_sim. Intended to fill
corpus/index.jsonl (hundreds of hard-successful runs).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from math import log10
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from corpus.index import load_index  # noqa: E402


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    mac_bin = "/Applications/qucs-s.app/Contents/MacOS"
    mac_tools = f"{mac_bin}/bin"
    env["PATH"] = f"{mac_bin}:{mac_tools}:" + env.get("PATH", "")
    return subprocess.run(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=check,
    )


def _uv(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return _run(["uv", "run", *args], check=check)


def _state(run_id: str) -> dict:
    return json.loads((ROOT / "runs" / run_id / "state.json").read_text())


def _db(mag: float) -> float:
    return 20.0 * log10(mag) if mag > 0 else float("-inf")


def _fit_ro_for_target(st: dict, tf_ghz: float) -> float | None:
    """Two-point fit f = k/(ro + c); return predicted ro for tf, else None."""
    pts = []
    for e in st["history"]:
        cost = e.get("cost") or {}
        params = e.get("params") or {}
        if not cost or "ro" not in params:
            continue
        f = cost["best_freq_hz"] / 1e9
        ro = float(params["ro"])
        if f > 0.5:
            pts.append((ro, f))
    # unique by ro
    by_ro: dict[float, float] = {}
    for ro, f in pts:
        by_ro[round(ro, 4)] = f
    pts = sorted(by_ro.items())
    if len(pts) < 2:
        return None
    (ro1, f1), (ro2, f2) = pts[-2], pts[-1]
    if abs(f1 - f2) < 1e-6 or abs(ro1 - ro2) < 1e-9:
        return None
    # f = k/(ro+c) => f1*(ro1+c)=f2*(ro2+c) => c=(f2*ro2-f1*ro1)/(f1-f2)
    c = (f2 * ro2 - f1 * ro1) / (f1 - f2)
    k = f1 * (ro1 + c)
    if abs(tf_ghz) < 1e-9:
        return None
    return k / tf_ghz - c


def decide(st: dict) -> tuple[dict[str, str], str, str, bool]:
    goal = st["goal"]
    hist = [h for h in st["history"] if h.get("intent")]
    h = st["history"][-1]
    c, p = h["cost"], h["params"]
    tf = goal["target_freq_hz"] / 1e9
    td = float(goal["target_depth_db"])
    bf = c["best_freq_hz"] / 1e9
    cur_db = _db(c["total_cost"])
    ratio = tf / max(bf, 0.1)
    met = cur_db <= td
    df = bf - tf  # >0 means notch too high in frequency
    notch_db = _db(c["best_s21_mag"])

    # Recovery: resonator collapsed — grow fan radius to restore a notch
    if notch_db > -20.0:
        intent = {"ro": "increase_strong"} if p["ro"] < 13.5 else {"ri": "increase"}
        note = f"notch collapsed ({notch_db:.1f}dB); restore resonator"
        think = (
            f"Deepest |S21| is only {notch_db:.1f} dB — the shunt stub is not resonating. "
            f"Restore a transmission zero with {intent} before targeting {tf:.3f} GHz."
        )
        return intent, note, think, met

    # Phase A: coarse frequency move
    if abs(df) > 0.25:
        pred = _fit_ro_for_target(st, tf)
        if pred is not None and abs(pred - p["ro"]) > 0.05:
            if pred < p["ro"] - 0.8:
                intent = {"ro": "decrease_strong"}
            elif pred < p["ro"] - 0.25:
                intent = {"ro": "decrease"}
            elif pred < p["ro"]:
                intent = {"ro": "decrease_slight"}
            elif pred > p["ro"] + 0.8:
                intent = {"ro": "increase_strong"}
            elif pred > p["ro"] + 0.25:
                intent = {"ro": "increase"}
            else:
                intent = {"ro": "increase_slight"}
            note = f"fit wants ro~{pred:.3f} (now {p['ro']:.3f}); notch {bf:.2f}->{tf:.2f}"
            think = (
                f"Two-point radial fit targets ro≈{pred:.3f} mm for {tf:.3f} GHz "
                f"(now notch {bf:.3f} GHz at ro={p['ro']:.3f}). Intent {intent}."
            )
            if p["ro"] <= 3.05 and pred < p["ro"]:
                intent = {"ri": "decrease_strong"}
                note = "ro at floor; shrink ri per fit"
            if p["ro"] >= 13.9 and pred > p["ro"]:
                intent = {"ri": "increase_strong"}
                note = "ro at ceil; grow ri per fit"
            return intent, note, think, met

        if p["ro"] <= 3.05 and bf < tf - 0.1:
            intent = {"ri": "decrease_strong"}
            note = "ro at floor; shrink ri to raise notch"
        elif p["ro"] >= 13.9 and bf > tf + 0.1:
            intent = {"ri": "increase_strong"}
            note = "ro at ceil; grow ri to lower notch"
        elif ratio > 1.3:
            intent = {"ro": "decrease_strong"}
            note = f"notch {bf:.2f}GHz vs {tf:.2f}; shrink ro strong"
        elif ratio > 1.08:
            intent = {"ro": "decrease"}
            note = f"notch {bf:.2f}GHz still low; decrease ro"
        elif df < -0.05:
            intent = {"ro": "decrease_slight"}
            note = f"notch {bf:.2f}GHz slightly low"
        elif ratio < 0.77:
            intent = {"ro": "increase_strong"}
            note = f"notch {bf:.2f}GHz too high; grow ro strong"
        elif ratio < 0.93:
            intent = {"ro": "increase"}
            note = f"notch {bf:.2f}GHz high; increase ro"
        else:
            intent = {"ro": "increase_slight"}
            note = f"notch {bf:.2f}GHz slightly high"
        think = (
            f"Evidence: deepest notch at {bf:.3f} GHz; |S21| at target is {cur_db:.1f} dB. "
            f"Target {tf:.3f} GHz implies frequency ratio ~{ratio:.2f}. "
            f"Apply {intent} to move resonance before chasing depth."
        )
        return intent, note, think, met

    # Phase B: fine frequency (notch near target but |S21|@target still shallow)
    if abs(df) > 0.03 and cur_db > td + 2:
        if df < 0:
            intent = {"ro": "decrease_slight"} if abs(df) < 0.12 else {"ri": "decrease_slight"}
        else:
            intent = {"ro": "increase_slight"} if abs(df) < 0.12 else {"ri": "increase_slight"}
        # avoid repeating same dead move
        if hist and hist[-1].get("intent") == intent:
            intent = {"ri": "decrease_slight"} if df < 0 else {"ri": "increase_slight"}
        note = f"fine-tune freq {bf:.3f}->{tf:.3f} (df={df:+.3f})"
        think = (
            f"Notch {bf:.3f} GHz is close to {tf:.3f} but target-bin |S21| is {cur_db:.1f} dB. "
            f"Nudge {intent} to land the transmission zero on the goal frequency."
        )
        return intent, note, think, met

    # Phase C: deepen transmission zero
    last = (hist[-1].get("intent") if hist else {}) or {}
    last_keys = list(last.keys())
    if cur_db > td + 15:
        intent = {"alpha": "increase"} if p["alpha"] < 110 else {"alpha": "decrease"}
        if last_keys == ["alpha"]:
            intent = {"Lc": "decrease_slight"} if p["Lc"] > 1.0 else {"Wf": "decrease_slight"}
    elif cur_db > td + 5:
        intent = {"alpha": "increase_slight"}
        if "alpha" in last:
            intent = {"ri": "decrease_slight"} if df <= 0 else {"ri": "increase_slight"}
    else:
        intent = {"alpha": "increase_slight"}
        if last == intent:
            intent = {"ri": "decrease_slight"} if df <= 0 else {"ri": "increase_slight"}
        if last_keys == list(intent.keys()):
            intent = {"Lc": "increase_slight"} if p["Lc"] < 8 else {"Lc": "decrease_slight"}
    note = f"freq~{bf:.2f}; deepen {cur_db:.1f}dB toward {td:.1f}dB"
    think = (
        f"Frequency aligned (~{bf:.3f} vs {tf:.3f} GHz). "
        f"Depth {cur_db:.1f} dB vs goal {td:.1f} dB; refine with {intent}."
    )
    return intent, note, think, met


def generate_one(run_id: str, *, seed: int, prefer_coverage: bool) -> dict:
    assign = ["qucs-corpus", "assign", "--run", run_id, "--seed", str(seed)]
    if prefer_coverage:
        assign.append("--prefer-coverage")
    r = _uv(*assign, check=False)
    if r.returncode != 0:
        return {"run_id": run_id, "ok": False, "stage": "assign", "err": r.stderr or r.stdout}

    r = _uv("python", "run_step.py", "init", "--run", run_id, check=False)
    if r.returncode != 0:
        return {"run_id": run_id, "ok": False, "stage": "init", "err": r.stderr or r.stdout}

    think_dir = ROOT / "runs" / run_id / "think"
    think_dir.mkdir(parents=True, exist_ok=True)

    for step_i in range(19):
        st = _state(run_id)
        intent, note, think, met = decide(st)
        if met:
            break
        think_path = think_dir / f"step_{step_i:02d}.md"
        think_path.write_text(think + "\n", encoding="utf-8")
        r = _uv(
            "python",
            "run_step.py",
            "step",
            "--run",
            run_id,
            "--intent",
            json.dumps(intent),
            "--note",
            note,
            "--thinking-file",
            str(think_path),
            check=False,
        )
        if r.returncode != 0:
            return {
                "run_id": run_id,
                "ok": False,
                "stage": f"step_{step_i}",
                "err": r.stderr or r.stdout,
            }
    else:
        st = _state(run_id)
        cur = _db(st["history"][-1]["cost"]["total_cost"])
        return {
            "run_id": run_id,
            "ok": False,
            "stage": "cap",
            "best_db": cur,
            "goal_db": st["goal"]["target_depth_db"],
        }

    r = _uv("qucs-corpus", "gate", "--run", run_id, check=False)
    st = _state(run_id)
    corpus = st.get("corpus") or {}
    return {
        "run_id": run_id,
        "ok": bool(corpus.get("eligible")),
        "stage": "gate",
        "best_db": corpus.get("best_db"),
        "goal": st.get("goal"),
        "gate_out": (r.stdout or "")[-400:],
        "err": None if r.returncode == 0 else (r.stderr or r.stdout),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=50, help="how many new runs to attempt")
    parser.add_argument("--start-index", type=int, default=2, help="first numeric id (tNNNN)")
    parser.add_argument("--prefix", default="t")
    parser.add_argument("--base-seed", type=int, default=10_000, help="unique seed = base + n*97")
    parser.add_argument("--prefer-coverage", action="store_true", default=True)
    parser.add_argument("--no-prefer-coverage", action="store_false", dest="prefer_coverage")
    parser.add_argument("--log", default="outputs/logs/teacher_gen.log")
    args = parser.parse_args()

    log_path = ROOT / args.log
    log_path.parent.mkdir(parents=True, exist_ok=True)
    index_path = ROOT / "corpus" / "index.jsonl"

    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        if sys.stdout.isatty():
            print(line, flush=True)

    before = len(load_index(index_path)) if index_path.exists() else 0
    log(f"start count={args.count} index_before={before}")

    ok_n = 0
    for i in range(args.count):
        n = args.start_index + i
        run_id = f"{args.prefix}{n:04d}"
        seed = args.base_seed + n * 97
        if (ROOT / "runs" / run_id).exists():
            log(f"skip existing {run_id}")
            continue
        t0 = time.time()
        result = generate_one(run_id, seed=seed, prefer_coverage=args.prefer_coverage)
        dt = time.time() - t0
        if result.get("ok"):
            ok_n += 1
            log(
                f"OK {run_id} best_db={result.get('best_db'):.2f} "
                f"goal_db={result['goal']['target_depth_db']:.1f} "
                f"freq={result['goal']['target_freq_hz']/1e9:.3f}GHz {dt:.1f}s"
            )
        else:
            log(
                f"FAIL {run_id} stage={result.get('stage')} "
                f"best_db={result.get('best_db')} goal_db={result.get('goal_db')} "
                f"{dt:.1f}s err={result.get('err')!r}"
            )

    after = len(load_index(index_path)) if index_path.exists() else 0
    log(f"done ok={ok_n}/{args.count} index {before}->{after}")


if __name__ == "__main__":
    main()
