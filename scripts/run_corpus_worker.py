#!/usr/bin/env python3
"""Parameterized hard-success corpus worker (prefer-coverage, trash failures).

Reuses the RF teacher decide() plus light overrides from the proven worker-C
loop. Stops when corpus/index.jsonl reaches --stop-at-index (default 100).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from math import log10
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

INDEX = ROOT / "corpus" / "index.jsonl"
MAX_STEPS = 19


def _log(path: Path, msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    mac_bin = "/Applications/qucs-s.app/Contents/MacOS"
    env["PATH"] = f"{mac_bin}:{mac_bin}/bin:" + env.get("PATH", "")
    return subprocess.run(cmd, cwd=str(ROOT), env=env, text=True, capture_output=True)


def _uv(*args: str) -> subprocess.CompletedProcess:
    return _run(["uv", "run", *args])


def _trash(run_id: str) -> None:
    run_dir = ROOT / "runs" / run_id
    if not run_dir.exists():
        return
    trash = shutil.which("trash")
    if trash:
        subprocess.run([trash, str(run_dir)], check=False)
    else:
        shutil.rmtree(run_dir, ignore_errors=True)


def _index_len() -> int:
    if not INDEX.is_file():
        return 0
    return sum(1 for line in INDEX.open(encoding="utf-8") if line.strip())


def _db(mag: float) -> float:
    return 20.0 * log10(mag) if mag > 0 else float("-inf")


def _load_teacher():
    path = ROOT / "scripts" / "generate_teacher_trajectories.py"
    spec = importlib.util.spec_from_file_location("teacher_gen", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.decide, mod._state


def _near_bound(p: dict, key: str, direction: str) -> bool:
    bounds = {
        "ri": (0.1, 1.5),
        "ro": (3.0, 14.0),
        "alpha": (30.0, 150.0),
        "Wf": (0.2, 2.5),
        "Lc": (0.2, 10.0),
    }
    lo, hi = bounds[key]
    v = float(p[key])
    if "increase" in direction:
        return v >= hi - 1e-6
    if "decrease" in direction:
        return v <= lo + 1e-6
    return False


def decide_plus(st: dict, teacher_decide) -> tuple[dict, str, str, bool]:
    intent, note, think, met = teacher_decide(st)
    if met:
        return intent, note, think, met

    h = st["history"][-1]
    c, p = h["cost"], h["params"]
    tf = st["goal"]["target_freq_hz"] / 1e9
    td = float(st["goal"]["target_depth_db"])
    bf = c["best_freq_hz"] / 1e9
    cur_db = _db(c["total_cost"])
    df = bf - tf
    gap = cur_db - td
    hist = [x for x in st["history"] if x.get("intent")]

    if len(st["history"]) >= 2 and hist:
        prev_db = _db(st["history"][-2]["cost"]["total_cost"])
        last_intent = hist[-1].get("intent") or {}
        if cur_db > prev_db + 4.0 and last_intent:
            k, v = next(iter(last_intent.items()))
            if "increase" in v:
                rev = v.replace("increase", "decrease")
            elif "decrease" in v:
                rev = v.replace("decrease", "increase")
            else:
                rev = None
            if rev and not _near_bound(p, k, rev):
                intent = {k: rev}
                note = f"regret undo {last_intent}"
                think = f"Undo worsening move {last_intent} via {intent}."
                return intent, note, think, False

    if abs(df) <= 0.08 and cur_db <= -45.0 and gap > 2:
        if p["alpha"] < 148:
            intent = {"alpha": "increase_slight"}
        elif p["Lc"] > 0.5:
            intent = {"Lc": "decrease_slight"}
        else:
            intent = {"ri": "decrease_slight"} if df <= 0 else {"ri": "increase_slight"}
        note = f"goldilocks deepen via {intent}"
        think = f"On-target deepen with {intent}."
        return intent, note, think, False

    if abs(df) < 0.25 and gap > 3 and intent.get("alpha", "").startswith("decrease"):
        if p["alpha"] < 148:
            intent = {"alpha": "increase_slight"}
        elif p["Lc"] > 1.0:
            intent = {"Lc": "decrease_slight"}
        else:
            intent = {"ri": "decrease_slight"} if df <= 0 else {"ri": "increase_slight"}
        note = f"block alpha-decrease; {intent}"
        think = note
        return intent, note, think, False

    if intent:
        k, v = next(iter(intent.items()))
        if _near_bound(p, k, v):
            alts = [
                {"ro": "decrease_slight"} if df < 0 else {"ro": "increase_slight"},
                {"ri": "decrease_slight"} if df < 0 else {"ri": "increase_slight"},
                {"alpha": "increase_slight"},
                {"Lc": "decrease_slight"},
                {"Wf": "decrease_slight"},
            ]
            for alt in alts:
                ak, av = next(iter(alt.items()))
                if ak != k and not _near_bound(p, ak, av):
                    intent = alt
                    note = f"escape bound on {k}; try {intent}"
                    think = note
                    return intent, note, think, False

    if len(hist) >= 3:
        last3 = [x.get("intent") for x in hist[-3:]]
        if last3[0] == last3[1] == last3[2] == intent:
            intent = (
                {"Lc": "decrease_slight"}
                if p["Lc"] > 1.5
                else {"alpha": "increase_slight"}
            )
            note = "break repeated intent stall"
            think = note
            return intent, note, think, False

    return intent, note, think, False


def run_one(run_id: str, seed: int, teacher_decide, _state) -> dict:
    r = _uv(
        "qucs-corpus",
        "assign",
        "--run",
        run_id,
        "--seed",
        str(seed),
        "--prefer-coverage",
        "--index",
        "corpus/index.jsonl",
    )
    if r.returncode != 0:
        return {"ok": False, "stage": "assign", "err": (r.stderr or r.stdout)[-500:]}

    r = _uv("python", "run_step.py", "init", "--run", run_id)
    if r.returncode != 0:
        return {"ok": False, "stage": "init", "err": (r.stderr or r.stdout)[-500:]}

    think_dir = ROOT / "tmp" / "agent_think" / run_id
    think_dir.mkdir(parents=True, exist_ok=True)

    for step_i in range(MAX_STEPS):
        _uv("python", "run_step.py", "observe", "--run", run_id)
        st = _state(run_id)
        intent, note, think, met = decide_plus(st, teacher_decide)
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
        )
        if r.returncode != 0:
            return {
                "ok": False,
                "stage": f"step_{step_i}",
                "err": (r.stderr or r.stdout)[-500:],
            }

    st = _state(run_id)
    cur = _db(st["history"][-1]["cost"]["total_cost"])
    td = float(st["goal"]["target_depth_db"])
    if cur > td:
        return {"ok": False, "stage": "cap", "best_db": cur, "goal_db": td}

    r = _uv("qucs-corpus", "gate", "--run", run_id)
    st = _state(run_id)
    corpus = st.get("corpus") or {}
    return {
        "ok": bool(corpus.get("eligible")),
        "stage": "gate",
        "best_db": corpus.get("best_db"),
        "goal": st.get("goal"),
        "err": None if r.returncode == 0 else (r.stderr or r.stdout),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="worker label for logs")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--max-inits", type=int, default=120)
    parser.add_argument("--stop-at-index", type=int, default=100)
    parser.add_argument("--prefix", default="t")
    args = parser.parse_args()

    log_path = ROOT / "outputs" / "logs" / f"agent_worker_{args.name}.log"
    teacher_decide, _state = _load_teacher()
    attempts = 0
    successes = 0
    ok_ids: list[str] = []
    before = _index_len()
    _log(
        log_path,
        f"START worker={args.name} range={args.prefix}{args.start:04d}-"
        f"{args.prefix}{args.end:04d} max_inits={args.max_inits} "
        f"stop_at={args.stop_at_index} index={before}",
    )

    for n in range(args.start, args.end + 1):
        if _index_len() >= args.stop_at_index:
            _log(log_path, f"STOP index>={args.stop_at_index}")
            break
        if attempts >= args.max_inits:
            _log(log_path, f"STOP max_inits={args.max_inits}")
            break
        run_id = f"{args.prefix}{n:04d}"
        run_dir = ROOT / "runs" / run_id
        if run_dir.exists():
            try:
                st = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
                eligible = bool((st.get("corpus") or {}).get("eligible"))
            except Exception:  # noqa: BLE001
                eligible = False
            if eligible:
                _log(log_path, f"SKIP {run_id} eligible exists")
                continue
            _trash(run_id)
        seed = args.base_seed + n * 97
        attempts += 1
        _log(log_path, f"ATTEMPT {attempts}/{args.max_inits} {run_id} seed={seed}")
        t0 = time.time()
        try:
            result = run_one(run_id, seed, teacher_decide, _state)
        except Exception as exc:  # noqa: BLE001
            result = {"ok": False, "stage": "exception", "err": str(exc)}
        dt = time.time() - t0
        if result.get("ok"):
            successes += 1
            ok_ids.append(run_id)
            _log(
                log_path,
                f"OK {run_id} dt={dt:.1f}s best_db={result.get('best_db')} "
                f"index={_index_len()} successes={successes}/{attempts}",
            )
        else:
            _log(
                log_path,
                f"FAIL {run_id} stage={result.get('stage')} dt={dt:.1f}s "
                f"best_db={result.get('best_db')} goal_db={result.get('goal_db')} "
                f"err={result.get('err')}",
            )
            _trash(run_id)

    _log(
        log_path,
        f"DONE worker={args.name} attempts={attempts} successes={successes} "
        f"index {before}->{_index_len()} ok_ids={ok_ids}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
