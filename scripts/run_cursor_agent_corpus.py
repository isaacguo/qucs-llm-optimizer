#!/usr/bin/env python3
"""Drive Cursor ``agent`` CLI to grow the hard-success corpus.

Each attempt:
  1. assign --prefer-coverage (empty / least-filled bins first)
  2. Cursor agent plays strategy via run_step init/observe/step
  3. gate into corpus/index.jsonl on hard success
  4. trash the run directory on failure

Designed to run detached (tmux / start_new_session) for long corpus fills.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "corpus" / "index.jsonl"
RUNS = ROOT / "runs"


def _log(path: Path, msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def _index_len() -> int:
    if not INDEX.is_file():
        return 0
    n = 0
    with INDEX.open(encoding="utf-8") as fh:
        for raw in fh:
            if raw.strip():
                n += 1
    return n


def _run_indexed(run_id: str) -> bool:
    if not INDEX.is_file():
        return False
    with INDEX.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if json.loads(line).get("run_id") == run_id:
                return True
    return False


def _eligible(run_id: str) -> bool:
    state_path = RUNS / run_id / "state.json"
    if not state_path.is_file():
        return False
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return bool((state.get("corpus") or {}).get("eligible"))


def _trash_run(run_id: str) -> None:
    run_dir = RUNS / run_id
    if not run_dir.exists():
        return
    trash = shutil.which("trash")
    if trash:
        subprocess.run([trash, str(run_dir)], check=False)
    else:
        shutil.rmtree(run_dir, ignore_errors=True)


def _agent_prompt(run_id: str, seed: int) -> str:
    return f"""You are the Cursor RF strategy agent for the qucs-llm-optimizer corpus cold-start.

Workspace root: {ROOT}
Do all work only inside this repo. Do NOT commit or push. Do NOT edit training code.

Your ONLY job for this turn is ONE trajectory: run_id={run_id}, assign seed hint={seed}.

Procedure (follow exactly):
1) Ensure Qucs is on PATH:
   export PATH="/Applications/qucs-s.app/Contents/MacOS:/Applications/qucs-s.app/Contents/MacOS/bin:$PATH"
2) Assign goal with coverage preference:
   cd {ROOT}
   uv run qucs-corpus assign --run {run_id} --seed {seed} --prefer-coverage --index corpus/index.jsonl
3) Init baseline simulation:
   uv run python run_step.py init --run {run_id}
4) Strategy loop (max 19 steps):
   - uv run python run_step.py observe --run {run_id}
   - Read runs/{run_id}/state.json. Reason like an RF engineer about notch freq vs target and depth vs target_depth_db.
   - Choose a qualitative intent dict using keys among ri,ro,alpha,Wf,Lc and values among
     increase_strong|increase|increase_slight|hold|decrease_slight|decrease|decrease_strong
     (omit hold keys; only emit keys you change).
   - Heuristics from prior successful runs: notch freq roughly follows f≈k/(ro+c); ri moves frequency ~4× more per mm than ro; alpha is the fine depth knob; if deepest notch is shallow (>-20 dB) restore resonator first (grow ro/ri) before chasing the target.
   - Write short reasoning to tmp/agent_think/{run_id}/step_XX.md
   - uv run python run_step.py step --run {run_id} --intent '<json>' --note '<short>' --thinking-file tmp/agent_think/{run_id}/step_XX.md
   - Stop early when |S21| at target (total_cost in dB) is <= goal.target_depth_db.
5) Gate:
   uv run qucs-corpus gate --run {run_id}
6) If NOT eligible / gate did not append: delete the run with `trash runs/{run_id}` (or rm -rf if trash missing).
7) Print a final one-liner exactly like:
   RESULT {run_id} OK index=N
   or
   RESULT {run_id} FAIL reason=...

Be decisive; prefer fewer high-quality steps over wandering.
"""


def _run_agent(run_id: str, seed: int, *, model: str, timeout_s: int) -> int:
    cmd = [
        "agent",
        "-p",
        "--force",
        "--trust",
        "--workspace",
        str(ROOT),
        "--model",
        model,
        "--output-format",
        "text",
        _agent_prompt(run_id, seed),
    ]
    env = os.environ.copy()
    mac_bin = "/Applications/qucs-s.app/Contents/MacOS"
    env["PATH"] = f"{mac_bin}:{mac_bin}/bin:" + env.get("PATH", "")
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or "") + "\n" + (exc.stderr or "")
        (ROOT / "outputs" / "logs" / f"agent_{run_id}.out").write_text(out, encoding="utf-8")
        return 124
    out = (proc.stdout or "") + "\n--- STDERR ---\n" + (proc.stderr or "")
    (ROOT / "outputs" / "logs" / f"agent_{run_id}.out").write_text(out, encoding="utf-8")
    return int(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=200, help="attempt budget")
    parser.add_argument("--start-index", type=int, default=201, help="first tNNNN id")
    parser.add_argument("--prefix", default="t")
    parser.add_argument("--base-seed", type=int, default=30_000)
    parser.add_argument("--model", default="auto")
    parser.add_argument("--timeout-s", type=int, default=900, help="per-trajectory agent timeout")
    parser.add_argument("--log", default="outputs/logs/cursor_agent_corpus.log")
    parser.add_argument(
        "--stop-at-index",
        type=int,
        default=0,
        help="optional soft stop when index length reaches N (0=never)",
    )
    args = parser.parse_args()

    log_path = ROOT / args.log
    before = _index_len()
    _log(log_path, f"start count={args.count} index_before={before} model={args.model}")

    ok_n = 0
    for i in range(args.count):
        if args.stop_at_index and _index_len() >= args.stop_at_index:
            _log(log_path, f"soft-stop index>={args.stop_at_index}")
            break
        n = args.start_index + i
        run_id = f"{args.prefix}{n:04d}"
        seed = args.base_seed + n * 97
        if (RUNS / run_id).exists():
            _log(log_path, f"skip existing {run_id}")
            continue
        t0 = time.time()
        _log(log_path, f"BEGIN {run_id} seed={seed}")
        rc = _run_agent(run_id, seed, model=args.model, timeout_s=args.timeout_s)
        dt = time.time() - t0
        success = _run_indexed(run_id) or _eligible(run_id)
        if success and not _run_indexed(run_id):
            # Agent marked eligible but skipped append — try gate once more.
            subprocess.run(
                ["uv", "run", "qucs-corpus", "gate", "--run", run_id],
                cwd=str(ROOT),
                check=False,
            )
            success = _run_indexed(run_id)
        if success:
            ok_n += 1
            _log(log_path, f"OK {run_id} rc={rc} {dt:.0f}s index={_index_len()}")
        else:
            _trash_run(run_id)
            _log(log_path, f"FAIL {run_id} rc={rc} {dt:.0f}s trashed index={_index_len()}")

    _log(log_path, f"done ok={ok_n}/{args.count} index {before}->{_index_len()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
