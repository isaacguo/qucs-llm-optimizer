#!/usr/bin/env python3
"""Generalization probe: random CFs × BW=15 MHz under skills-policy v3.

Start from task initial_guess (no analytic target L/C). Compares against
the previous BW=5 linspace ablation only as a headline rate reference.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bpf_tuning_skills import (  # noqa: E402
    choose_intent_from_skills,
    format_skills_reasoning,
)
from goals_bpf import BpfGoalSpec  # noqa: E402

os.environ.setdefault("QUCS_S", "/Applications/qucs-s.app/Contents/MacOS/qucs-s")
os.environ.setdefault(
    "QUCSATOR_RF", "/Applications/qucs-s.app/Contents/MacOS/bin/qucsator_rf"
)

PY = sys.executable
MAX_ITER = 24


def goal_for(f0: float, bw_hz: float) -> BpfGoalSpec:
    return BpfGoalSpec(
        f_low_hz=f0 - bw_hz / 2,
        f_high_hz=f0 + bw_hz / 2,
        passband_il_max_db=-1.0,
        stopband_atten_min_db=-20.0,
        stopband_guard_hz=10e6,
        sweep_hz=(0.0, 300e6),
    )


def run_cmd(args: list[str]) -> str:
    r = subprocess.run([PY, *args], capture_output=True, text=True, cwd=ROOT)
    out = r.stdout + r.stderr
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed {args}\n{out}")
    return out


def last_state(run: str) -> dict:
    return json.loads((ROOT / "runs" / run / "state.json").read_text())


def skills_crawl(run: str, goal: BpfGoalSpec) -> dict:
    think_dir = ROOT / "runs" / run / "think"
    think_dir.mkdir(parents=True, exist_ok=True)
    prev_cost = None
    prev_thinking = None
    while True:
        st = last_state(run)
        hist = st["history"][-1]
        cost = hist["cost"]
        if cost.get("goal_met"):
            return cost | {"iteration": st["iteration"]}
        if st["iteration"] >= MAX_ITER:
            return cost | {"iteration": st["iteration"]}
        intent, diag = choose_intent_from_skills(
            cost,
            goal,
            st["iteration"],
            prev_cost=prev_cost,
            prev_thinking=prev_thinking,
        )
        if all(v == "hold" for v in intent.values()):
            return cost | {"iteration": st["iteration"]}
        obs = subprocess.run(
            [PY, "run_step.py", "observe", "--run", run],
            capture_output=True,
            text=True,
            cwd=ROOT,
        ).stdout
        if "=== SYSTEM: BPF tuning skills" not in obs:
            raise RuntimeError("skills system block missing from observe output")
        think = think_dir / f"step_{st['iteration']:02d}.md"
        think.write_text(
            format_skills_reasoning(
                cost, goal, intent, diag, iteration=st["iteration"]
            )
        )
        run_cmd(
            [
                "run_step.py",
                "step",
                "--run",
                run,
                "--intent",
                json.dumps(intent),
                "--note",
                "skills-policy v3 gen step",
                "--thinking-file",
                str(think),
            ]
        )
        prev_cost = cost
        prev_thinking = diag


def sample_cfs(
    *,
    n: int,
    seed: int,
    cf_lo_mhz: float,
    cf_hi_mhz: float,
    bw_mhz: float,
) -> list[float]:
    half = bw_mhz / 2
    lo = max(cf_lo_mhz, half + 1.0)  # keep passband above ~0
    hi = min(cf_hi_mhz, 300.0 - half - 1.0)
    if hi <= lo:
        raise ValueError(f"CF range empty after BW margin: lo={lo} hi={hi}")
    rng = np.random.default_rng(seed)
    raw = rng.uniform(lo, hi, size=n)
    return sorted(float(x) * 1e6 for x in raw)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20260911)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--bw-mhz", type=float, default=15.0)
    ap.add_argument("--cf-lo-mhz", type=float, default=40.0)
    ap.add_argument("--cf-hi-mhz", type=float, default=220.0)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    bw_hz = args.bw_mhz * 1e6
    bw_tag = f"{args.bw_mhz:g}".replace(".", "p")
    tag = args.tag or f"gen_bw{bw_tag}_s{args.seed}"
    cfs = sample_cfs(
        n=args.n,
        seed=args.seed,
        cf_lo_mhz=args.cf_lo_mhz,
        cf_hi_mhz=args.cf_hi_mhz,
        bw_mhz=args.bw_mhz,
    )

    rows = []
    print(
        f"Skills generalization seed={args.seed} BW={args.bw_mhz:g} MHz "
        f"n={args.n} tag={tag}"
    )
    print(f"CFs (MHz): {[round(c / 1e6, 2) for c in cfs]}")

    for f0 in cfs:
        cf_mhz = round(f0 / 1e6, 2)
        cf_tag = f"{cf_mhz:.1f}".replace(".", "p")
        run = f"bpf_cf{cf_tag}_bw{bw_tag}_{tag}"
        d = ROOT / "runs" / run
        if d.exists():
            shutil.rmtree(d)
        g = goal_for(f0, bw_hz)
        run_cmd(
            [
                "run_step.py",
                "init",
                "--run",
                run,
                "--task",
                "butterworth_bpf5",
                "--goal-json",
                json.dumps(g.to_dict()),
                "--note",
                f"skills-gen start CF={cf_mhz}MHz BW={args.bw_mhz:g} seed={args.seed}",
            ]
        )
        seed_cost = last_state(run)["history"][-1]["cost"]
        final = skills_crawl(run, g)
        run_cmd(
            [
                "run_step.py",
                "conclude",
                "--run",
                run,
                "--thinking",
                f"Skills-gen finished CF={cf_mhz} MHz BW={args.bw_mhz:g}; "
                f"goal_met={final.get('goal_met')} iter={final['iteration']}.",
            ]
        )
        run_cmd(["run_step.py", "html", "--run", run])
        row = {
            "seed": args.seed,
            "bw_mhz": args.bw_mhz,
            "cf_mhz": cf_mhz,
            "f_low_mhz": round((f0 - bw_hz / 2) / 1e6, 2),
            "f_high_mhz": round((f0 + bw_hz / 2) / 1e6, 2),
            "init_goal_met": bool(seed_cost.get("goal_met")),
            "init_pb_db": round(float(seed_cost["passband_min_s21_db"]), 2),
            "init_sb_db": round(float(seed_cost["stopband_max_s21_db"]), 2),
            "skills_goal_met": bool(final.get("goal_met")),
            "skills_iter": final["iteration"],
            "skills_pb_db": round(float(final["passband_min_s21_db"]), 2),
            "skills_sb_db": round(float(final["stopband_max_s21_db"]), 2),
            "skills_cost": round(float(final["total_cost"]), 4),
            "run": run,
        }
        rows.append(row)
        print(
            f"CF={cf_mhz:6.1f} [{row['f_low_mhz']}-{row['f_high_mhz']}] "
            f"| init met={row['init_goal_met']} PB={row['init_pb_db']} SB={row['init_sb_db']} "
            f"| skills met={row['skills_goal_met']} iter={row['skills_iter']} "
            f"PB={row['skills_pb_db']} SB={row['skills_sb_db']}"
        )

    out = ROOT / "runs" / f"bpf_skills_{tag}_summary.json"
    meta = {
        "seed": args.seed,
        "bw_mhz": args.bw_mhz,
        "n": args.n,
        "cf_lo_mhz": args.cf_lo_mhz,
        "cf_hi_mhz": args.cf_hi_mhz,
        "tag": tag,
        "policy": "skills-v3",
        "max_iter": MAX_ITER,
        "rows": rows,
        "skills_goal_met_count": sum(1 for r in rows if r["skills_goal_met"]),
        "init_goal_met_count": sum(1 for r in rows if r["init_goal_met"]),
    }
    out.write_text(json.dumps(meta, indent=2))
    n = len(rows)
    print(f"\nWrote {out}")
    print(f"init goal_met:   {meta['init_goal_met_count']}/{n}")
    print(f"skills goal_met: {meta['skills_goal_met_count']}/{n}")


if __name__ == "__main__":
    state_py = ROOT / "src" / "state.py"
    text = state_py.read_text()
    if "MAX_ITERATIONS = 20" in text:
        state_py.write_text(text.replace("MAX_ITERATIONS = 20", "MAX_ITERATIONS = 25"))
        restored = True
    else:
        restored = False
    try:
        main()
    finally:
        if restored:
            state_py.write_text(
                state_py.read_text().replace("MAX_ITERATIONS = 25", "MAX_ITERATIONS = 20")
            )
            print("MAX_ITERATIONS restored to 20")
