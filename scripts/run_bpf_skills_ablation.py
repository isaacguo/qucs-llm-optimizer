#!/usr/bin/env python3
"""Ablation: cold-start tune 10 CFs with skills system prompt + skill policy.

Baseline (previous batch) used a directed crawl toward analytic L/C targets.
This script uses only observation diagnostics + the skills system prompt policy
(no target L/C table), matching how a model that reads the system card would act.
"""
from __future__ import annotations

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

BW = 5e6
CFS = [float(x) for x in np.linspace(55e6, 205e6, 10)]
PY = sys.executable
MAX_ITER = 24


def goal_for(f0: float) -> BpfGoalSpec:
    return BpfGoalSpec(
        f_low_hz=f0 - BW / 2,
        f_high_hz=f0 + BW / 2,
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
    think = Path("/tmp/bpf_skills_think.md")
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
        # Reasoning only — intent goes to <intent> via run_step / format_agent_completion.
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
                "skills-policy v3 step",
                "--thinking-file",
                str(think),
            ]
        )
        prev_cost = cost
        prev_thinking = diag


def main() -> None:
    baseline_path = ROOT / "runs" / "bpf_batch_10cf_summary.json"
    baseline = {
        float(r["cf_mhz"]): r for r in json.loads(baseline_path.read_text())
    }
    rows = []
    print(f"Skills ablation CFs (MHz): {[round(c/1e6, 2) for c in CFS]}")
    for f0 in CFS:
        cf_mhz = round(f0 / 1e6, 2)
        tag = f"{cf_mhz:.1f}".replace(".", "p")
        run = f"bpf_cf{tag}_bw5_skills_v3"
        d = ROOT / "runs" / run
        if d.exists():
            shutil.rmtree(d)
        g = goal_for(f0)
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
                f"skills-system v3 cold start CF={cf_mhz}MHz BW=5",
            ]
        )
        final = skills_crawl(run, g)
        run_cmd(
            [
                "run_step.py",
                "conclude",
                "--run",
                run,
                "--thinking",
                f"Skills-system v3 policy finished CF={cf_mhz} MHz; "
                f"goal_met={final.get('goal_met')} iter={final['iteration']}.",
            ]
        )
        run_cmd(["run_step.py", "html", "--run", run])
        base = baseline.get(cf_mhz, {})
        v2 = {}
        v2_path = ROOT / "runs" / "bpf_skills_ablation_v2_summary.json"
        if v2_path.exists():
            for row in json.loads(v2_path.read_text()):
                if abs(float(row["cf_mhz"]) - cf_mhz) < 0.01:
                    v2 = row
                    break
        row = {
            "cf_mhz": cf_mhz,
            "skills_v3_goal_met": bool(final.get("goal_met")),
            "skills_v3_iter": final["iteration"],
            "skills_v3_pb_db": round(float(final["passband_min_s21_db"]), 2),
            "skills_v3_sb_db": round(float(final["stopband_max_s21_db"]), 2),
            "skills_v3_cost": round(float(final["total_cost"]), 4),
            "skills_v3_run": run,
            "skills_v2_goal_met": v2.get("skills_v2_goal_met"),
            "skills_v2_iter": v2.get("skills_v2_iter"),
            "skills_v2_pb_db": v2.get("skills_v2_pb_db"),
            "skills_v2_sb_db": v2.get("skills_v2_sb_db"),
            "baseline_tune_goal_met": base.get("tune_goal_met"),
            "baseline_tune_iter": base.get("tune_iter"),
            "baseline_tune_pb_db": base.get("tune_pb_db"),
            "baseline_tune_sb_db": base.get("tune_sb_db"),
        }
        rows.append(row)
        print(
            f"CF={cf_mhz:6.1f} | v3 met={row['skills_v3_goal_met']} "
            f"iter={row['skills_v3_iter']} PB={row['skills_v3_pb_db']} SB={row['skills_v3_sb_db']} "
            f"| v2 met={row['skills_v2_goal_met']} PB={row['skills_v2_pb_db']} SB={row['skills_v2_sb_db']} "
            f"| oracle met={row['baseline_tune_goal_met']}"
        )

    out = ROOT / "runs" / "bpf_skills_ablation_v3_summary.json"
    out.write_text(json.dumps(rows, indent=2))
    n = len(rows)
    print(f"\nWrote {out}")
    print(f"skills v3 goal_met: {sum(1 for r in rows if r['skills_v3_goal_met'])}/{n}")
    print(f"skills v2 goal_met: {sum(1 for r in rows if r['skills_v2_goal_met'])}/{n}")
    print(
        f"oracle tune goal_met: "
        f"{sum(1 for r in rows if r['baseline_tune_goal_met'])}/{n}"
    )


if __name__ == "__main__":
    # Allow up to MAX_ITER steps (iteration index 0..MAX_ITER)
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
