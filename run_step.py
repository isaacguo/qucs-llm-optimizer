#!/usr/bin/env python3
"""
run_step.py — CLI driver for the butterfly band-stop LLM-optimizer pipeline.

Architecture (per design discussion): the strategy layer ("LLM", currently
played by a human/cursor-agent, later swappable for a fine-tuned small
model) only ever supplies a *qualitative intent* per free variable
(increase/decrease/hold, with a magnitude qualifier). All precise numeric
arithmetic — step size, iteration-based decay, bound clamping — lives in
intent.py and is never touched by the strategy layer.

Usage:
    python run_step.py init  [--run NAME]
    python run_step.py step  --intent '{"ro":"decrease","alpha":"hold"}' \
                              [--note "text"] [--run NAME]
    python run_step.py report [--run NAME]
    python run_step.py best   [--run NAME]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cost import TARGET_DEPTH_DB, TARGET_NOTCH_HZ, TARGET_S21_MAG, evaluate, s21_db
from intent import BOUNDS, INITIAL_GUESS, VARIABLES, apply_intent
from qucs_sim import simulate
from state import RunState

RUNS_ROOT = Path(__file__).resolve().parent / "runs"
TARGET_BAND_HZ = (4e9, 6e9)


def _print_report(entry: dict) -> None:
    p, c = entry["params"], entry["cost"]
    target_mag = c.get("target_s21_mag", c["total_cost"])
    target_hz = c.get("target_freq_hz", TARGET_NOTCH_HZ)
    print(f"--- iteration {entry['iteration']} ---")
    print("params: " + ", ".join(f"{k}={p[k]:.3f}" for k in VARIABLES))
    print(
        f"total_cost |S21| @ {target_hz/1e9:.2f} GHz = {target_mag:.6f} "
        f"({s21_db(target_mag):.2f} dB)  [goal {TARGET_DEPTH_DB:.0f} dB / {TARGET_S21_MAG:.3e}]"
    )
    print(f"deepest notch |S21|       = {c['best_s21_mag']:.6f} "
          f"({s21_db(c['best_s21_mag']):.2f} dB) @ {c['best_freq_hz']/1e9:.3f} GHz")
    if "stopband_max_s21" in c:
        print(f"stopband max |S21| (aux)  = {c['stopband_max_s21']:.4f} "
              f"@ {c['worst_freq_hz']/1e9:.3f} GHz")
    print(f"mean |S21| in stopband             = {c['mean_cost']:.4f}")
    print(f"low_edge  ({TARGET_BAND_HZ[0]/1e9:.1f} GHz)  |S21| = {c['low_edge_cost']:.4f}")
    print(f"band mid  ({(sum(TARGET_BAND_HZ)/2)/1e9:.1f} GHz)  |S21| = {c['center_cost']:.4f}")
    print(f"high_edge ({TARGET_BAND_HZ[1]/1e9:.1f} GHz)  |S21| = {c['high_edge_cost']:.4f}")
    if "passband_low_mean" in c:
        print(f"passband |S21| mean (<{TARGET_BAND_HZ[0]/1e9:.1f} GHz) = {c['passband_low_mean']:.4f}")
        print(f"passband |S21| mean (>{TARGET_BAND_HZ[1]/1e9:.1f} GHz) = {c['passband_high_mean']:.4f}")
    if "zin_norm_at_center" in c:
        print(f"|Zin|/Z0 at stopband mid (aux) = {c['zin_norm_at_center']:.4f}")
    if entry.get("intent"):
        print(f"intent used: {entry['intent']}")
    if entry.get("note"):
        print(f"note: {entry['note']}")


def cmd_init(args):
    run_dir = RUNS_ROOT / args.run
    state = RunState(run_dir)
    if state.iteration >= 0:
        print(f"run '{args.run}' already initialized at iteration {state.iteration}; "
              f"use 'report' to inspect or pick a new --run name", file=sys.stderr)
        sys.exit(1)
    params = dict(INITIAL_GUESS)
    res = simulate(params, workdir=run_dir / "iter_000")
    cost = asdict(evaluate(res, TARGET_BAND_HZ, target_hz=TARGET_NOTCH_HZ))
    it = state.record(params, cost, intent=None, note="baseline initial guess")
    _print_report(state.history[it])


def cmd_step(args):
    run_dir = RUNS_ROOT / args.run
    state = RunState(run_dir)
    if state.iteration < 0:
        print(f"run '{args.run}' not initialized; run 'init' first", file=sys.stderr)
        sys.exit(1)
    intent = json.loads(args.intent)
    unknown = set(intent) - set(VARIABLES)
    if unknown:
        print(f"unknown variable(s) in intent: {unknown}; valid: {VARIABLES}", file=sys.stderr)
        sys.exit(1)
    new_iteration = state.iteration + 1
    new_params = apply_intent(state.params, intent, iteration=new_iteration)
    res = simulate(new_params, workdir=run_dir / f"iter_{new_iteration:03d}")
    cost = asdict(evaluate(res, TARGET_BAND_HZ, target_hz=TARGET_NOTCH_HZ))
    it = state.record(new_params, cost, intent=intent, note=args.note or "")
    _print_report(state.history[it])


def cmd_report(args):
    run_dir = RUNS_ROOT / args.run
    state = RunState(run_dir)
    if state.iteration < 0:
        print(f"run '{args.run}' has no history yet", file=sys.stderr)
        sys.exit(1)
    for entry in state.history:
        _print_report(entry)
        print()
    best = state.best()
    print("=== BEST SO FAR ===")
    _print_report(best)


def cmd_best(args):
    run_dir = RUNS_ROOT / args.run
    state = RunState(run_dir)
    best = state.best()
    if best is None:
        print(f"run '{args.run}' has no history yet", file=sys.stderr)
        sys.exit(1)
    _print_report(best)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init")
    p_init.add_argument("--run", default="default")
    p_init.set_defaults(func=cmd_init)

    p_step = sub.add_parser("step")
    p_step.add_argument("--run", default="default")
    p_step.add_argument("--intent", required=True, help="JSON dict, e.g. '{\"ro\":\"decrease\"}'")
    p_step.add_argument("--note", default="")
    p_step.set_defaults(func=cmd_step)

    p_report = sub.add_parser("report")
    p_report.add_argument("--run", default="default")
    p_report.set_defaults(func=cmd_report)

    p_best = sub.add_parser("best")
    p_best.add_argument("--run", default="default")
    p_best.set_defaults(func=cmd_best)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
