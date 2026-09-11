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
    python run_step.py init  [--run NAME] [--task butterfly_stub|butterworth_bpf5]
    python run_step.py step  --intent '{"ro":"decrease","alpha":"hold"}' \
                              [--note "text"] [--run NAME]
    python run_step.py report [--run NAME]
    python run_step.py best   [--run NAME]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cost import TARGET_DEPTH_DB, TARGET_NOTCH_HZ, evaluate, s21_db
from cost_bpf import evaluate as evaluate_bpf
from decision_log import append_record, format_agent_completion
from goals_bpf import BpfGoalSpec, default_goal as default_bpf_goal
from goals_bpf import goal_from_dict as bpf_goal_from_dict
from intent import BOUNDS, VARIABLES, apply_intent
from qucs_sim import simulate
from report_html import write_report
from state import RunState
from tasks import TaskConfig, get_task

# GoalSpec lives under training/; keep import local-friendly for the CLI script.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from training.goals import GoalSpec  # noqa: E402

RUNS_ROOT = Path(__file__).resolve().parent / "runs"
TARGET_BAND_HZ = (4e9, 6e9)
_BPF_TASK = "butterworth_bpf5"
_DEFAULT_TASK = "butterfly_stub"


def _cli_task_name(args) -> str:
    raw = getattr(args, "task", None)
    if isinstance(raw, str) and raw in (_DEFAULT_TASK, _BPF_TASK):
        return raw
    return _DEFAULT_TASK


def _task_name_from_state(state: RunState) -> str:
    return state.task if state.task else _DEFAULT_TASK


def _is_bpf(task_name: str | None) -> bool:
    return task_name == _BPF_TASK


def _legacy_goal_dict() -> dict:
    return {
        "target_freq_hz": TARGET_NOTCH_HZ,
        "band_hz": [TARGET_BAND_HZ[0], TARGET_BAND_HZ[1]],
        "target_depth_db": TARGET_DEPTH_DB,
    }


def _goal_from_state(state: RunState) -> GoalSpec:
    """Resolve this run's GoalSpec; missing goal falls back to legacy constants."""
    raw = state.goal
    if not raw:
        return GoalSpec(
            target_freq_hz=TARGET_NOTCH_HZ,
            band_hz=TARGET_BAND_HZ,
            target_depth_db=TARGET_DEPTH_DB,
        )
    band = raw["band_hz"]
    return GoalSpec(
        target_freq_hz=float(raw["target_freq_hz"]),
        band_hz=(float(band[0]), float(band[1])),
        target_depth_db=float(raw.get("target_depth_db", TARGET_DEPTH_DB)),
    )


def _bpf_goal_from_state(state: RunState) -> BpfGoalSpec:
    raw = state.goal
    if not raw or "f_low_hz" not in raw:
        return default_bpf_goal()
    return bpf_goal_from_dict(raw)


def _evaluate_for_run(res, goal: GoalSpec) -> dict:
    return asdict(evaluate(res, goal.band_hz, target_hz=goal.target_freq_hz))


def _evaluate_bpf_for_run(res, goal: BpfGoalSpec) -> dict:
    return evaluate_bpf(res, goal).to_dict()


def _simulate_for_task(params: dict, workdir: Path, task: TaskConfig, bpf_goal: BpfGoalSpec | None):
    if _is_bpf(task.name):
        assert bpf_goal is not None
        f0 = math.sqrt(bpf_goal.f_low_hz * bpf_goal.f_high_hz)
        return simulate(
            params,
            workdir=workdir,
            template_path=task.template_path,
            export_layout=task.export_layout,
            sweep_start_hz=bpf_goal.sweep_hz[0],
            sweep_stop_hz=bpf_goal.sweep_hz[1],
            sweep_points=task.sweep_points,
            f0_hz=f0,
        )
    return simulate(params, workdir=workdir)


def _cost_db_for_log(cost: dict) -> float:
    if "passband_min_s21_db" in cost:
        return float(cost["passband_min_s21_db"])
    return s21_db(cost["total_cost"])


def format_report(
    entry: dict,
    goal: GoalSpec | None = None,
    *,
    task: str | None = None,
) -> str:
    """Render one history entry as the plain-text block the CLI prints."""
    if _is_bpf(task):
        return _format_report_bpf(entry)
    return _format_report_butterfly(entry, goal=goal)


def _format_report_butterfly(entry: dict, goal: GoalSpec | None = None) -> str:
    p, c = entry["params"], entry["cost"]
    target_mag = c.get("target_s21_mag", c["total_cost"])
    if goal is None:
        target_hz = c.get("target_freq_hz", TARGET_NOTCH_HZ)
        depth_db = TARGET_DEPTH_DB
        band = TARGET_BAND_HZ
    else:
        target_hz = c.get("target_freq_hz", goal.target_freq_hz)
        depth_db = goal.target_depth_db
        band = goal.band_hz
    goal_mag = 10 ** (depth_db / 20.0)
    lines = [
        f"--- iteration {entry['iteration']} ---",
        "params: " + ", ".join(f"{k}={p[k]:.3f}" for k in VARIABLES),
        f"total_cost |S21| @ {target_hz/1e9:.2f} GHz = {target_mag:.6f} "
        f"({s21_db(target_mag):.2f} dB)  [goal {depth_db:.0f} dB / {goal_mag:.3e}]",
        f"deepest notch |S21|       = {c['best_s21_mag']:.6f} "
        f"({s21_db(c['best_s21_mag']):.2f} dB) @ {c['best_freq_hz']/1e9:.3f} GHz",
    ]
    if "stopband_max_s21" in c:
        lines.append(f"stopband max |S21| (aux)  = {c['stopband_max_s21']:.4f} "
                     f"@ {c['worst_freq_hz']/1e9:.3f} GHz")
    lines += [
        f"mean |S21| in stopband             = {c['mean_cost']:.4f}",
        f"low_edge  ({band[0]/1e9:.1f} GHz)  |S21| = {c['low_edge_cost']:.4f}",
        f"band mid  ({(sum(band)/2)/1e9:.1f} GHz)  |S21| = {c['center_cost']:.4f}",
        f"high_edge ({band[1]/1e9:.1f} GHz)  |S21| = {c['high_edge_cost']:.4f}",
    ]
    if "passband_low_mean" in c:
        lines.append(f"passband |S21| mean (<{band[0]/1e9:.1f} GHz) = {c['passband_low_mean']:.4f}")
        lines.append(f"passband |S21| mean (>{band[1]/1e9:.1f} GHz) = {c['passband_high_mean']:.4f}")
    if "zin_norm_at_center" in c:
        lines.append(f"|Zin|/Z0 at stopband mid (aux) = {c['zin_norm_at_center']:.4f}")
    if entry.get("intent"):
        lines.append(f"intent used: {entry['intent']}")
    if entry.get("note"):
        lines.append(f"note: {entry['note']}")
    return "\n".join(lines)


def _format_report_bpf(entry: dict) -> str:
    p, c = entry["params"], entry["cost"]
    vars_ = tuple(p.keys()) if p else get_task(_BPF_TASK).variables
    lines = [
        f"--- iteration {entry['iteration']} ---",
        "params: " + ", ".join(f"{k}={p[k]:.4g}" for k in vars_),
        f"total_cost = {c['total_cost']:.6f}",
        f"passband min S21 = {c['passband_min_s21_db']:.2f} dB "
        f"(f_low={c.get('f_low_hz', 0)/1e6:.1f} MHz, f_high={c.get('f_high_hz', 0)/1e6:.1f} MHz)",
        f"stopband max S21 = {c['stopband_max_s21_db']:.2f} dB",
    ]
    if "passband_mean_s21_db" in c:
        lines.append(f"passband mean S21 = {c['passband_mean_s21_db']:.2f} dB")
    if "s11_passband_max_db" in c:
        lines.append(f"passband max S11 = {c['s11_passband_max_db']:.2f} dB")
    if entry.get("intent"):
        lines.append(f"intent used: {entry['intent']}")
    if entry.get("note"):
        lines.append(f"note: {entry['note']}")
    return "\n".join(lines)


def format_observation(
    entry: dict,
    goal: GoalSpec | None = None,
    *,
    task: str | None = None,
) -> str:
    """
    Render the exact evidence block handed to the strategy layer before it
    decides the next intent: the latest measurement plus the bounds it must
    stay inside. Stored verbatim on the next history entry.
    """
    task_name = task or _DEFAULT_TASK
    cfg = get_task(task_name) if _is_bpf(task_name) else None
    variables = cfg.variables if cfg else VARIABLES
    bounds = cfg.bounds if cfg else BOUNDS
    p = entry["params"]
    bounds_txt = "\n".join(
        f"  {v:<6} = {p[v]:>10.4g}   bounds [{bounds[v][0]}, {bounds[v][1]}]"
        for v in variables
    )
    return (
        f"{format_report(entry, goal=goal, task=task_name)}\n\n"
        f"free variables and bounds:\n{bounds_txt}"
    )


def _print_report(entry: dict, goal: GoalSpec | None = None, *, task: str | None = None) -> None:
    print(format_report(entry, goal=goal, task=task))


def _read_thinking(args) -> str:
    """Resolve --thinking / --thinking-file into one reasoning string."""
    if getattr(args, "thinking_file", None):
        return Path(args.thinking_file).read_text().strip()
    return (getattr(args, "thinking", "") or "").strip()


def _goal_json_arg(args) -> str:
    raw = getattr(args, "goal_json", "")
    return raw if isinstance(raw, str) else ""


def cmd_init(args):
    run_dir = RUNS_ROOT / args.run
    state = RunState(run_dir)
    if state.iteration >= 0:
        print(f"run '{args.run}' already initialized at iteration {state.iteration}; "
              f"use 'report' to inspect or pick a new --run name", file=sys.stderr)
        sys.exit(1)

    task_name = _cli_task_name(args)
    if state.task is not None and state.task != task_name:
        print(f"run task mismatch: state={state.task} cli={task_name}", file=sys.stderr)
        sys.exit(1)

    task = get_task(task_name)
    state.set_run_meta(task=task_name)

    goal_json = _goal_json_arg(args)
    bpf_goal: BpfGoalSpec | None = None
    butterfly_goal: GoalSpec | None = None

    if _is_bpf(task_name):
        if goal_json.strip():
            bpf_goal = bpf_goal_from_dict(json.loads(goal_json))
            state.set_run_meta(goal=bpf_goal.to_dict())
        elif state.goal is None or "f_low_hz" not in state.goal:
            bpf_goal = default_bpf_goal()
            state.set_run_meta(goal=bpf_goal.to_dict())
        else:
            bpf_goal = _bpf_goal_from_state(state)
    else:
        if goal_json.strip():
            state.set_run_meta(goal=json.loads(goal_json))
        elif state.goal is None:
            state.set_run_meta(goal=_legacy_goal_dict())
        butterfly_goal = _goal_from_state(state)

    params = dict(state.initial_params) if state.initial_params else dict(task.initial_guess)
    res = _simulate_for_task(params, run_dir / "iter_000", task, bpf_goal)
    if _is_bpf(task_name):
        cost = _evaluate_bpf_for_run(res, bpf_goal)  # type: ignore[arg-type]
    else:
        cost = _evaluate_for_run(res, butterfly_goal)  # type: ignore[arg-type]

    it = state.record(
        params,
        cost,
        intent=None,
        note=args.note or "baseline initial guess",
        thinking=_read_thinking(args),
        observation=None,
    )
    thinking = _read_thinking(args)
    if thinking:
        append_record(
            run_dir / "completions.jsonl",
            {
                "source": "agent",
                "run": args.run,
                "iteration": it,
                "prompt": "init baseline (no prior observation)",
                "completion": format_agent_completion(thinking, None),
                "valid": True,
                "intent": None,
                "stopped": False,
                "db_before": None,
                "db_after": _cost_db_for_log(cost),
            },
        )
    _print_report(state.history[it], goal=butterfly_goal, task=task_name)


def cmd_step(args):
    run_dir = RUNS_ROOT / args.run
    state = RunState(run_dir)
    if state.iteration < 0:
        print(f"run '{args.run}' not initialized; run 'init' first", file=sys.stderr)
        sys.exit(1)

    task_name = _task_name_from_state(state)
    task = get_task(task_name)
    intent = json.loads(args.intent)
    unknown = set(intent) - set(task.variables)
    if unknown:
        print(f"unknown variable(s) in intent: {unknown}; valid: {task.variables}", file=sys.stderr)
        sys.exit(1)

    bpf_goal = _bpf_goal_from_state(state) if _is_bpf(task_name) else None
    butterfly_goal = None if _is_bpf(task_name) else _goal_from_state(state)

    prev_entry = state.history[-1]
    observation = {
        "from_iteration": prev_entry["iteration"],
        "report_text": format_observation(prev_entry, goal=butterfly_goal, task=task_name),
    }
    new_iteration = state.iteration + 1
    new_params = apply_intent(
        state.params,
        intent,
        iteration=new_iteration,
        variables=task.variables,
        bounds=task.bounds,
    )
    res = _simulate_for_task(new_params, run_dir / f"iter_{new_iteration:03d}", task, bpf_goal)
    if _is_bpf(task_name):
        cost = _evaluate_bpf_for_run(res, bpf_goal)  # type: ignore[arg-type]
    else:
        cost = _evaluate_for_run(res, butterfly_goal)  # type: ignore[arg-type]
    thinking = _read_thinking(args)
    it = state.record(
        new_params,
        cost,
        intent=intent,
        note=args.note or "",
        thinking=thinking,
        observation=observation,
    )
    append_record(
        run_dir / "completions.jsonl",
        {
            "source": "agent",
            "run": args.run,
            "iteration": it,
            "prompt": observation["report_text"],
            "completion": format_agent_completion(thinking, intent),
            "valid": True,
            "intent": intent,
            "stopped": False,
            "db_before": _cost_db_for_log(prev_entry["cost"]),
            "db_after": _cost_db_for_log(cost),
        },
    )
    _print_report(state.history[it], goal=butterfly_goal, task=task_name)


def cmd_report(args):
    run_dir = RUNS_ROOT / args.run
    state = RunState(run_dir)
    if state.iteration < 0:
        print(f"run '{args.run}' has no history yet", file=sys.stderr)
        sys.exit(1)
    task_name = _task_name_from_state(state)
    goal = None if _is_bpf(task_name) else _goal_from_state(state)
    for entry in state.history:
        _print_report(entry, goal=goal, task=task_name)
        print()
    best = state.best()
    print("=== BEST SO FAR ===")
    _print_report(best, goal=goal, task=task_name)


def cmd_best(args):
    run_dir = RUNS_ROOT / args.run
    state = RunState(run_dir)
    best = state.best()
    if best is None:
        print(f"run '{args.run}' has no history yet", file=sys.stderr)
        sys.exit(1)
    task_name = _task_name_from_state(state)
    goal = None if _is_bpf(task_name) else _goal_from_state(state)
    _print_report(best, goal=goal, task=task_name)


def cmd_observe(args):
    """Print the evidence block the strategy layer should reason over next."""
    run_dir = RUNS_ROOT / args.run
    state = RunState(run_dir)
    if state.iteration < 0:
        print(f"run '{args.run}' has no history yet", file=sys.stderr)
        sys.exit(1)
    task_name = _task_name_from_state(state)
    goal = None if _is_bpf(task_name) else _goal_from_state(state)
    print(format_observation(state.history[-1], goal=goal, task=task_name))


def cmd_conclude(args):
    run_dir = RUNS_ROOT / args.run
    state = RunState(run_dir)
    if state.iteration < 0:
        print(f"run '{args.run}' has no history yet", file=sys.stderr)
        sys.exit(1)
    text = _read_thinking(args)
    if not text:
        print("nothing to record; pass --thinking or --thinking-file", file=sys.stderr)
        sys.exit(1)
    state.conclude(text)
    print(f"recorded conclusion for run '{args.run}' ({len(text)} chars)")


def cmd_html(args):
    run_dir = RUNS_ROOT / args.run
    state = RunState(run_dir)
    if state.iteration < 0:
        print(f"run '{args.run}' has no history yet", file=sys.stderr)
        sys.exit(1)
    out = Path(args.out) if args.out else run_dir / "report.html"
    write_report(
        args.run,
        run_dir,
        state.history,
        out,
        conclusion=state.conclusion,
        task=state.task,
    )
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_thinking_args(p):
        p.add_argument("--thinking", default="",
                       help="strategy-layer reasoning recorded verbatim with this step")
        p.add_argument("--thinking-file", default="",
                       help="read the reasoning from a file instead (better for long text)")

    p_init = sub.add_parser("init")
    p_init.add_argument("--run", default="default")
    p_init.add_argument("--note", default="")
    p_init.add_argument(
        "--task",
        default=_DEFAULT_TASK,
        choices=[_DEFAULT_TASK, _BPF_TASK],
        help="optimization task (default: butterfly_stub)",
    )
    p_init.add_argument(
        "--goal-json",
        default="",
        help="JSON goal dict for the selected task; "
             "omitted: keep existing state.goal or fall back to task defaults",
    )
    add_thinking_args(p_init)
    p_init.set_defaults(func=cmd_init)

    p_step = sub.add_parser("step")
    p_step.add_argument("--run", default="default")
    p_step.add_argument("--intent", required=True, help="JSON dict, e.g. '{\"ro\":\"decrease\"}'")
    p_step.add_argument("--note", default="")
    add_thinking_args(p_step)
    p_step.set_defaults(func=cmd_step)

    p_report = sub.add_parser("report")
    p_report.add_argument("--run", default="default")
    p_report.set_defaults(func=cmd_report)

    p_best = sub.add_parser("best")
    p_best.add_argument("--run", default="default")
    p_best.set_defaults(func=cmd_best)

    p_observe = sub.add_parser("observe")
    p_observe.add_argument("--run", default="default")
    p_observe.set_defaults(func=cmd_observe)

    p_conclude = sub.add_parser("conclude")
    p_conclude.add_argument("--run", default="default")
    add_thinking_args(p_conclude)
    p_conclude.set_defaults(func=cmd_conclude)

    p_html = sub.add_parser("html")
    p_html.add_argument("--run", default="default")
    p_html.add_argument("--out", default="", help="output path (default runs/<run>/report.html)")
    p_html.set_defaults(func=cmd_html)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
