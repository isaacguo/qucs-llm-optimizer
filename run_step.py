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
    python run_step.py init  [--run NAME] [--task TASK_NAME]
    python run_step.py step  --intent '{"ro":"decrease","alpha":"hold"}' \
                              [--note "text"] [--run NAME]
    python run_step.py report [--run NAME]
    python run_step.py best   [--run NAME]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cost import TARGET_DEPTH_DB, TARGET_NOTCH_HZ, evaluate, s21_db
from decision_log import append_record, format_agent_completion
from intent import BOUNDS, VARIABLES, apply_intent
from qucs_sim import simulate
from report_html import write_report
from state import RunState
from task_registry import get_plugin
from tasks import get_task

# GoalSpec lives under training/; keep import local-friendly for the CLI script.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from training.common.goals import GoalSpec  # noqa: E402

RUNS_ROOT = Path(__file__).resolve().parent / "runs"
TARGET_BAND_HZ = (4e9, 6e9)
_DEFAULT_TASK = "butterfly_stub"


def discover_job_plugins(repo_root: Path) -> None:
    """Load jobs/*/register.py and call register()."""
    jobs_root = repo_root / "jobs"
    if not jobs_root.is_dir():
        return
    root_s = str(repo_root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    for reg in sorted(jobs_root.glob("*/register.py")):
        job_id = reg.parent.name
        mod_name = f"jobs.{job_id}.register"
        if mod_name in sys.modules:
            module = sys.modules[mod_name]
        else:
            spec = importlib.util.spec_from_file_location(mod_name, reg)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)
        module.register()


discover_job_plugins(_ROOT)


def _try_get_plugin(task_name: str | None):
    if not task_name:
        return None
    try:
        return get_plugin(task_name)
    except ValueError:
        return None


def _cli_task_name(args) -> str:
    raw = getattr(args, "task", None)
    if not isinstance(raw, str) or not raw.strip():
        return _DEFAULT_TASK
    try:
        get_task(raw)
    except ValueError:
        print(f"unknown task: {raw}", file=sys.stderr)
        sys.exit(1)
    return raw


def _task_name_from_state(state: RunState) -> str:
    return state.task if state.task else _DEFAULT_TASK


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


def _evaluate_for_run(res, goal: GoalSpec) -> dict:
    return asdict(evaluate(res, goal.band_hz, target_hz=goal.target_freq_hz))


def _goal_to_dict(goal: Any) -> dict:
    if hasattr(goal, "to_dict"):
        return goal.to_dict()
    if isinstance(goal, dict):
        return goal
    raise TypeError(f"cannot serialize goal of type {type(goal)!r}")


def _resolve_plugin_goal(plugin, goal_json: str, state: RunState):
    """Parse / default plugin goal and validate; exit(1) on bad input."""
    try:
        if goal_json.strip():
            goal = plugin.goal_from_state(json.loads(goal_json))
        else:
            goal = plugin.goal_from_state(state.goal)
        plugin.validate_goal(goal)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"invalid goal: {exc}", file=sys.stderr)
        sys.exit(1)
    return goal


def _cost_db_for_log(cost: dict) -> float:
    if "passband_min_s21_db" in cost:
        return float(cost["passband_min_s21_db"])
    return s21_db(cost["total_cost"])


def format_report(
    entry: dict,
    goal: GoalSpec | None = None,
    *,
    task: str | None = None,
    bpf_goal: Any = None,
) -> str:
    """Render one history entry as the plain-text block the CLI prints."""
    task_name = task or _DEFAULT_TASK
    plugin = _try_get_plugin(task_name)
    if plugin is not None:
        plugin_goal = bpf_goal if bpf_goal is not None else goal
        if plugin_goal is None:
            plugin_goal = plugin.goal_from_state(None)
        return plugin.format_report(entry, plugin_goal)
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


def format_observation(
    entry: dict,
    goal: GoalSpec | None = None,
    *,
    task: str | None = None,
    bpf_goal: Any = None,
    include_skills_system: bool = True,
) -> str:
    """
    Render the exact evidence block handed to the strategy layer before it
    decides the next intent: the latest measurement plus the bounds it must
    stay inside. Stored verbatim on the next history entry.

    When a job plugin is registered for ``task``, delegate formatting
    (including skills system text) to the plugin.
    """
    task_name = task or _DEFAULT_TASK
    plugin = _try_get_plugin(task_name)
    if plugin is not None:
        plugin_goal = bpf_goal if bpf_goal is not None else goal
        if plugin_goal is None:
            plugin_goal = plugin.goal_from_state(None)
        return plugin.format_observation(
            entry, plugin_goal, include_skills_system=include_skills_system
        )

    p = entry["params"]
    bounds_txt = "\n".join(
        f"  {v:<6} = {p[v]:>10.4g}   bounds [{BOUNDS[v][0]}, {BOUNDS[v][1]}]"
        for v in VARIABLES
    )
    body = (
        f"{format_report(entry, goal=goal, task=task_name)}\n\n"
        f"free variables and bounds:\n{bounds_txt}"
    )
    return body


def _print_report(
    entry: dict,
    goal: GoalSpec | None = None,
    *,
    task: str | None = None,
    bpf_goal: Any = None,
) -> None:
    print(format_report(entry, goal=goal, task=task, bpf_goal=bpf_goal))


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
    plugin = _try_get_plugin(task_name)
    plugin_goal = None
    butterfly_goal: GoalSpec | None = None

    if plugin is not None:
        plugin_goal = _resolve_plugin_goal(plugin, goal_json, state)
        state.set_run_meta(goal=_goal_to_dict(plugin_goal))
        params = dict(state.initial_params) if state.initial_params else dict(task.initial_guess)
        res = plugin.simulate(params, run_dir / "iter_000", plugin_goal)
        cost = plugin.evaluate(res, plugin_goal)
    else:
        if goal_json.strip():
            state.set_run_meta(goal=json.loads(goal_json))
        elif state.goal is None:
            state.set_run_meta(goal=_legacy_goal_dict())
        butterfly_goal = _goal_from_state(state)
        params = dict(state.initial_params) if state.initial_params else dict(task.initial_guess)
        res = simulate(params, workdir=run_dir / "iter_000")
        cost = _evaluate_for_run(res, butterfly_goal)

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
    _print_report(state.history[it], goal=butterfly_goal, task=task_name, bpf_goal=plugin_goal)


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

    plugin = _try_get_plugin(task_name)
    plugin_goal = plugin.goal_from_state(state.goal) if plugin is not None else None
    butterfly_goal = None if plugin is not None else _goal_from_state(state)

    prev_entry = state.history[-1]
    observation = {
        "from_iteration": prev_entry["iteration"],
        "report_text": format_observation(
            prev_entry, goal=butterfly_goal, task=task_name, bpf_goal=plugin_goal
        ),
    }
    new_iteration = state.iteration + 1
    new_params = apply_intent(
        state.params,
        intent,
        iteration=new_iteration,
        variables=task.variables,
        bounds=task.bounds,
        step_mode=task.step_mode,
    )
    if plugin is not None:
        res = plugin.simulate(new_params, run_dir / f"iter_{new_iteration:03d}", plugin_goal)
        cost = plugin.evaluate(res, plugin_goal)
    else:
        res = simulate(new_params, workdir=run_dir / f"iter_{new_iteration:03d}")
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
    _print_report(
        state.history[it], goal=butterfly_goal, task=task_name, bpf_goal=plugin_goal
    )


def cmd_report(args):
    run_dir = RUNS_ROOT / args.run
    state = RunState(run_dir)
    if state.iteration < 0:
        print(f"run '{args.run}' has no history yet", file=sys.stderr)
        sys.exit(1)
    task_name = _task_name_from_state(state)
    plugin = _try_get_plugin(task_name)
    plugin_goal = plugin.goal_from_state(state.goal) if plugin is not None else None
    goal = None if plugin is not None else _goal_from_state(state)
    for entry in state.history:
        _print_report(entry, goal=goal, task=task_name, bpf_goal=plugin_goal)
        print()
    best = state.best()
    print("=== BEST SO FAR ===")
    _print_report(best, goal=goal, task=task_name, bpf_goal=plugin_goal)


def cmd_best(args):
    run_dir = RUNS_ROOT / args.run
    state = RunState(run_dir)
    best = state.best()
    if best is None:
        print(f"run '{args.run}' has no history yet", file=sys.stderr)
        sys.exit(1)
    task_name = _task_name_from_state(state)
    plugin = _try_get_plugin(task_name)
    plugin_goal = plugin.goal_from_state(state.goal) if plugin is not None else None
    goal = None if plugin is not None else _goal_from_state(state)
    _print_report(best, goal=goal, task=task_name, bpf_goal=plugin_goal)


def cmd_observe(args):
    """Print the evidence block the strategy layer should reason over next."""
    run_dir = RUNS_ROOT / args.run
    state = RunState(run_dir)
    if state.iteration < 0:
        print(f"run '{args.run}' has no history yet", file=sys.stderr)
        sys.exit(1)
    task_name = _task_name_from_state(state)
    plugin = _try_get_plugin(task_name)
    plugin_goal = plugin.goal_from_state(state.goal) if plugin is not None else None
    goal = None if plugin is not None else _goal_from_state(state)
    print(format_observation(state.history[-1], goal=goal, task=task_name, bpf_goal=plugin_goal))


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
        help="optimization task name (registered plugin or builtin; "
             f"default: {_DEFAULT_TASK})",
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
