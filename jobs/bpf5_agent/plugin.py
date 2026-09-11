"""Full TaskPlugin for butterworth_bpf5."""

from __future__ import annotations

import math
from typing import Any

from jobs.bpf5_agent import cost as cost_mod
from jobs.bpf5_agent import goals as goals_mod
from jobs.bpf5_agent import skills as skills_mod
from jobs.bpf5_agent import task as task_mod
from qucs_sim import simulate
from tasks import TaskConfig


def _fmt_param_value(val: float) -> str:
    """Prefer fixed decimals that preserve round(..., 4) without .4g collapse."""
    return f"{val:.4f}".rstrip("0").rstrip(".") if "." in f"{val:.4f}" else f"{val:.4f}"


class Bpf5Plugin:
    name = "butterworth_bpf5"

    @property
    def config(self) -> TaskConfig:
        return TaskConfig(
            name=task_mod.TASK_NAME,
            variables=task_mod.VARIABLES,
            bounds=task_mod.BOUNDS,
            initial_guess=dict(task_mod.INITIAL_GUESS),
            template_path=task_mod.TEMPLATE_PATH,
            export_layout=task_mod.EXPORT_LAYOUT,
            sweep_points=task_mod.SWEEP_POINTS,
            step_mode=task_mod.STEP_MODE,
        )

    def default_goal(self) -> dict:
        return goals_mod.default_goal().to_dict()

    def goal_from_state(self, raw: dict | None) -> Any:
        # None / empty → defaults; non-empty dict → strict parse (CLI --goal-json).
        if not raw:
            return goals_mod.default_goal()
        return goals_mod.goal_from_dict(raw)

    def validate_goal(self, goal) -> None:
        goals_mod.validate_goal(goal)

    def evaluate(self, sim_result, goal) -> dict:
        report = cost_mod.evaluate(sim_result, goal)
        out = report.to_dict()
        out["goal_met"] = goals_mod.is_goal_met(out, goal)
        return out

    def simulate(self, params, workdir, goal):
        f0 = math.sqrt(goal.f_low_hz * goal.f_high_hz)
        cfg = self.config
        return simulate(
            params,
            workdir=workdir,
            template_path=cfg.template_path,
            export_layout=cfg.export_layout,
            sweep_start_hz=goal.sweep_hz[0],
            sweep_stop_hz=goal.sweep_hz[1],
            sweep_points=cfg.sweep_points,
            f0_hz=f0,
        )

    def param_unit(self, var: str) -> str:
        if var.startswith("L"):
            return "nH"
        if var.startswith("C"):
            return "pF"
        return ""

    def format_report(self, entry, goal) -> str:
        p, c = entry["params"], entry["cost"]
        vars_ = tuple(p.keys()) if p else self.config.variables
        g = goal if goal is not None else goals_mod.default_goal()
        goal_met = c.get("goal_met")
        if goal_met is None:
            goal_met = goals_mod.is_goal_met(c, g)
        status = "goal met" if goal_met else "goal not met"
        param_bits = []
        for k in vars_:
            unit = self.param_unit(k)
            suffix = f" {unit}" if unit else ""
            param_bits.append(f"{k}={_fmt_param_value(p[k])}{suffix}")
        sweep_lo, sweep_hi = g.sweep_hz
        lines = [
            f"--- iteration {entry['iteration']} ---",
            "params: " + ", ".join(param_bits),
            f"total_cost = {c['total_cost']:.6f}",
            f"passband min S21 = {c['passband_min_s21_db']:.2f} dB "
            f"(f_low={c.get('f_low_hz', g.f_low_hz)/1e6:.1f} MHz, "
            f"f_high={c.get('f_high_hz', g.f_high_hz)/1e6:.1f} MHz)",
            f"stopband max S21 = {c['stopband_max_s21_db']:.2f} dB",
            f"goal thresholds: passband_il_max_db={g.passband_il_max_db:.2f} dB, "
            f"stopband_atten_min_db={g.stopband_atten_min_db:.2f} dB  [{status}]",
            f"sweep window: {sweep_lo/1e6:.1f} .. {sweep_hi/1e6:.1f} MHz",
        ]
        if "passband_mean_s21_db" in c:
            lines.append(f"passband mean S21 = {c['passband_mean_s21_db']:.2f} dB")
        if "s11_passband_max_db" in c:
            lines.append(f"passband max S11 = {c['s11_passband_max_db']:.2f} dB")
        if "s21_peak_freq_hz" in c:
            lines.append(f"S21 peak frequency = {c['s21_peak_freq_hz']/1e6:.2f} MHz")
        if entry.get("intent"):
            lines.append(f"intent used: {entry['intent']}")
        if entry.get("note"):
            lines.append(f"note: {entry['note']}")
        return "\n".join(lines)

    def format_observation(self, entry, goal, *, include_skills_system: bool = True) -> str:
        cfg = self.config
        p = entry["params"]
        bounds_txt = "\n".join(
            f"  {v:<6} = {_fmt_param_value(p[v]):>10} {self.param_unit(v):<2}  "
            f"bounds [{cfg.bounds[v][0]}, {cfg.bounds[v][1]}]"
            for v in cfg.variables
        )
        body = (
            f"{self.format_report(entry, goal)}\n\n"
            f"free variables and bounds:\n{bounds_txt}"
        )
        if include_skills_system:
            skills = skills_mod.load_skills_system_prompt()
            return (
                "=== SYSTEM: BPF tuning skills (read every round) ===\n"
                f"{skills}\n"
                "=== END SYSTEM ===\n\n"
                f"{body}"
            )
        return body
