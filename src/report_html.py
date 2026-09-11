"""
report_html.py — self-contained static HTML report for one optimization run.

Each iteration becomes a collapsible panel holding the full decision record:
what the strategy layer was shown (observation), what it reasoned (thinking),
what it emitted (intent), and what the simulator returned (result), next to
the Qucs-RFlayout copper geometry for that iteration (when present).

No external assets: the layout SVGs are inlined, CSS/JS are embedded.
"""
from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path

VARIABLES = ("ri", "ro", "alpha", "Wf", "Lc")
UNITS = {"ri": "mm", "ro": "mm", "alpha": "deg", "Wf": "mm", "Lc": "mm"}
TARGET_DEPTH_DB = -70.0
BPF_TASK = "butterworth_bpf5"

# Rendering of an intent token: (arrow, human label, css class).
INTENT_TOKENS = {
    "increase_strong": ("\u21d1", "increase strong", "up"),
    "increase": ("\u2191", "increase", "up"),
    "increase_slight": ("\u2197", "increase slight", "up"),
    "hold": ("\u2192", "hold", "hold"),
    "decrease_slight": ("\u2198", "decrease slight", "down"),
    "decrease": ("\u2193", "decrease", "down"),
    "decrease_strong": ("\u21d3", "decrease strong", "down"),
}


def db(mag: float) -> float:
    return 20.0 * math.log10(mag) if mag > 0 else float("-inf")


def _esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def _is_bpf(task: str | None, cost: dict | None = None) -> bool:
    if task == BPF_TASK:
        return True
    if cost is not None and "passband_min_s21_db" in cost:
        return True
    return False


def _unit_for(var: str) -> str:
    if var in UNITS:
        return UNITS[var]
    if var.startswith("L"):
        return "nH"
    if var.startswith("C"):
        return "pF"
    return ""


def _param_vars(params: dict) -> tuple[str, ...]:
    ordered = tuple(v for v in VARIABLES if v in params)
    return ordered if ordered else tuple(params.keys())


def _inline_layout_svg(svg_path: Path) -> str:
    """
    Inline a Qucs-RFlayout SVG. Drops the XML prolog and the fixed mm
    width/height so CSS can scale it; the viewBox keeps the aspect ratio.
    """
    if not svg_path.exists():
        return '<p class="missing">no layout.svg for this iteration</p>'
    svg = svg_path.read_text()
    svg = re.sub(r"<\?xml[^>]*\?>\s*", "", svg)
    svg = re.sub(r"<!--.*?-->\s*", "", svg, flags=re.S)
    svg = re.sub(r'\s(width|height)="[^"]*"', "", svg, count=2)
    return svg.strip()


def _intent_chips(intent: dict | None) -> str:
    if not intent:
        return '<span class="chip none">baseline &mdash; no intent</span>'
    chips = []
    for var, token in intent.items():
        arrow, label, cls = INTENT_TOKENS.get(token, ("?", token, "hold"))
        chips.append(
            f'<span class="chip {cls}" title="{_esc(label)}">'
            f"{_esc(var)}&nbsp;{arrow}</span>"
        )
    return "".join(chips)


def _param_table(params: dict, prev: dict | None) -> str:
    rows = []
    for v in _param_vars(params):
        now = params[v]
        if prev is None or v not in prev:
            delta_cell = '<td class="dim">&mdash;</td><td class="dim">&mdash;</td>'
        else:
            before = prev[v]
            change = now - before
            if abs(change) < 1e-9:
                delta_cell = (f'<td class="dim">{before:.4f}</td>'
                              f'<td class="dim">unchanged</td>')
            else:
                cls = "up" if change > 0 else "down"
                delta_cell = (f"<td>{before:.4f}</td>"
                              f'<td class="{cls}">{change:+.4f}</td>')
        rows.append(
            f"<tr><th>{_esc(v)}</th>{delta_cell}"
            f'<td class="now">{now:.4f}</td><td class="dim">{_unit_for(v)}</td></tr>'
        )
    return (
        '<table class="params"><thead><tr>'
        "<th>var</th><th>before</th><th>delta</th><th>after</th><th></th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _cost_table(cost: dict) -> str:
    if _is_bpf(None, cost):
        rows = [
            ("total_cost", f"{cost['total_cost']:.6f}", "primary"),
            ("passband_min_s21_db",
             f"{cost['passband_min_s21_db']:.2f} dB", ""),
            ("stopband_max_s21_db",
             f"{cost['stopband_max_s21_db']:.2f} dB", ""),
            ("passband_mean_s21_db",
             f"{cost.get('passband_mean_s21_db', float('nan')):.2f} dB", ""),
            ("s11_passband_max_db",
             f"{cost.get('s11_passband_max_db', float('nan')):.2f} dB", ""),
            ("passband (Hz)",
             f"{cost.get('f_low_hz', 0):.3e} .. {cost.get('f_high_hz', 0):.3e}", ""),
        ]
    else:
        total = cost["total_cost"]
        target_hz = cost.get("target_freq_hz", 5.5e9)
        rows = [
            ("total_cost = |S21| @ %.2f GHz" % (target_hz / 1e9),
             f"{total:.6f}  ({db(total):.2f} dB)", "primary"),
            ("deepest notch in sweep",
             f"{cost['best_s21_mag']:.6f}  ({db(cost['best_s21_mag']):.2f} dB) "
             f"@ {cost['best_freq_hz']/1e9:.3f} GHz", ""),
            ("mean |S21| in 4-6 GHz", f"{cost['mean_cost']:.4f}", ""),
            ("stopband max |S21|",
             f"{cost.get('stopband_max_s21', float('nan')):.4f} "
             f"@ {cost.get('worst_freq_hz', 0)/1e9:.3f} GHz", ""),
            ("passband mean |S21| (&lt;4 GHz / &gt;6 GHz)",
             f"{cost.get('passband_low_mean', float('nan')):.4f} / "
             f"{cost.get('passband_high_mean', float('nan')):.4f}", ""),
        ]
    body = "".join(
        f'<tr class="{cls}"><th>{label}</th><td>{_esc(value)}</td></tr>'
        for label, value, cls in rows
    )
    return f'<table class="cost"><tbody>{body}</tbody></table>'


def _convergence_svg(history: list[dict], *, bpf: bool = False) -> str:
    """Line chart of progress against iteration (notch dB or BPF passband IL)."""
    if bpf:
        vals = [float(e["cost"]["passband_min_s21_db"]) for e in history]
        aria = "passband_min_s21_db against iteration"
        goal_db: float | None = None
        goal_label = ""
    else:
        vals = [db(e["cost"]["total_cost"]) for e in history]
        aria = "total_cost in dB against iteration"
        goal_db = TARGET_DEPTH_DB
        goal_label = f"goal {TARGET_DEPTH_DB:.0f} dB"

    n = len(vals)
    lo = min(vals) - 4
    hi = max(vals) + 2
    if goal_db is not None:
        lo = min(lo, goal_db - 4)
        hi = max(hi, -5.0)
    w, h = 720.0, 220.0
    pad_l, pad_r, pad_t, pad_b = 52.0, 14.0, 14.0, 30.0

    def px(i: int) -> float:
        return pad_l if n < 2 else pad_l + i * (w - pad_l - pad_r) / (n - 1)

    def py(v: float) -> float:
        return pad_t + (hi - v) * (h - pad_t - pad_b) / (hi - lo)

    # Align chart "best" with panels/cards: always min(total_cost).
    best_i = min(
        range(n),
        key=lambda i: float(history[i]["cost"]["total_cost"]),
    )

    grid, ticks = [], []
    step = 10
    first = int(math.ceil(lo / step) * step)
    for level in range(first, int(hi) + 1, step):
        y = py(level)
        grid.append(f'<line class="grid" x1="{pad_l}" y1="{y:.1f}" x2="{w-pad_r}" y2="{y:.1f}"/>')
        ticks.append(f'<text class="tick" x="{pad_l-8}" y="{y+3.5:.1f}">{level}</text>')

    goal_svg = ""
    if goal_db is not None:
        goal_y = py(goal_db)
        goal_svg = (
            f'<line class="goal" x1="{pad_l}" y1="{goal_y:.1f}" '
            f'x2="{w-pad_r}" y2="{goal_y:.1f}"/>'
            f'<text class="goal-label" x="{w-pad_r-4}" y="{goal_y-5:.1f}" '
            f'text-anchor="end">{_esc(goal_label)}</text>'
        )

    poly = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(vals))
    dots = "".join(
        f'<circle class="dot{" best" if i == best_i else ""}" '
        f'cx="{px(i):.1f}" cy="{py(v):.1f}" r="{3.6 if i == best_i else 2.4}">'
        f"<title>iter {i}: {v:.2f} dB</title></circle>"
        for i, v in enumerate(vals)
    )
    xlabels = "".join(
        f'<text class="tick" x="{px(i):.1f}" y="{h-10:.1f}" text-anchor="middle">{i}</text>'
        for i in range(n) if n <= 24 or i % 2 == 0
    )
    return f"""<svg class="chart" viewBox="0 0 {w:.0f} {h:.0f}" role="img"
     aria-label="{aria}">
  {''.join(grid)}
  {goal_svg}
  <polyline class="trace" points="{poly}"/>
  {dots}
  {''.join(ticks)}
  {xlabels}
  <text class="axis" x="{pad_l-8}" y="{pad_t-2:.1f}" text-anchor="end">dB</text>
  <text class="axis" x="{(w+pad_l)/2:.0f}" y="{h-1:.0f}" text-anchor="middle">iteration</text>
</svg>"""


def _panel(entry: dict, prev: dict | None, run_dir: Path, is_best: bool, *, bpf: bool) -> str:
    it = entry["iteration"]
    cost = entry["cost"]
    if bpf:
        metric_db = float(cost["passband_min_s21_db"])
        hit_goal = False
        layout_caption = "lumped LC &mdash; no qucsrflayout for this task"
    else:
        metric_db = db(cost["total_cost"])
        hit_goal = metric_db <= TARGET_DEPTH_DB
        layout_caption = "Qucs-RFlayout copper geometry"
    thinking = (entry.get("thinking") or "").strip()
    note = (entry.get("note") or "").strip()
    observation = entry.get("observation") or {}
    obs_text = (observation.get("report_text") or "").strip()
    layout = _inline_layout_svg(run_dir / f"iter_{it:03d}" / "layout.svg")

    if obs_text:
        obs_block = (f'<p class="hint">state after iteration '
                     f'{observation.get("from_iteration", it - 1)}, handed to the strategy '
                     f'layer before it chose this move</p>'
                     f"<pre>{_esc(obs_text)}</pre>")
    else:
        obs_block = ('<p class="missing">not recorded &mdash; this run predates '
                     "observation capture</p>")

    if thinking:
        think_block = f'<div class="thinking">{_esc(thinking)}</div>'
    elif note:
        think_block = ('<p class="missing">no full reasoning recorded; only the '
                       f'one-line note survives</p><div class="thinking">{_esc(note)}</div>')
    else:
        think_block = '<p class="missing">no reasoning recorded</p>'

    intent_json = json.dumps(entry.get("intent") or {}, indent=2)
    badges = []
    if is_best:
        badges.append('<span class="badge best">best</span>')
    if hit_goal:
        badges.append('<span class="badge goal">goal met</span>')

    return f"""<details class="iter{' is-best' if is_best else ''}" id="iter-{it:03d}">
  <summary>
    <span class="it">iter {it:03d}</span>
    <span class="db{' good' if hit_goal else ''}">{metric_db:.2f} dB</span>
    <span class="chips">{_intent_chips(entry.get("intent"))}</span>
    <span class="tag">{_esc(note) if note else ''}</span>
    {''.join(badges)}
  </summary>
  <div class="body">
    <div class="left">
      <div class="layout">{layout}</div>
      <p class="caption">{layout_caption}</p>
      {_param_table(entry["params"], prev["params"] if prev else None)}
    </div>
    <div class="right">
      <section><h4><span class="step">1</span> Input &mdash; what the strategy layer saw</h4>
        {obs_block}</section>
      <section><h4><span class="step">2</span> Thinking &mdash; why this move</h4>
        {think_block}</section>
      <section><h4><span class="step">3</span> Intent &mdash; the strategy layer's output</h4>
        <pre class="intent">{_esc(intent_json)}</pre>
        <p class="hint">intent.py turns these qualitative tokens into numbers:
          step = fraction x 0.93^iteration x range, then clamped to bounds</p></section>
      <section><h4><span class="step">4</span> Result &mdash; what the simulator returned</h4>
        {_cost_table(cost)}</section>
    </div>
  </div>
</details>"""


CSS = """
:root {
  --bg: #f7f4ef; --card: #ffffff; --line: #d8d2c6; --ink: #23201b;
  --dim: #7a7367; --copper: #c4a35a; --green: #2a6f4e; --blue: #0b5fff;
  --red: #b4442b;
}
* { box-sizing: border-box; }
body { margin: 0; padding: 28px 20px 60px; background: var(--bg); color: var(--ink);
  font: 14px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
.wrap { max-width: 1180px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; letter-spacing: -0.2px; }
h1 .run { font-family: ui-monospace, Menlo, monospace; color: var(--copper); }
.sub { color: var(--dim); margin: 0 0 20px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 10px; margin-bottom: 18px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 8px;
  padding: 10px 12px; }
.card .k { font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: var(--dim); }
.card .v { font-size: 19px; font-weight: 700; font-family: ui-monospace, Menlo, monospace; }
.card.win .v { color: var(--green); }
.chartbox { background: var(--card); border: 1px solid var(--line); border-radius: 8px;
  padding: 8px 10px; margin-bottom: 18px; }
.chart { width: 100%; height: auto; }
.chart .grid { stroke: #e8e3d8; stroke-width: 1; }
.chart .goal { stroke: var(--green); stroke-width: 1.2; stroke-dasharray: 5 4; }
.chart .goal-label, .chart .tick, .chart .axis {
  font-family: ui-monospace, Menlo, monospace; font-size: 10px; fill: var(--dim); }
.chart .goal-label { fill: var(--green); }
.chart .trace { fill: none; stroke: var(--blue); stroke-width: 2; stroke-linejoin: round; }
.chart .dot { fill: var(--blue); }
.chart .dot.best { fill: var(--green); }
.toolbar { display: flex; gap: 8px; margin-bottom: 12px; }
.toolbar button { font: inherit; font-size: 13px; padding: 5px 12px; cursor: pointer;
  background: var(--card); border: 1px solid var(--line); border-radius: 6px; color: var(--ink); }
.toolbar button:hover { border-color: var(--copper); }
details.iter, details.conclusion { background: var(--card); border: 1px solid var(--line);
  border-radius: 8px; margin-bottom: 8px; overflow: hidden; }
details.conclusion { margin-top: 18px; border-color: var(--copper); }
details.conclusion > summary { background: #fbf6ea; }
details.conclusion .thinking { margin: 14px; }
details.iter.is-best { border-color: var(--green); border-width: 2px; }
summary { cursor: pointer; padding: 10px 14px; display: flex; align-items: center;
  gap: 10px; flex-wrap: wrap; list-style: none; }
summary::-webkit-details-marker { display: none; }
summary::before { content: "\\25B8"; color: var(--dim); font-size: 11px; }
details[open] > summary::before { content: "\\25BE"; }
details[open] > summary { border-bottom: 1px solid var(--line); background: #fbf9f5; }
summary .it { font-family: ui-monospace, Menlo, monospace; font-weight: 700; }
summary .db { font-family: ui-monospace, Menlo, monospace; min-width: 76px;
  text-align: right; color: var(--red); }
summary .db.good { color: var(--green); }
summary .tag { color: var(--dim); font-size: 12.5px; flex: 1; min-width: 120px; }
.chip { display: inline-block; font-family: ui-monospace, Menlo, monospace; font-size: 12px;
  padding: 1px 7px; border-radius: 20px; margin-right: 4px; border: 1px solid var(--line); }
.chip.up { color: #8a5a12; background: #fdf3e0; border-color: #edd9b0; }
.chip.down { color: #12518a; background: #e8f1fd; border-color: #b8d3f2; }
.chip.hold, .chip.none { color: var(--dim); background: #f2efe9; }
.badge { font-size: 11px; padding: 2px 8px; border-radius: 20px; font-weight: 700;
  text-transform: uppercase; letter-spacing: .05em; }
.badge.best { background: var(--green); color: #fff; }
.badge.goal { background: #e8f4ee; color: var(--green); border: 1px solid var(--green); }
.body { display: grid; grid-template-columns: 300px 1fr; gap: 18px; padding: 16px 14px; }
@media (max-width: 900px) { .body { grid-template-columns: 1fr; } }
.layout { background: #fbf9f5; border: 1px solid var(--line); border-radius: 6px; padding: 10px; }
.layout svg { width: 100%; height: auto; display: block; }
.layout svg path, .layout svg rect { fill: var(--copper); }
.is-best .layout svg path, .is-best .layout svg rect { fill: var(--green); }
.caption { font-size: 11.5px; color: var(--dim); margin: 6px 0 12px; text-align: center; }
section { margin-bottom: 16px; }
h4 { font-size: 13px; margin: 0 0 6px; display: flex; align-items: center; gap: 7px; }
h4 .step { display: inline-flex; align-items: center; justify-content: center;
  width: 18px; height: 18px; border-radius: 50%; background: var(--ink); color: var(--card);
  font-size: 11px; font-weight: 700; }
pre { font-family: ui-monospace, Menlo, monospace; font-size: 12px; line-height: 1.45;
  background: #fbf9f5; border: 1px solid var(--line); border-radius: 6px;
  padding: 9px 11px; margin: 0; overflow-x: auto; white-space: pre; }
pre.intent { background: #eef4ff; border-color: #cfdffb; }
.thinking { background: #fffdf6; border: 1px solid #ece3c9; border-left: 3px solid var(--copper);
  border-radius: 6px; padding: 10px 12px; white-space: pre-wrap; font-size: 13px; }
.hint, .missing { font-size: 12px; color: var(--dim); margin: 0 0 6px; }
.missing { font-style: italic; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; }
table th, table td { text-align: left; padding: 4px 8px; border-bottom: 1px solid #eee8dc; }
table.params th { font-family: ui-monospace, Menlo, monospace; }
table.params td { font-family: ui-monospace, Menlo, monospace; text-align: right; }
table.params thead th { font-size: 11px; color: var(--dim); text-transform: uppercase; }
table.cost th { font-weight: 400; color: var(--dim); width: 46%; }
table.cost td { font-family: ui-monospace, Menlo, monospace; }
table.cost tr.primary th, table.cost tr.primary td { font-weight: 700; color: var(--ink); }
.up { color: #8a5a12; }
.down { color: #12518a; }
.dim { color: var(--dim); }
.now { font-weight: 700; }
footer { color: var(--dim); font-size: 12px; margin-top: 24px; }
"""

JS = """
document.getElementById('expand').onclick = function () {
  document.querySelectorAll('details.iter').forEach(function (d) { d.open = true; });
};
document.getElementById('collapse').onclick = function () {
  document.querySelectorAll('details.iter').forEach(function (d) { d.open = false; });
};
"""


def build_report(
    run_name: str,
    run_dir: Path,
    history: list[dict],
    conclusion: str = "",
    task: str | None = None,
) -> str:
    bpf = _is_bpf(task, history[0]["cost"] if history else None)
    best = min(history, key=lambda e: e["cost"]["total_cost"])
    best_it = best["iteration"]
    if bpf:
        start_db = float(history[0]["cost"]["passband_min_s21_db"])
        best_db = float(best["cost"]["passband_min_s21_db"])
        title = f"Butterworth BPF5 &mdash; run <span class=\"run\">{_esc(run_name)}</span>"
        page_title = f"Butterworth BPF5 &mdash; run {_esc(run_name)}"
        subtitle = (
            "Objective: passband insertion loss and stopband attenuation on a "
            "5th-order lumped LC ladder. No qucsrflayout; open circuit.sch in Qucs-S."
        )
        footer_extra = "BPF runs omit layout.svg."
        win_cls = ""
    else:
        start_db = db(history[0]["cost"]["total_cost"])
        best_db = db(best["cost"]["total_cost"])
        title = f"Butterfly radial stub &mdash; run <span class=\"run\">{_esc(run_name)}</span>"
        page_title = f"Butterfly stub optimization &mdash; run {_esc(run_name)}"
        subtitle = (
            f"Objective: minimize |S21| at 5.5 GHz on a shunt butterfly notch, "
            f"goal {TARGET_DEPTH_DB:.0f} dB. The strategy layer sees only the measured "
            f"report and emits qualitative intent; <code>intent.py</code> owns every number."
        )
        footer_extra = "Layouts are Qucs-RFlayout output, recoloured with CSS."
        win_cls = "win" if best_db <= TARGET_DEPTH_DB else ""

    reasoned = sum(1 for e in history if (e.get("thinking") or "").strip())

    panels = []
    for i, entry in enumerate(history):
        prev = history[i - 1] if i > 0 else None
        panels.append(
            _panel(entry, prev, run_dir, is_best=entry["iteration"] == best_it, bpf=bpf)
        )

    cards = [
        ("iterations", f"{len(history)}", ""),
        ("baseline", f"{start_db:.1f} dB", ""),
        ("best", f"{best_db:.1f} dB", win_cls),
        ("best iteration", f"{best_it:03d}", ""),
        ("reasoning captured", f"{reasoned}/{len(history)}", ""),
    ]
    card_html = "".join(
        f'<div class="card {cls}"><div class="k">{k}</div><div class="v">{v}</div></div>'
        for k, v, cls in cards
    )
    conclusion_html = (
        f'<details class="conclusion" open><summary><span class="it">why the run stopped</span>'
        f'</summary><div class="thinking">{_esc(conclusion.strip())}</div></details>'
        if conclusion.strip() else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{page_title}</title>
<style>{CSS}</style>
</head><body><div class="wrap">
<h1>{title}</h1>
<p class="sub">{subtitle}</p>
<div class="cards">{card_html}</div>
<div class="chartbox">{_convergence_svg(history, bpf=bpf)}</div>
<div class="toolbar">
  <button id="expand" type="button">expand all</button>
  <button id="collapse" type="button">collapse all</button>
</div>
{''.join(panels)}
{conclusion_html}
<footer>Generated by <code>run_step.py html --run {_esc(run_name)}</code>.
{footer_extra}</footer>
</div><script>{JS}</script></body></html>"""


def write_report(
    run_name: str,
    run_dir: Path,
    history: list[dict],
    out_path: Path,
    conclusion: str = "",
    task: str | None = None,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        build_report(run_name, run_dir, history, conclusion, task=task)
    )
    return out_path
