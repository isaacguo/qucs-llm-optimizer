"""Operator CLI for corpus assign / gate / coverage / export / import-run.

Workflow:
  qucs-corpus assign --run ID --goal-config configs/goal_distribution.yaml
  # then: run_step.py init / observe / step ... (assign does not simulate)
  qucs-corpus gate --run ID
  qucs-corpus coverage --index corpus/index.jsonl --goal-config ...
  qucs-corpus export --index corpus/index.jsonl --out path.jsonl

Optional legacy import (rewrites state.json; copy the run first):
  cp -R runs/llm1 runs/llm1-import
  qucs-corpus import-run --run llm1-import
  # refuses if existing goal differs unless --force

``--prefer-coverage`` (assign): samples a short seed window and picks the goal
whose coverage bin is least filled in the current index. If the index is empty
or missing, sampling is identical to the normal path.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from state import RunState  # noqa: E402

from training.corpus import (  # noqa: E402
    append_index,
    coverage_counts,
    export_from_index,
    gate_run,
    index_record_from_run,
    load_index,
)
from training.environment import sample_params  # noqa: E402
from training.goals import (  # noqa: E402
    GoalDistributionConfig,
    GoalSpec,
    band_for,
    load_goal_distribution,
    sample_goal_from_distribution,
)

_PREFER_COVERAGE_TRIES = 32

# Reference goal for the historical runs/llm1 trajectory (5.5 GHz / −70 dB).
_LLM1_TARGET_FREQ_HZ = 5.5e9
_LLM1_TARGET_DEPTH_DB = -70.0


def _llm1_reference_goal() -> dict:
    band = band_for(_LLM1_TARGET_FREQ_HZ)
    return {
        "target_freq_hz": _LLM1_TARGET_FREQ_HZ,
        "band_hz": [float(band[0]), float(band[1])],
        "target_depth_db": _LLM1_TARGET_DEPTH_DB,
    }


def _goal_dict(goal: GoalSpec) -> dict:
    raw = asdict(goal)
    band = raw["band_hz"]
    raw["band_hz"] = [float(band[0]), float(band[1])]
    return raw


def _bin_for_goal(goal: GoalSpec, dist: GoalDistributionConfig) -> tuple[int, int]:
    freq_bin = int((goal.target_freq_hz - dist.freq_min_hz) // dist.freq_bin_hz)
    depth_bin = int((goal.target_depth_db - dist.depth_db_min) // dist.depth_bin_db)
    return (freq_bin, depth_bin)


def _sample_goal_prefer_coverage(
    seed: int,
    dist: GoalDistributionConfig,
    index: list[dict],
) -> tuple[GoalSpec, int]:
    """Pick seed in [seed, seed+N) whose goal lands in the least-covered bin."""
    counts = coverage_counts(index, dist)
    best_seed = seed
    best_goal = sample_goal_from_distribution(seed, dist)
    best_count = counts.get(_bin_for_goal(best_goal, dist), 0)
    for offset in range(1, _PREFER_COVERAGE_TRIES):
        cand_seed = seed + offset
        cand_goal = sample_goal_from_distribution(cand_seed, dist)
        cand_count = counts.get(_bin_for_goal(cand_goal, dist), 0)
        if cand_count < best_count:
            best_seed = cand_seed
            best_goal = cand_goal
            best_count = cand_count
            if best_count == 0:
                break
    return best_goal, best_seed


def cmd_assign(args: argparse.Namespace) -> int:
    dist = load_goal_distribution(args.goal_config)
    seed = int(args.seed)
    if args.prefer_coverage:
        index = load_index(Path(args.index))
        goal, seed = _sample_goal_prefer_coverage(seed, dist, index)
    else:
        goal = sample_goal_from_distribution(seed, dist)

    run_dir = Path(args.runs_root) / args.run
    state = RunState(run_dir)
    if state.history or state.iteration >= 0:
        print(
            f"warning: run {args.run} already has history/iteration; "
            "overwriting goal metadata only",
            file=sys.stderr,
        )
    state.set_run_meta(
        goal=_goal_dict(goal),
        start_seed=seed,
        initial_params=sample_params(seed),
    )
    print(
        f"assigned {args.run}: seed={seed} "
        f"goal={goal.target_freq_hz / 1e9:.3f} GHz / {goal.target_depth_db:.1f} dB"
    )
    print("next: run_step.py init (preserves goal/initial_params; simulates baseline)")
    return 0


def _gate_and_maybe_index(
    *,
    run_id: str,
    runs_root: Path,
    index_path: Path,
) -> int:
    run_dir = Path(runs_root) / run_id
    state_path = run_dir / "state.json"
    if not state_path.is_file():
        raise SystemExit(f"missing state: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    corpus = gate_run(state)
    run_state = RunState(run_dir)
    run_state.set_run_meta(corpus=corpus)
    # Re-read so index record sees any goal/corpus just persisted.
    state = json.loads(state_path.read_text(encoding="utf-8"))
    print(json.dumps(corpus, sort_keys=True))
    if corpus.get("eligible"):
        record = index_record_from_run(
            run_id,
            str(Path(run_id)),
            state,
            corpus,
        )
        append_index(Path(index_path), record)
        print(f"appended to {index_path}")
    else:
        print(f"not indexed ({corpus.get('reason')})")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    return _gate_and_maybe_index(
        run_id=args.run,
        runs_root=Path(args.runs_root),
        index_path=Path(args.index),
    )


def _goals_match_llm1_reference(existing: dict, reference: dict) -> bool:
    """True when freq/depth match the llm1 reference (band derived from freq)."""
    try:
        return (
            float(existing["target_freq_hz"]) == float(reference["target_freq_hz"])
            and float(existing["target_depth_db"]) == float(reference["target_depth_db"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def cmd_import_run(args: argparse.Namespace) -> int:
    """Attach the llm1 reference goal (5.5 GHz / −70 dB), then gate + index."""
    run_dir = Path(args.runs_root) / args.run
    state_path = run_dir / "state.json"
    if not state_path.is_file():
        raise SystemExit(f"missing state: {state_path}")
    goal = _llm1_reference_goal()
    run_state = RunState(run_dir)
    existing = run_state.goal
    if existing is not None and not _goals_match_llm1_reference(existing, goal):
        if not getattr(args, "force", False):
            raise SystemExit(
                f"import-run {args.run}: existing state.goal differs from llm1 "
                f"reference (5.5e9 Hz / -70 dB); refuse to overwrite without --force "
                f"(got target_freq_hz={existing.get('target_freq_hz')!r} "
                f"target_depth_db={existing.get('target_depth_db')!r})"
            )
    elif existing is None:
        print(
            f"import-run {args.run}: warning: writing llm1 reference goal onto a "
            f"previously goal-less run (rewrites state.json)",
            file=sys.stderr,
        )
    run_state.set_run_meta(goal=goal)
    print(
        f"import-run {args.run}: attached goal="
        f"{goal['target_freq_hz'] / 1e9:.3f} GHz / {goal['target_depth_db']:.1f} dB"
    )
    return _gate_and_maybe_index(
        run_id=args.run,
        runs_root=Path(args.runs_root),
        index_path=Path(args.index),
    )


def cmd_coverage(args: argparse.Namespace) -> int:
    dist = load_goal_distribution(args.goal_config)
    index = load_index(Path(args.index))
    counts = coverage_counts(index, dist)
    total = sum(counts.values())
    print(f"total: {total}")
    print(f"bins: {len(counts)}")
    for (freq_bin, depth_bin), n in sorted(counts.items()):
        print(f"  freq_bin={freq_bin} depth_bin={depth_bin} count={n}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    records = export_from_index(
        Path(args.index),
        Path(args.runs_root),
        history_window=int(args.history_window),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"wrote {len(records)} records to {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qucs-corpus",
        description="Corpus assign / gate / coverage / export / import-run",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_assign = sub.add_parser("assign", help="Sample goal + initial_params into runs/<id>/state.json")
    p_assign.add_argument("--run", required=True, help="Run id under --runs-root")
    p_assign.add_argument(
        "--goal-config",
        default="configs/goal_distribution.yaml",
        help="Goal distribution YAML",
    )
    p_assign.add_argument("--seed", type=int, default=0)
    p_assign.add_argument(
        "--prefer-coverage",
        action="store_true",
        help="Bias seed toward under-covered bins using --index",
    )
    p_assign.add_argument(
        "--index",
        default="corpus/index.jsonl",
        help="Corpus index used only with --prefer-coverage",
    )
    p_assign.add_argument("--runs-root", default="runs")
    p_assign.set_defaults(func=cmd_assign)

    p_gate = sub.add_parser("gate", help="Gate a finished run and append to index if eligible")
    p_gate.add_argument("--run", required=True)
    p_gate.add_argument("--index", default="corpus/index.jsonl")
    p_gate.add_argument("--runs-root", default="runs")
    p_gate.set_defaults(func=cmd_gate)

    p_cov = sub.add_parser("coverage", help="Print coverage bin counts from index")
    p_cov.add_argument("--index", default="corpus/index.jsonl")
    p_cov.add_argument(
        "--goal-config",
        default="configs/goal_distribution.yaml",
    )
    p_cov.set_defaults(func=cmd_coverage)

    p_export = sub.add_parser("export", help="Export multiturn SFT JSONL from index")
    p_export.add_argument("--index", default="corpus/index.jsonl")
    p_export.add_argument("--out", required=True)
    p_export.add_argument("--runs-root", default="runs")
    p_export.add_argument("--history-window", type=int, default=8)
    p_export.set_defaults(func=cmd_export)

    p_import = sub.add_parser(
        "import-run",
        help="Attach llm1 reference goal (5.5 GHz / −70 dB), gate, index if eligible",
    )
    p_import.add_argument("--run", required=True, help="Run id under --runs-root (e.g. llm1)")
    p_import.add_argument("--index", default="corpus/index.jsonl")
    p_import.add_argument("--runs-root", default="runs")
    p_import.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing state.goal that differs from the llm1 reference",
    )
    p_import.set_defaults(func=cmd_import_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
