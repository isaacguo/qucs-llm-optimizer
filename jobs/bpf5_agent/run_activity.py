# jobs/bpf5_agent/run_activity.py
"""Drive Cursor ``agent`` (``--model auto``) for BPF5 teacher collection.

Each attempt: assign → agent (init/observe/step via prompt) → gate → progress
or delete failed rollout. Intent selection is left to the agent CLI only.
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from jobs.bpf5_agent.assign import assign_rollout, make_activity_id
from jobs.bpf5_agent.buckets import pick_bucket, sample_goal
from jobs.bpf5_agent.gate import gate_rollout, on_success
from jobs.bpf5_agent.progress import ProgressStore

AGENT_MODEL = "auto"
REPO_ROOT = Path(__file__).resolve().parents[2]
_PROMPT_PATH = Path(__file__).resolve().parent / "agent_prompt.md"
_DEFAULT_TIMEOUT_S = 900


def build_agent_cmd(repo: Path | str, prompt: str) -> list[str]:
    """Build Cursor agent CLI; model is always Auto (``auto``)."""
    return [
        "agent",
        "-p",
        "--force",
        "--trust",
        "--workspace",
        str(repo),
        "--model",
        AGENT_MODEL,
        "--output-format",
        "text",
        prompt,
    ]


def load_prompt_template() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8")


def build_rollout_prompt(
    *,
    repo: Path | str,
    run_id: str,
    bucket: int,
    cf_mhz: float,
    bw_mhz: float,
) -> str:
    template = load_prompt_template()
    return template.format(
        repo=str(repo),
        run_id=run_id,
        bucket=int(bucket),
        cf_mhz=float(cf_mhz),
        bw_mhz=float(bw_mhz),
    )


def invoke_agent(
    cmd: list[str],
    *,
    cwd: Path | str,
    env: dict[str, str] | None = None,
    timeout_s: int = _DEFAULT_TIMEOUT_S,
) -> int:
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    mac_bin = "/Applications/qucs-s.app/Contents/MacOS"
    run_env["PATH"] = f"{mac_bin}:{mac_bin}/bin:" + run_env.get("PATH", "")
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=run_env,
            text=True,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return 124
    return int(proc.returncode)


def _delete_rollout(rollout_dir: Path) -> None:
    if not rollout_dir.exists():
        return
    trash = shutil.which("trash")
    if trash:
        subprocess.run([trash, str(rollout_dir)], check=False)
    else:
        shutil.rmtree(rollout_dir, ignore_errors=True)


def _count_rollout_dirs(activity_dir: Path) -> int:
    if not activity_dir.is_dir():
        return 0
    return sum(
        1
        for p in activity_dir.iterdir()
        if p.is_dir() and p.name.startswith("rollout_")
    )


def _attempt_rng(seed: int, attempt: int, prior_rollouts: int) -> random.Random:
    """Derive a per-attempt RNG so resume with the same seed samples differently."""
    mixed = (
        (int(seed) & 0xFFFFFFFF)
        ^ ((int(attempt) * 0x9E3779B9) & 0xFFFFFFFF)
        ^ ((int(prior_rollouts) * 0x85EBCA6B) & 0xFFFFFFFF)
    )
    return random.Random(mixed)


def run_one_rollout(
    *,
    runs_root: Path,
    activity_id: str,
    progress: ProgressStore,
    rng: random.Random,
    repo: Path,
    timeout_s: int = _DEFAULT_TIMEOUT_S,
) -> bool:
    data = progress.load()
    bucket = pick_bucket(list(data["counts"]), rng)
    goal, cf_mhz, bw_mhz = sample_goal(bucket, rng)
    rollout_dir = assign_rollout(
        runs_root,
        activity_id,
        bucket,
        goal,
        cf_mhz=cf_mhz,
        bw_mhz=bw_mhz,
    )
    run_id = f"{activity_id}/{rollout_dir.name}"
    prompt = build_rollout_prompt(
        repo=repo,
        run_id=run_id,
        bucket=bucket,
        cf_mhz=cf_mhz,
        bw_mhz=bw_mhz,
    )
    cmd = build_agent_cmd(repo, prompt)
    invoke_agent(cmd, cwd=repo, timeout_s=timeout_s)
    if gate_rollout(rollout_dir, expected_goal=goal):
        on_success(progress, bucket)
        return True
    _delete_rollout(rollout_dir)
    return False


def run_activity(
    *,
    runs_root: Path | str,
    activity_id: str,
    max_rollouts: int,
    seed: int,
    repo: Path | str | None = None,
    timeout_s: int = _DEFAULT_TIMEOUT_S,
) -> dict:
    runs_root = Path(runs_root)
    repo = Path(repo) if repo is not None else REPO_ROOT
    activity_dir = runs_root / activity_id
    activity_dir.mkdir(parents=True, exist_ok=True)
    progress = ProgressStore(activity_dir / "progress.json")
    ok_n = 0
    attempts = 0
    for _ in range(int(max_rollouts)):
        if progress.is_complete():
            break
        attempts += 1
        prior = _count_rollout_dirs(activity_dir)
        rng = _attempt_rng(int(seed), attempts, prior)
        if run_one_rollout(
            runs_root=runs_root,
            activity_id=activity_id,
            progress=progress,
            rng=rng,
            repo=repo,
            timeout_s=timeout_s,
        ):
            ok_n += 1
    return {"ok": ok_n, "attempts": attempts, "activity_id": activity_id}


def _resolve_activity_id(raw: str) -> str:
    if raw == "auto" or not raw.strip():
        return make_activity_id(datetime.now().astimezone())
    return raw.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "BPF5 Cursor-agent teacher activity driver "
            f"(always --model {AGENT_MODEL})."
        )
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=REPO_ROOT / "runs",
        help="Root directory for activity/rollout outputs (default: <repo>/runs)",
    )
    parser.add_argument(
        "--activity",
        default="auto",
        help='Activity id, or "auto" for bpf_agent_YYYYMMDD_HHMMSS',
    )
    parser.add_argument(
        "--max-rollouts",
        type=int,
        default=1,
        help="Maximum assign→agent→gate attempts this invocation",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for bucket/goal")
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=_DEFAULT_TIMEOUT_S,
        help="Per-rollout agent subprocess timeout",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=REPO_ROOT,
        help="Workspace passed to agent --workspace",
    )
    args = parser.parse_args(argv)

    activity_id = _resolve_activity_id(args.activity)
    summary = run_activity(
        runs_root=args.runs_root,
        activity_id=activity_id,
        max_rollouts=args.max_rollouts,
        seed=args.seed,
        repo=args.repo,
        timeout_s=args.timeout_s,
    )
    print(
        f"activity={summary['activity_id']} "
        f"ok={summary['ok']}/{summary['attempts']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
