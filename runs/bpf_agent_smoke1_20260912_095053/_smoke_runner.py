import os, sys
from pathlib import Path
REPO = Path('/Volumes/Datasets/workspace_ssd/qucs-llm-optimizer')
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
os.chdir(REPO)
os.environ["PYTHONPATH"] = f"{REPO / 'src'}:{REPO}"

import jobs.bpf5_agent.run_activity as ra

activity_dir = Path('/Volumes/Datasets/workspace_ssd/qucs-llm-optimizer/runs/bpf_agent_smoke1_20260912_095053')
_orig_invoke = ra.invoke_agent
_orig_delete = ra._delete_rollout
_attempt = {"n": 0}

def invoke_agent(cmd, *, cwd, env=None, timeout_s=900):
    _attempt["n"] += 1
    out_path = activity_dir / f"agent_attempt_{_attempt['n']:02d}.out"
    print(f"[smoke] invoke agent -> {out_path}", flush=True)
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    mac = "/Applications/qucs-s.app/Contents/MacOS"
    run_env["PATH"] = f"{mac}:{mac}/bin:" + run_env.get("PATH", "")
    import subprocess
    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd), env=run_env, text=True,
            capture_output=True, timeout=timeout_s, check=False,
        )
        out_path.write_text(
            (proc.stdout or "") + "\n--- STDERR ---\n" + (proc.stderr or ""),
            encoding="utf-8",
        )
        print(f"[smoke] agent rc={proc.returncode} bytes={out_path.stat().st_size}", flush=True)
        return int(proc.returncode)
    except subprocess.TimeoutExpired as exc:
        out_path.write_text(
            str(exc.stdout or "") + "\n--- STDERR ---\n" + str(exc.stderr or "") + "\nTIMEOUT\n",
            encoding="utf-8",
        )
        print("[smoke] agent TIMEOUT", flush=True)
        return 124

def _keep_failed(rollout_dir: Path) -> None:
    print(f"[smoke] KEEP failed rollout {rollout_dir}", flush=True)

ra.invoke_agent = invoke_agent
ra._delete_rollout = _keep_failed
raise SystemExit(ra.main([
    "--max-rollouts", "1",
    "--activity", 'bpf_agent_smoke1_20260912_095053',
    "--seed", str(5446612),
    "--timeout-s", "900",
    "--runs-root", str('/Volumes/Datasets/workspace_ssd/qucs-llm-optimizer/runs'),
    "--repo", str('/Volumes/Datasets/workspace_ssd/qucs-llm-optimizer'),
]))
