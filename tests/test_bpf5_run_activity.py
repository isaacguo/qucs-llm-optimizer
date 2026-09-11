# tests/test_bpf5_run_activity.py
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from jobs.bpf5_agent.progress import ProgressStore
from jobs.bpf5_agent.run_activity import (
    AGENT_MODEL,
    build_agent_cmd,
    build_rollout_prompt,
    run_activity,
)


class TestBuildAgentCmd(unittest.TestCase):
    def test_agent_cmd_uses_auto_model(self):
        repo = Path("/tmp/repo")
        cmd = build_agent_cmd(repo, prompt="hi")
        self.assertEqual(cmd[0], "agent")
        self.assertIn("-p", cmd)
        self.assertIn("--force", cmd)
        self.assertIn("--trust", cmd)
        self.assertIn("--workspace", cmd)
        self.assertEqual(cmd[cmd.index("--workspace") + 1], str(repo))
        self.assertIn("--model", cmd)
        self.assertEqual(cmd[cmd.index("--model") + 1], "auto")
        self.assertEqual(cmd[cmd.index("--model") + 1], AGENT_MODEL)
        self.assertIn("--output-format", cmd)
        self.assertEqual(cmd[cmd.index("--output-format") + 1], "text")
        self.assertEqual(cmd[-1], "hi")


class TestBuildRolloutPrompt(unittest.TestCase):
    def test_prompt_mentions_run_step_and_completions_not_think_tree(self):
        prompt = build_rollout_prompt(
            repo=Path("/repo"),
            run_id="bpf_agent_20260912_021530/rollout_b03_cf112p5_bw5",
            bucket=3,
            cf_mhz=112.5,
            bw_mhz=5.0,
        )
        self.assertIn("run_step.py", prompt)
        self.assertIn("init", prompt)
        self.assertIn("observe", prompt)
        self.assertIn("step", prompt)
        self.assertIn("--thinking", prompt)
        self.assertIn("completions.jsonl", prompt)
        self.assertIn("goal_met", prompt)
        # Explicit: do not require a think/ directory layout
        self.assertRegex(prompt, r"(?i)do not.*think/")
        self.assertNotRegex(prompt, r"(?i)write .* to .*think/")


class TestRunActivityMocked(unittest.TestCase):
    def test_one_rollout_invokes_agent_with_auto_and_records_success(self):
        with tempfile.TemporaryDirectory() as td:
            runs_root = Path(td) / "runs"
            repo = Path(td) / "repo"
            repo.mkdir()
            activity_id = "bpf_agent_20260912_021530"

            def fake_agent(cmd, *, cwd, env=None, timeout_s=None):
                self.assertIn("--model", cmd)
                self.assertEqual(cmd[cmd.index("--model") + 1], "auto")
                # Agent "succeeds": write a goal_met history into the assigned rollout.
                # Find the rollout dir under activity (created by assign before invoke).
                activity_dir = runs_root / activity_id
                rollouts = [p for p in activity_dir.iterdir() if p.is_dir()]
                self.assertEqual(len(rollouts), 1)
                rollout = rollouts[0]
                state_path = rollout / "state.json"
                state = json.loads(state_path.read_text(encoding="utf-8"))
                state["history"] = [
                    {
                        "iteration": 3,
                        "cost": {"goal_met": True},
                        "params": {},
                        "intent": None,
                        "note": "",
                    }
                ]
                state["iteration"] = 3
                state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
                return 0

            with mock.patch(
                "jobs.bpf5_agent.run_activity.invoke_agent",
                side_effect=fake_agent,
            ) as inv:
                summary = run_activity(
                    runs_root=runs_root,
                    activity_id=activity_id,
                    max_rollouts=1,
                    seed=1,
                    repo=repo,
                )

            self.assertEqual(inv.call_count, 1)
            cmd = inv.call_args.args[0]
            self.assertEqual(cmd[cmd.index("--model") + 1], "auto")
            self.assertEqual(summary["ok"], 1)
            self.assertEqual(summary["attempts"], 1)
            progress = ProgressStore(runs_root / activity_id / "progress.json")
            self.assertEqual(sum(progress.load()["counts"]), 1)
            # Successful rollout kept
            activity_dir = runs_root / activity_id
            kept = [p for p in activity_dir.iterdir() if p.is_dir()]
            self.assertEqual(len(kept), 1)

    def test_failed_rollout_is_deleted_and_not_counted(self):
        with tempfile.TemporaryDirectory() as td:
            runs_root = Path(td) / "runs"
            repo = Path(td) / "repo"
            repo.mkdir()
            activity_id = "bpf_agent_fail_case"

            def fake_agent(cmd, *, cwd, env=None, timeout_s=None):
                activity_dir = runs_root / activity_id
                rollout = next(p for p in activity_dir.iterdir() if p.is_dir())
                state_path = rollout / "state.json"
                state = json.loads(state_path.read_text(encoding="utf-8"))
                state["history"] = [
                    {
                        "iteration": 0,
                        "cost": {"goal_met": False},
                        "params": {},
                        "intent": None,
                        "note": "",
                    }
                ]
                state["iteration"] = 0
                state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
                return 0

            with mock.patch(
                "jobs.bpf5_agent.run_activity.invoke_agent",
                side_effect=fake_agent,
            ):
                summary = run_activity(
                    runs_root=runs_root,
                    activity_id=activity_id,
                    max_rollouts=1,
                    seed=42,
                    repo=repo,
                )

            self.assertEqual(summary["ok"], 0)
            self.assertEqual(summary["attempts"], 1)
            progress = ProgressStore(runs_root / activity_id / "progress.json")
            self.assertEqual(sum(progress.load()["counts"]), 0)
            activity_dir = runs_root / activity_id
            rollouts = [p for p in activity_dir.iterdir() if p.is_dir()]
            self.assertEqual(rollouts, [])

    def test_module_does_not_import_choose_intent_from_skills(self):
        import ast

        import jobs.bpf5_agent.run_activity as mod

        tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported.append(alias.name)
                if node.module:
                    imported.append(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported.append(alias.name)
        self.assertNotIn("choose_intent_from_skills", imported)
        self.assertTrue(
            all("choose_intent_from_skills" not in name for name in imported)
        )


if __name__ == "__main__":
    unittest.main()
