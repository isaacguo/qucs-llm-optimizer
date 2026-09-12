# BPF5 Cursor-agent 教师采集 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 BPF 任务从 `src/` 迁到 `jobs/bpf5_agent/`，用通用插件注册解耦框架；删除规则式采集与旧 `corpus/` 混放包；实现带时间戳 activity、桶配额、Cursor agent（`--model auto`）教师采集。

**Architecture:** `src/task_registry.py` 只提供 `register_plugin` / `get_plugin` / `get_task`；`run_step.py`（仓库根，非 `src/`）扫描 `jobs/*/register.py` 完成发现。`jobs/bpf5_agent/` 持有目标/代价/task/skills/prompt/模板/采样/assign/gate/驱动。框架不 `import jobs…`。

**Tech Stack:** Python 3.12、现有 `run_step.py` / Qucs、`agent` CLI、`unittest`/`pytest`、`uv`

## Global Constraints

- 回复与 **spec/plan 正文** 用简体中文；代码标识符/路径/命令保持原文
- `src/` = 通用框架；`jobs/` = 具体任务；`runs/` = 仅产出；`src/` **不得** `import jobs`
- 教师仅为 Cursor `agent`，固定 `--model auto`；禁止 `choose_intent_from_skills` 代选 intent
- 无全局 `runs/index.*`；进度在 activity 内 `progress.json`
- 正式推理落盘：`completions.jsonl` + `state.json` history；不要求 `think/` 布局
- 禁用词：`cold-start` / `coldstart`
- 设计权威：[`docs/superpowers/specs/2026-09-12-bpf-agent-teacher-collection-design.md`](../specs/2026-09-12-bpf-agent-teacher-collection-design.md)

## 文件结构（锁定）

```text
src/
  task_registry.py          # NEW：插件协议 + 注册表（无 jobs 依赖）
  tasks/butterfly_stub.py   # 保留 demo
  tasks/__init__.py         # 只注册 butterfly + 委托 get_task
  # 删除：bpf_tuning_skills.py, goals_bpf.py, cost_bpf.py, tasks/butterworth_bpf5.py
  qucs_sim.py               # 去掉 BPF_TEMPLATE_PATH 硬编码（用调用方 template_path）

jobs/
  __init__.py
  bpf5_agent/
    __init__.py
    register.py             # 向 task_registry 注册插件
    plugin.py               # TaskPlugin 实现（goal/cost/sim/observe 钩子）
    goals.py                # 自 goals_bpf 迁入
    cost.py                 # 自 cost_bpf 迁入
    task.py                 # 自 tasks/butterworth_bpf5 迁入
    skills.py               # 自 bpf_tuning_skills 迁入（仅 prompt/诊断辅助；驱动不用选 intent）
    prompts/bpf_tuning_skills_system.md
    templates/butterworth_bpf5.sch.tpl
    buckets.py              # 20 桶采样
    progress.py             # progress.json
    assign.py               # 创建 rollout + 写 goal/meta
    gate.py                 # 成功判定 + 更新进度
    run_activity.py         # activity 驱动（调 agent --model auto）

run_step.py                 # discover_plugins()；经 plugin 调度，去掉 _is_bpf 硬编码 import

scripts/run_cursor_agent_corpus.py  # 改为明确失败提示（notch 待另 job）或删除 CLI 入口
# 删除：scripts/run_bpf_skills_ablation.py, run_bpf_skills_generalization.py, corpus/
```

**命名锁定：** `job_id = bpf5_agent`；activity 名 = `bpf_agent_YYYYMMDD_HHMMSS`（本地时区墙钟）；rollout 名 = `rollout_b{bucket:02d}_cf{cf_mhz}p{frac}_bw{bw}`（实现时用可读且文件系统安全的格式）。

**`--run` 语义：** 继续传相对 `runs/` 的路径段。rollout 的 run id 形如：`bpf_agent_20260912_021530/rollout_b03_cf112p5_bw5`（`RunState` 已按 `runs/<run>/` 工作即可；若现实现假设单层目录，本计划 Task 3 须验证/修正）。

---

### Task 1: 通用 `TaskPlugin` 注册表

**Files:**
- Create: `src/task_registry.py`
- Modify: `src/tasks/__init__.py`
- Test: `tests/test_task_registry.py`

**Interfaces:**
- Produces:
  - `class TaskPlugin`（Protocol / ABC）含至少：`name: str`、`config: TaskConfig`、`default_goal() -> dict`、`goal_from_state(raw: dict | None) -> Any`、`validate_goal(goal) -> None`、`evaluate(sim_result, goal) -> dict`（须含 `goal_met`）、`simulate(params, workdir, goal)`、`format_report(entry, goal) -> str`、`format_observation(entry, goal, *, include_skills_system: bool) -> str`、`param_unit(var: str) -> str`
  - `register_plugin(plugin: TaskPlugin) -> None`
  - `get_plugin(name: str) -> TaskPlugin`
  - `get_task(name: str) -> TaskConfig`（plugin.config；未知则 ValueError）
  - `clear_plugins_for_tests() -> None`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_task_registry.py
import unittest
from task_registry import clear_plugins_for_tests, get_plugin, get_task, register_plugin
from tasks import TaskConfig
from pathlib import Path

class _Dummy:
    name = "dummy_task"
    @property
    def config(self):
        return TaskConfig(
            name="dummy_task",
            variables=("x",),
            bounds={"x": (0.0, 1.0)},
            initial_guess={"x": 0.5},
            template_path=Path("."),
            export_layout=False,
        )
    def default_goal(self): return {"g": 1}
    def goal_from_state(self, raw): return raw or self.default_goal()
    def validate_goal(self, goal): return None
    def evaluate(self, sim_result, goal): return {"goal_met": False}
    def simulate(self, params, workdir, goal): raise NotImplementedError
    def format_report(self, entry, goal): return "r"
    def format_observation(self, entry, goal, *, include_skills_system=True): return "o"
    def param_unit(self, var): return ""

class TestTaskRegistry(unittest.TestCase):
    def setUp(self):
        clear_plugins_for_tests()
    def test_register_and_get(self):
        register_plugin(_Dummy())
        self.assertEqual(get_plugin("dummy_task").name, "dummy_task")
        self.assertEqual(get_task("dummy_task").name, "dummy_task")
    def test_unknown(self):
        with self.assertRaises(ValueError):
            get_task("nope")
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /Volumes/Datasets/workspace_ssd/qucs-llm-optimizer && PYTHONPATH=src python -m pytest tests/test_task_registry.py -v`  
Expected: FAIL（`task_registry` 缺失）

- [ ] **Step 3: 实现 `src/task_registry.py` + 改 `tasks/__init__.py`**

`tasks/__init__.py`：保留 `TaskConfig`；`get_task` 改为调用 `task_registry.get_task`；在模块加载时用内联方式注册 `butterfly_stub` 为最小内置 plugin（可把 butterfly 的 evaluate/simulate 仍留给 `run_step` 默认路径——见 Task 3）。本 Task 至少保证：`get_task("butterfly_stub")` 仍返回现有 `TaskConfig` 字段。

最小做法：`task_registry` 同时支持「仅 config 的内置任务」与完整 `TaskPlugin`。若 butterfly 暂无完整 plugin，允许 `register_builtin_config(TaskConfig)`；`get_plugin` 仅对完整 plugin 可用。**BPF 必须是完整 plugin。**

- [ ] **Step 4: 测试通过**

Run: `PYTHONPATH=src python -m pytest tests/test_task_registry.py tests/test_intent_bpf.py -v`（此时 BPF 仍在旧路径则 intent 测试可能仍绿；若已迁则等 Task 2）

- [ ] **Step 5: Commit**

```bash
git add src/task_registry.py src/tasks/__init__.py tests/test_task_registry.py
git commit -m "$(cat <<'EOF'
Add generic task plugin registry for job decoupling.

EOF
)"
```

---

### Task 2: 迁出 BPF 模块到 `jobs/bpf5_agent/`

**Files:**
- Create: `jobs/__init__.py`, `jobs/bpf5_agent/{__init__,register,plugin,goals,cost,task,skills}.py`
- Create: `jobs/bpf5_agent/prompts/bpf_tuning_skills_system.md`（自 `prompts/` 复制或移动）
- Create: `jobs/bpf5_agent/templates/butterworth_bpf5.sch.tpl`（自 `templates/` 移动）
- Delete: `src/bpf_tuning_skills.py`, `src/goals_bpf.py`, `src/cost_bpf.py`, `src/tasks/butterworth_bpf5.py`
- Modify: `pyproject.toml`（`packages.find` 含 `jobs*`，去掉 `corpus*`）
- Modify: 相关测试 import 路径
- Modify: `src/qucs_sim.py` 删除 `BPF_TEMPLATE_PATH` 常量（若不再被引用）

**Interfaces:**
- Produces: `jobs.bpf5_agent.register.register()` → `register_plugin(Bpf5Plugin())`
- `Bpf5Plugin.name == "butterworth_bpf5"`
- `task.TEMPLATE_PATH` 指向 `jobs/bpf5_agent/templates/butterworth_bpf5.sch.tpl`
- `skills.DEFAULT_SKILLS_PATH` 指向 `jobs/bpf5_agent/prompts/bpf_tuning_skills_system.md`

- [ ] **Step 1: 写/改失败测试（插件加载后 get_task）**

```python
# tests/test_bpf5_job_plugin.py
import importlib
import unittest
from task_registry import clear_plugins_for_tests, get_task

class TestBpf5JobPlugin(unittest.TestCase):
    def setUp(self):
        clear_plugins_for_tests()
        # 重新注册 butterfly 若需要
        from tasks import ensure_builtin_tasks
        ensure_builtin_tasks()
        importlib.import_module("jobs.bpf5_agent.register").register()
    def test_task_config(self):
        t = get_task("butterworth_bpf5")
        self.assertEqual(t.name, "butterworth_bpf5")
        self.assertTrue(t.template_path.is_file())
        self.assertEqual(len(t.variables), 10)
```

- [ ] **Step 2: 运行确认失败**

Run: `PYTHONPATH=src:. python -m pytest tests/test_bpf5_job_plugin.py -v`  
Expected: FAIL

- [ ] **Step 3: 迁移文件并实现 `plugin.py` / `register.py`**

迁入后修正内部 import，例如：
- `from jobs.bpf5_agent.goals import ...`
- `from jobs.bpf5_agent.cost import ...`
- `from jobs.bpf5_agent.task import VARIABLES, ...`

`skills.py`：**保留** `load_skills_system_prompt` 与诊断辅助；`choose_intent_from_skills` 可留供单测，但 **任何新驱动不得调用它选 intent**。

更新既有测试：
- `tests/test_goals_bpf.py` → import `jobs.bpf5_agent.goals`
- `tests/test_cost_bpf.py` → `jobs.bpf5_agent.cost`
- `tests/test_intent_bpf.py` / `test_butterworth_bpf5_template.py` / `test_run_step_bpf.py`：先经 `register()` 再测

- [ ] **Step 4: 在 `run_step.py` 顶部加入发现（临时，Task 3 完善）**

```python
def discover_job_plugins(repo_root: Path) -> None:
    jobs_root = repo_root / "jobs"
    if not jobs_root.is_dir():
        return
    for reg in sorted(jobs_root.glob("*/register.py")):
        # importlib.util.spec_from_file_location + exec，调用 register()
        ...
```

**禁止**在 `src/` 内实现 `discover_job_plugins`。

- [ ] **Step 5: 全量相关测试**

Run: `PYTHONPATH=src:. python -m pytest tests/test_bpf5_job_plugin.py tests/test_goals_bpf.py tests/test_cost_bpf.py tests/test_intent_bpf.py tests/test_butterworth_bpf5_template.py tests/test_run_step_bpf.py -v`  
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add jobs src tests pyproject.toml prompts templates
git commit -m "$(cat <<'EOF'
Move butterworth_bpf5 task modules into jobs/bpf5_agent.

EOF
)"
```

---

### Task 3: `run_step.py` 经 plugin 调度（去掉 BPF 硬编码）

**Files:**
- Modify: `run_step.py`
- Modify: `tests/test_run_step_bpf.py`
- Test: 扩展 `tests/test_run_step_bpf.py` 断言 observe 含 skills 文案且来自 job prompt

**Interfaces:**
- Consumes: `get_plugin(task_name)`；若无 plugin（butterfly），走现有 notch 路径
- 删除顶层 `from cost_bpf import ...` / `from goals_bpf import ...` / `from bpf_tuning_skills import ...`
- `_cli_task_name`：允许任意已注册 task 名，不只两个常量

- [ ] **Step 1: 写失败测试（无硬编码模块）**

```python
def test_run_step_does_not_import_jobs_bpf_modules_at_top_level(self):
    import run_step
    import sys
    # 顶层加载后，sys.modules 可以有 jobs（因 discover），但 run_step 源码不得出现
    src = Path(run_step.__file__).read_text(encoding="utf-8")
    self.assertNotIn("from cost_bpf", src)
    self.assertNotIn("from goals_bpf", src)
    self.assertNotIn("from bpf_tuning_skills", src)
    self.assertNotIn("_is_bpf", src)  # 用 get_plugin 有无代替
```

- [ ] **Step 2: 重构 `cmd_init` / `cmd_step` / `format_*`**

伪代码：

```python
plugin = None
try:
    plugin = get_plugin(task_name)
except ValueError:
    plugin = None

if plugin is not None:
    goal = plugin.goal_from_state(...)
    res = plugin.simulate(...)
    cost = plugin.evaluate(res, goal)
    ...
else:
    # butterfly 旧路径
    ...
```

`format_observation`：若 plugin 有 `format_observation`，委托之（内含 skills system）。

- [ ] **Step 3: 测试**

Run: `PYTHONPATH=src:. python -m pytest tests/test_run_step_bpf.py tests/test_run_step_goal.py tests/test_run_step_decision_log.py -v`  
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add run_step.py tests/test_run_step_bpf.py
git commit -m "$(cat <<'EOF'
Route run_step BPF path through job plugins only.

EOF
)"
```

---

### Task 4: 删除规则式脚本与 `corpus/`，保住可导入性

**Files:**
- Delete: `scripts/run_bpf_skills_ablation.py`, `scripts/run_bpf_skills_generalization.py`
- Delete: `corpus/` 包（`cli.py`, `index.py`, `__init__.py`, `.gitkeep`）
- Delete: `tests/test_corpus.py`, `tests/test_corpus_package_import.py`
- Modify: `pyproject.toml` 去掉 `qucs-corpus` script
- Modify: `scripts/run_cursor_agent_corpus.py` → 开头 `sys.exit` 并打印：notch 采集待迁入独立 job；本仓库 BPF 请用 `jobs/bpf5_agent/run_activity.py`
- Modify: `tests/test_sft_data.py` / `tests/test_sft_cli.py`：去掉对 `corpus.index` 的依赖。若 SFT 仍需读 jsonl 索引，把**通用** `append_index` / 读行辅助迁到 `training/common/jsonl_index.py`（无 notch 业务），并改测试指向临时文件；默认路径字符串改为显式参数，不再默认 `corpus/index.jsonl`（若改默认属行为变化，在提交说明写清）。

- [ ] **Step 1: 确认无生产代码再 import `corpus`**

Run: `rg -n "from corpus|import corpus|qucs-corpus" --glob '!docs/**' --glob '!**/plans/**'`  
Expected: 仅测试/脚本待改处；改完后为零（或仅 deprecation 提示字符串）

- [ ] **Step 2: 按上表删除/修改并跑测试**

Run: `PYTHONPATH=src:. python -m pytest tests/ -v --ignore=...`（全量；修红）  
Expected: PASS（与 Qucs 无关的测试）

- [ ] **Step 3: Commit**

```bash
git add -A scripts corpus tests pyproject.toml training
git commit -m "$(cat <<'EOF'
Remove rule-based BPF collectors and mixed corpus package.

EOF
)"
```

---

### Task 5: 桶采样 + `progress.json`

**Files:**
- Create: `jobs/bpf5_agent/buckets.py`, `jobs/bpf5_agent/progress.py`
- Test: `tests/test_bpf5_buckets.py`, `tests/test_bpf5_progress.py`

**Interfaces:**
- `N_BUCKETS = 20`, `BUCKET_WIDTH_MHZ = 15`, `SWEEP_MHZ = (0.0, 300.0)`, `BW_CHOICES_MHZ = (5, 10, 15)`, `SUCCESS_PER_BUCKET = 10`, `MAX_STEPS = 20`
- `sample_goal(bucket: int, rng: random.Random) -> tuple[dict, float, float]`  
  返回 `(goal_dict, cf_mhz, bw_mhz)`，`goal_dict` 可被 `goals.goal_from_dict` 接受；通带在 sweep 内
- `pick_bucket(counts: list[int], rng) -> int`：优先 `counts[b] < 10` 的桶
- `ProgressStore(path)`：`load()` / `record_success(bucket: int)` / `is_complete() -> bool`  
  文件格式示例：`{"version": 1, "success_per_bucket": 10, "counts": [0]*20}`

- [ ] **Step 1: 失败测试（边界与重抽）**

```python
def test_sample_passband_inside_sweep(self):
    rng = random.Random(0)
    for b in range(20):
        for _ in range(50):
            goal, cf, bw = sample_goal(b, rng)
            half = bw / 2
            self.assertGreaterEqual(cf - half, 0.0)
            self.assertLessEqual(cf + half, 300.0)
            self.assertEqual(goal["f_low_hz"], (cf - half) * 1e6)
```

- [ ] **Step 2: 实现并测试通过**

Run: `PYTHONPATH=src:. python -m pytest tests/test_bpf5_buckets.py tests/test_bpf5_progress.py -v`

- [ ] **Step 3: Commit**

```bash
git add jobs/bpf5_agent/buckets.py jobs/bpf5_agent/progress.py tests/test_bpf5_buckets.py tests/test_bpf5_progress.py
git commit -m "$(cat <<'EOF'
Add BPF5 bucket sampling and activity progress helpers.

EOF
)"
```

---

### Task 6: assign + gate（activity / rollout）

**Files:**
- Create: `jobs/bpf5_agent/assign.py`, `jobs/bpf5_agent/gate.py`
- Test: `tests/test_bpf5_assign_gate.py`

**Interfaces:**
- `make_activity_id(now: datetime) -> str` → `bpf_agent_YYYYMMDD_HHMMSS`
- `make_rollout_id(bucket: int, cf_mhz: float, bw_mhz: float) -> str`
- `assign_rollout(runs_root, activity_id, bucket, goal_dict, *, cf_mhz, bw_mhz) -> Path`  
  创建目录；写入可被 `run_step init` 消费的 meta（`task=butterworth_bpf5` + goal）。优先调用现有 `RunState.set_run_meta` / 等价 API。
- `gate_rollout(rollout_dir: Path, *, max_steps: int = 20) -> bool`  
  读 `state.json`：`goal_met` 且达成迭代 `<= max_steps` 为成功
- `on_success(progress: ProgressStore, bucket: int) -> None`

- [ ] **Step 1: 失败测试（临时 runs 目录）**

用 `tempfile.TemporaryDirectory` 模拟 `runs/`，assign 后检查 `state` 或 meta 文件字段；构造假 `state.json` 测 gate。

- [ ] **Step 2: 实现并通过**

Run: `PYTHONPATH=src:. python -m pytest tests/test_bpf5_assign_gate.py -v`

- [ ] **Step 3: Commit**

```bash
git add jobs/bpf5_agent/assign.py jobs/bpf5_agent/gate.py tests/test_bpf5_assign_gate.py
git commit -m "$(cat <<'EOF'
Add BPF5 activity assign and success gate.

EOF
)"
```

---

### Task 7: Cursor agent activity 驱动（`--model auto`）

**Files:**
- Create: `jobs/bpf5_agent/run_activity.py`
- Create: `jobs/bpf5_agent/agent_prompt.md`（或 `.py` 内常量）：角色/任务/harness/actions/做法；明确写 reasoning 经 `--thinking` 进入 `completions.jsonl`；**不要**要求维护 `think/` 树
- Test: `tests/test_bpf5_run_activity.py`（mock `subprocess.run`，断言 cmd 含 `--model`, `auto`）

**Interfaces:**
- CLI：`python -m jobs.bpf5_agent.run_activity [--runs-root runs] [--activity ID|auto] [--max-rollouts N] [--seed S]`
- 每条 rollout：assign → 调 `agent -p --force --trust --workspace <repo> --model auto --output-format text <prompt>`  
  prompt 指示 agent：对该 rollout 跑 `run_step init/observe/step`，intent 自选，thinking 用 `--thinking` 或临时文件，成功条件 `goal_met` 且 ≤20 步
- 驱动在 agent 返回后 `gate_rollout`；成功则 `progress.record_success`；失败可删 rollout 目录
- **禁止** import `choose_intent_from_skills` 作为教师

- [ ] **Step 1: 测试 agent 命令行**

```python
def test_agent_cmd_uses_auto_model(self):
    cmd = build_agent_cmd(repo, prompt="hi")
    self.assertIn("--model", cmd)
    self.assertEqual(cmd[cmd.index("--model") + 1], "auto")
```

- [ ] **Step 2: 实现 `run_activity.py` + 单测 mock 一轮调度（不真调 agent/Qucs）**

- [ ] **Step 3: Dry-run 文档化命令**

```bash
PYTHONPATH=src:. python -m jobs.bpf5_agent.run_activity --help
```

- [ ] **Step 4: Commit**

```bash
git add jobs/bpf5_agent/run_activity.py jobs/bpf5_agent/agent_prompt.md tests/test_bpf5_run_activity.py
git commit -m "$(cat <<'EOF'
Add BPF5 Cursor-agent activity driver with Auto model.

EOF
)"
```

---

### Task 8: System prompt 六块结构润色（可审）

**Files:**
- Modify: `jobs/bpf5_agent/prompts/bpf_tuning_skills_system.md`

- [ ] **Step 1: 按设计六块加清晰小标题**（角色 / 任务 / Harness / Actions / 做法 / 如何更好），**不改变**定性 intent 约束与技能优先级语义
- [ ] **Step 2: 单测 `load_skills_system_prompt` 仍非空且含 `increase`/`decrease` 等关键词**
- [ ] **Step 3: Commit**

```bash
git add jobs/bpf5_agent/prompts/bpf_tuning_skills_system.md tests/
git commit -m "$(cat <<'EOF'
Clarify BPF skills system prompt role and harness sections.

EOF
)"
```

---

### Task 9: 验收清单（人工 + 自动）

- [ ] **Step 1: 自动**

```bash
rg -n "from cost_bpf|from goals_bpf|bpf_tuning_skills|choose_intent_from_skills" jobs/bpf5_agent/run_activity.py
# 期望：run_activity 无 choose_intent_from_skills
test ! -e src/goals_bpf.py
test ! -e src/cost_bpf.py
test ! -e src/bpf_tuning_skills.py
test ! -e src/tasks/butterworth_bpf5.py
test ! -d corpus || test ! -f corpus/cli.py
PYTHONPATH=src:. python -m pytest tests/ -v
```

- [ ] **Step 2: Dry-run（可无 Qucs）**

```bash
PYTHONPATH=src:. python - <<'PY'
from pathlib import Path
import random
from jobs.bpf5_agent.buckets import sample_goal
from jobs.bpf5_agent.progress import ProgressStore
g, cf, bw = sample_goal(3, random.Random(1))
print(cf, bw, g["f_low_hz"], g["f_high_hz"])
p = ProgressStore(Path("/tmp/bpf_progress_test.json"))
p.load(); p.record_success(3); print(p.load())
PY
```

- [ ] **Step 3: 有 Qucs 时（可选）** 起一个短 activity `--max-rollouts 1`，确认 rollout 下出现 `completions.jsonl`，且无强制 `think/` 目录要求

- [ ] **Step 4: 最终 commit（若有残留）**

---

## 规格覆盖自检

| 规格项 | Task |
|--------|------|
| `--model auto` | 7 |
| `jobs/` 采样/assign/gate | 5–6 |
| 迁出 BPF 出 `src/` | 2–3 |
| `src` 不 import `jobs` | 1–3（发现在 `run_step`） |
| 无全局 index | 5–6 |
| `completions.jsonl` / 无强制 think/ | 3, 7 |
| 删规则脚本 + corpus | 4 |
| system prompt 保留/润色 | 2, 8 |
| 20×10×≤20 步 | 5–7 |

## 占位符扫描

计划内无 TBD/TODO 实现洞；插件 `discover` 的 `importlib` 细节在 Task 2 Step 4 由实现者按标准库写法补全（不得改回 `src` 内发现）。
