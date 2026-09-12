# Cursor agent 教师采集：Butterworth BPF5

**日期：** 2026-09-12  
**状态：** 已批准（2026-09-12）  
**方案：** activity 落在 `runs/`；**`src/` = 通用训练/优化框架，`jobs/` = 具体任务实现，二者完全解耦**；教师仅为 Cursor `agent`，模型固定 **Auto**（`--model auto`）

## 目标

以 **Cursor agent（Auto Model）** 为唯一教师，采集 **完整且成功** 的 BPF5 调谐轨迹：在 0–300 MHz 上填满 **20 个频率桶**，桶内随机中心频率，带宽取 `{5, 10, 15}` MHz，每桶至少 **10 条成功** rollout，且每条成功须在 **≤ 20 步** 内达成。

## 动机

基于规则的 BPF 驱动（`run_bpf_skills_ablation.py`、`run_bpf_skills_generalization.py`）不能当教师。现有 notch 的 Cursor-agent 采集也不是 BPF。产出只能落在 `runs/`。**职责必须彻底分开：** `src/` 只做通用框架；凡「针对某一具体采集/调谐任务」的设计与实现一律进 `jobs/<job_id>/`，**禁止**混进 `src/`。**run** 指覆盖多条 rollout 的一次 **activity**，不是单条轨迹。

## 术语（已锁定）

| 术语 | 含义 |
|------|------|
| **job** | 一类**具体任务**的全部实现（例如 BPF5 agent 教师采集）。桶规则、assign、gate、进度、采集驱动、任务专属 prompt/策略等**全部**放在 `jobs/<job_id>/`。 |
| **run（activity）** | 一次独立的教师活动，目标填满该 job 的配额（对本 job：20 桶）。名称须含 **时间戳**。各 run 互不共享状态。 |
| **rollout（completion）** | activity 内、针对单个 `(cf, bw)` 目标的一条完整优化 **轨迹**。 |
| **成功** | `butterworth_bpf5` 上 `goal_met` **且** `iteration ≤ 20`。失败不计入桶配额。 |
| **教师** | 仅 Cursor `agent` CLI，**`--model auto`（Auto Model）**。intent JSON 由 agent 选定，绝不用 `choose_intent_from_skills` 或其他 Python 策略代选。 |

## 非目标

- 采集时用规则 skills 代码 **选择** intent
- 全局 `runs/index.json` / `runs/index.jsonl`（已拒绝：选项 A）
- 在 `src/` 混入任何具体采集任务逻辑（桶采样、assign、gate、job 配额/调度、任务专属教师策略）
- 让 `src/` import `jobs/`（框架反向依赖具体任务）
- 把生产代码放到 `runs/` 下
- 在 rollout 目录中维护独立的 `think/` 树作为正式产物（见下文「推理落盘」）
- 未经单独批准就改 BPF 拓扑、代价公式或 `BpfGoalSpec` 默认值
- 恢复禁用措辞（`cold-start` / `coldstart`）

## 决策（已锁定）

| 主题 | 选择 |
|------|------|
| 教师 | Cursor `agent` |
| 模型 | **Auto Model**，CLI 固定 `--model auto`（与现有 notch 采集脚本默认一致） |
| 采样 | 0–300 MHz → 20 个等宽 15 MHz 桶；桶内随机 `cf` + `bw ∈ {5,10,15}` MHz，通带须落在 sweep 内 |
| 配额 | 每桶 ≥ 10 条成功 rollout |
| 步数上限 | 每条 rollout 20 步 |
| Job 代码 | `jobs/<job_id>/`：**该具体任务的全部实现**（含从 `src/` **一并迁出** 的 BPF 目标/代价/task 模块、桶采样、assign、gate、进度、采集驱动、任务专属文案等） |
| 框架代码 | `src/`：**仅**通用训练/优化框架与跨任务共享基础设施；与具体采集任务**完全解耦**；**不得**再保留 BPF 任务专属模块 |
| 依赖方向 | `jobs/` → 可调用 `src/`；`src/` → **不得**依赖 `jobs/`（任务经框架注册/发现接口加载，见下文） |
| 迁出 `src/`（已批准） | `bpf_tuning_skills.py`、`goals_bpf.py`、`cost_bpf.py`、`tasks/butterworth_bpf5.py` → 对应 `jobs/<job_id>/`；框架侧改为通用 task 注册/发现，去掉对 BPF 的硬编码分支 |
| 产出根目录 | 仅 `runs/<带时间戳的 activity>/` |
| 全局索引 | **无** |
| activity 进度 | 写在该 activity 目录内（例如 `progress.json`） |
| 推理落盘 | **写入 `completions.jsonl`（及 `state.json` history 的 `thinking` 字段）**；**不**再把独立 `think/` 目录当作正式布局要求 |
| System prompt | **保留** BPF skills markdown，作为 agent 的 **system prompt**：角色、任务、harness、允许 action、怎么做、如何更好 |
| 删除 | `scripts/run_bpf_skills_ablation.py`、`scripts/run_bpf_skills_generalization.py`；删除曾混放索引与包代码的旧顶层目录 |
| 任务 id | 经现有 `run_step.py` 使用 `butterworth_bpf5` |

## 设计

### 1. 分层（完全解耦）

**硬边界（已锁定）：**

| 树 | 放什么 | 不放什么 |
|----|--------|----------|
| **`src/`** | 通用训练/优化框架：`run_step` 契约、state、decision_log、通用 intent 算术、**通用 task 注册/发现**、与「某一个采集 job」无关的共享能力 | BPF 或其它具体任务的目标/代价/变量表、桶采样、assign、gate、job 进度/配额、教师采集驱动、任务专属策略 |
| **`jobs/<job_id>/`** | 该具体任务的**全部**设计实现：迁入的目标/代价/task、采样、assign、gate、进度、驱动、任务专属 prompt/策略 | 通用框架内核 |
| **`runs/`** | activity / rollout **产出** | 任何业务包 |

```text
jobs/<job_id>/     具体任务实现（与 src 完全解耦）
src/               通用训练/优化框架（不感知具体采集 job）
scripts/           可选：薄启动器（选 job、起 activity），不含任务业务
prompts/           可被 job 引用的 prompt 源（如 BPF skills）
training/          本阶段不改

runs/              仅产出
  <activity_ts>/
    progress.json
    rollout_…/
      state.json
      completions.jsonl    # 每步 prompt + completion（含 reasoning）
      iter_*/…
```

本阶段 BPF job 示例路径：`jobs/bpf5_agent/`（具体 `job_id` 在实现计划中固定）。

**迁出 `src/`（已批准，实现阶段执行）：**

| 现位置 | 迁入 |
|--------|------|
| `src/bpf_tuning_skills.py` | `jobs/<job_id>/` |
| `src/goals_bpf.py` | `jobs/<job_id>/` |
| `src/cost_bpf.py` | `jobs/<job_id>/` |
| `src/tasks/butterworth_bpf5.py` | `jobs/<job_id>/` |

配套要求：

- `src/tasks/__init__.py`（或等价注册表）**不得**再 `elif name == "butterworth_bpf5"` 硬编码 import；改为通用注册/发现（例如 job 声明 task id + 入口，框架按名加载）。
- `run_step.py` / `qucs_sim.py` 中仅服务 BPF 的分支，改为经 task 插件接口调用 job 实现；框架不 import `jobs.*` 具体模块路径写死在业务里（发现机制在实现计划中定）。
- `prompts/bpf_tuning_skills_system.md` 可由 job 引用；归属以 job 为准（实现计划决定是否物理挪到 `jobs/<job_id>/`）。
- `templates/butterworth_bpf5.sch.tpl` 属该任务资源，迁出时一并归 job（或由 job 声明路径）；实现计划固定。
- 默认示例任务 `butterfly_stub` 可暂留框架侧，作为通用 demo；若日后也要完全任务化，另开设计。
- 测试随模块迁移更新 import；删除规则式 BPF 采集脚本后，其依赖一并消失。

依赖方向仍硬约束：`src/` **不** `import jobs…`；由注册表返回已加载的 task 对象/可调用接口。
### 2. Activity 目录布局

示例：

```text
runs/
  bpf_agent_20260912_021530/
    progress.json
    rollout_b03_cf112p5_bw5/
      state.json
      completions.jsonl
      …
```

- activity 名称 **必须含时间戳**（具体格式在实现计划中固定）。
- 每条单目标轨迹（成功或尝试）是 activity 下的一个 **rollout** 子目录。
- 失败 rollout 可删可留且不标记成功；**不得**增加 `progress.json` 中的桶计数。

### 3. 推理落盘（回答：为何不要单独 `think/`）

现状里 `think/step_XX.md` 只是给 `run_step.py step --thinking-file …` 的**临时喂入**方式；`run_step` 真正持久化时已经：

1. 把 reasoning 写入 `state.json` 对应 history 条目的 `thinking`；
2. 经 `decision_log.append_record` 写入该 rollout 的 **`completions.jsonl`**（字段 `completion` 含 reasoning + intent）。

因此：**正式产物以 `completions.jsonl` 为准**（文件名与现有 harness 一致；不另造 `completion.jsonl`）。  
本设计 **取消**「每个 rollout 必须有 `think/` 目录」的布局要求。若实现上 agent 仍需短暂写临时文件再传 `--thinking-file`，该临时文件可放在 rollout 内任意暂存路径或用 `--thinking` 直传，**不**升格为对外约定的目录结构。

### 4. 桶采样（逻辑在 job 内）

实现放在 `jobs/bpf5_agent/`（名称以实现计划为准），**不**放进 `src/`。

- Sweep：`[0, 300]` MHz（`BpfGoalSpec.sweep_hz`）。
- 桶 `b ∈ {0..19}` 覆盖 `[b*15, (b+1)*15)` MHz（末桶按需含 300 MHz）。
- `bw` 在 `{5, 10, 15}` MHz 上均匀抽取。
- `cf` 在桶内均匀抽取，再设 `f_low = cf - bw/2`、`f_high = cf + bw/2`。
- 若通带越出 `[0, 300]` MHz 则拒绝并重抽。
- 用现有 IL/阻带默认构造 `BpfGoalSpec(f_low_hz=…, f_high_hz=…)`，除非另行批准改默认。
- 调度下一条 rollout 时，优先选成功数仍不足 10 的桶。

### 5. Cursor-agent rollout 循环

驱动调用：`agent -p --force --trust --model auto …`（**禁止**默认成其它 model）。

对 activity 内每个调度到的 `(bucket, cf, bw)`：

1. Job 的 assign：创建 rollout 目录；写入 goal 与 `task=butterworth_bpf5`。
2. `run_step.py init --run <rollout 路径或 id> --task butterworth_bpf5`（精确 CLI 在实现计划中固定；不得破坏现有 butterfly 默认）。
3. 最多 20 次：`observe` → agent 读观察（含 **system prompt**）→ 选定 intent，经 `--thinking` / `--thinking-file` 把推理交给 `step` → harness 追加 **`completions.jsonl`**。
4. 若 20 步内 `goal_met`：job 的 gate / 进度逻辑更新 activity 的 `progress.json` 并保留 rollout。
5. 否则：不计配额；可选删除 rollout 目录。

全部桶均 ≥ 10 条成功，或操作者停止驱动时，activity 结束。

### 6. System prompt（保留并写清职责）

**保留** `prompts/bpf_tuning_skills_system.md`（当前经 `bpf_tuning_skills.load_skills_system_prompt` 注入 BPF 的 `observe`）。

**用途（已锁定）：** 作为 Cursor-agent 教师的 **system prompt** 一部分，须交代清楚：

1. **角色** — 5 阶 series-first LC BPF 的 strategy 层  
2. **任务** — 仅用定性 intent 达到 `goal_met`  
3. **Harness** — `run_step` 的 observe/step、`state.json`、`completions.jsonl`、bounds、cost 字段  
4. **Actions** — 允许的 intent 键值；禁止数值 L/C  
5. **做法** — diagnose → ladder 块 / BW / 中心移动  
6. **如何更好** — hysteresis、anti-oscillation、slight vs strong 纪律  

**拟议实现注意（非改目标）：** 实现时可润色该 markdown，使上述六块更醒目，但**不得**用 Python `choose_intent_from_skills` 代替 agent 选 intent。若润色文案，在实现计划中单独列为可审改动。

### 7. 删除与迁移

- 删除上文列出的规则式 BPF 采集脚本。
- 删除曾用于索引 + 包代码的旧顶层混放目录；本 job 的 assign/gate/桶逻辑新建在 `jobs/bpf5_agent/`（或实现计划中的等价 `job_id`），路径指向 `runs/<activity>/…`。
- 通用框架留在 `src/` / `run_step.py`；**新**采集逻辑一律进 `jobs/`。
- 现有 notch Cursor-agent 脚本日后可迁成另一个 `jobs/…`；**本 spec 交付重点是 BPF5 job**。因删除混放目录而必须改的 notch 路径，仅以保证仓库可运行为限（在实现计划中写明）。

### 8. 验证

- Dry-run：创建 activity 目录，采样一条合法 `(cf,bw)`，init + 一次 observe 可见 system prompt 与 BPF report。
- 一步 `step` 后 rollout 出现 **`completions.jsonl`**，其中含 reasoning；**不**要求存在 `think/` 目录。
- Agent 调用带 `--model auto`。
- 桶采样 / assign / gate 模块路径落在 `jobs/…`，不在 `src/`。
- 若有 Qucs：一条真实成功 rollout 能更新 `progress.json`。
- 确认新驱动未 import 规则式 intent 选择器。
- 确认 `src/` 中不再存在 `bpf_tuning_skills` / `goals_bpf` / `cost_bpf` / `tasks/butterworth_bpf5`；`get_task("butterworth_bpf5")` 经注册/发现仍可用。
- 确认未创建全局 `runs/index.*`。

## 成功标准

1. BPF 采集的 intent 教师仅为 Cursor agent，且使用 **Auto Model**（`--model auto`）。  
2. 一次带时间戳的 activity 能按采样规则填满 20×10 成功 rollout。  
3. **`src/` 与 `jobs/` 完全解耦**：BPF 任务模块已从 `src/` 迁出；新采集/任务逻辑只在 `jobs/<job_id>/`；`src/` 只保留通用框架；产出仅在 `runs/<activity>/`；`src/` 不依赖 `jobs/`。  
4. 无全局 runs 索引文件。  
5. 推理以 `completions.jsonl`（及 state history）为正式落盘；无强制 `think/` 布局。  
6. System prompt 保留，并按角色 / 任务 / harness / actions / 做法 / 更好 写清。  
7. 规则式 BPF 采集脚本已删除。
