# Split SFT and GRPO into separate packages

**Date:** 2026-09-12  
**Status:** Pending user review of this file  
**Approach:** Four packages under `training/` (Approach A + flat layout, no old-path shims)

## Goal

把现在混在同一个 `training/` 目录里的 SFT 与 multi-turn GRPO 代码，按功能拆成不同文件夹，避免以后再当成一套东西改。

## Motivation

当前 `training/sft.py`、`training/multiturn_train.py`、`training/data.py`、`training/rollout.py`、`training/corpus*.py` 平铺在一起。SFT 数据加载还会从 GRPO 的 `rollout` 里拿系统提示词。结果是：改 GRPO 容易误伤 SFT，新人也不知道哪段属于哪条训练线。

## Terminology (locked)

| 说法 | 含义 |
|------|------|
| **GRPO cold-start** | 用已训好的 SFT LoRA 启动 multi-turn GRPO（`qucs-multiturn --resume-adapter`）。cold-start 的主语是 GRPO，不是 SFT。 |
| **SFT warm-up** | 用 corpus 里合格教师轨迹做一轮监督微调，产出 LoRA，供上面的 GRPO cold-start 使用。 |
| **corpus** | 教师轨迹的采集与验收（assign / gate / coverage / export）。**不是** 本仓库 policy 的 inference，也不是训练循环本身。 |

流水线（仅作背景，本改动不改算法）：

`qucs-corpus` 收合格轨迹 → `qucs-sft` warm-up → `qucs-multiturn --resume-adapter` 做 GRPO cold-start。

## Non-goals

- 不改训练算法、reward、YAML 字段语义
- 不把 GRPO 再拆成更深的 `train/rollout/reward` 子包（那是否决的方案 2）
- 不保留旧 import 路径的兼容层（否决的方案 3）
- 不把 corpus 并进 SFT 或 common（用户要求按功能单独成包）
- 不改对外命令名：`qucs-sft` / `qucs-multiturn` / `qucs-corpus` 保持不变

## Decisions (locked)

| Topic | Choice |
|-------|--------|
| Layout | `training/common/` + `training/sft/` + `training/grpo/` + `training/corpus/` |
| Dependencies | 见下表；**禁止** `sft` ↔ `grpo` 互相 import |
| SYSTEM_PROMPT / multi-turn 拼 prompt | 抽到 `training.common.prompts`，corpus / sft / grpo 都从 common 读 |
| Old modules | 搬家后删除旧扁平文件；无 shim |
| CLI names | 不变；只改 `pyproject.toml` entry points |
| Tests | import 改到新路径；测试文件按包重命名对齐 |

## Design

### 1. Packages and dependencies

```text
training/
  __init__.py              # 只写包说明，不 re-export
  common/                  # 两边都要用的基础设施
  corpus/                  # 教师轨迹采集 / 索引 / 导出
  sft/                     # SFT 数据装载 + 训练入口
  grpo/                    # multi-turn GRPO 训练、仿真循环、reward、评测
```

| 包 | 可以依赖 | 不可以依赖 |
|----|----------|------------|
| `common` | 标准库 / 第三方 / 现有 `src`（与今天一致） | `sft`、`grpo`、`corpus` |
| `corpus` | `common` | `sft`、`grpo` |
| `sft` | `common`、`corpus` | `grpo` |
| `grpo` | `common` | `sft`、`corpus` |

说明：GRPO 在线自己跑仿真循环，不读 `corpus/index.jsonl`。SFT 通过 `corpus` 的 export/index API 读教师数据。

### 2. File move map

| 新路径 | 来源 |
|--------|------|
| `training/common/modeling.py` | `training/modeling.py` |
| `training/common/goals.py` | `training/goals.py` |
| `training/common/environment.py` | `training/environment.py` |
| `training/common/contracts.py` | `training/contracts.py` |
| `training/common/preflight.py` | `training/preflight.py` |
| `training/common/prompts.py` | **新建**：从 `rollout.py` 抽出 `SYSTEM_PROMPT`、`TurnRecord`、`build_multiturn_prompt`（corpus export 与 SFT 装载都要用；`Trajectory` / `run_trajectory` 仍留在 `grpo.rollout`） |
| `training/corpus/index.py` | `training/corpus.py`（库逻辑） |
| `training/corpus/cli.py` | `training/corpus_cli.py` |
| `training/sft/data.py` | `training/data.py` |
| `training/sft/train.py` | `training/sft.py` |
| `training/grpo/config.py` | `training/config.py` |
| `training/grpo/train.py` | `training/multiturn_train.py` |
| `training/grpo/rollout.py` | `training/rollout.py`（去掉已抽到 common 的 prompt） |
| `training/grpo/reward_math.py` | `training/reward_math.py` |
| `training/grpo/starts.py` | `training/starts.py` |
| `training/grpo/diagnostics.py` | `training/diagnostics.py` |
| `training/grpo/trl_rollout.py` | `training/trl_rollout.py` |
| `training/grpo/evaluate.py` | `training/evaluate_multiturn_generalization.py` |

`run_step.py` 等根目录脚本若 import `training.goals`，改为 `training.common.goals`。

### 3. CLI entry points

| 命令 | 新入口 |
|------|--------|
| `qucs-sft` | `training.sft.train:main` |
| `qucs-multiturn` | `training.grpo.train:main` |
| `qucs-corpus` | `training.corpus.cli:main` |

`pyproject.toml` 的 `[tool.setuptools.packages.find] include = ["training*"]` 保持可发现子包即可。

### 4. Shared prompts extraction

今天的问题：`training/data.py`（SFT）和 `training/corpus.py` 都 `from training.rollout import …`，等于 SFT/corpus 绑在 GRPO 模块上。

改法：

1. 新建 `training.common.prompts`
2. 把 `SYSTEM_PROMPT`、`TurnRecord`、`build_multiturn_prompt` 放进去（`Trajectory` 与 `run_trajectory` 留在 `grpo.rollout`）
3. `grpo.rollout`、`corpus.index`、`sft.data` 都从 `common.prompts` import
4. 保证 export 出来的 SFT 样本与 GRPO 训练时看到的 prompt **仍然同构**（行为不变，只换存放位置）

### 5. Tests and docs

- 更新所有 `from training.…` 到新路径
- 测试文件按包重命名，例如：
  - `tests/test_common_*.py`
  - `tests/test_sft_*.py`
  - `tests/test_grpo_*.py`
  - `tests/test_corpus_*.py`（已有 `test_corpus.py` 可保留名或对齐）
- README / Colab 脚本：只改模块路径说明；命令名与用法不变
- 历史 plan/spec 里旧路径可不批量改写；本文件为准

### 6. Error handling and verification

- 搬家后跑现有相关 unittest（common / sft / grpo / corpus）全部通过
- 用一次 `uv run qucs-sft --help`、`qucs-multiturn --help`、`qucs-corpus --help` 确认入口
- 可选：用 `rg` 确认仓库 live 代码里不再出现旧模块路径（`training.multiturn_train`、`training.sft` 作为模块文件等）

## Out of scope for this change

- 把 reasoning 文件路径改到 `runs/<id>/think/`（已在并行清理中完成，不属于本拆包语义）
- 升级/更换 TRL 训练循环实现
- 删除或重写 BPF skills 相关代码

## Success criteria

1. `training/sft` 与 `training/grpo` 目录分离，互不 import
2. `training/corpus` 独立，只依赖 `common`
3. 对外三条 CLI 仍可用，行为与拆包前一致
4. 相关单测通过
5. README 用词区分：**SFT warm-up** vs **GRPO cold-start**；corpus 不写成 inference
