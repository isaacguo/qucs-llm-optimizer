#!/usr/bin/env bash
# Reports current progress of the active multiturn GRPO run to Slack
# via an openclaw cron job (command payload, delivery=announce).
# Override RUN_DIR / TITLE with cron --command-env when the job changes.
# The LAST line printed to stdout becomes the delivered Slack message.
set -uo pipefail

RUN_DIR="${RUN_DIR:-/home/yue1guo2/workspace/qucs-llm-optimizer/outputs/multiturn-from-ckpt20}"
TITLE="${TITLE:-QUCS GRPO from-ckpt20 训练进展}"
LOG="$RUN_DIR/train.log"
PIDFILE="$RUN_DIR/train.pid"
NVIDIA_SMI="/usr/lib/wsl/lib/nvidia-smi"

TS="$(date '+%Y-%m-%d %H:%M:%S %Z')"

if [ ! -f "$LOG" ]; then
    echo "*${TITLE}*  ($TS)"
    echo "日志文件尚未生成: $LOG (训练可能还未启动或仍在加载模型)"
    exit 0
fi

# --- process status ---
STATUS="未知"
if [ -f "$PIDFILE" ]; then
    PID="$(cat "$PIDFILE" 2>/dev/null || echo "")"
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        STATUS="运行中 (pid $PID)"
    else
        STATUS="进程未运行 (pidfile 中的 pid $PID 已不存在)"
    fi
else
    STATUS="未找到 pidfile"
fi

# --- GPU memory ---
if [ -x "$NVIDIA_SMI" ]; then
    GPU_LINE="$("$NVIDIA_SMI" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)"
    MEM_USED="$(echo "$GPU_LINE" | cut -d, -f1 | tr -d ' ')"
    MEM_TOTAL="$(echo "$GPU_LINE" | cut -d, -f2 | tr -d ' ')"
    GPU_TXT="${MEM_USED:-?} / ${MEM_TOTAL:-?} MiB"
else
    GPU_TXT="n/a (nvidia-smi not found)"
fi

# --- latest completed step summary line ---
# format: [HH:MM:SS] step X/Y loss=... reward_mean=... reward_std=... turns_mean=... goal_met=A/B stop=C patience=D kl_mean=E
LAST_STEP_LINE="$(grep -E '^\[[0-9:]+\] step [0-9]+/[0-9]+ loss=' "$LOG" | tail -1)"

# --- latest step_stats json line (richer per-step aggregate) ---
LAST_STATS_LINE="$(grep 'step_stats=' "$LOG" | tail -1)"

# --- latest raw log line (to show liveness / current rollout position) ---
LAST_RAW_LINE="$(tail -1 "$LOG")"

MSG="*${TITLE}*  ($TS)
状态: $STATUS
GPU 显存: $GPU_TXT"

if [ -n "$LAST_STEP_LINE" ]; then
    MSG="$MSG
最近完成的 step: \`${LAST_STEP_LINE#*] }\`"
else
    MSG="$MSG
尚无已完成的 step（可能仍在第一步 rollout 中）"
fi

if [ -n "$LAST_STATS_LINE" ]; then
    STATS_JSON="${LAST_STATS_LINE#*step_stats=}"
    GOAL_MET_PCT="$(echo "$STATS_JSON" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(f"{d.get(\"goal_met_pct\",0):.1f}")' 2>/dev/null || echo "?")"
    MAX_TURNS_PCT="$(echo "$STATS_JSON" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(f"{d.get(\"max_turns_pct\",0):.1f}")' 2>/dev/null || echo "?")"
    N_TURNS="$(echo "$STATS_JSON" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("n_turns","?"))' 2>/dev/null || echo "?")"
    MSG="$MSG
最新 step_stats: goal_met=${GOAL_MET_PCT}% max_turns=${MAX_TURNS_PCT}% n_turns(反向传播总turn数)=${N_TURNS}"
fi

MSG="$MSG
最新日志行: \`${LAST_RAW_LINE#*] }\`"

echo "$MSG"
