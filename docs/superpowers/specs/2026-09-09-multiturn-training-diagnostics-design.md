# Multiturn training diagnostics, start filter, and reward shaping

**Date:** 2026-09-09  
**Status:** Approved (user confirmed; includes agent-path parity)

## Goal

Make multiturn GRPO runs diagnosable from logs, reduce noisy / untrainable starts, soften reward right-skew, reward correct early stops, and ensure the human/agent `run_step.py` loop can emit the same decision records (prompt + reasoning + intent).

## Non-goals

- One-step GRPO warm-up wired into multiturn (deferred)
- Offline `summarize_multiturn_log.py` script (deferred)
- Penalty for “destroy best and refuse to stop” (deferred; stop bonus only)

## Design

### Shared decision log

New module `src/decision_log.py` appends one JSON object per decision to a JSONL file.

Required fields (both training and agent):

| Field | Meaning |
|-------|---------|
| `source` | `"multiturn"` or `"agent"` |
| `timestamp` | UTC ISO-8601 |
| `prompt` | What the strategy saw (chat messages list or report text) |
| `completion` | Full model/agent output including reasoning + intent |
| `valid` | Whether intent parsed / was applied |
| `intent` | Parsed intent dict or null |
| `stopped` | Whether this turn was an explicit stop |
| `db_before` / `db_after` | \|S21\| dB at target before/after (agent: prev vs new) |
| `run` / `step` / `task` / `gen` / `turn` / `iteration` | Identity (nullable by source) |
| `seed` / `target_freq_hz` / `terminated_reason` | Optional context |

**Multiturn:** `outputs/<run>/completions.jsonl` — one line per turn.  
**Agent:** `runs/<run>/completions.jsonl` — one line per `run_step.py step` (and init if useful).  
`train.log` stays short; full text never dumped into the human-facing step summary.

### Multiturn `train.log` step_stats

After each optimizer step, append one line:

```text
step_stats={...json...}
```

Including: reward mean/p25/p50/p75, within-group std/range means, group_best_mean, valid_turn_rate, pos_delta_rate, best_improve mean/p50, termination reason percentages, zero_adv_frac, n_resampled_starts.

### Start headroom filter

Before rolling out a GRPO group, sample params and simulate. Reject starts where:

`initial_db <= target_depth_db + min_start_headroom_db`

Default `min_start_headroom_db=5`. Retry up to 20 times with advancing seeds; if still failing, keep last sample and log a warning. Count resamples in `n_resampled_starts`.

### Defaults

- `--patience` default **5** (was 3)
- `--max-turns` default **15** (was 8)
- `--min-start-headroom-db` default **5**

### Reward shaping

- `REWARD_CLIP` default **20** (was 40)
- If trajectory ends with legal `stop` and `best_db < initial_db - 0.2`, add **+1.0** to mixed terminal reward, then clip to `[-clip, clip]`

### Agent parity

`run_step.py step` (and `init` with thinking when present) appends the same schema via `decision_log.append_record`, with:

- `prompt` = observation `report_text` (or init banner)
- `completion` = `<reasoning>…</reasoning>\n<intent>…</intent>` built from `--thinking` / intent JSON
- `source` = `"agent"`

No change to HTML report or `state.json` schema beyond the extra JSONL file.

## Testing

Unit tests for: headroom predicate/resample, clip 20, stop bonus on/off, decision_log append, step_stats keys, run_step writing completions.jsonl (temp dir).
