# Colab training smoke (post-Qucs success)

**Date:** 2026-09-10  
**Status:** Approved via user "ok go" after Qucs smoke succeeded

## Goal

A Colab notebook that reuses the proven Qucs AppImage + Xvfb path, then runs the smallest useful training checks:

1. `multiturn_train --dry-run` (real Qucs, no model)
2. Optional GPU micro-run: 1 step × 1 task × 2 generations × few turns

## Non-goals

- Full 60-step nightly replacement
- Drive sync / HF upload automation (later)
- Fixing Cursor embedded Colab kernel

## Design

- Runtime: **GPU required** for phase 2; phase 1 works on CPU
- Bootstrap: clone/pull repo → apt xvfb → AppImage extract → `apply_qucs_env(..., extract_root=...)`
- Python: use `uv` with Python 3.12 to match `pyproject.toml`
- Success criteria:
  - dry-run prints a JSON trajectory with finite `reward`
  - micro-run creates `outputs/.../train.log` containing `step_stats=` and at least one checkpoint or final_lora

## Files

- `notebooks/colab_train_smoke.ipynb`
- Optional: keep `notebooks/colab_qucs_smoke.ipynb` as Qucs-only
