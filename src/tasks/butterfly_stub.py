from __future__ import annotations

from pathlib import Path

from intent import BOUNDS, INITIAL_GUESS, VARIABLES

TASK_NAME = "butterfly_stub"
TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "templates" / "butterfly_stub.sch.tpl"
EXPORT_LAYOUT = True

__all__ = [
    "BOUNDS",
    "EXPORT_LAYOUT",
    "INITIAL_GUESS",
    "TASK_NAME",
    "TEMPLATE_PATH",
    "VARIABLES",
]
