"""Register butterworth_bpf5 as a full TaskPlugin."""

from __future__ import annotations

from jobs.bpf5_agent.plugin import Bpf5Plugin
from task_registry import register_plugin


def register() -> None:
    register_plugin(Bpf5Plugin())
