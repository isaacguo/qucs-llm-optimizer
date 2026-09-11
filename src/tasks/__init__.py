from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskConfig:
    name: str
    variables: tuple[str, ...]
    bounds: dict
    initial_guess: dict
    template_path: Path
    export_layout: bool
    sweep_points: int = 181
    step_mode: str = "range"


def get_task(name: str) -> TaskConfig:
    if name == "butterfly_stub":
        from tasks import butterfly_stub as m
    elif name == "butterworth_bpf5":
        from tasks import butterworth_bpf5 as m
    else:
        raise ValueError(f"unknown task: {name!r}")
    return TaskConfig(
        name=m.TASK_NAME,
        variables=m.VARIABLES,
        bounds=m.BOUNDS,
        initial_guess=dict(m.INITIAL_GUESS),
        template_path=m.TEMPLATE_PATH,
        export_layout=m.EXPORT_LAYOUT,
        sweep_points=getattr(m, "SWEEP_POINTS", 181),
        step_mode=getattr(m, "STEP_MODE", "range"),
    )
