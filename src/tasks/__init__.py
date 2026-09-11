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


def _config_from_module(m) -> TaskConfig:
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


def _register_builtins() -> None:
    from task_registry import register_builtin_config
    from tasks import butterfly_stub as butterfly_stub_m

    register_builtin_config(_config_from_module(butterfly_stub_m))


def ensure_builtin_tasks() -> None:
    """Re-register config-only builtins (e.g. after clear_plugins_for_tests)."""
    _register_builtins()


def get_task(name: str) -> TaskConfig:
    from task_registry import get_task as _get_task

    try:
        return _get_task(name)
    except ValueError:
        # Registry unit tests may call clear_plugins_for_tests(); restore builtins.
        _register_builtins()
        return _get_task(name)


_register_builtins()
