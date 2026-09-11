"""Generic task plugin registry (no jobs dependency)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from tasks import TaskConfig

_plugins: dict[str, TaskPlugin] = {}
_builtin_configs: dict[str, TaskConfig] = {}


@runtime_checkable
class TaskPlugin(Protocol):
    name: str

    @property
    def config(self) -> TaskConfig: ...

    def default_goal(self) -> dict: ...

    def goal_from_state(self, raw: dict | None) -> Any: ...

    def validate_goal(self, goal) -> None: ...

    def evaluate(self, sim_result, goal) -> dict: ...

    def simulate(self, params, workdir, goal): ...

    def format_report(self, entry, goal) -> str: ...

    def format_observation(self, entry, goal, *, include_skills_system: bool) -> str: ...

    def param_unit(self, var: str) -> str: ...


def register_plugin(plugin: TaskPlugin) -> None:
    _plugins[plugin.name] = plugin
    _builtin_configs.pop(plugin.name, None)


def register_builtin_config(config: TaskConfig) -> None:
    """Register a config-only builtin task (no full TaskPlugin)."""
    if config.name in _plugins:
        # Full plugin already registered; leave it alone.
        return
    _builtin_configs[config.name] = config


def get_plugin(name: str) -> TaskPlugin:
    try:
        return _plugins[name]
    except KeyError as exc:
        raise ValueError(f"unknown task plugin: {name!r}") from exc


def get_task(name: str) -> TaskConfig:
    if name in _plugins:
        return _plugins[name].config
    if name in _builtin_configs:
        return _builtin_configs[name]
    raise ValueError(f"unknown task: {name!r}")


def clear_plugins_for_tests() -> None:
    _plugins.clear()
    _builtin_configs.clear()
