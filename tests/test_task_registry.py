# tests/test_task_registry.py
import unittest
from task_registry import clear_plugins_for_tests, get_plugin, get_task, register_plugin
from tasks import TaskConfig
from pathlib import Path

class _Dummy:
    name = "dummy_task"
    @property
    def config(self):
        return TaskConfig(
            name="dummy_task",
            variables=("x",),
            bounds={"x": (0.0, 1.0)},
            initial_guess={"x": 0.5},
            template_path=Path("."),
            export_layout=False,
        )
    def default_goal(self): return {"g": 1}
    def goal_from_state(self, raw): return raw or self.default_goal()
    def validate_goal(self, goal): return None
    def evaluate(self, sim_result, goal): return {"goal_met": False}
    def simulate(self, params, workdir, goal): raise NotImplementedError
    def format_report(self, entry, goal): return "r"
    def format_observation(self, entry, goal, *, include_skills_system=True): return "o"
    def param_unit(self, var): return ""

class TestTaskRegistry(unittest.TestCase):
    def setUp(self):
        clear_plugins_for_tests()
    def test_register_and_get(self):
        register_plugin(_Dummy())
        self.assertEqual(get_plugin("dummy_task").name, "dummy_task")
        self.assertEqual(get_task("dummy_task").name, "dummy_task")
    def test_unknown(self):
        with self.assertRaises(ValueError):
            get_task("nope")
