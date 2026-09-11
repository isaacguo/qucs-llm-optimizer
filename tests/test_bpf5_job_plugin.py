# tests/test_bpf5_job_plugin.py
import importlib
import unittest
from task_registry import clear_plugins_for_tests, get_task


class TestBpf5JobPlugin(unittest.TestCase):
    def setUp(self):
        clear_plugins_for_tests()
        # 重新注册 butterfly 若需要
        from tasks import ensure_builtin_tasks

        ensure_builtin_tasks()
        importlib.import_module("jobs.bpf5_agent.register").register()

    def test_task_config(self):
        t = get_task("butterworth_bpf5")
        self.assertEqual(t.name, "butterworth_bpf5")
        self.assertTrue(t.template_path.is_file())
        self.assertEqual(len(t.variables), 10)
