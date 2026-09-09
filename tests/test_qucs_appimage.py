"""Tests for AppImage binary discovery (no network)."""
from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.qucs_appimage import (
    apply_qucs_env,
    find_named_binary,
    find_qt_plugins_dir,
    find_qucs_binaries,
    prepare_appimage_runtime,
)


def _touch_exec(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class QucsAppImageFindTests(unittest.TestCase):
    def test_prefers_bin_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            decoy = root / "share" / "qucs-s"
            preferred = root / "usr" / "bin" / "qucs-s"
            _touch_exec(decoy)
            _touch_exec(preferred)
            self.assertEqual(find_named_binary(root, "qucs-s"), preferred)

    def test_find_qucs_binaries_requires_both(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _touch_exec(root / "usr" / "bin" / "qucs-s")
            with self.assertRaises(FileNotFoundError):
                find_qucs_binaries(root)
            _touch_exec(root / "usr" / "bin" / "qucsator_rf")
            found = find_qucs_binaries(root)
            self.assertEqual(found["QUCS_S"].name, "qucs-s")
            self.assertEqual(found["QUCSATOR_RF"].name, "qucsator_rf")

    def test_find_qt_plugins_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plugin = root / "usr" / "plugins" / "platforms" / "libqxcb.so"
            plugin.parent.mkdir(parents=True)
            plugin.write_text("")
            self.assertEqual(find_qt_plugins_dir(root), root / "usr" / "plugins")

    def test_prepare_appimage_runtime_uses_xcb_not_offscreen(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "usr" / "lib").mkdir(parents=True)
            plugin = root / "usr" / "plugins" / "platforms" / "libqxcb.so"
            plugin.parent.mkdir(parents=True)
            plugin.write_text("")
            with mock.patch(
                "scripts.qucs_appimage.ensure_virtual_display", return_value=":99"
            ):
                updates = prepare_appimage_runtime(root)
            self.assertEqual(updates["QT_QPA_PLATFORM"], "xcb")
            self.assertEqual(updates["DISPLAY"], ":99")
            self.assertIn(str(root / "usr" / "lib"), updates["LD_LIBRARY_PATH"])

    def test_apply_qucs_env_sets_process_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            qs = root / "bin" / "qucs-s"
            qa = root / "bin" / "qucsator_rf"
            _touch_exec(qs)
            _touch_exec(qa)
            mapping = apply_qucs_env({"QUCS_S": qs, "QUCSATOR_RF": qa})
            self.assertEqual(os.environ["QUCS_S"], mapping["QUCS_S"])
            self.assertEqual(os.environ["QUCSATOR_RF"], mapping["QUCSATOR_RF"])


if __name__ == "__main__":
    unittest.main()
