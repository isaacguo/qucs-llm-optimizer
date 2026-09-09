"""Locate Qucs-S / qucsator_rf binaries inside an extracted AppImage tree."""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import time
from pathlib import Path


def _is_executable(path: Path) -> bool:
    try:
        mode = path.stat().st_mode
    except OSError:
        return False
    return path.is_file() and bool(mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))


def find_named_binary(root: Path, name: str) -> Path | None:
    """Return the first executable named ``name`` under ``root`` (depth-first)."""
    matches: list[Path] = []
    for path in root.rglob(name):
        if _is_executable(path):
            matches.append(path)
    if not matches:
        return None
    # Prefer .../bin/<name> over random copies.
    matches.sort(key=lambda p: (0 if p.parent.name == "bin" else 1, len(p.parts), str(p)))
    return matches[0]


def find_qucs_binaries(extract_root: Path) -> dict[str, Path]:
    """Find ``qucs-s`` and ``qucsator_rf`` under an AppImage extract directory."""
    root = Path(extract_root)
    qucs_s = find_named_binary(root, "qucs-s")
    qucsator = find_named_binary(root, "qucsator_rf")
    missing = [n for n, p in (("qucs-s", qucs_s), ("qucsator_rf", qucsator)) if p is None]
    if missing:
        raise FileNotFoundError(
            f"missing binaries under {root}: {', '.join(missing)}"
        )
    assert qucs_s is not None and qucsator is not None
    return {"QUCS_S": qucs_s, "QUCSATOR_RF": qucsator}


def find_qt_plugins_dir(extract_root: Path) -> Path | None:
    """Return the Qt plugins directory that contains platforms/libqxcb.so if present."""
    root = Path(extract_root)
    for path in root.rglob("libqxcb.so"):
        # .../plugins/platforms/libqxcb.so -> plugins
        if path.parent.name == "platforms":
            return path.parent.parent
    return None


def find_appimage_lib_dir(extract_root: Path) -> Path | None:
    root = Path(extract_root)
    for candidate in (root / "usr" / "lib", root / "lib"):
        if candidate.is_dir():
            return candidate
    return None


def ensure_virtual_display(display: str = ":99") -> str:
    """Start Xvfb when no usable DISPLAY exists (Colab / headless CI).

    The Qucs-S AppImage ships the ``xcb`` Qt platform plugin but not
    ``offscreen``, so headless hosts must provide a fake X server.
    """
    existing = os.environ.get("DISPLAY", "").strip()
    if existing:
        return existing
    if shutil.which("Xvfb") is None:
        raise RuntimeError(
            "Xvfb not found. On Colab/Debian run: apt-get install -y xvfb"
        )
    # Already started by a previous cell?
    lock = Path(f"/tmp/.X{display.lstrip(':')}-lock")
    if not lock.exists():
        subprocess.Popen(
            ["Xvfb", display, "-screen", "0", "1280x1024x24", "-ac"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.5)
    os.environ["DISPLAY"] = display
    return display


def prepare_appimage_runtime(extract_root: Path) -> dict[str, str]:
    """Configure env so extracted AppImage qucs-s can run headlessly."""
    root = Path(extract_root)
    updates: dict[str, str] = {}

    lib_dir = find_appimage_lib_dir(root)
    if lib_dir is not None:
        prev = os.environ.get("LD_LIBRARY_PATH", "")
        updates["LD_LIBRARY_PATH"] = (
            f"{lib_dir}:{prev}" if prev else str(lib_dir)
        )

    plugins = find_qt_plugins_dir(root)
    if plugins is not None:
        updates["QT_PLUGIN_PATH"] = str(plugins)
        updates["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(plugins / "platforms")

    # AppImage Qt typically only bundles xcb — do not force offscreen.
    updates["QT_QPA_PLATFORM"] = "xcb"
    display = ensure_virtual_display()
    updates["DISPLAY"] = display

    os.environ.update(updates)
    # Drop a stale offscreen preference if a previous cell set it.
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        os.environ["QT_QPA_PLATFORM"] = "xcb"
    return updates


def apply_qucs_env(
    binaries: dict[str, Path],
    *,
    extract_root: Path | None = None,
) -> dict[str, str]:
    """Set process env for this repo's resolvers; return the mapping used."""
    mapping = {key: str(path.resolve()) for key, path in binaries.items()}
    os.environ.update(mapping)
    if extract_root is not None:
        mapping.update(prepare_appimage_runtime(extract_root))
    return mapping
