"""Locate Qucs-S / qucsator_rf binaries inside an extracted AppImage tree."""
from __future__ import annotations

import os
import stat
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


def apply_qucs_env(binaries: dict[str, Path]) -> dict[str, str]:
    """Set process env for this repo's resolvers; return the mapping used."""
    mapping = {key: str(path.resolve()) for key, path in binaries.items()}
    os.environ.update(mapping)
    # Headless Colab / CI: avoid Qt trying to open a display for qucs-s -n.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return mapping
