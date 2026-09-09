#!/usr/bin/env python3
"""Download/extract the Qucs-S Linux AppImage and print env exports.

Usage (local or Colab):
  python scripts/setup_qucs_appimage.py --dir /content/qucs-s
"""
from __future__ import annotations

import argparse
import os
import stat
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.qucs_appimage import apply_qucs_env, find_qucs_binaries

DEFAULT_URL = (
    "https://github.com/ra3xdh/qucs_s/releases/download/26.1.1/"
    "Qucs-S-26.1.1-linux-x86_64.AppImage"
)


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"reusing cached AppImage: {dest}")
        return dest
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, dest)
    dest.chmod(dest.stat().st_mode | stat.S_IXUSR)
    return dest


def extract_appimage(appimage: Path, extract_dir: Path) -> Path:
    squash = extract_dir / "squashfs-root"
    if squash.exists() and any(squash.rglob("qucsator_rf")):
        print(f"reusing extract: {squash}")
        return squash
    extract_dir.mkdir(parents=True, exist_ok=True)
    # Colab / many servers lack FUSE; --appimage-extract unpacks without mounting.
    print(f"extracting {appimage} -> {extract_dir}")
    subprocess.run(
        [str(appimage), "--appimage-extract"],
        cwd=extract_dir,
        check=True,
    )
    if not squash.exists():
        raise RuntimeError(f"expected {squash} after --appimage-extract")
    return squash


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="tmp/qucs-s-appimage", help="work directory")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="only locate binaries under --dir/squashfs-root",
    )
    args = parser.parse_args(argv)
    work = Path(args.dir).resolve()
    if args.skip_download:
        squash = work / "squashfs-root"
    else:
        appimage = download(args.url, work / Path(args.url).name)
        squash = extract_appimage(appimage, work)
    binaries = find_qucs_binaries(squash)
    mapping = apply_qucs_env(binaries)
    for key, value in mapping.items():
        print(f"export {key}={value}")
    # Smoke: binaries respond to -h / --help or at least exist and execute.
    for key, path in binaries.items():
        print(f"ok {key}: {path} exists={path.exists()} exec={os.access(path, os.X_OK)}")


if __name__ == "__main__":
    main()
