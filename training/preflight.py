"""Runtime checks shared by the training commands."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def verify_runtime(*, require_cuda: bool = True) -> dict[str, str]:
    from qucs_sim import resolve_qucs_s, resolve_qucsator

    info = {
        "qucs_s": resolve_qucs_s(),
        "qucsator_rf": resolve_qucsator(),
    }
    if require_cuda:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available inside this uv environment")
        info["gpu"] = torch.cuda.get_device_name(0)
        info["cuda"] = torch.version.cuda or "unknown"
    return info

