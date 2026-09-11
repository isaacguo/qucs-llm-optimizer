# jobs/bpf5_agent/progress.py
from __future__ import annotations

import json
from pathlib import Path

from jobs.bpf5_agent.buckets import N_BUCKETS, SUCCESS_PER_BUCKET


class ProgressStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def _default(self) -> dict:
        return {
            "version": 1,
            "success_per_bucket": SUCCESS_PER_BUCKET,
            "counts": [0] * N_BUCKETS,
        }

    def load(self) -> dict:
        if not self.path.is_file():
            data = self._default()
            self._write(data)
            return data
        with self.path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")

    def record_success(self, bucket: int) -> None:
        if not 0 <= bucket < N_BUCKETS:
            raise ValueError(f"bucket must be in 0..{N_BUCKETS - 1}, got {bucket}")
        data = self.load()
        data["counts"][bucket] = int(data["counts"][bucket]) + 1
        self._write(data)

    def is_complete(self) -> bool:
        data = self.load()
        target = int(data.get("success_per_bucket", SUCCESS_PER_BUCKET))
        counts = data["counts"]
        return all(int(c) >= target for c in counts)
