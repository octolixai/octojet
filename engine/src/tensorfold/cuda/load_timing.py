"""Load-phase timing behind ``OCTOJET_LOAD_TIMING=1``: where a checkpoint's start spends its seconds.

The loaders wrap their phases in ``PHASES.phase(name)``; disabled (the default) the context manager costs one
attribute read. ``report(total)`` prints every phase, its call count, and ``other`` (the total not covered by any
non-wrapper phase). Spec: docs/superpowers/specs/2026-09-29-f2-release-design.md section 3.2.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager

ENV = "OCTOJET_LOAD_TIMING"
WRAPPERS = ("layer",)        # phases that contain other phases: reported, not subtracted from ``other``


class Phases:
    def __init__(self, enabled: bool | None = None) -> None:
        self.enabled = (os.environ.get(ENV) == "1") if enabled is None else bool(enabled)
        self.seconds: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.layers: list[dict[str, float]] = []     # per ``layer`` wrapper: {"total": s, <inner phase>: s, ...}

    def reset(self) -> None:
        self.seconds.clear()
        self.counts.clear()
        self.layers.clear()

    @contextmanager
    def phase(self, name: str):
        if not self.enabled:
            yield
            return
        t = time.perf_counter()
        before = dict(self.seconds) if name in WRAPPERS else None
        try:
            yield
        finally:
            dt = time.perf_counter() - t
            self.seconds[name] = self.seconds.get(name, 0.0) + dt
            self.counts[name] = self.counts.get(name, 0) + 1
            if before is not None:
                inner = {k: v - before.get(k, 0.0) for k, v in self.seconds.items()
                         if k != name and v - before.get(k, 0.0) > 0.0}
                self.layers.append({"total": dt, **inner})

    def report(self, total: float) -> str:
        parts = [f"{k} {v:.1f} s ({self.counts.get(k, 0)})" for k, v in self.seconds.items()]
        covered = sum(v for k, v in self.seconds.items() if k not in WRAPPERS)
        parts.append(f"other {max(0.0, total - covered):.1f} s")
        parts.append(f"total {total:.1f} s")
        return "[octojet] load phases: " + ", ".join(parts)


PHASES = Phases()
