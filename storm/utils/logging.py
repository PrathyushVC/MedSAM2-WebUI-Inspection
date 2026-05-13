"""
Lightweight metric logging utilities.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from collections import defaultdict
from typing import Dict, Optional


def setup_logging(
    log_dir: Optional[str] = None,
    rank: int = 0,
    level_primary: str = "INFO",
    level_secondary: str = "WARNING",
):
    """
    Configure root logger.

    Primary rank (0) logs at `level_primary` to stdout and to a file if
    `log_dir` is provided. All other ranks log at `level_secondary` to stdout.
    """
    level = logging.getLevelName(level_primary if rank == 0 else level_secondary)
    handlers: list = [logging.StreamHandler(sys.stdout)]

    if rank == 0 and log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, "train.log"))
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [rank%(process)d] %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


class _SmoothedValue:
    """Tracks a running deque-smoothed average of a scalar."""

    def __init__(self, window: int = 20):
        self._window = window
        self._values: list = []
        self._total = 0.0
        self._count = 0

    def update(self, value: float):
        self._values.append(value)
        if len(self._values) > self._window:
            self._values.pop(0)
        self._total += value
        self._count += 1

    @property
    def avg(self) -> float:
        return self._total / max(self._count, 1)

    @property
    def smoothed(self) -> float:
        return sum(self._values) / max(len(self._values), 1)

    @property
    def latest(self) -> float:
        return self._values[-1] if self._values else 0.0


class MetricLogger:
    """
    Accumulates per-step scalars and provides epoch-averaged summaries.

    Usage::

        logger = MetricLogger()
        for batch in loader:
            ...
            logger.update(loss=loss.item(), lr=lr)
        summary = logger.summary()
    """

    def __init__(self, delimiter: str = "  "):
        self._meters: Dict[str, _SmoothedValue] = defaultdict(lambda: _SmoothedValue())
        self.delimiter = delimiter
        self._start = time.time()

    def update(self, **kwargs: float):
        for k, v in kwargs.items():
            self._meters[k].update(float(v))

    def summary(self) -> Dict[str, float]:
        return {k: m.avg for k, m in self._meters.items()}

    def __str__(self) -> str:
        parts = [f"{k}: {m.smoothed:.4f}" for k, m in self._meters.items()]
        return self.delimiter.join(parts)

    def elapsed(self) -> float:
        return time.time() - self._start

    def reset(self):
        self._meters.clear()
        self._start = time.time()
