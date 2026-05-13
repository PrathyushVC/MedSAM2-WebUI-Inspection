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
    """Configure the root logger for distributed or single-GPU training.

    Rank-0 logs at ``level_primary`` and writes to a ``train.log`` file if
    ``log_dir`` is given. All other ranks log at ``level_secondary`` to stdout
    only.

    Args:
        log_dir: Directory to write ``train.log``; only used for rank 0.
        rank: Global process rank.
        level_primary: Log level string for rank 0 (e.g. ``"INFO"``).
        level_secondary: Log level string for all other ranks.
    """
    level = logging.getLevelName(level_primary if rank == 0 else level_secondary)
    handlers: list = [logging.StreamHandler(sys.stdout)]

    if rank == 0 and log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(os.path.join(log_dir, "train.log")))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [rank%(process)d] %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


class _SmoothedValue:
    """Running windowed average of a scalar metric.

    Args:
        window: Number of recent values to use for the smoothed average.
    """

    def __init__(self, window: int = 20):
        self._window = window
        self._values: list = []
        self._total = 0.0
        self._count = 0

    def update(self, value: float):
        """Add a new observation.

        Args:
            value: Scalar value to record.
        """
        self._values.append(value)
        if len(self._values) > self._window:
            self._values.pop(0)
        self._total += value
        self._count += 1

    @property
    def avg(self) -> float:
        """Running mean over all observations."""
        return self._total / max(self._count, 1)

    @property
    def smoothed(self) -> float:
        """Mean over the most recent ``window`` observations."""
        return sum(self._values) / max(len(self._values), 1)

    @property
    def latest(self) -> float:
        """Most recently recorded value."""
        return self._values[-1] if self._values else 0.0


class MetricLogger:
    """Accumulates per-step scalar metrics and provides epoch-level summaries.

    Example:
        >>> logger = MetricLogger()
        >>> for batch in loader:
        ...     logger.update(loss=loss.item(), lr=lr)
        >>> print(logger.summary())
        {'loss': 0.423, 'lr': 1e-4}

    Args:
        delimiter: String used to join metric strings in ``__str__``.
    """

    def __init__(self, delimiter: str = "  "):
        self._meters: Dict[str, _SmoothedValue] = defaultdict(lambda: _SmoothedValue())
        self.delimiter = delimiter
        self._start = time.time()

    def update(self, **kwargs: float):
        """Record one step's worth of scalar metrics.

        Args:
            **kwargs: Metric name to float value pairs.
        """
        for k, v in kwargs.items():
            self._meters[k].update(float(v))

    def summary(self) -> Dict[str, float]:
        """Return a dict of running-mean values for all tracked metrics.

        Returns:
            Dict mapping metric names to their epoch-averaged values.
        """
        return {k: m.avg for k, m in self._meters.items()}

    def __str__(self) -> str:
        parts = [f"{k}: {m.smoothed:.4f}" for k, m in self._meters.items()]
        return self.delimiter.join(parts)

    def elapsed(self) -> float:
        """Seconds elapsed since this logger was created."""
        return time.time() - self._start

    def reset(self):
        """Clear all accumulated metrics and reset the timer."""
        self._meters.clear()
        self._start = time.time()
