"""Rate-limited progress lines with a rate and an ETA."""

import logging
import time
from typing import Callable

logger = logging.getLogger(__name__)


def format_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


class Progress:
    """Logs `label: done/total unit, rate, ETA` at most once per `interval` seconds.

    `done` may start above zero (a resumed run); the rate counts only work done since the start.
    """

    def __init__(self, label: str, total: float, unit: str = "items", done: float = 0.0,
                 interval: float = 30.0, log: Callable[[str], None] = logger.info,
                 clock: Callable[[], float] = time.monotonic):
        self.label, self.total, self.unit, self.interval = label, total, unit, interval
        self.log, self.clock = log, clock
        self.start_done, self.start_time = done, clock()
        self.last_logged = self.start_time
        self.done = done

    def line(self, done: float) -> str:
        elapsed = self.clock() - self.start_time
        rate = (done - self.start_done) / elapsed if elapsed > 0 else 0.0
        eta = f", ETA {format_duration((self.total - done) / rate)}" if rate > 0 else ""
        percent = 100.0 * done / self.total if self.total else 100.0
        return (f"{self.label}: {done:.0f}/{self.total:.0f} {self.unit} ({percent:.0f}%), "
                f"{rate:.1f} {self.unit}/s{eta}")

    def update(self, done: float, force: bool = False) -> str | None:
        """Log when the interval has passed (or `force`); returns the line that was logged."""
        self.done = done
        now = self.clock()
        if not force and now - self.last_logged < self.interval:
            return None
        self.last_logged = now
        text = self.line(done)
        self.log(text)
        return text
