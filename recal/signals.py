"""Rebuild signals. All counters count main-path requests only."""

from collections import deque

import numpy as np


class Signals:
    def __init__(self, p, updater=None):
        self.p = p
        self.updater = updater
        self.events: list = []
        self._reset(0)

    def _reset(self, t: int) -> None:
        p = self.p
        self.window = max(p.drop_floor, int(p.drop_window * p.H))
        self.base: list = []
        self.recent: deque = deque(maxlen=self.window)
        self.next_check = max(int(p.drop_from * p.H), 2 * self.window)
        self.pool_at_build = None

    def after_construction(self, system) -> None:
        self._reset(system.t)
        self.pool_at_build = system.core.n_data_slots()
        if self.updater is not None:
            self.updater.after_construction()

    # ---- the signals -----------------------------------------------------------------------------

    def after_request(self, system, score: float | None) -> None:
        if system.built is None:
            return
        since = system.t - system.last_construction
        if score is not None:
            if len(self.base) < self.window:
                self.base.append(score)
            self.recent.append(score)
        fired = self.fired(system, since)
        if fired and since >= self.p.min_interval * self.p.H:
            self.events.append({"t": system.t, "since": since, "signal": fired})
            system.rebuild_requested = fired

    def fired(self, system, since: int) -> str | None:
        p = self.p
        if p.no_rebuild_signals:
            return "R4 periodic refresh" if since >= p.H else None
        if self.pool_at_build and system.core.n_data_slots() < p.shrink * self.pool_at_build:
            return "R2 shrinking pool"
        if self.updater is not None and self.updater.frozen >= p.frozen * max(system.core.n_data_slots(), 1):
            return "R3 frozen inserts"
        if since >= p.H:
            return "R4 periodic refresh"
        if self.score_drop(since):
            return "R5 score drop"
        full = len(system.status.flags) >= system.status.span
        if full and system.status.share() >= p.outage:
            return "R6 outage"
        return None

    def score_drop(self, since: int) -> bool:
        """Mean score of the last window is 3 standard errors below the mean of the first window."""
        if since < self.next_check or len(self.base) < self.window or len(self.recent) < self.window:
            return False
        self.next_check = since + self.p.drop_every
        base, recent = np.asarray(self.base), np.asarray(self.recent)
        se = np.sqrt(base.var(ddof=1) / len(base) + recent.var(ddof=1) / len(recent))
        return bool(recent.mean() < base.mean() - self.p.drop_se * max(se, 1e-12))
