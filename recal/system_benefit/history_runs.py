"""h(s) and ReCal's background work at each split, from construction runs on the history only."""

import copy

import numpy as np

from recal import sim
from recal.records import Records


def history_run(args, budget_fraction: float, history_len: int, train_frac: float = 0.7) -> dict:
    """ReCal on the first history_len requests: build from the first part, serve the rest.

    Returns the hit rate on that rest and the run's background work (searches by kind, re-read vectors).
    """
    a = copy.copy(args)
    a.budget = budget_fraction
    a.max_requests = history_len
    a.history_frac = train_frac
    records = Records()
    out = sim.run(a, records)
    n = int(out["n_scored"])
    cols = records.arrays()
    hit = float(np.isin(cols["path"], (0, 1)).mean()) if n else 0.0
    return {"hit_rate": hit, "n_scored": n, "background": out["background"],
            "rereads": int(cols["rereads"].sum()), "within_M": out["within_M"]}


def split_runs(args, total_fraction: float, history_len: int, splits) -> dict:
    """{s: history_run} for s > 0; s = 0 has no ReCal (no hits, no background)."""
    runs = {}
    for s in splits:
        runs[s] = ({"hit_rate": 0.0, "n_scored": 0, "background": {}, "rereads": 0, "within_M": True}
                   if s == 0 else history_run(args, s * total_fraction, history_len))
    return runs
