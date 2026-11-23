"""The oracle: admit main-path requests by true candidate recall while the mean stays >= T."""

import numpy as np

from recal.records import MINI


def oracle(cols: dict, target: float) -> dict:
    """Best possible admission over the main-path requests that had a candidate answer.

    Requests are admitted in order of decreasing true candidate recall, as many as keep
    the running mean of recall at or above the target. Mini cache hits count as hits of
    recall 1 on top, as in the run.
    """
    n = len(cols["path"])
    main_path = cols["path"] != MINI
    cand = cols["cand_recall"][main_path].astype(np.float64)
    cand = np.sort(cand[np.isfinite(cand)])[::-1]
    running = np.cumsum(cand) / np.arange(1, len(cand) + 1)
    admitted = int((running >= target).nonzero()[0].max() + 1) if (running >= target).any() else 0
    mini = int((~main_path).sum())
    return {
        "admitted": admitted,
        "main_path_requests": int(main_path.sum()),
        "main_hit_rate": admitted / n if n else float("nan"),
        "hit_rate": (admitted + mini) / n if n else float("nan"),
        "mean_recall_admitted": float(running[admitted - 1]) if admitted else float("nan"),
    }
