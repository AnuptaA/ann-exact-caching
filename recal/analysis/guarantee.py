"""The recall guarantee: recall of main-cache hits in windows of main-path requests."""

import numpy as np

from recal.records import MAIN, MINI


def guarantee_table(cols: dict, target: float, window: int = 5000) -> dict:
    """Consecutive windows of `window` main-path requests (mini hits excluded).

    A window with at least one main-cache hit has a recall of hits; the table gives
    the share of those windows below the target and the mean of recall minus target.
    """
    main_path = cols["path"] != MINI
    path = cols["path"][main_path]
    recall = cols["recall"][main_path].astype(np.float64)
    values = []
    for start in range(0, len(path) - window + 1, window):
        hits = path[start:start + window] == MAIN
        if hits.any():
            values.append(recall[start:start + window][hits].mean())
    values = np.asarray(values)
    return {
        "window": window,
        "n_windows": len(values),
        "recall_of_hits": values,
        "share_below_target": float((values < target).mean()) if len(values) else float("nan"),
        "mean_margin": float((values - target).mean()) if len(values) else float("nan"),
        "min_margin": float((values - target).min()) if len(values) else float("nan"),
    }
