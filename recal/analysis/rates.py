"""Hit rates, system recall and recall of hits."""

import numpy as np

from recal.records import ABSTAIN, MAIN, MINI, MISS


def hit_mask(cols: dict) -> np.ndarray:
    return np.isin(cols["path"], (MINI, MAIN))


def rates(cols: dict) -> dict:
    """Shares of scored requests by path, system recall and recall of hits."""
    n = len(cols["path"])
    path, recall = cols["path"], cols["recall"].astype(np.float64)
    hits = hit_mask(cols)
    return {
        "n": n,
        "hit_rate": float(hits.mean()) if n else float("nan"),
        "mini_hit_rate": float((path == MINI).mean()) if n else float("nan"),
        "main_hit_rate": float((path == MAIN).mean()) if n else float("nan"),
        "miss_rate": float((path == MISS).mean()) if n else float("nan"),
        "abstain_rate": float((path == ABSTAIN).mean()) if n else float("nan"),
        "system_recall": float(recall.mean()) if n else float("nan"),
        "recall_of_hits": float(recall[hits].mean()) if hits.any() else float("nan"),
        "recall_of_main_hits": float(recall[path == MAIN].mean())
        if (path == MAIN).any() else float("nan"),
    }
