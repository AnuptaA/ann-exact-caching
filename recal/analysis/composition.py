"""Hit composition: mini cache, one main-cache entry, several entries."""

import numpy as np

from recal.records import MAIN, MINI


def hit_composition(cols: dict) -> dict:
    path, contrib = cols["path"], cols["n_contrib"]
    mini = int((path == MINI).sum())
    main = path == MAIN
    known = main & (contrib >= 1)
    one = int((known & (contrib == 1)).sum())
    several = int((known & (contrib > 1)).sum())
    unknown = int(main.sum()) - one - several
    total = mini + one + several + unknown
    share = lambda x: float(x / total) if total else float("nan")  # noqa: E731
    return {"hits": total, "mini": share(mini), "one_entry": share(one),
            "several_entries": share(several), "unknown": share(unknown),
            "mean_entries": float(np.mean(contrib[known])) if known.any() else float("nan")}
