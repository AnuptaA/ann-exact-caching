"""Per-stream statistics, including empirical against theoretical Zipf."""

import numpy as np

from workloads.q1.streams import zipf_probabilities
from workloads.q1.universe import Universe

REPORT_RANKS = (1, 2, 5, 10, 50, 100, 500, 1000)


def stream_stats(uni: Universe, stream: np.ndarray, s: float | None = None) -> dict:
    """Distinct count, repeat share, top group share and Zipf comparison."""
    stream = np.asarray(stream)
    n = len(stream)
    distinct = len(np.unique(stream))
    out = {"requests": n, "distinct": distinct,
           "repeat_share": 0.0 if n == 0 else (n - distinct) / n}
    in_members = stream < uni.n_members
    if n and in_members.all():
        counts = np.bincount(stream // uni.group_size, minlength=uni.n_groups)
        freq = np.sort(counts)[::-1] / n
        out["top_group_share"] = float(freq[0])
        if s is not None:
            theory = zipf_probabilities(uni.n_groups, s)
            out["zipf"] = [{"rank": r, "empirical": float(freq[r - 1]),
                            "theoretical": float(theory[r - 1])}
                           for r in REPORT_RANKS if r <= uni.n_groups]
    return out


def format_stats(name: str, stats: dict) -> str:
    lines = [f"{name}: requests {stats['requests']}, distinct {stats['distinct']}, "
             f"exact-repeat share {stats['repeat_share']:.4f}"]
    if "top_group_share" in stats:
        lines.append(f"  top group share {stats['top_group_share']:.4f}")
    for row in stats.get("zipf", []):
        lines.append(f"  rank {row['rank']:>5}: empirical {row['empirical']:.5f} "
                     f"theoretical {row['theoretical']:.5f}")
    return "\n".join(lines)
