"""ReCal's hit rate at a given system recall, interpolated along the T curve."""

import numpy as np


def hit_rate_at_recall(points: list[tuple[float, float]], system_recall: float) -> float:
    """points: (system recall, hit rate) of ReCal runs at several targets T.

    Linear interpolation between the neighboring runs; NaN outside their range.
    """
    pts = sorted(set(points))
    if len(pts) < 2:
        return float("nan")
    recalls, hits = zip(*pts)
    if not recalls[0] <= system_recall <= recalls[-1]:
        return float("nan")
    return float(np.interp(system_recall, recalls, hits))
