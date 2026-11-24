"""Working set of a workload: the smallest cache budget beyond which the history hit rate stops growing.

ReCal builds its cache from the history only (no serving). The history requests are then looked up in the
built cache: the share answered from the main cache is the history hit rate at that budget.
"""

import tempfile
from pathlib import Path

import numpy as np

from recal import accounting
from recal.core import MAIN_HIT
from recal.params import Params
from recal.system import ReCal

Q1_SHARES = (0.1, 0.3, 1.0)
MIN_GAIN = 0.01


def geometric_sweep(low: float, high: float, factor: float = 2.0) -> list[float]:
    out, b = [], low
    while b < high * (1 + 1e-9):
        out.append(b)
        b *= factor
    return out


def history_hit_rate(backend, vectors, ks, params: Params, budget_bytes: float, workdir: Path) -> float:
    system = ReCal(backend, params, budget_bytes, workdir)
    system.ingest_history(vectors, ks)
    if system.built is None:
        return 0.0
    m = 0 if params.idealized else system.built.m
    hits = 0
    for vec, k in zip(vectors, ks):
        res = system.core.lookup(np.ascontiguousarray(vec, dtype=np.float32), int(k), system.lam, m, False)
        hits += res.path == MAIN_HIT
    return hits / len(vectors)


def working_set(budgets: list[float], rates: list[float], min_gain: float = MIN_GAIN) -> float:
    """The smallest budget after which every further step of the sweep adds less than min_gain."""
    for i in range(len(budgets)):
        if all(rates[j + 1] - rates[j] < min_gain for j in range(i, len(budgets) - 1)):
            return budgets[i]
    return budgets[-1]


def sweep_rates(backend, vectors, ks, params: Params, n_vectors: int, dim: int, budgets: list[float],
                coord_bytes: float) -> list[float]:
    rates = []
    for b in budgets:
        with tempfile.TemporaryDirectory(prefix="working_set_") as work:
            rates.append(history_hit_rate(backend, vectors, ks, params,
                                          accounting.budget_bytes(n_vectors, dim, b, coord_bytes), Path(work)))
    return rates


def budgets_file(budgets: list[float], rates: list[float], w: float) -> dict:
    """Budgets as fractions of the dataset's stored bytes."""
    return {"unit": "fraction of stored bytes", "sweep": [{"budget": b, "history_hit_rate": r}
                                                          for b, r in zip(budgets, rates)],
            "working_set_fraction": w, "q1_budgets": [s * w for s in Q1_SHARES], "standin_5pct": 0.3 * w}
