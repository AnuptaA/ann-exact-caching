"""Small builders shared by the ReCal tests."""

import numpy as np

from recal.core import Core

ID_BASE = 1000


def exact_order(data: np.ndarray, point: np.ndarray) -> np.ndarray:
    dist = np.linalg.norm(data.astype(np.float64) - point.astype(np.float64), axis=1)
    return np.argsort(dist, kind="stable")


def build_core(data, centers, K, js, k_max=100, budget=0.0, ef_search=200, model=None):
    """Entries around `centers` with response lists of j*K exact neighbors and shells of ranks
    jK+1..(j+2)K (the core drops shell vectors that are in the data pool)."""
    ids = np.arange(len(data), dtype=np.int64) + ID_BASE
    core = Core(data.shape[1], k_max)
    core.set_quantizer(data.min(axis=0), data.max(axis=0))
    core.set_budget(budget)
    lists = []
    for center, j in zip(centers, js):
        order = exact_order(data, center)
        top = order[: j * K]
        core.add_entry(center, K, int(j), ids[top], data[top])
        lists.append(order[: (j + 2) * K])
    for e, (j, order) in enumerate(zip(js, lists)):
        shell = order[j * K:]
        core.add_shell(e, ids[shell], data[shell])
    core.finish_build(16, 200, ef_search, 100)
    if model is not None:
        core.set_model(*model)
    return core, ids, lists


def pool_ids(core: Core, K, js, lists):
    ids = set()
    for j, order in zip(js, lists):
        ids.update((order[: j * K] + ID_BASE).tolist())
    return ids
