"""Regression tests for the M6 fixes in the advisor's cache core (cachecore.cpp)."""
import numpy as np
import pytest

cc = pytest.importorskip("simulations.approximate_hits.cachecore")

D, K = 16, 10


def small_core(graph: bool):
    rng = np.random.default_rng(0)
    db = rng.normal(size=(800, D)).astype(np.float32)
    q = rng.normal(size=(60, D)).astype(np.float32)
    core = cc.Core(D, K, 3, graph=graph)
    core.set_model([0.0] * 7, [1.0] * 7, [0.0] * 7, 0.0)
    for c in range(8):
        d2 = ((db - q[c]) ** 2).sum(axis=1)
        order = np.argsort(d2)
        b, s = order[:30], order[30:60]
        core.add_entry(c, q[c], float(np.sqrt(d2[b[-1]])), b.astype(np.int64), db[b], s.astype(np.int64), db[s])
    return core, q


def test_graph_search_on_a_core_without_graph_raises_instead_of_crashing():
    core, q = small_core(graph=False)
    with pytest.raises(ValueError):
        core.lookup_batch(q[:8], 0.0, 0, 64)      # ef2 > 0 with the graph off used to dereference a null index
    with pytest.raises(ValueError):
        core.lookup_batch(q[:8], 0.0, 64, 0)
    hit, *_ = core.lookup_batch(q[:8], 0.0, 0, 0)   # exact batch lookups still work
    assert len(hit) == 8
