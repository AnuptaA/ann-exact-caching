import numpy as np
import pytest

from workloads.q1 import shifts
from workloads.q1.universe import Universe


def small_universe(groups=64, size=5):
    return Universe(np.arange(groups), np.zeros((groups, size), dtype=np.int64),
                    np.zeros((groups, size)), n_official=groups)


def clustered_points(k=8, per=30, d=4, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=10.0, size=(k, d))
    return np.vstack([c + rng.normal(size=(per, d)) for c in centers]).astype(np.float32), centers


def test_kmeans_recovers_separated_clusters_and_assigns_nearest():
    x, centers = clustered_points()
    cent = shifts.kmeans_centroids(x, k=8, seed=0)
    labels = shifts.assign(x, cent)
    assert len(np.unique(labels)) == 8
    for c in range(8):
        assert len(np.unique(labels[c * 30:(c + 1) * 30])) == 1
    assert np.array_equal(labels, shifts.assign(x, shifts.kmeans_centroids(x, k=8, seed=0)))


def test_cluster_sets_are_disjoint_and_fixed():
    old, new = shifts.cluster_sets(16)
    assert len(old) == len(new) == 8 and not set(old) & set(new)
    assert np.array_equal(old, shifts.cluster_sets(16)[0])


def test_probability_schedules():
    ab = shifts.new_set_probability(10, "abrupt", 4)
    assert ab.tolist() == [0] * 4 + [1] * 6
    gr = shifts.new_set_probability(10, "gradual", 2, 4)
    assert gr[2] == 0 and gr[4] == 0.5 and gr[6:].tolist() == [1.0] * 4
    with pytest.raises(ValueError):
        shifts.new_set_probability(10, "gradual", 2, 0)


def stream(kind, **kw):
    uni = small_universe()
    gc = np.arange(uni.n_groups) % 8
    return uni, gc, shifts.shift_stream(uni, gc, kind, 4000, start=1000, span=kw.get("span", 0), n_clusters=8, seed=1)


def test_abrupt_shift_swaps_the_popular_clusters():
    uni, gc, out = stream("abrupt")
    old, new = shifts.cluster_sets(8)
    cluster = gc[out["ids"] // uni.group_size]
    assert np.isin(cluster[:1000], old).all() and np.isin(cluster[1000:], new).all()
    assert out["phase"][:1000].max() == 0 and out["phase"][1000] == 1 and out["phase"][-1] == 2


def test_gradual_shift_moves_traffic_over_the_span():
    uni, gc, out = stream("gradual", span=2000)
    _, new = shifts.cluster_sets(8)
    share = np.isin(gc[out["ids"] // uni.group_size], new).astype(float)
    assert share[:1000].mean() == 0.0 and share[3000:].mean() == 1.0
    assert 0.3 < share[1800:2200].mean() < 0.7


def test_correlated_inserts_are_new_cluster_held_out_vectors_over_the_span():
    vc = np.arange(400) % 8
    held = np.arange(300, 400)
    ins = shifts.correlated_inserts(vc, held, start=1000, span=2000, n_inserts=20, n_clusters=8)
    _, new = shifts.cluster_sets(8)
    assert np.isin(vc[ins["id"]], new).all() and np.isin(ins["id"], held).all()
    assert ins["t"].min() >= 1000 and ins["t"].max() < 3000 and len(set(ins["id"])) == 20
    with pytest.raises(ValueError):
        shifts.correlated_inserts(vc, held, 0, 10, 1000, n_clusters=8)
