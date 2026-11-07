"""Independent cross-checks: naive references against the production code and the C++ drivers."""

import numpy as np
import pytest

from harness import config
from harness.runner import manifest, replay, results
from harness.runner.prepare import prepare_dataset
from tests.harness.test_replay import DRIVERS, ExactBackend, needs_driver
from workloads.q1 import K
from workloads.q1.recall import recall_batch

TIE_RTOL = 1e-6


def naive_recall(query, answer, base, kth):
    """One loop, no vectorization: distinct valid ids at distance <= kth(1+tol), over the first K."""
    seen, good = set(), 0
    for vid in list(answer)[:K]:
        if vid < 0 or vid >= len(base) or vid in seen:
            continue
        seen.add(vid)
        dist = sum((float(a) - float(b)) ** 2 for a, b in zip(query, base[vid])) ** 0.5
        good += dist <= kth * (1 + TIE_RTOL)
    return good / K


def test_recall_matches_a_naive_loop_with_ties_short_lists_and_duplicates():
    rng = np.random.default_rng(5)
    base = rng.integers(-2, 3, size=(80, 3)).astype(np.float32)        # integer grid: many exact ties
    base[40:50] = base[0]                                              # duplicate vectors
    queries = rng.integers(-2, 3, size=(60, 3)).astype(np.float32)
    answers = rng.integers(-1, 81, size=(60, K))                       # -1 padding, 80 out of range
    answers[::7, 3] = answers[::7, 2]                                  # repeated ids
    answers[::5, 5:] = -1                                              # short lists
    exact = np.sort(np.linalg.norm(base[None] - queries[:, None], axis=2), axis=1)
    kth = exact[:, K - 1]
    got = recall_batch(queries, answers, base, kth)
    want = [naive_recall(q, a, base, k) for q, a, k in zip(queries, answers, kth)]
    assert np.allclose(got, want)


def naive_proximity(vectors, ids, thresh):
    """Hit when the id was served before, or a stored query lies within thresh (squared L2)."""
    stored, known, hits = [], set(), []
    for vec, uid in zip(vectors, ids):
        if uid in known:
            hits.append(True)
            continue
        near = any(float(((vec - s) ** 2).sum()) <= thresh for s in stored)
        if near:
            known.add(uid)
        else:
            stored.append(vec)
            known.add(uid)
        hits.append(near)
    return np.array(hits)


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    rng = np.random.default_rng(1)
    centers = rng.normal(size=(8, 6)).astype(np.float32) * 5
    base = centers[rng.integers(0, 8, 2000)] + rng.normal(size=(2000, 6)).astype(np.float32) * 0.5
    official = centers[rng.integers(0, 8, 40)] + rng.normal(size=(40, 6)).astype(np.float32) * 0.5
    root = tmp_path_factory.mktemp("xc")
    prepare_dataset(root, "toy", base, official, ExactBackend(base), ef_search=50, list_size=20,
                    n_groups=20, group_size=3, n_requests=400, zipf=(0.99,), no_locality=True)
    return root


@needs_driver
def test_proximity_driver_agrees_with_a_naive_reference_when_nothing_is_evicted(toy, tmp_path):
    ds = replay.DatasetDir(toy, "toy")
    rng = np.random.default_rng(2)
    distinct = ds.universe.pool_id(0) + rng.choice(40, size=12, replace=False)    # 12 queries x 10 = 120 <= pool
    stream = rng.choice(distinct, size=300).astype(np.int64)
    ds.streams["custom"] = stream
    cell = manifest.Cell("toy", "custom", "proximity", "p1", 10)
    header, arrays = results.read_result(replay.run_cell(cell, ds, DRIVERS, tmp_path / "r", tmp_path / "w"))
    assert header["settings"]["pool_size"] >= 10 * len(distinct)
    want = naive_proximity(ds.vectors_of(stream), stream, header["settings"]["global_thresh"])
    got = arrays["outcome"] != 0
    assert (got != want).sum() <= 3, f"{(got != want).sum()} of {len(got)} requests disagree"


@needs_driver
def test_no_locality_stream_has_no_equality_hits_and_exact_hits_score_one(toy, tmp_path):
    ds = replay.DatasetDir(toy, "toy")
    cell = manifest.Cell("toy", "no_locality", "aker", "paper", 10)
    header, arrays = results.read_result(replay.run_cell(cell, ds, DRIVERS, tmp_path / "r", tmp_path / "w"))
    assert header["equality_hits_in_window"] == 0 and not arrays["equality"].any()
    cell = manifest.Cell("toy", "zipf_0.99", "aker", "paper", 10)
    _, arrays = results.read_result(replay.run_cell(cell, ds, DRIVERS, tmp_path / "r2", tmp_path / "w"))
    assert arrays["equality"].any()
    assert (arrays["recall"][arrays["equality"]] == 1.0).all()        # own entry from an exact backend


def test_uniform_stream_uses_only_the_base_queries(toy):
    ds = replay.DatasetDir(toy, "toy")
    uni, ids = ds.universe, ds.streams["uniform"]
    groups = ids - uni.n_members
    assert groups.min() >= 0 and groups.max() < uni.n_groups           # group base ids, never member ids
    assert np.array_equal(ds.vectors_of(ids), np.asarray(ds.official[uni.base_idx[groups]]))   # no interpolation


def test_percentile_thresholds_match_an_exact_percentile():
    from harness import thresholds
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(300, 4)).astype(np.float32)
    exact = np.sort(((vecs[:, None] - vecs[None]) ** 2).sum(2)[np.triu_indices(300, 1)])
    got = thresholds.pairwise_sq_percentiles(vecs, (0.1, 1.0), seed=0)
    for pct, value in got.items():
        assert value == pytest.approx(np.percentile(exact, pct), rel=0.05)
    assert config.pool_size_for(1_814_924, 1) == 18_149
