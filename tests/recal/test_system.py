import numpy as np
import pytest

from recal.backend import ExactBackend
from recal.params import Params
from recal.system import ReCal


def make_world(seed=0, n=4000, d=12, clusters=40):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=3.0, size=(clusters, d))
    data = (centers[rng.integers(0, clusters, n)] + rng.normal(size=(n, d))).astype(np.float32)
    return rng, data, centers


def make_stream(rng, centers, n, repeat_share=0.3):
    d = centers.shape[1]
    stream, pool = [], []
    for _ in range(n):
        if pool and rng.random() < repeat_share:
            stream.append(pool[rng.integers(0, len(pool))])
        else:
            q = (centers[rng.integers(0, len(centers))] + rng.normal(size=d)).astype(np.float32)
            pool.append(q)
            stream.append(q)
    return stream


@pytest.fixture
def world(tmp_path):
    rng, data, centers = make_world()
    params = Params(K=5, k_max=20, H=3000, cold_start=1500, min_ess=30.0, seed=1)
    backend = ExactBackend(data)
    M = 0.6 * data.size * 4
    system = ReCal(backend, params, M, tmp_path)
    return rng, data, centers, params, backend, system, M


def test_cold_start_construction_and_logs(world):
    rng, data, centers, p, backend, system, M = world
    stream = make_stream(rng, centers, 2500)
    for q in stream:
        r = system.serve(q, 5)
        assert len(r.ids) == 5 and np.all(np.diff(r.dists) >= -1e-6)
    assert system.constructions == 1 and system.built.n_entries > 0
    # mini cache hits are in the hash log but not in the history
    assert system.log.hash_count == system.requests == 2500
    assert system.log.count == system.t == system.requests - system.counts["mini_hits"]
    assert system.counts["mini_hits"] > 0
    b = system.bytes_by_structure()
    assert b["total"] <= M
    # the floor of 256 entries can exceed 10% of M at this tiny scale
    assert b["mini_cache"] <= max(p.mini_cap * M, p.mini_floor * (40 + 8 * p.K)) + 1


def test_main_hits_audited_and_lambda_recalibrates(world):
    rng, data, centers, p, backend, system, M = world
    for q in make_stream(rng, centers, 2600):
        system.serve(q, 5)
    assert system.window is not None and system.built is not None
    stream = make_stream(rng, centers, 4000)
    n_before = len(system.window)
    sources = [system.serve(q, 5).source for q in stream]
    assert "main" in sources and "backend" in sources and "mini" in sources
    assert system.counts["audits"] > 0
    assert len(system.window) > 0 and len(system.window) != n_before or system.window.W > 0
    report = system.status_report()
    assert set(report) >= {"lambda", "reason", "undefined_share", "last_lambda", "audited_recall"}
    # audits are about a fraction `a` of main-cache hits
    assert system.counts["audits"] / max(system.counts["main_hits"], 1) == pytest.approx(p.a, abs=0.06)


def test_main_hits_meet_target_on_stationary_stream(world):
    rng, data, centers, p, backend, system, M = world
    for q in make_stream(rng, centers, 2600):
        system.serve(q, 5)
    stream = make_stream(rng, centers, 6000, repeat_share=0.0)
    truth, _ = backend.search(np.stack(stream), 5, kind="eval")
    recalls = []
    for q, t in zip(stream, truth):
        r = system.serve(q, 5)
        if r.source == "main":
            recalls.append(np.isin(t, r.ids).mean())
    assert len(recalls) > 100
    assert np.mean(recalls) >= p.T - 0.03


def test_k_above_max_goes_to_backend_and_is_not_logged(world):
    rng, data, centers, p, backend, system, M = world
    q = data[0]
    before = system.log.count
    r = system.serve(q, 25)
    assert r.source == "backend" and len(r.ids) == 25
    assert system.log.count == before and system.t == 0


def test_construction_keeps_mini_cache(world):
    rng, data, centers, p, backend, system, M = world
    stream = make_stream(rng, centers, 5000)
    size_before = 0
    for q in stream:
        if system.built is None:
            size_before = system.core.mini_size()
        system.serve(q, 5)
        if system.built is not None:
            break
    assert system.built is not None and size_before > 0
    assert system.core.mini_size() >= min(size_before, system.built.R)


def test_mini_cache_without_repeats_falls_to_the_floor(tmp_path):
    rng, data, centers = make_world(seed=5)
    params = Params(K=5, k_max=20, H=3000, cold_start=1500, min_ess=30.0, seed=1)
    system = ReCal(ExactBackend(data), params, 0.6 * data.size * 4, tmp_path)
    for q in make_stream(rng, centers, 1600, repeat_share=0.0):
        system.serve(q, 5)
    assert system.built is not None
    assert system.built.R == params.mini_floor
    assert system.core.mini_capacity() == params.mini_floor
    first = system.first_construction_bytes
    assert first["mini_cache"] <= params.mini_floor * (40 + 8 * 5)
    assert first["total"] <= 0.6 * data.size * 4
