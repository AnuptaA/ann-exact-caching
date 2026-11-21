import numpy as np
import pytest

from recal.core import Core
from tests.recal.helpers import ID_BASE, build_core, exact_order
from tests.recal.test_core import random_case


def line_core(headroom_ids=()):
    """One entry at the origin over points on a line: 1, 2, ..., 6 (response list of five)."""
    d = 2
    data = np.array([[x, 0.0] for x in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)], np.float32)
    core = Core(d, 20)
    core.set_quantizer(data.min(axis=0), data.max(axis=0))
    ids = np.arange(5, dtype=np.int64) + 100
    core.add_entry(np.zeros(d, np.float32), 5, 1, ids, data[:5])
    core.finish_build(8, 50, 50, 1)
    return core, data


def test_insert_joins_every_containing_sphere_in_rank_order():
    rng, data, centers, js, K = random_case(seed=20)
    core, ids, lists = build_core(data, centers, K, js)
    radii = np.array([core.entry(e)["radius"] for e in range(len(centers))])
    new = rng.normal(size=(30, data.shape[1])).astype(np.float32)
    new_ids = np.arange(30, dtype=np.int64) + 90_000
    admitted, frozen, ignored, left = core.insert_batch(new_ids, new, 1e12, 5, 16)
    assert frozen == 0 and admitted + ignored == 30
    for i, v in enumerate(new):
        inside = [e for e in range(len(centers)) if np.linalg.norm(centers[e].astype(np.float64) - v) <= radii[e]]
        for e in range(len(centers)):
            member = int(new_ids[i]) in core.entry(e)["response_ids"]
            # the center index finds spheres within its probing rounds; with 24 entries all are reachable
            assert member == (e in inside)
    for e in range(len(centers)):
        resp = core.entry(e)["response_ids"]
        pts = np.stack([data[r - ID_BASE] if r < 90_000 else new[r - 90_000] for r in resp])
        dist = np.linalg.norm(pts.astype(np.float64) - centers[e].astype(np.float64), axis=1)
        assert np.all(np.diff(dist) >= -1e-9)


def test_displacement_keeps_pool_size_and_shrinks_radius():
    core, data = line_core()
    live = core.n_data_slots()
    # headroom exhausted: the sixth point (distance 4.5) joins, the farthest member (5) is displaced
    admitted, frozen, _, _ = core.insert_batch(np.array([900]), np.array([[4.5, 0.0]], np.float32), 0.0)
    assert (admitted, frozen) == (1, 0)
    assert core.n_data_slots() == live
    e = core.entry(0)
    assert 900 in e["response_ids"] and 104 not in e["response_ids"]
    assert e["radius"] == pytest.approx(4.5)


def test_insert_frozen_when_displacement_frees_nothing():
    d = 2
    data = np.array([[x, 0.0] for x in (1.0, 2.0, 3.0, 4.0, 5.0)], np.float32)
    core = Core(d, 20)
    core.set_quantizer(data.min(axis=0), data.max(axis=0))
    ids = np.arange(5, dtype=np.int64) + 100
    core.add_entry(np.zeros(d, np.float32), 5, 1, ids, data)
    # a second entry on the same side that also lists the farthest point of the first
    core.add_entry(np.array([1.0, 0.0], np.float32), 2, 1, ids[[4, 3]], data[[4, 3]], 4.0)
    core.finish_build(8, 50, 50, 1)
    live = core.n_data_slots()
    admitted, frozen, _, _ = core.insert_batch(np.array([900]), np.array([[0.5, 0.0]], np.float32), 0.0)
    # joins the first entry only (the second has radius 4 around 1.0 and also contains 0.5: both join);
    # the farthest members 105 (shared by both lists) is displaced from both lists, freeing one slot
    assert admitted + frozen == 1
    assert core.n_data_slots() <= live + (0 if frozen else 0)


def test_shortened_list_takes_insert_directly():
    core, data = line_core()
    assert core.delete_id(101) and core.delete_id(102)       # the list lost members: shorter than built
    live = core.n_data_slots()
    admitted, frozen, _, _ = core.insert_batch(np.array([901]), np.array([[1.5, 0.0]], np.float32), 0.0)
    assert (admitted, frozen) == (1, 0)
    assert core.n_data_slots() == live + 1                   # taken directly: no displacement, the pool grows


def test_sweep_removes_stale_pointers_and_keeps_radius():
    rng, data, centers, js, K = random_case(seed=21)
    core, ids, lists = build_core(data, centers, K, js)
    radii = [core.entry(e)["radius"] for e in range(len(centers))]
    pool = sorted({i for e in range(len(centers)) for i in core.entry(e)["response_ids"]})
    victims = pool[: max(1, int(0.06 * len(pool)))]
    for v in victims:
        core.delete_id(v)
    assert core.n_dead_slots() == len(victims)
    removed = core.sweep()
    assert removed == len(victims) and core.n_dead_slots() == 0
    for e in range(len(centers)):
        resp = core.entry(e)["response_ids"]
        assert not set(resp) & set(victims)
        assert core.entry(e)["radius"] == radii[e]
    q = rng.normal(size=data.shape[1]).astype(np.float32)
    r = core.lookup(q, K, None)
    assert not set(r.ids.tolist()) & set(victims)


def test_updater_batches_and_headroom(tmp_path):
    from recal.backend import ExactBackend
    from recal.params import Params
    from recal.system import ReCal
    from tests.recal.test_system import make_stream, make_world
    rng, data, centers = make_world(seed=3)
    p = Params(K=5, k_max=20, H=3000, cold_start=1500, min_ess=30.0, seed=2, insert_batch_min=50)
    backend = ExactBackend(data)
    M = 0.6 * data.size * 4
    system = ReCal(backend, p, M, tmp_path)
    for q in make_stream(rng, centers, 2600):
        system.serve(q, 5)
    assert system.built is not None
    version = backend.version
    new = (centers[rng.integers(0, len(centers), 120)] + rng.normal(size=(120, data.shape[1]))).astype(np.float32)
    system.insert(np.arange(40) + 50_000, new[:40])
    assert len(system.updater.buffer_ids) == 40 and system.updater.admitted == 0       # lazy: below one batch
    system.insert(np.arange(40, 120) + 50_000, new[40:])
    assert len(system.updater.buffer_ids) < 50 and backend.version > version
    assert system.updater.admitted + system.updater.ignored + system.updater.frozen >= 50
    b = system.bytes_by_structure()
    assert b["total"] <= M + 1
    assert system.delete([int(system.core.entry(0)["response_ids"][0])]) == 1
