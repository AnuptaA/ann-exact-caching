import dataclasses

import numpy as np
import pytest

from recal.backend import ExactBackend
from recal.params import Params
from recal.system import ReCal
from tests.recal.test_system import make_stream, make_world

BASE = dict(K=5, k_max=20, H=3000, cold_start=1500, min_ess=30.0, seed=1)
ABLATIONS = ("no_union", "no_expansion", "no_shells", "no_calibration", "no_recalibration", "no_mini_cache")


def run(tmp_path, n=3200, **flags):
    rng, data, centers = make_world(seed=3)
    system = ReCal(ExactBackend(data), Params(**BASE, **flags), 0.6 * data.size * 4, tmp_path)
    responses = [system.serve(q, 5) for q in make_stream(np.random.default_rng(9), centers, n)]
    return system, responses


@pytest.fixture(scope="module")
def default_run(tmp_path_factory):
    return run(tmp_path_factory.mktemp("default"))


def test_all_switches_off_equals_the_default(default_run, tmp_path):
    system, responses = default_run
    again, other = run(tmp_path, **{name: False for name in ABLATIONS}, lru=False)
    assert [r.source for r in responses] == [r.source for r in other]
    assert system.counts == again.counts and system.lam == again.lam


def test_no_expansion_keeps_only_own_top_k(tmp_path):
    system, _ = run(tmp_path, no_expansion=True)
    sizes = set(system.core.entry_sizes().tolist())
    assert system.built is not None and sizes == {5}
    assert system.core.n_shell_slots() > 0


def test_no_shells_has_no_shell_pool_and_a_two_feature_estimator(tmp_path):
    system, _ = run(tmp_path, no_shells=True)
    assert system.core.n_shell_slots() == 0
    assert system.built.model["coef"][2] == 0.0
    assert system.bytes_by_structure()["total"] <= system.M


def test_no_union_answers_from_one_entry(tmp_path):
    system, responses = run(tmp_path, no_union=True)
    system.enable_records()
    served = [r for r in responses if r.source == "main"]
    assert served and system.built is not None
    q = np.random.default_rng(1).normal(size=system.backend.dim).astype(np.float32)
    res = system.core.lookup(q, 5, system.lam, system.built.m, False)
    if res.n_candidates >= 5:
        assert res.n_contrib == 1
        entry = system.core.entry(res.best_entry)
        assert set(res.ids.tolist()) <= set(entry["response_ids"])


def test_no_calibration_uses_target_as_threshold_and_never_changes_it(tmp_path):
    system, responses = run(tmp_path, no_calibration=True)
    assert system.lam == system.p.T
    assert {r.lam_used for r in responses if r.lam_used is not None} == {system.p.T}


def test_no_recalibration_keeps_the_construction_threshold(tmp_path):
    system, responses = run(tmp_path, n=3600, no_recalibration=True)
    lams = {r.lam_used for r in responses if r.lam_used is not None}
    assert len(lams) <= system.constructions and system.counts["audits"] > 0


def test_default_recalibrates(default_run):
    system, responses = default_run
    assert len({r.lam_used for r in responses if r.lam_used is not None}) > system.constructions


def test_no_mini_cache_sends_every_request_to_the_main_cache(tmp_path):
    system, responses = run(tmp_path, no_mini_cache=True)
    assert system.counts["mini_hits"] == 0 and system.core.mini_capacity() == 0
    assert all(r.source != "mini" for r in responses)
    assert system.log.count == system.requests


def test_ablation_flags_reach_params_from_the_command_line():
    from recal import sim
    args = sim.parser().parse_args(["--prep", "x", "--out", "y", "--no-union", "--lru", "--no-mini-cache"])
    flags = sim.ablations(args)
    assert flags["no_union"] and flags["lru"] and flags["no_mini_cache"] and not flags["no_shells"]
    assert dataclasses.replace(Params(), **flags).no_union


def test_lru_admits_on_misses_holds_2k_entries_and_stays_within_m(tmp_path):
    system, responses = run(tmp_path, n=3600, lru=True, T=0.5)
    assert system.lru is not None and system.built is not None
    assert set(system.lru.entry_sizes().tolist()) == {10}
    assert system.counts["misses"] > 0 and system.lru.n_entries() > 0
    assert system.bytes_by_structure()["total"] <= system.M
    assert system.counts["main_hits"] > 0
    depths = {d for r in responses for _, d in r.background if _ == "miss"}
    assert 20 in depths


def test_lru_touch_keeps_used_entries_and_evicts_the_oldest():
    from recal.lru import LruCache
    cache = LruCache(2, budget=3000.0, coord_bytes=4.0)
    rng = np.random.default_rng(0)
    first = None
    for i in range(40):
        ids = np.arange(i * 20, i * 20 + 20)
        vecs = rng.normal(size=(20, 2)).astype(np.float32)
        dists = np.sort(rng.random(20)).astype(np.float32)
        cache.admit(rng.normal(size=2).astype(np.float32), 5, ids, dists, vecs)
        if i == 0:
            first = next(iter(cache.entries))
        else:
            cache.touch([first])
    assert first in cache.entries and cache.total_bytes() <= 3000.0 and cache.n_entries() < 40


def test_lru_leaves_default_path_and_refresh_is_a_noop(tmp_path):
    system, _ = run(tmp_path, n=3200, lru=True)
    built = system.built
    assert system.construct() is built and system.constructions == 1


def test_no_rebuild_signals_leaves_only_the_periodic_refresh(tmp_path):
    from recal.signals import Signals

    class Sys:
        built, t, last_construction = object(), 0, 0

        class core:
            @staticmethod
            def n_data_slots():
                return 1

    p = Params(**BASE, no_rebuild_signals=True)
    sig = Signals(p)
    sig.pool_at_build = 1000
    assert sig.fired(Sys, since=10) is None
    assert sig.fired(Sys, since=p.H) == "R4 periodic refresh"
    default = Signals(Params(**BASE))
    default.pool_at_build = 1000
    assert default.fired(Sys, since=10) == "R2 shrinking pool"


def test_choose_m_rule_and_the_lru_needed_count_match_the_core_semantics():
    from recal import build as builder
    from recal.lru import LruCache
    needed = np.array([1, 2, 3, 10, -1, 4])
    assert builder.choose_m(needed, 50, 95.0) == 10
    assert builder.choose_m(needed, 6, 95.0) == 6
    assert builder.choose_m(np.array([-1, -1]), 7, 95.0) == 7 and builder.choose_m(needed, 50, 95.0, True) == 0
    cache = LruCache(2, budget=1e9)
    far = np.array([10.0, 0.0], np.float32)
    near = np.array([0.0, 0.0], np.float32)
    zeros = np.zeros(2, np.float32)
    for center, ids in ((far, np.arange(0, 10)), (near, np.arange(10, 20))):
        cache.admit(center, 5, ids, np.linspace(0.1, 1.0, 10).astype(np.float32), np.zeros((20, 2), np.float32))
    q = np.array([[0.5, 0.0]], np.float32)
    assert cache.entries_needed(q, np.array([[10, 11, 12, 13, 14]])).tolist() == [1]
    assert cache.entries_needed(q, np.array([[10, 0, 99, 98, 97]])).tolist() == [2]
    assert cache.entries_needed(q, np.array([[90, 91, 92, 93, 94]])).tolist() == [-1]
    assert zeros.shape == (2,)


def test_lru_and_recal_report_m_from_the_same_function(tmp_path, monkeypatch):
    from recal import build as builder
    calls = []
    real = builder.choose_m

    def spy(needed, n_entries, percentile, idealized=False):
        m = real(needed, n_entries, percentile, idealized)
        calls.append(m)
        return m

    monkeypatch.setattr(builder, "choose_m", spy)
    recal, _ = run(tmp_path / "a", n=2600)
    lru, _ = run(tmp_path / "b", n=3000, lru=True, T=0.5)
    assert recal.construction_log[0]["m"] == calls[0] and recal.construction_log[0]["m_source"] == "construction"
    assert lru.construction_log[0]["m"] == calls[-1] and lru.construction_log[0]["m_source"] == "history_once"
    assert lru.built.m == calls[-1] and lru.built.m <= lru.lru.n_entries()
