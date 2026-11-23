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
