from types import SimpleNamespace

import numpy as np

from recal.params import Params
from recal.policy import Status
from recal.signals import Signals


def fake_system(p, slots=1000, t=0, last=0):
    core = SimpleNamespace(slots=slots, n_data_slots=lambda: core.slots)
    system = SimpleNamespace(t=t, last_construction=last, core=core, built=object(), status=Status(int(p.drop_window * p.H)),
                             rebuild_requested=None)
    return system


def small_params(**kw):
    return Params(H=1000, drop_floor=100, drop_every=50, min_interval=0.2, **kw)


def run(signals, system, scores, lam=1.0):
    for s in scores:
        system.t += 1
        system.status.record(lam)
        signals.after_request(system, s)
        if system.rebuild_requested:
            return system.rebuild_requested
    return None


def test_stationary_stream_fires_nothing_before_the_periodic_refresh():
    p = small_params()
    system = fake_system(p)
    sig = Signals(p)
    sig.after_construction(system)
    rng = np.random.default_rng(0)
    assert run(sig, system, rng.normal(0.6, 0.1, size=900)) is None


def test_periodic_refresh_at_H():
    p = small_params()
    system = fake_system(p)
    sig = Signals(p)
    sig.after_construction(system)
    fired = run(sig, system, np.full(1200, 0.5))
    assert fired == "R4 periodic refresh" and system.t == p.H


def test_score_drop_fires_and_respects_min_interval():
    p = small_params()
    system = fake_system(p)
    sig = Signals(p)
    sig.after_construction(system)
    rng = np.random.default_rng(1)
    first = rng.normal(0.7, 0.05, size=300)      # baseline window (100) and beyond
    drop = rng.normal(0.4, 0.05, size=300)
    fired = run(sig, system, np.r_[first, drop])
    assert fired == "R5 score drop"
    assert system.t >= p.min_interval * p.H        # never before 0.2H
    assert system.t >= max(int(p.drop_from * p.H), 2 * sig.window)


def test_score_drop_not_before_minimum_interval():
    p = small_params()
    system = fake_system(p)
    sig = Signals(p)
    sig.after_construction(system)
    rng = np.random.default_rng(2)
    run(sig, system, np.r_[rng.normal(0.7, 0.01, size=100), rng.normal(0.1, 0.01, size=99)])
    assert system.rebuild_requested is None and sig.events == []


def test_shrinking_pool_and_frozen_inserts_and_outage():
    p = small_params()
    system = fake_system(p, slots=1000)
    sig = Signals(p, updater=SimpleNamespace(frozen=0, after_construction=lambda: None))
    sig.after_construction(system)
    system.core.slots = 790
    assert run(sig, system, np.full(300, 0.5)) == "R2 shrinking pool"
    system = fake_system(p, slots=1000)
    sig = Signals(p, updater=SimpleNamespace(frozen=0, after_construction=lambda: None))
    sig.after_construction(system)
    sig.updater.frozen = 50                      # 5% of the pool
    assert run(sig, system, np.full(300, 0.5)) == "R3 frozen inserts"
    system = fake_system(p)
    sig = Signals(p)
    sig.after_construction(system)
    assert run(sig, system, np.full(900, 0.5), lam=None) == "R6 outage"
