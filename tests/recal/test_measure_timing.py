import os
import itertools

import pytest

from simulations.approximate_hits import timing_env as te


def fake_samples(n, cores=0.1, mbps=1.0, every=2.0):
    return [(i * every, cores, mbps) for i in range(n)]


def verdict(samples, duration=100.0, **kw):
    return te.compute_verdict(samples, duration, 2.0, 1.0, 50.0, kw.pop("start_ok", True), **kw)


def test_full_samples_quiet():
    assert verdict(fake_samples(50))[0] == "quiet"


def test_seventy_percent_unknown():
    v, reason = verdict(fake_samples(35))
    assert v == "unknown"
    assert "35 of 50" in reason


def test_empty_unknown_and_allow_unsampled():
    assert verdict([])[0] == "unknown"
    assert verdict([], allow_unsampled=True)[0] == "quiet"


def test_contended_cpu_and_io():
    assert verdict(fake_samples(50, cores=3.0))[0] == "contended"
    assert verdict(fake_samples(50, mbps=200.0))[0] == "contended"
    assert verdict(fake_samples(10, cores=3.0))[0] == "contended"


def test_start_check_failure():
    assert verdict(fake_samples(50), start_ok=False)[0] == "contended"


def test_expected_samples():
    assert te.expected_samples(100.0, 2.0) == 50
    assert te.expected_samples(0.5, 2.0) == 1


def test_report_stores_fields():
    g = te.QuietGuard()
    g.start_check = {"ok": True}
    g.samples = fake_samples(10)
    g.duration_s = 100.0
    rep = g.report()
    assert rep["verdict"] == "unknown"
    assert rep["during"]["samples"] == 10
    assert rep["during"]["expected_samples"] == 50
    assert rep["reason"]
    g.allow_unsampled = True
    assert g.report()["verdict"] == "quiet"


def test_register_exempt(tmp_path, monkeypatch):
    monkeypatch.setenv("RECAL_QUEUE_DIR", str(tmp_path))
    assert te.register_exempt(123) is None
    (tmp_path / "registry").mkdir()
    path = te.register_exempt(123)
    assert open(path).read().split()[1] == "timing"
    te.unregister_exempt(123)
    assert not os.path.exists(path)


@pytest.mark.parametrize("k,delta,big_r", [(10, 5, 500), (10, 0, 20), (100, 5, 500)])
def test_every_miss_charged(k, delta, big_r):
    base = [0.004, 0.006, 0.005]
    charges, approx = [], set()
    depths = [k, big_r, k + delta]
    for b, depth in itertools.product(base, depths):
        c, a = te.backend_charge(b, depth, k)
        assert c >= b
        charges.append(c)
        if a:
            approx.add(depth)
    out = te.summarize_charges(charges, len(charges), approx)
    assert out["n_charged"] == out["n_misses"]
    assert out["min_charge_s"] >= min(base)
    assert big_r in out["approximated_depths"]


def test_charge_never_zero():
    with pytest.raises(ValueError):
        te.backend_charge(0.0, 10, 10)
    with pytest.raises(ValueError):
        te.backend_charge(None, 10, 10)
