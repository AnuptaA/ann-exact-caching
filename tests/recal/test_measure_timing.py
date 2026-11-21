import json
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


def table():
    return te.DepthLatency([10, 20, 100, 500], [0.002, 0.0025, 0.004, 0.012])


@pytest.mark.parametrize("k,delta,big_r", [(10, 5, 500), (10, 0, 20), (10, 5, 70)])
def test_every_miss_charged_at_its_depth(k, delta, big_r):
    base = [0.004, 0.006, 0.005]
    charges, depths_seen = [], set()
    depths = [k, big_r, k + delta]
    for b, depth in itertools.product(base, depths):
        c, approx = te.backend_charge(b, depth, k, table())
        assert c >= b and not approx
        charges.append(c)
        depths_seen.add(depth)
    out = te.summarize_charges(charges, len(charges), depths_seen)
    assert out["n_charged"] == out["n_misses"]
    assert out["min_charge_s"] >= min(base)
    assert big_r in out["depths_charged"]


def test_deeper_misses_cost_more_than_depth_k():
    t = table()
    c10, _ = te.backend_charge(0.004, 10, 10, t)
    c500, _ = te.backend_charge(0.004, 500, 10, t)
    assert c10 == 0.004
    assert c500 == pytest.approx(0.004 * 0.012 / 0.002)


def test_depth_table_interpolates_clamps_below_and_fails_above():
    t = table()
    assert t.latency(60) == pytest.approx(0.00325)              # midway between 20 and 100
    assert t.latency(5) == pytest.approx(0.002)                 # below the smallest depth: no extrapolation
    assert t.ratio(5, 10) == 1.0
    with pytest.raises(ValueError):
        t.latency(501)


def test_deep_call_without_a_table_is_refused_not_approximated():
    assert te.backend_charge(0.004, 10, 10)[0] == 0.004
    with pytest.raises(ValueError):
        te.backend_charge(0.004, 70, 10)


def test_depth_table_loads_from_probe_json(tmp_path):
    path = tmp_path / "d.json"
    path.write_text(json.dumps({"depth_latency": {"depths": [100, 10], "mean_ms": [4.0, 2.0]}}))
    t = te.DepthLatency.load(path)
    assert t.depths == [10.0, 100.0] and t.latency(10) == pytest.approx(0.002)


def test_charge_never_zero():
    with pytest.raises(ValueError):
        te.backend_charge(0.0, 10, 10)
    with pytest.raises(ValueError):
        te.backend_charge(None, 10, 10)
