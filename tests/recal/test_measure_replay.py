import json
import queue
import threading
import time

import numpy as np
import pytest

from recal.measure import common_sample, pagecache, write_json_atomic
from simulations.approximate_hits import throughput_all, throughput_replay as tr


def recorded(warmup, n, with_ids, total=None, background=None):
    total = warmup + n if total is None else total
    ids = list(range(total))
    hit = [bool(i % 2) for i in ids]
    dt = [0.1] * total
    timing = {"hit": hit, "dt_ms": dt}
    if with_ids:
        timing["request_ids"] = ids
    else:
        timing["hit"], timing["dt_ms"] = hit[warmup:], dt[warmup:]
    r = {"args": {"warmup": warmup}, "timing": timing, "audit_pos": [warmup + 1, warmup + 3]}
    if background is not None:
        r["background"] = background
    return r


def test_load_arm_new_and_old_formats_agree():
    new = tr.load_arm(recorded(5, 20, True))
    old = tr.load_arm(recorded(5, 20, False))
    assert np.array_equal(new["pos"], old["pos"])
    assert np.array_equal(new["pos"], np.arange(5, 25))
    assert np.array_equal(new["hit"], old["hit"])
    assert new["total"] == old["total"] == 25


def test_load_arm_rejects_length_mismatch_and_missing_warmup():
    r = recorded(5, 20, True)
    r["timing"]["hit"] = r["timing"]["hit"][:-1]
    with pytest.raises(SystemExit):
        tr.load_arm(r)
    r = recorded(5, 20, True)
    del r["args"]["warmup"]
    with pytest.raises(SystemExit):
        tr.load_arm(r)


def test_alignment_checks():
    a = tr.load_arm(recorded(5, 20, True))
    tr.check_alignment([("none", None), ("a", a), ("b", tr.load_arm(recorded(5, 20, False)))])
    with pytest.raises(SystemExit, match="warmup"):
        tr.check_alignment([("a", a), ("b", tr.load_arm(recorded(6, 20, True)))])
    with pytest.raises(SystemExit, match="replayed requests"):
        tr.check_alignment([("a", a), ("b", tr.load_arm(recorded(5, 19, True)))])
    shifted = tr.load_arm(recorded(5, 20, True))
    shifted["pos"] = shifted["pos"] + 1
    with pytest.raises(SystemExit, match="positions differ"):
        tr.check_alignment([("a", a), ("b", shifted)])


@pytest.mark.parametrize("n,total", [(10, 3), (10, 25), (7, 0), (4, 4)])
def test_spread_counts_sums_and_is_even(n, total):
    c = tr.spread_counts(n, total)
    assert len(c) == n and c.sum() == total
    assert c.max() - c.min() <= 1
    assert np.array_equal(c, tr.spread_counts(n, total))


def test_background_plan_scales_and_skips_miss():
    bg = {"miss": {"searches": 100, "depth_sum": 1000}, "audit": {"searches": 40, "depth_sum": 400},
          "candidate": {"searches": 10, "depth_sum": 1280}, "label": {"searches": 0, "depth_sum": 0}}
    plan, k_miss = tr.background_plan(bg, n=50, total_recorded=200)
    assert k_miss == 10
    assert [(int(c.sum()), k) for c, k in plan] == [(10, 10), (2, 128)]
    assert tr.background_plan(None, 50, 200) == ([], None)


def test_common_sample_is_deterministic_and_nested():
    a = common_sample(1000, 100, seed=3)
    assert np.array_equal(a, common_sample(1000, 100, seed=3))
    assert len(set(a)) == 100 and (np.diff(a) > 0).all()
    assert set(common_sample(1000, 30, seed=3)) <= set(a)
    assert not np.array_equal(a, common_sample(1000, 100, seed=4))
    assert np.array_equal(common_sample(5, 9, seed=0), np.arange(5))


def test_shuffled_arms():
    arms = ["a", "b", "c", "d", "e"]
    o = pagecache.shuffled_arms(arms, 1, 0)
    assert sorted(o) == arms
    assert o == pagecache.shuffled_arms(arms, 1, 0)
    assert any(pagecache.shuffled_arms(arms, 1, r) != o for r in range(1, 6))
    assert arms == ["a", "b", "c", "d", "e"]


def test_write_json_atomic_leaves_no_temp(tmp_path):
    p = tmp_path / "sub" / "x.json"
    write_json_atomic(p, [1, 2])
    assert json.loads(p.read_text()) == [1, 2]
    assert [f.name for f in p.parent.iterdir()] == ["x.json"]


def test_pagecache_unsupported_paths(tmp_path, monkeypatch):
    f = tmp_path / "f.bin"
    f.write_bytes(b"x" * 10000)
    monkeypatch.setattr(pagecache.sys, "platform", "darwin")
    assert pagecache.evict([f])["state"] == "unsupported"
    assert pagecache.residency([f])["state"] == "unknown"
    assert pagecache.record({"a": 1}, {"b": 2}) == {"before": {"a": 1}, "after": {"b": 2}}


def test_pagecache_evict_linux_with_fake_fadvise(tmp_path, monkeypatch):
    (tmp_path / "d" / "e").mkdir(parents=True)
    (tmp_path / "d" / "a.bin").write_bytes(b"1")
    (tmp_path / "d" / "e" / "b.bin").write_bytes(b"2")
    (tmp_path / "c.bin").write_bytes(b"3")
    calls = []
    monkeypatch.setattr(pagecache.sys, "platform", "linux")
    monkeypatch.setattr(pagecache.os, "posix_fadvise", lambda fd, o, n, adv: calls.append((o, n, adv)), raising=False)
    monkeypatch.setattr(pagecache.os, "POSIX_FADV_DONTNEED", 4, raising=False)
    r = pagecache.evict([tmp_path / "d", tmp_path / "c.bin", tmp_path / "missing"])
    assert r == {"state": "evicted", "files": 3, "errors": 0}
    assert calls == [(0, 0, 4)] * 3


def test_residency_on_this_machine_is_well_formed(tmp_path):
    f = tmp_path / "f.bin"
    f.write_bytes(b"x" * 100000)
    r = pagecache.residency([f])
    assert r["state"] in ("ok", "unknown")
    if r["state"] == "ok":
        assert 0.0 <= r["fraction"] <= 1.0


def test_direct_io_detection_returns_none_without_diskannpy(monkeypatch):
    monkeypatch.setattr(pagecache.importlib.util, "find_spec", lambda name: None)
    assert pagecache.diskann_uses_direct_io() is None


def test_direct_io_detection_scans_so_files(tmp_path, monkeypatch):
    class Spec:
        submodule_search_locations = [str(tmp_path)]
    monkeypatch.setattr(pagecache.importlib.util, "find_spec", lambda name: Spec())
    (tmp_path / "a.so").write_bytes(b"abc")
    assert pagecache.diskann_uses_direct_io() is False
    (tmp_path / "b.so").write_bytes(b"x" * 100 + b"O_DIRECT" + b"y")
    assert pagecache.diskann_uses_direct_io() is True


class FakeProc:
    def __init__(self):
        self.alive = True
        self.pid = 1234

    def is_alive(self):
        return self.alive


def fake_backend(jobs, done, stop, delay=0.0):
    def loop():
        while not stop.is_set():
            try:
                i, qi, k = jobs.get(timeout=0.05)
            except queue.Empty:
                continue
            time.sleep(delay)
            done.put(("bg" if i < -1 else "done", i, time.perf_counter()))
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


def run_fake(n=20, background=(), audit=None, procs=None, timeout=10.0):
    jobs, done = queue.Queue(), queue.Queue()
    stop = threading.Event()
    sent = []
    orig_put = jobs.put
    jobs.put = lambda job: (sent.append(job), orig_put(job))
    fake_backend(jobs, done, stop)
    hit = np.arange(n) % 2 == 0
    try:
        r = tr.run_arm("x", hit, np.zeros(n), np.arange(n), 4, jobs, done, procs or [FakeProc()],
                       time.monotonic() + timeout, audit=audit, k_miss=10, background=background)
    finally:
        stop.set()
    return r, sent


def test_run_arm_issues_background_searches_with_depth():
    n = 20
    r, sent = run_fake(n, background=[(tr.spread_counts(n, 5), 128)])
    bg = [j for j in sent if j[0] == -2]
    assert len(bg) == 5 and all(j[2] == 128 for j in bg)
    assert r["background_searches"] == 5
    assert len([j for j in sent if j[0] >= 0]) == n // 2


def test_run_arm_audits_only_on_hits():
    n = 10
    audit = np.ones(n, bool)
    r, sent = run_fake(n, audit=audit)
    assert len([j for j in sent if j[0] == -2]) == n // 2


def test_run_arm_raises_when_worker_dies():
    proc = FakeProc()
    proc.alive = False
    jobs, done = queue.Queue(), queue.Queue()
    with pytest.raises(RuntimeError, match="died"):
        tr.run_arm("x", np.zeros(3, bool), np.zeros(3), np.arange(3), 2, jobs, done, [proc], time.monotonic() + 5)


def test_run_arm_times_out_instead_of_hanging():
    jobs, done = queue.Queue(), queue.Queue()
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        tr.run_arm("x", np.zeros(3, bool), np.zeros(3), np.arange(3), 2, jobs, done, [FakeProc()], time.monotonic() + 0.3)
    assert time.monotonic() - t0 < 5


def test_throughput_all_helpers(tmp_path):
    good, bad = tmp_path / "g.json", tmp_path / "b.json"
    good.write_text(json.dumps([{"a": 1}]))
    bad.write_text('[{"a": 1')
    assert throughput_all.valid_output(good)
    assert not throughput_all.valid_output(bad)
    assert not throughput_all.valid_output(tmp_path / "none.json")
    run = tmp_path / "run"
    run.mkdir()
    assert throughput_all.find_recording(run) is None
    (run / "r1.json").write_text("[]")
    assert throughput_all.find_recording(run) is None
    (run / "DONE").write_text("")
    assert throughput_all.find_recording(run).endswith("r1.json")
    (run / "r2.json").write_text("[]")
    with pytest.raises(RuntimeError, match="several"):
        throughput_all.find_recording(run)


def test_run_to_file_checks_exit_code_and_output(tmp_path, monkeypatch):
    out = tmp_path / "o.json"

    def fake_run(cmd, env, check):
        part = cmd[cmd.index("--out") + 1]
        if "fail" in cmd:
            raise throughput_all.subprocess.CalledProcessError(2, cmd)
        content = "[" if "partial" in cmd else '[{"x": 1}]'
        open(part, "w").write(content)
    monkeypatch.setattr(throughput_all.subprocess, "run", fake_run)
    with pytest.raises(throughput_all.subprocess.CalledProcessError):
        throughput_all.run_to_file(["py", "s", "fail"], out)
    with pytest.raises(RuntimeError):
        throughput_all.run_to_file(["py", "s", "partial"], out)
    assert not out.exists()
    throughput_all.run_to_file(["py", "s"], out)
    assert json.loads(out.read_text()) == [{"x": 1}]
    assert not (tmp_path / "o.json.part").exists()
