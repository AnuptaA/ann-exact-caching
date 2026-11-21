#!/usr/bin/env python3
"""Closed-loop throughput of a cache in front of the real DiskANN backend (doc 9ap), by replay.

A timed cache run (online_sim --timing-every 1) recorded, for every request after the warm-up, whether it was
a hit and how long the cache path took (dt_ms). This script replays those requests in order:
  - one "cache thread" (the main process) handles requests one at a time and spends each request's measured
    cache-path time (a busy wait; hits finish here);
  - a miss is sent to a pool of backend worker processes that run the REAL DiskANN search (answers discarded;
    DiskANN's Python binding holds the GIL, hence processes), so SSD reads and CPU are real;
  - closed loop: at most --clients requests are in flight; the next request starts when one finishes.
Alignment: a recorded run carries timing.request_ids (stream positions, new) or only hit/dt_ms starting at
args.warmup (old). The replayed query of a request is stream[position]. All arms must have the same positions
(and lengths); --legacy-alignment restores the old min(20000, len // 5) computation without checks.
Page cache: index files are evicted before every arm (--no-evict disables), arm order is shuffled per repetition
with a recorded seed, and the residency before and after each arm is stored.
--io-complete counts background work as backend searches: audits, deep candidate fetches and label fetches taken
from the recorded run's "background" stats (searches and mean depth, scaled to the replayed requests and spread
evenly), plus misses at the recorded mean miss depth. Runs without the field behave as before (audit_pos at k 10;
misses of arms named ours* at --k-deep). Still NOT counted: admission vector fetches, shell-sample reads and
rebuild/greedy/eviction CPU. Each backend process opens its own index and node cache.
Reports completed requests per second (QPS) and the latency (issue to completion) percentiles. "none" = no cache
(every request goes straight to the backend, cache time 0). Same backend processes and node-cache setting for
every arm. Needs a quiet machine.

  PYTHONPATH=. .venv-ann/bin/python simulations/approximate_hits/throughput_replay.py --workload tripclick-ip \
      --arm none --arm ours=<run.json> --arm aker=<run.json> --clients 16 --n 20000
"""
import sys
import json
import time
import argparse
import logging
import queue
import multiprocessing as mp
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import numpy as np

from recal.measure import common_sample, write_json_atomic
from recal.measure import pagecache

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
RES = Path("simulations/approximate_hits/results")


def query_file(w):
    """q.npy of the workload's prep, else the ESCI query vectors (online_sim.load)."""
    f = w.prep_dir / "q.npy"
    return f if f.exists() else Path("datasets/esci/data_vectors/esci_queries.npy")


def worker(info, qfile, jobs, done, dtype):
    from simulations.approximate_hits.diskann_aio import open_disk_index
    q = np.load(qfile, mmap_mode="r")
    ix = open_disk_index(max_wait=3600, index_directory=info["index_dir"], num_threads=2,
                         num_nodes_to_cache=info.get("nodes", 0), cache_mechanism=1 if info.get("nodes", 0) else 0,
                         index_prefix="ann")
    done.put(("ready", 0, 0.0))
    while True:
        job = jobs.get()
        if job is None:
            return
        i, qi, k = job
        # a deep list of k > L needs a search list of at least k (SPACEV / ESCI: L 100 < 128)
        ix.search(np.ascontiguousarray(q[qi], dtype=dtype), k_neighbors=k, complexity=max(info["L"], k),
                  beam_width=info["W"])
        done.put(("bg" if i < -1 else "done", i, time.perf_counter()))


def spin(seconds):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


def load_arm(r):
    """Aligned view of a recorded run: stream positions, hit, dt_ms, warmup, audit positions, background stats."""
    t = r["timing"]
    try:
        warmup = int(r["args"]["warmup"])
    except KeyError:
        raise SystemExit("recorded run has no args.warmup")
    hit = np.asarray(t["hit"], bool)
    dt = np.asarray(t["dt_ms"], float)
    if len(hit) != len(dt):
        raise SystemExit(f"recorded hit ({len(hit)}) and dt_ms ({len(dt)}) differ in length")
    if "request_ids" in t:
        ids = np.asarray(t["request_ids"], dtype=np.int64)
        if len(ids) != len(hit):
            raise SystemExit(f"recorded request_ids ({len(ids)}) and hit ({len(hit)}) differ in length")
        keep = ids >= warmup
        pos, hit, dt = ids[keep], hit[keep], dt[keep]
        total = len(ids)
    else:
        pos = warmup + np.arange(len(hit), dtype=np.int64)
        total = warmup + len(hit)
    return {"pos": pos, "hit": hit, "dt_ms": dt, "warmup": warmup, "total": total, "background": r.get("background"),
            "audit_pos": np.asarray(r["audit_pos"], dtype=np.int64) if "audit_pos" in r else None}


def check_alignment(arms):
    """arms: list of (name, loaded arm or None). All recorded arms must replay the same positions."""
    recorded = [(n, t) for n, t in arms if t is not None]
    if not recorded:
        return
    name0, t0 = recorded[0]
    for name, t in recorded[1:]:
        if t["warmup"] != t0["warmup"]:
            raise SystemExit(f"arm {name}: warmup {t['warmup']} != {t0['warmup']} of arm {name0}")
        if len(t["pos"]) != len(t0["pos"]):
            raise SystemExit(f"arm {name}: {len(t['pos'])} replayed requests != {len(t0['pos'])} of arm {name0}")
        if not np.array_equal(t["pos"], t0["pos"]):
            raise SystemExit(f"arm {name}: query positions differ from arm {name0}")


def spread_counts(n, total):
    """Deterministic per-request counts summing to total, spread evenly over n requests."""
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    cum = (np.arange(n + 1, dtype=np.int64) * int(total)) // n
    return np.diff(cum)


def background_plan(background, n, total_recorded):
    """[(per-request counts, k)] for the background kinds other than miss, scaled to n replayed requests.
    Also returns the mean miss depth (None when the recording has none)."""
    plan, k_miss = [], None
    if not background:
        return plan, k_miss
    for kind, e in background.items():
        s = int(e.get("searches", 0))
        if s <= 0:
            continue
        depth = max(1, int(round(e["depth_sum"] / s)))
        if kind == "miss":
            k_miss = depth
            continue
        plan.append((spread_counts(n, int(round(s * n / max(total_recorded, 1)))), depth))
    return plan, k_miss


def get_message(done, procs, deadline, block):
    """Next message from the workers, None if nothing is ready. Raises when a worker died or the time is up."""
    if not block:
        try:
            return done.get_nowait()
        except queue.Empty:
            return None
    while True:
        if time.monotonic() > deadline:
            raise TimeoutError("throughput replay exceeded --timeout-s")
        dead = [p.pid for p in procs if not p.is_alive()]
        if dead:
            raise RuntimeError(f"backend worker(s) {dead} died")
        try:
            return done.get(timeout=1.0)
        except queue.Empty:
            pass


def run_arm(name, hit, dt, qids, clients, jobs, done, procs, deadline, audit=None, k_miss=10, background=()):
    """audit: requests whose hit is audited (old recordings); each audit is a backend search issued as
    background work. background: [(counts per request, k)] extra backend searches issued after each request.
    Misses search with k_miss results. Wall time runs until the background work is done too."""
    n = len(qids)
    issue = np.zeros(n)
    finish = np.zeros(n)
    inflight = 0
    ndone = 0
    bg_out = 0
    bg_total = 0

    def drain(block):
        nonlocal inflight, ndone, bg_out
        msg = get_message(done, procs, deadline, block)
        if msg is None:
            return False
        kind, i, t = msg
        if kind == "bg":
            bg_out -= 1
            return True
        finish[i] = t
        inflight -= 1
        ndone += 1
        return True

    def issue_bg(i, count, k):
        nonlocal bg_out, bg_total
        for _ in range(int(count)):
            jobs.put((-2, int(qids[i]), int(k)))
            bg_out += 1
            bg_total += 1

    t0 = time.perf_counter()
    for i in range(n):
        while inflight >= clients:
            drain(True)
        while drain(False):
            pass
        issue[i] = time.perf_counter()
        if dt[i] > 0:
            spin(dt[i])                       # the cache thread's work on this request
        if hit[i]:
            finish[i] = time.perf_counter()
            ndone += 1
            if audit is not None and audit[i]:
                issue_bg(i, 1, 10)
        else:
            jobs.put((i, int(qids[i]), k_miss))
            inflight += 1
        for counts, k in background:
            issue_bg(i, counts[i], k)
    while inflight > 0 or bg_out > 0:
        drain(True)
    wall = time.perf_counter() - t0
    lat = (finish - issue) * 1e3
    return {"arm": name, "n": n, "wall_s": wall, "qps": n / wall, "hit_rate": float(np.mean(hit)),
            "audits": int(audit.sum()) if audit is not None else 0, "background_searches": bg_total, "k_miss": k_miss,
            "mean_ms": float(lat.mean()), "p50_ms": float(np.percentile(lat, 50)), "p90_ms": float(np.percentile(lat, 90)),
            "p99_ms": float(np.percentile(lat, 99)), "clients": clients}


def build_arms(specs):
    arms = []
    for spec in specs:
        if spec == "none":
            arms.append(("none", None))
        else:
            name, path = spec.split("=", 1)
            arms.append((name, load_arm(json.loads(Path(path).read_text()))))
    return arms


def main():
    from simulations.approximate_hits.workloads import get
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True)
    ap.add_argument("--arm", action="append", required=True, help="none | name=<timed run json>")
    ap.add_argument("--clients", type=int, nargs="+", default=[16])
    ap.add_argument("--backend-procs", type=int, default=8)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--nodes", type=int, default=None, help="backend node cache (default: the workload's base node cache)")
    ap.add_argument("--io-complete", action="store_true",
                    help="count the recorded background work (audits, candidate and label fetches) as backend searches; "
                         "old recordings: audit_pos at k 10 and misses of arms named ours* with k = --k-deep")
    ap.add_argument("--k-deep", type=int, default=128)
    ap.add_argument("--legacy-alignment", action="store_true",
                    help="old behaviour: warm-up min(20000, len // 5), first n requests, no alignment checks")
    ap.add_argument("--no-evict", action="store_true", help="old behaviour: no page cache eviction between arms")
    ap.add_argument("--repetitions", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0, help="arm order and query sample")
    ap.add_argument("--timeout-s", type=float, default=6 * 3600, help="total time limit of the run")
    a = ap.parse_args()
    deadline = time.monotonic() + a.timeout_s
    w = get(a.workload)
    info = w.backend_info()
    info["nodes"] = w.node_cache() if a.nodes is None else a.nodes
    stream = np.asarray(np.load(w.prep_dir / "streams.npz")[w.stream], dtype=np.int64)
    arms = build_arms(a.arm)
    if a.legacy_alignment:
        warm = min(20000, len(stream) // 5)
        post = stream[warm:]
        n = min(a.n, len(post), *[len(t["hit"]) for _, t in arms if t])
        qids = post[:n]
        sel = {name: np.arange(n) for name, _ in arms}
        starts = {name: warm for name, _ in arms}
    else:
        check_alignment(arms)
        recorded = [t for _, t in arms if t]
        warm = recorded[0]["warmup"] if recorded else min(20000, len(stream) // 5)
        off = common_sample(len(stream) - warm, a.n, a.seed)
        positions = warm + off
        n = len(positions)
        qids = stream[positions]
        sel = {}
        for name, t in arms:
            if t is None:
                continue
            idx = np.searchsorted(t["pos"], positions)
            if (idx >= len(t["pos"])).any() or not np.array_equal(t["pos"][np.minimum(idx, len(t["pos"]) - 1)], positions):
                raise SystemExit(f"arm {name}: recording does not cover the sampled positions")
            sel[name] = idx
    ctx = mp.get_context("fork")
    jobs, done = ctx.Queue(), ctx.Queue()
    procs = [ctx.Process(target=worker, args=(info, str(query_file(w)), jobs, done, np.int8 if w.name.startswith("spacev") else np.float32), daemon=True)
             for _ in range(a.backend_procs)]
    for p in procs:
        p.start()
    try:
        for _ in procs:
            get_message(done, procs, deadline, True)
        logger.info(f"{w.name}: {a.backend_procs} backend processes ready; L {info['L']} beam {info['W']}; {n} requests")

        def warm_page_cache():
            for i in range(300):
                jobs.put((-1, int(qids[i % n]), 10))
            for i in range(300):
                get_message(done, procs, deadline, True)

        index_paths = [info["index_dir"]]
        if a.no_evict:
            warm_page_cache()
        out = []
        for c in a.clients:
            for rep in range(a.repetitions):
                order = arms if a.no_evict else pagecache.shuffled_arms(arms, a.seed, rep)
                for name, t in order:
                    state = {}
                    if not a.no_evict:
                        state["evict"] = pagecache.evict(index_paths)
                        warm_page_cache()
                    before = pagecache.residency(index_paths)
                    hit = np.zeros(n, bool) if t is None else t["hit"][sel[name]]
                    dt = np.zeros(n) if t is None else t["dt_ms"][sel[name]] / 1e3
                    aud = None
                    plan, km = [], 10
                    if t is not None and a.io_complete:
                        plan, rec_k = background_plan(t["background"], n, t["total"])
                        if t["background"]:
                            km = rec_k or 10
                        else:
                            km = a.k_deep if name.startswith("ours") else 10
                            if t["audit_pos"] is not None:
                                aud = np.isin(t["pos"][sel[name]], t["audit_pos"])
                    r = run_arm(name, hit, dt, qids, c, jobs, done, procs, deadline, audit=aud, k_miss=km, background=plan)
                    r["nodes"] = info["nodes"]
                    r["repetition"] = rep
                    r["order_seed"] = a.seed
                    r["warmup"] = warm
                    r["page_cache"] = pagecache.record(dict(before, **state), pagecache.residency(index_paths))
                    out.append(r)
                    logger.info(f"clients {c:3d} {name:12s} hit {100 * r['hit_rate']:5.1f}% bg {r['background_searches']} audits {r['audits']} k {km}  "
                                f"{r['qps']:7.1f} QPS  mean {r['mean_ms']:.1f} ms  p50 {r['p50_ms']:.1f}  p99 {r['p99_ms']:.1f}")
        for _ in procs:
            jobs.put(None)
    except BaseException:
        for p in procs:
            if p.is_alive():
                p.terminate()
        raise
    path = Path(a.out or RES / "throughput" / f"replay_{w.name}.json")
    write_json_atomic(path, out)
    logger.info(f"wrote {path}")


if __name__ == "__main__":
    main()
