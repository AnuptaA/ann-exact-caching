#!/usr/bin/env python3
"""Closed-loop throughput of a cache in front of the real DiskANN backend (doc 9ap), by replay.

A timed cache run (online_sim --timing-every 1) recorded, for every request after the warm-up, whether it was
a hit and how long the cache path took (dt_ms). This script replays those requests in order:
  - one "cache thread" (the main process) handles requests one at a time and spends each request's measured
    cache-path time (a busy wait; hits finish here);
  - a miss is sent to a pool of backend worker processes that run the REAL DiskANN search (answers discarded;
    DiskANN's Python binding holds the GIL, hence processes), so SSD reads and CPU are real;
  - closed loop: at most --clients requests are in flight; the next request starts when one finishes.
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
import multiprocessing as mp
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import numpy as np

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
    reader = None
    while True:
        job = jobs.get()
        if job is None:
            return
        i, qi, k = job
        if i == -3:                       # doc 9as: re-read qi vectors (random 4 KB sectors of the disk index, O_DIRECT)
            if reader is None:
                reader = SectorReader(str(Path(info["index_dir"]) / "ann_disk.index"))
            reader.read(qi)
            done.put(("bg", i, time.perf_counter()))
            continue
        # a deep list of k > L needs a search list of at least k (SPACEV / ESCI: L 100 < 128)
        ix.search(np.ascontiguousarray(q[qi], dtype=dtype), k_neighbors=k, complexity=max(info["L"], k),
                  beam_width=info["W"])
        done.put(("bg" if i < -1 else "done", i, time.perf_counter()))


class SectorReader:
    """Random 4 KB reads of the disk index with O_DIRECT (no page cache), 8 at a time: the cost of re-reading
    vectors chosen for the cache (one read per vector: an upper bound when several nodes share a sector)."""
    def __init__(self, path, threads=8):
        import os
        import mmap
        from concurrent.futures import ThreadPoolExecutor
        self.os = os
        self.fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        self.n_sec = os.fstat(self.fd).st_size // 4096
        self.bufs = [mmap.mmap(-1, 4096) for _ in range(threads)]
        self.pool = ThreadPoolExecutor(threads)
        self.rng = np.random.default_rng()

    def read(self, m):
        secs = self.rng.integers(1, self.n_sec, size=int(m))
        nb = len(self.bufs)
        list(self.pool.map(lambda j: [self.os.preadv(self.fd, [self.bufs[j]], int(x) * 4096) for x in secs[j::nb]],
                           range(nb)))


def spin(seconds):
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


def run_arm(name, hit, dt, qids, clients, jobs, done, audit=None, k_miss=10, fetch=None):
    """audit (doc 9as, I/O-complete): requests whose hit is audited; each audit is a real backend search issued as
    background work (it occupies a backend process but no client waits for it). Misses search with k_miss results
    (a deep list taken from the answer search costs the same at L >= k_miss). Wall time runs until the background
    work is done too."""
    n = len(qids)
    issue = np.zeros(n)
    finish = np.zeros(n)
    inflight = 0
    ndone = 0
    bg_out = 0

    def drain(block):
        nonlocal inflight, ndone, bg_out
        try:
            kind, i, t = done.get(block=block, timeout=60 if block else None) if block else done.get_nowait()
        except Exception:
            return False
        if kind == "bg":
            bg_out -= 1
            return True
        finish[i] = t
        inflight -= 1
        ndone += 1
        return True

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
                jobs.put((-2 - i, int(qids[i]), 10))
                bg_out += 1
        else:
            jobs.put((i, int(qids[i]), k_miss))
            inflight += 1
        if fetch is not None and fetch[i] > 0:
            m = int(fetch[i])
            for c0 in range(0, m, 64):            # background vector re-reads, in chunks of 64
                jobs.put((-3, min(64, m - c0), 0))
                bg_out += 1
    while inflight > 0 or bg_out > 0:
        drain(True)
    wall = time.perf_counter() - t0
    lat = (finish - issue) * 1e3
    return {"arm": name, "n": n, "wall_s": wall, "qps": n / wall, "hit_rate": float(np.mean(hit)),
            "audits": int(audit.sum()) if audit is not None else 0, "k_miss": k_miss,
            "vector_rereads": int(fetch.sum()) if fetch is not None else 0,
            "mean_ms": float(lat.mean()), "p50_ms": float(np.percentile(lat, 50)), "p90_ms": float(np.percentile(lat, 90)),
            "p99_ms": float(np.percentile(lat, 99)), "clients": clients}


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
                    help="doc 9as: audited hits of runs logged with --log-served become background searches; misses of "
                         "arms named ours* search with k = --k-deep")
    ap.add_argument("--k-deep", type=int, default=128)
    ap.add_argument("--no-reread", action="store_true", help="--io-complete without the vector re-reads")
    a = ap.parse_args()
    w = get(a.workload)
    info = w.backend_info()
    info["nodes"] = w.node_cache() if a.nodes is None else a.nodes
    stream = np.asarray(np.load(w.prep_dir / "streams.npz")[w.stream], dtype=np.int64)
    warm = min(20000, len(stream) // 5)
    post = stream[warm:]
    arms = []
    for spec in a.arm:
        if spec == "none":
            arms.append(("none", None))
        else:
            name, path = spec.split("=", 1)
            r = json.loads(Path(path).read_text())
            t = dict(r["timing"])
            if a.io_complete and "audit_pos" in r:
                t["audit_post"] = np.asarray(r["audit_pos"], dtype=np.int64) - r["args"]["warmup"]
            if a.io_complete and not a.no_reread and r["log"] and "fetched" in r["log"][0]:
                # vectors the cache read for chosen candidates (ball members + shell samples), per batch: re-read
                # from disk at the batch's end (candidates are kept as ids until chosen)
                W = r["args"]["warmup"]
                # shell-sample vectors: samp_new (11-21: only ones no entry sampled already) where logged, else
                # samp_fetched (every re-coding counted: ~1.4x too many on TripClick 10%)
                cum = np.array([x["fetched"] + x.get("samp_new", x["samp_fetched"]) for x in r["log"]], dtype=np.int64)
                tt = np.array([x["t"] for x in r["log"]], dtype=np.int64) - W - 1
                t["reread_at"] = (tt[1:], np.diff(cum))
            arms.append((name, t))
    n = min(a.n, len(post), *[len(t["hit"]) for _, t in arms if t])
    qids = post[:n]
    ctx = mp.get_context("fork")
    jobs, done = ctx.Queue(), ctx.Queue()
    procs = [ctx.Process(target=worker, args=(info, str(query_file(w)), jobs, done, np.int8 if w.name.startswith("spacev") else np.float32), daemon=True)
             for _ in range(a.backend_procs)]
    for p in procs:
        p.start()
    for _ in procs:
        done.get(timeout=3600)
    logger.info(f"{w.name}: {a.backend_procs} backend processes ready; L {info['L']} beam {info['W']}; {n} requests")
    import os
    flag = os.environ.get("QUIET_FLAG")      # doc 9at: the arms need a quiet machine (quiet_tp.sh pauses other jobs once set)
    if flag:
        Path(flag).touch()
        time.sleep(15)
    for i in range(300):                                       # warm the page cache with real misses
        jobs.put((-1, int(qids[i]), 10))
    for i in range(300):
        done.get(timeout=600)
    out = []
    for c in a.clients:
        for name, t in arms:
            hit = np.zeros(n, bool) if t is None else np.asarray(t["hit"][:n], bool)
            dt = np.zeros(n) if t is None else np.asarray(t["dt_ms"][:n]) / 1e3
            aud = None
            if t is not None and "audit_post" in t:
                aud = np.zeros(n, bool)
                ap_ = t["audit_post"]
                aud[ap_[(ap_ >= 0) & (ap_ < n)]] = True
            km = a.k_deep if (a.io_complete and name.startswith("ours")) else 10
            fe = None
            if t is not None and "reread_at" in t:
                fe = np.zeros(n, np.int64)
                at, m = t["reread_at"]
                keep = (at >= 0) & (at < n)
                np.add.at(fe, at[keep], m[keep])
            r = run_arm(name, hit, dt, qids, c, jobs, done, audit=aud, k_miss=km, fetch=fe)
            r["nodes"] = info["nodes"]
            out.append(r)
            logger.info(f"clients {c:3d} {name:12s} hit {100 * r['hit_rate']:5.1f}% audits {r['audits']} rereads {r['vector_rereads']} k {km}  {r['qps']:7.1f} QPS  "
                        f"mean {r['mean_ms']:.1f} ms  p50 {r['p50_ms']:.1f}  p99 {r['p99_ms']:.1f}")
    if flag:
        Path(flag).unlink(missing_ok=True)
    for _ in procs:
        jobs.put(None)
    path = Path(a.out or RES / "throughput" / f"replay_{w.name}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    logger.info(f"wrote {path}")


if __name__ == "__main__":
    main()
