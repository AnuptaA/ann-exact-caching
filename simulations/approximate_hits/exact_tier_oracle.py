#!/usr/bin/env python3
"""
Offline oracle for a second-tier exact-repeat cache (context doc 9as).

The tier maps a query to the backend's answer for it (stored when the backend answered that query: a miss of the
main cache, an audited hit, or a warm-up request) and serves later exact repeats of it; it sits in front of the main
cache and is outside the hit model and calibration. Question: what share x of the memory budget B (at most 10%)
should go to it? Giving it x costs the main cache x of its memory and gains the tier's hits on requests the main
cache misses.

Inputs: two logged runs of the main cache (online_sim --log-served) at budgets B and 0.9 B. Per x:
  extra(x)  hits of the tier on requests the main cache (at B) misses, tier capacity C = x B / cost per entry;
            two tier policies: Belady (perfect knowledge of the future: keep the entries whose next useful
            request, an exact repeat the main cache misses, comes soonest; bypass when the newcomer's is later)
            and LRU (admit every backend answer, any request of the query refreshes it)
  loss(x)   the main cache's hits lost by shrinking it to (1 - x) B: linear between the B and 0.9 B runs
  total(x)  = hits(B) - loss(x) + extra(x), as a share of the requests after the warm-up
Cost per entry: --entry-cost vectors (default k = 10: the answer's vectors, the main cache's accounting; an
ids-only entry, a query hash plus 10 ids, is ~50 bytes ~ 0.016 vectors at 769-d float32).
Approximations: the tier's extra hits are counted against the misses of the main cache at B (at (1 - x) B it misses a
little more, so extra is slightly low), and the main cache is assumed unaffected by the tier taking some requests.

  .venv/bin/python simulations/approximate_hits/exact_tier_oracle.py --workload tripclick-ip
"""

import json
import heapq
import argparse
from pathlib import Path
from collections import OrderedDict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = Path("simulations/approximate_hits/results")
INF = 1 << 60


def load_run(path):
    r = json.loads(Path(path).read_text())
    a = r["args"]
    return r, np.array(r["served_pos"], dtype=np.int64), np.array(r["audit_pos"], dtype=np.int64), a


def next_useful(q, useful):
    """For each position t: the next position t' > t with q[t'] == q[t] and useful[t'] (else INF)."""
    nxt = np.full(len(q), INF, dtype=np.int64)
    last = {}
    for t in range(len(q) - 1, -1, -1):
        nxt[t] = last.get(int(q[t]), INF)
        if useful[t]:
            last[int(q[t])] = t
    return nxt


def belady(q, answered, miss, nxt, C, warmup):
    """Extra hits (requests >= warmup that the main cache misses and the tier serves) of the offline-optimal tier
    of C entries with bypass."""
    if C <= 0:
        return 0
    key = {}                     # query -> its current next useful position
    heap = []                    # (-next use, query), lazily invalidated
    extra = 0
    for t in range(len(q)):
        x = int(q[t])
        if x in key:
            if t >= warmup and miss[t]:
                extra += 1
            key[x] = int(nxt[t])
            heapq.heappush(heap, (-key[x], x))
            continue
        if not answered[t] or nxt[t] >= INF:
            continue
        nu = int(nxt[t])
        if len(key) >= C:
            while heap and key.get(heap[0][1]) != -heap[0][0]:
                heapq.heappop(heap)          # stale
            far, y = -heap[0][0], heap[0][1]
            if far <= nu:
                continue                     # bypass: every cached entry is needed sooner
            heapq.heappop(heap)
            del key[y]
        key[x] = nu
        heapq.heappush(heap, (-nu, x))
    return extra


def lru(q, answered, miss, C, warmup):
    if C <= 0:
        return 0
    od = OrderedDict()
    extra = 0
    for t in range(len(q)):
        x = int(q[t])
        if x in od:
            od.move_to_end(x)
            if t >= warmup and miss[t]:
                extra += 1
        elif answered[t]:
            od[x] = True
            if len(od) > C:
                od.popitem(last=False)
    return extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="tripclick-ip")
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.01, 0.1, 0.3])
    ap.add_argument("--variants", nargs="+", default=["lru_R128", "og_R128"])
    ap.add_argument("--entry-cost", type=float, default=10.0, help="memory per tier entry, in vectors")
    ap.add_argument("--xs", type=float, nargs="+", default=[0, .0025, .005, .01, .015, .02, .03, .04, .05, .06, .08, .1])
    a = ap.parse_args()
    root = RES / "exact_oracle" / a.workload
    n_db = np.load(f"datasets/aker/{a.workload}-docs.npy", mmap_mode="r").shape[0]
    out = {}
    fig, axes = plt.subplots(len(a.variants), len(a.budgets), figsize=(5 * len(a.budgets), 3.8 * len(a.variants)),
                             squeeze=False)
    for i, v in enumerate(a.variants):
        for j, B in enumerate(a.budgets):
            fB = root / f"b{B:g}" / v
            f9 = root / f"b{0.9 * B:g}" / v
            js = [next(iter(sorted(d.glob("*.json"))), None) for d in (fB, f9)]
            if not all(js) or not all((d / "DONE").exists() for d in (fB, f9)):
                print(f"{v} {B:g}: runs missing")
                continue
            rB, sB, aB, args = load_run(js[0])
            r9, s9, _, _ = load_run(js[1])
            streams = np.load(Path(args["prep"]) / "streams.npz")
            q = streams[args["stream"]][:args["max_queries"]].astype(np.int64)
            W = args["warmup"]
            n = len(q) - W
            hit = np.zeros(len(q), bool)
            hit[sB] = True
            aud = np.zeros(len(q), bool)
            aud[aB] = True
            miss = ~hit
            miss[:W] = False                      # warm-up requests are not counted
            answered = np.zeros(len(q), bool)
            answered[:W] = True
            answered[W:] = miss[W:] | aud[W:]
            nxt = next_useful(q, miss)
            hB, h9 = len(sB) / n, len(s9) / n
            cap = a.entry_cost / (B * n_db)       # share of the budget per entry
            rows = []
            for x in a.xs:
                C = int(x / cap)
                eb, el = belady(q, answered, miss, nxt, C, W) / n, lru(q, answered, miss, C, W) / n
                loss = (hB - h9) * x / 0.1
                rows.append({"x": x, "entries": C, "extra_belady": eb, "extra_lru": el, "main_loss": loss,
                             "total_belady": hB - loss + eb, "total_lru": hB - loss + el})
                print(f"{v} {100 * B:g}% x {100 * x:5.2f}% C {C:6d}  extra belady {100 * eb:5.2f} lru {100 * el:5.2f}"
                      f"  loss {100 * loss:5.2f}  total belady {100 * (hB - loss + eb):5.2f} lru {100 * (hB - loss + el):5.2f}",
                      flush=True)
            unl = belady(q, answered, miss, nxt, len(q), W) / n
            out[f"{v}/{B:g}"] = {"hit_B": hB, "hit_0.9B": h9, "extra_unlimited": unl, "entry_cost": a.entry_cost,
                                 "rows": rows}
            ax = axes[i][j]
            xs = [100 * r["x"] for r in rows]
            ax.plot(xs, [100 * r["total_belady"] for r in rows], "o-", color="#D55E00", label="total, oracle tier")
            ax.plot(xs, [100 * r["total_lru"] for r in rows], "s-", color="#0072B2", label="total, LRU tier")
            ax.plot(xs, [100 * (hB - r["main_loss"]) for r in rows], "--", color="grey", label="main cache alone")
            ax.axhline(100 * hB, color="k", lw=0.8)
            ax.set_title(f"{v}, cache {100 * B:g}%  (no tier {100 * hB:.1f}%; unlimited tier +{100 * unl:.1f})",
                         fontsize=9, loc="left")
            ax.set_xlabel("share of the budget for the exact tier (%)")
            ax.set_ylabel("hit rate (%)")
            ax.grid(alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.suptitle(f"{a.workload}: exact-repeat tier, {a.entry_cost:g} vectors per entry", fontsize=11)
    fig.tight_layout()
    tag = f"_cost{a.entry_cost:g}"
    fig.savefig(root / f"exact_tier{tag}.png", dpi=130)
    (root / f"exact_tier{tag}.json").write_text(json.dumps(out, indent=1))
    print("wrote", root / f"exact_tier{tag}.png")


if __name__ == "__main__":
    main()
