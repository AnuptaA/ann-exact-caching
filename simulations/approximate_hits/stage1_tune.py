#!/usr/bin/env python3
"""
Adaptive tuning of the cache's hit path (context doc 9ae, decided with Jialin 11-16).

The hit DECISION uses only the kNN radius (distance to the 10th nearest cached vector) and the
intruder count measured against it; the neighbors' identities matter only for the answer served on a
hit. So the lookup is split in two stages:
  stage 1 (every request)   cheap search over (possibly compressed) vectors -> radius -> score -> hit?
  stage 2 (hits only)       a more accurate search for the 10 points to serve.
Both are tuned ADAPTIVELY on warm-up requests against exact (brute-force) search, going down instead
of searching a grid:
  stage 1: for each vector precision, from the finest (fp32) down (8-bit, 6-bit, 4-bit scalar
           quantization, candidates re-ranked exactly with the fp32 vectors that the cache stores
           anyway): halve the search list from 512 while the decisions stay accurate, i.e. while
           (a) >= AGREE of the requests get the same hit / miss decision as with exact search at the
           simulator's cutoff and (b) the hit rate differs by <= 0.5 points; the coarser precision is
           tried only while the finer one still passed at its lowest search list; the cheapest
           accepted setting (measured ms per request) wins;
  stage 2: lowest search list (fp32 graph, exact re-rank of 5x candidates) whose served points keep the
           mean retention of the exact-hit queries within 0.003 of exact search's.

  .venv-ann/bin/python simulations/approximate_hits/stage1_tune.py --prep .../spacev-aker-0.6 \
      --state .../timing_v4/spacev-aker-0.6/state_b0.1.npz --out .../stage1_b0.1.json
"""

import sys
import json
import time
import argparse
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
RES = Path("simulations/approximate_hits/results")
AGREE, HIT_TOL, RET_TOL = 0.99, 0.005, 0.003
PRECISIONS = ["fp32", "sq8", "sq6", "sq4"]
K_FACTOR = 5          # candidates re-ranked exactly: K_FACTOR x k


def build_index(cv, precision, M=32, ef_construction=200):
    import faiss
    d = cv.shape[1]
    if precision == "fp32":
        ix = faiss.IndexHNSWFlat(d, M)
    else:
        qt = {"sq8": faiss.ScalarQuantizer.QT_8bit, "sq6": faiss.ScalarQuantizer.QT_6bit,
              "sq4": faiss.ScalarQuantizer.QT_4bit}[precision]
        ix = faiss.IndexHNSWSQ(d, qt, M)
        ix.train(cv[np.random.default_rng(0).choice(len(cv), min(len(cv), 100000), replace=False)])
    ix.hnsw.efConstruction = ef_construction
    ix.add(cv)
    return ix


def search(ix, cv, xq, ef, k=10, k_factor=K_FACTOR):
    """Top-k ids and exact squared distances: the index returns k_factor x k candidates, re-ranked
    with the fp32 vectors."""
    ix.hnsw.efSearch = max(ef, k * k_factor)
    _, cand = ix.search(xq, k * k_factor)
    ids = np.empty((len(xq), k), dtype=np.int64)
    d2 = np.empty((len(xq), k))
    for i in range(len(xq)):
        c = cand[i][cand[i] >= 0]
        dd = ((cv[c] - xq[i]) ** 2).sum(axis=1)
        o = np.argsort(dd)[:k]
        ids[i], d2[i] = c[o], dd[o]
    return ids, d2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--stream", default="trace")
    ap.add_argument("--target-recall", default="0.95")
    ap.add_argument("--n", type=int, default=600, help="warm-up requests used")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import faiss
    from simulations.approximate_hits.eval_timing import FastHitPath
    from simulations.approximate_hits.prep_io import load_prep
    prep = Path(args.prep)
    db, q, _, _ = load_prep(prep)
    st = dict(np.load(args.state, allow_pickle=False))
    ans = np.load(RES / "backends" / prep.name / "diskann.npz")[f"answer_{args.target_recall}"][:, :10]
    w = int(st["warmup"])
    req = np.asarray(np.load(prep / "streams.npz")[args.stream][:w], dtype=np.int64)[-args.n:]
    xq = np.ascontiguousarray(np.asarray(q[req], dtype=np.float32))
    hp = FastHitPath(db, q, st, "brute", "balls10")       # balls / shell parts and the model (no graph built)
    cv = np.ascontiguousarray(hp.cv, dtype=np.float32)
    ids_of = hp.ids
    lam = float(st["lam"])
    faiss.omp_set_num_threads(args.threads)
    _, gt = faiss.knn(xq, cv, 10)
    d2_ex = ((cv[gt] - xq[:, None, :]) ** 2).sum(axis=2)
    faiss.omp_set_num_threads(1)
    T = {"knn": 0.0, "balls": 0.0, "shell": 0.0, "model": 0.0}

    def scores(d2):
        out = []
        for x, dd in zip(xq, d2):
            out.append(_score(hp, x, float(np.sqrt(max(dd.max(), 0.0))) + 1e-9, T))
        return np.array(out)

    s_ex = scores(d2_ex)
    hit_ex = s_ex >= lam
    ov_ex = np.array([len(set(ids_of[g].tolist()) & set(ans[qi].tolist())) / 10 for g, qi in zip(gt, req)])
    logger.info(f"{prep.name}: {len(cv)} cached vectors; exact hit rate on {len(req)} warm-up requests {hit_ex.mean():.3f}")
    res = {"prep": prep.name, "n_cached": int(len(cv)), "n_requests": int(len(req)), "exact_hit_rate": float(hit_ex.mean()),
           "stage1_trials": [], "stage2_trials": []}

    # ---- stage 1: descend in search list, then in precision
    best = None
    for prec in PRECISIONS:
        t0 = time.time()
        ix = build_index(cv, prec)
        logger.info(f"stage 1, {prec}: index built in {time.time() - t0:.0f} s")
        last_ok = None
        ef = 512
        while ef >= 16:
            t = time.perf_counter()
            ids, d2 = search(ix, cv, xq, ef)
            ms = (time.perf_counter() - t) / len(xq) * 1e3
            s = scores(d2)
            hit = s >= lam
            agree, dh = float((hit == hit_ex).mean()), float(hit.mean() - hit_ex.mean())
            ratio = np.sqrt(d2.max(axis=1) / np.maximum(d2_ex.max(axis=1), 1e-12))
            ok = agree >= AGREE and abs(dh) <= HIT_TOL
            res["stage1_trials"].append({"precision": prec, "ef": ef, "ms": ms, "agree": agree, "dhit": dh,
                                         "radius_ratio_p95": float(np.percentile(ratio, 95)), "ok": ok})
            logger.info(f"  {prec} ef {ef}: agree {agree:.3f}, hit rate {hit.mean():.3f} ({dh:+.3f}), radius p95 x{np.percentile(ratio, 95):.3f}, "
                        f"{ms:.2f} ms -> {'ok' if ok else 'FAIL'}")
            if not ok:
                break
            last_ok = {"precision": prec, "ef": ef, "ms": ms}
            if best is None or ms < best["ms"]:
                best = dict(last_ok)
            ef //= 2
        del ix
        if last_ok is None:
            break                                      # even the highest search list failed: coarser will too
    res["stage1"] = best
    logger.info(f"stage 1 choice: {best}")

    # ---- stage 2: served points of the exact-hit queries, fp32 graph, exact re-rank
    ix = build_index(cv, "fp32")
    mask = hit_ex
    for ef in (2048, 1024, 512, 256, 128, 64, 32):
        ids, d2 = search(ix, cv, xq, ef)
        ov = np.array([len(set(ids_of[g].tolist()) & set(ans[qi].tolist())) / 10 for g, qi in zip(ids, req)])
        gap = float(ov_ex[mask].mean() - ov[mask].mean())
        ok = gap <= RET_TOL
        res["stage2_trials"].append({"ef": ef, "retention_gap": gap, "ok": ok})
        logger.info(f"stage 2 ef {ef}: retention of exact-hit queries {ov[mask].mean():.4f} (exact {ov_ex[mask].mean():.4f}) -> {'ok' if ok else 'FAIL'}")
        if not ok:
            break
        res["stage2"] = {"ef": ef, "retention_gap": gap}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    logger.info(f"wrote {args.out}")


def _score(hp, x, nr, t):
    """FastHitPath.decide after the kNN stage: balls, shell count, model score for a given radius."""
    from simulations.approximate_hits.eval_timing import N_BALLS, POOR_FIT
    x_sq = float(x @ x)
    de = np.sqrt(np.maximum(hp.cq_sq - 2.0 * (hp.cq @ x) + x_sq, 0.0))
    fit = (nr + de) / hp.radius
    m = min(N_BALLS, len(fit))
    best = np.argpartition(fit, m - 1)[:m] if m < len(fit) else np.arange(m)
    best = best[np.argsort(fit[best])]
    f = np.full(2 * N_BALLS + 1, POOR_FIT)
    f[0:2 * m:2] = de[best] / hp.radius[best]
    f[1:2 * m:2] = nr / hp.radius[best]
    if not len(hp.sv):
        cnt = 0
    else:
        sel = best if hp.nb <= m else (np.argpartition(fit, hp.nb - 1)[:hp.nb] if hp.nb < len(fit) else np.arange(len(fit)))
        pos = np.unique(np.concatenate([hp.es[hp.eo[e]:hp.eo[e + 1]] for e in sel]))
        cnt = int(((hp.sv_sq[pos] - 2.0 * (hp.sv[pos] @ x) + x_sq) < nr * nr).sum()) if len(pos) else 0
    f[-1] = np.log1p(cnt)
    return float(((f - hp.mu) / hp.sd) @ hp.coef + hp.b)


if __name__ == "__main__":
    main()
