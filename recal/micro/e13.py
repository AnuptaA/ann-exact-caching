"""E13: does the center index return the right entries?

For held-out queries: the exact c nearest centers against an HNSW over the centers (degree, ef): recall of the
c-set, whether the best-fit entry of the exact set is in the HNSW's set, how the v11 features (offset, size,
intruder rate with the core's shells) differ when computed from the HNSW's set, and the time per query.
"""

import time

import numpy as np

from recal.micro import e4, e10, intervals as iv, world as w

NAME = "E13"
CS = (8, 16, 32)
DEGREES = (8, 16)
EFS = (32, 64, 128, 256)
EF_CONSTRUCTION = 200
MAX_QUERIES = 3000
TARGET = 0.99
SAME = 1e-9


def queries(world: w.World) -> np.ndarray:
    vec = world.heldout()["vec"]
    if len(vec) > MAX_QUERIES:
        vec = vec[np.linspace(0, len(vec) - 1, MAX_QUERIES).astype(int)]
    return np.ascontiguousarray(vec, dtype=np.float32)


def best_fit_entries(feats: iv.Features, Q: np.ndarray, sets: np.ndarray, nr: np.ndarray) -> np.ndarray:
    """Best-fit entry of each row's set: argmin (d(q, c_j) + nr) / r_j."""
    centers, radius = feats.centers, feats.radius
    out = np.zeros(len(Q), np.int64)
    for i, S in enumerate(sets):
        d = np.linalg.norm(centers[S].astype(np.float64) - Q[i].astype(np.float64), axis=1)
        out[i] = S[np.argmin((d + nr[i]) / radius[S])]
    return out


def run(world: w.World, units=None) -> tuple[list[dict], dict]:
    feats = iv.Features(world, c=max(CS))
    if len(feats.centers) < max(CS) or len(feats.pool) < world.k:
        for degree in DEGREES:
            if units is not None:
                units.cached(f"degree{degree}", list)
        return [], dict(workload=world.name, budget=world.budget, note="fewer entries than c or pool smaller than K")
    Q = queries(world)
    Dc = iv._dist(Q, feats.centers)
    order = np.argsort(Dc, axis=1)
    nr = np.zeros(len(Q))
    for lo, hi in feats._blocks(len(Q)):
        nr[lo:hi] = np.partition(iv._dist(Q[lo:hi], feats.pool), world.k - 1, axis=1)[:, world.k - 1]
    exact_sets = {c: order[:, :c] for c in CS}
    reference = {c: feats.exact(Q, nr=nr, sets=list(exact_sets[c])) for c in CS}
    best = {c: best_fit_entries(feats, Q, exact_sets[c], nr) for c in CS}
    rows = []
    for degree in DEGREES:
        key = f"degree{degree}"
        compute = lambda degree=degree: degree_rows(world, feats, Q, nr, exact_sets, reference, best, degree)  # noqa
        rows += units.cached(key, compute, queries=len(Q)) if units is not None else compute()
    return rows, summary(world.name, world.budget, rows)


def degree_rows(world, feats, Q, nr, exact_sets, reference, best, degree) -> list[dict]:
    import hnswlib
    index = hnswlib.Index(space="l2", dim=feats.centers.shape[1])
    index.init_index(max_elements=len(feats.centers), ef_construction=EF_CONSTRUCTION, M=degree)
    index.add_items(feats.centers, np.arange(len(feats.centers)))
    index.set_num_threads(1)
    rows = []
    for ef in EFS:
        for c in CS:
            index.set_ef(max(ef, c))
            t0 = time.perf_counter()
            labels, _ = index.knn_query(Q, k=c, num_threads=1)
            per_query = (time.perf_counter() - t0) / len(Q)
            labels = labels.astype(np.int64)
            got = feats.exact(Q, nr=nr, sets=list(labels))
            ref = reference[c]
            diff = np.abs(got - ref)
            same_best = np.array([(best[c][i] == labels[i]).any() for i in range(len(Q))])
            identical = (diff <= SAME).all(axis=1)
            overlap = np.array([len(np.intersect1d(a, b)) for a, b in zip(labels, exact_sets[c])]) / c
            rows.append(dict(
                workload=world.name, budget=world.budget, c=c, degree=degree, ef=ef, n_queries=len(Q),
                n_entries=len(feats.centers), set_recall=float(overlap.mean()),
                best_fit_in_set_share=float(same_best.mean()), features_identical_share=float(identical.mean()),
                both_share=float((same_best & identical).mean()), offset_abs_diff_p95=e4.pct(diff[:, 0], 95),
                size_abs_diff_p95=e4.pct(diff[:, 1], 95), rate_abs_diff_p95=e4.pct(diff[:, 2], 95),
                us_per_query=per_query * 1e6))
    return rows


def summary(name: str, budget: float, rows: list[dict]) -> dict:
    """Per c: the smallest degree (then ef) with the best fit in the set and identical features for 99% of queries."""
    out = {}
    for c in CS:
        ok = [r for r in rows if r["c"] == c and r["best_fit_in_set_share"] >= TARGET
              and r["features_identical_share"] >= TARGET]
        best = min(ok, key=lambda r: (r["degree"], r["ef"])) if ok else None
        out[f"c{c}"] = None if best is None else dict(degree=best["degree"], ef=best["ef"],
                                                       us_per_query=best["us_per_query"])
    return dict(workload=name, budget=budget, n_entries=rows[0]["n_entries"] if rows else 0,
                smallest_for_99=out)
