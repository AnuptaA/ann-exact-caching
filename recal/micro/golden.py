"""Golden comparison: every quantity the experiments reimplement, computed both ways on a sample of queries.

The core (candidates_batch, lookup, entries_read_batch, entries_needed_batch) is compared with a float64 brute-force
reference built from its own entries and decoded shell codes. A mismatch above tolerance stops the run with a message
naming the quantity. It runs on a built cache before any experiment uses it (preflight and world_budget).
"""

import numpy as np

from recal import policy
from recal.micro import fit, intervals, intruder_check, topk, world as w

N_HELD = 200
N_CAL = 100
SCAN_ROWS = 3_000_000       # corpora up to this size also get a full exact scan for a few queries
BOUNDARY = 1e-9
TOL = dict(distance=1e-6, feature=1e-9, score=1e-9, features_all=1e-6)


class GoldenFailure(SystemExit):
    pass


def sample_queries(world: w.World, n_held: int, n_cal: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    held = world.heldout()
    h = rng.permutation(len(held["vec"]))[:n_held]
    cal_vec = world.role_rows("cal")[1]
    c = rng.permutation(len(cal_vec))[:n_cal]
    Q = np.vstack([held["vec"][h], cal_vec[c]]).astype(np.float32)
    truth = np.vstack([held["truth"][h], np.asarray(world.snap["ids"][world.snap["k"] == world.k]
                                                    [world.history()["role"] == w.ROLE_CODES["cal"]][c][:, :world.k])])
    return Q, truth.astype(np.int64)


def reference(world: w.World, entries: list[dict], q: np.ndarray, read: np.ndarray, decoded: np.ndarray,
              shell_slots: list[np.ndarray]) -> dict:
    """float64 definition of the main path for one query and the entries the core read."""
    K = world.k
    centers = np.asarray([e["center"] for e in entries], dtype=np.float32)
    radius = np.asarray([e["radius"] for e in entries], dtype=np.float64)
    q64 = q.astype(np.float64)
    d_c = np.linalg.norm(centers[read].astype(np.float64) - q64, axis=1)
    ids = np.unique(np.concatenate([np.asarray(entries[e]["response_ids"], dtype=np.int64) for e in read]))
    out = dict(n_candidates=len(ids), has=len(ids) >= K)
    if not out["has"]:
        return out
    dist = np.linalg.norm(np.asarray(world.db[ids], dtype=np.float64) - q64, axis=1)
    order = np.lexsort((ids, dist))
    nr = max(float(dist[order[K - 1]]), 1e-12)
    tie = len(order) > K and abs(dist[order[K]] - dist[order[K - 1]]) <= 1e-12 * max(nr, 1.0)
    fit_all = (d_c + nr) / np.maximum(radius[read], 1e-12)
    best = int(np.argmin(fit_all))                  # first minimum, like the core's strict <
    slots = np.unique(np.concatenate([shell_slots[e] for e in read])) if len(read) else np.zeros(0, np.int64)
    sq = ((decoded[slots] - q64) ** 2).sum(axis=1) if len(slots) else np.zeros(0)
    count = int((sq < nr * nr).sum())
    boundary = bool(len(sq) and (np.abs(sq / (nr * nr) - 1.0) < BOUNDARY).any())
    rad = max(float(radius[read[best]]), 1e-12)
    out.update(ids=ids[order[:K]], dists=dist[order[:K]], nr=nr, tie_at_k=bool(tie), best=int(read[best]),
               offset=float(d_c[best]) / rad, size=nr / rad, count=count, rate=float(np.log((count + 1.0) / K)),
               boundary=boundary)
    return out


def rank_of_needed(entries: list[dict], q: np.ndarray, truth: np.ndarray, centers: np.ndarray) -> int:
    """entries_needed_batch in Python: rank (1-based, nearest center first) of the entry set reaching every true
    neighbor the pool holds; -1 when none is pooled."""
    d = np.linalg.norm(centers.astype(np.float64) - q.astype(np.float64), axis=1)
    rank = np.empty(len(d), np.int64)
    rank[np.argsort(d, kind="stable")] = np.arange(len(d))
    holders: dict = {}
    for e, entry in enumerate(entries):
        for i in entry["response_ids"]:
            holders.setdefault(int(i), []).append(e)
    best = [min(rank[h] for h in holders[int(i)]) + 1 for i in truth if int(i) in holders]
    return max(best) if best else -1


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def off(a: float, b: float) -> float:
    """Difference relative to max(1, |a|, |b|): the distance expansion in Features has about 1e-7 absolute noise
    near zero (a query equal to a center), which a purely relative test would flag."""
    return abs(a - b) / max(1.0, abs(a), abs(b))


def check(world: w.World, n_held: int = N_HELD, n_cal: int = N_CAL, seed: int = 0, strict: bool = True,
          scan_rows: int = SCAN_ROWS) -> dict:
    """Run the comparison; returns the summary and raises GoldenFailure when strict and anything exceeds tolerance."""
    K, core = world.k, world.core
    if not world.built.n_entries:
        return dict(skipped="no entries")
    early = truth_checks(world, scan_rows)
    if early["truth_failures"]:
        failures = {name: "the stored answers do not describe the corpus vectors"
                    for name in early["truth_failures"]}
        if strict:
            raise GoldenFailure(f"golden comparison failed at {world.name} budget {world.budget:g}: "
                                + ", ".join(failures) + ": the stored top-500 answers do not describe the corpus "
                                "vectors (wrong file or metric); rerun the prep stage")
        return dict(early, failures={}, check_failures=failures, ok=False)
    Q, truth = sample_queries(world, n_held, n_cal, seed)
    m = world.m
    feat, cand_ids, cand_d = core.candidates_batch(Q, K, m)
    reads = np.asarray(core.entries_read_batch(Q, m))
    entries = world.entries()
    lo, step = core.quantizer()
    decoded = intruder_check.decode_codes(core.shell_codes(), lo, step)
    shell_slots = [np.asarray(e["shell_slots"], dtype=np.int64) for e in entries]
    centers = np.asarray([e["center"] for e in entries], dtype=np.float32)
    model = world.built.model
    bad: dict = {}
    stats = dict(n_queries=len(Q), m=int(m), boundary_queries=0, tie_at_k_queries=0, id_set_mismatch=0)

    checks: dict = {}

    def flag(name: str, i: int) -> None:
        bad.setdefault(name, []).append(i)

    def fail_check(name: str, detail: str) -> None:
        checks[name] = detail

    for i, q in enumerate(Q):
        read = reads[i][reads[i] >= 0]
        ref = reference(world, entries, q, read, decoded, shell_slots)
        has = bool(feat[i, 0] == 1.0)
        if has != ref["has"]:
            flag("has_candidate", i)
            continue
        if not has:
            continue
        stats["boundary_queries"] += int(ref["boundary"])
        stats["tie_at_k_queries"] += int(ref["tie_at_k"])
        if not ref["tie_at_k"] and set(cand_ids[i].tolist()) != set(ref["ids"].tolist()):
            flag("candidate_ids", i)
            stats["id_set_mismatch"] += 1
        if (np.abs(np.sort(cand_d[i]) - ref["dists"]) > TOL["distance"] * np.maximum(ref["dists"], 1e-12)).any():
            flag("candidate_distances", i)
        for name, got in (("nr", feat[i, 1]), ("offset", feat[i, 2]), ("size", feat[i, 3])):
            if rel(float(got), ref[name]) > TOL["feature"]:
                flag(name, i)
        if int(feat[i, 5]) != ref["count"] and not ref["boundary"]:
            flag("intruder_count", i)
        if not ref["boundary"] and abs(feat[i, 4] - ref["rate"]) > 1e-12:
            flag("rate", i)
        res = core.lookup(q, K, None, m, False)
        if res.best_entry != ref["best"] and rel(res.offset, ref["offset"]) > TOL["feature"]:
            flag("best_entry", i)
        if res.has_score:
            F = np.array([res.offset, res.size, res.rate])
            want = float(policy.score_features(F[None, :], model)[0])
            if abs(res.score - want) > TOL["score"]:
                flag("score", i)
        recall_ids = float(np.isin(truth[i][:K], cand_ids[i]).mean())
        if abs(recall_ids - float(world.features(q[None, :], truth[i][None, :]).recall[0])) > 1e-12:
            flag("recall_label", i)

    ne = min(50, len(Q))
    needed = np.asarray(core.entries_needed_batch(Q[:ne], truth[:ne]))
    for i in range(ne):
        if int(needed[i]) != rank_of_needed(entries, Q[i], truth[i], centers):
            flag("entries_needed", i)

    try:
        stats.update(all_entries_and_bounds(world, Q[:ne], flag, fail_check))
    except RuntimeError as error:
        fail_check("shell_membership", str(error))
        stats["shell_membership_error"] = str(error)
    stats.update(synthetic_certification(fail_check))
    stats.update(early)
    stats.update(duplicate_diagnostics(world, Q, truth, cand_ids))
    stats["read_set_recall_vs_exact"] = read_set_recall(world, Q[:ne], reads[:ne], centers, m)
    for name in stats.get("truth_failures", []):
        fail_check(name, "the stored answers do not describe the corpus vectors")
    failures = {name: len(set(rows)) for name, rows in bad.items()}
    summary = dict(stats, failures=failures, check_failures=checks, ok=not failures and not checks)
    if (failures or checks) and strict:
        parts = [f"{k}: {v} of {len(Q)} queries disagree" for k, v in failures.items()]
        parts += [f"{k} (whole check): {v}" for k, v in checks.items()]
        raise GoldenFailure("golden comparison failed (core and Python definitions disagree) at "
                            f"{world.name} budget {world.budget:g}: " + "; ".join(parts) +
                            ". Fix the Python side (or the core) before running experiments; "
                            "see recal/micro/golden.py for the tolerances.")
    return summary


def all_entries_and_bounds(world: w.World, Q: np.ndarray, flag, fail_check) -> dict:
    """Features over every entry equal the core at m = 0; bounds at e = 0 contain them and collapse."""
    K, core = world.k, world.core
    feats = intervals.Features(world, c=world.built.n_entries)
    exact = feats.exact(Q)
    feat0 = core.candidates_batch(Q, K, 0)[0]
    shell_boundary = 0
    for i in range(len(Q)):
        if not np.isfinite(exact[i]).all():
            continue
        for col, name in ((2, "all_offset"), (3, "all_size")):
            if off(float(feat0[i, col]), float(exact[i, col - 2])) > TOL["features_all"]:
                flag(name, i)
        rate_core = feat0[i, 4]
        if abs(rate_core - exact[i, 2]) > 1e-9:
            shell_boundary += 1
    if shell_boundary > max(1, len(Q) // 50):
        fail_check("all_rate", f"{shell_boundary} of {len(Q)} queries differ in rate over all entries "
                               f"(allowed {max(1, len(Q) // 50)})")
    low, high = feats.bounds(Q, np.zeros(len(Q)))
    ok = np.isfinite(exact).all(axis=1)
    for i in np.flatnonzero(ok):
        if (exact[i] < low[i] - 1e-9).any() or (exact[i] > high[i] + 1e-9).any():
            flag("interval_soundness", int(i))
        elif (np.abs(high[i] - low[i]) > 1e-9).any():
            flag("interval_collapse", int(i))
    return dict(all_entries_rate_mismatch=shell_boundary)


def synthetic_certification(fail_check) -> dict:
    """certify_worst_case with zero-width intervals equals the plain procedure (fit.certify)."""
    rng = np.random.default_rng(1)
    score = rng.uniform(size=800)
    y = np.clip(0.5 + 0.55 * score + rng.normal(0, 0.1, size=800), 0, 1)
    weights = np.where(rng.uniform(size=800) < 0.2, 10.0, 1.0)
    edges = policy.threshold_edges(score)
    plain = fit.certify(score, score, y, 0.9, 0.05, 30.0, weights)
    wc = intervals.certify_worst_case(edges, score, score, y, 0.9, 0.05, 30.0, weights)
    if plain.lam != wc["lam"]:
        fail_check("certify_worst_case", f"lambda {wc['lam']} against the plain procedure's {plain.lam}")
    return dict(certify_lambda=plain.lam)


def truth_checks(world: w.World, scan_rows: int) -> dict:
    """The stored answers describe the vectors in the corpus file; ties at rank K and an optional exact scan."""
    K = world.k
    held = world.heldout()
    n = min(100, len(held["vec"]))
    q = held["vec"][:n].astype(np.float64)
    top = np.asarray(world.top[held["unique"][:n]])
    depth = min(top.shape[1], K + 10)
    failures, ties = [], 0
    for i in range(n):
        d = np.linalg.norm(np.asarray(world.db[top[i, :depth]], dtype=np.float64) - q[i], axis=1)
        if (np.diff(d) < -1e-5 * np.maximum(d[1:], 1e-12)).any():
            failures.append("truth_order")
            break
        ties += int(depth > K and abs(d[K] - d[K - 1]) <= 1e-9 * max(d[K - 1], 1e-12))
    out = dict(truth_checked=n, truth_ties_at_k=ties, truth_failures=failures)
    if len(world.db) <= scan_rows and n:
        db = np.asarray(world.db, dtype=np.float32)
        for i in range(min(5, n)):
            d = np.linalg.norm(db.astype(np.float64) - q[i], axis=1)
            kth = np.partition(d, K - 1)[K - 1]
            got = np.linalg.norm(np.asarray(world.db[top[i, :K]], dtype=np.float64) - q[i], axis=1)
            if (got > kth * (1 + 1e-6) + 1e-9).any():
                out["truth_failures"] = out["truth_failures"] + ["truth_scan"]
                break
        out["truth_scanned"] = True
    else:
        out["truth_scanned"] = False
    return out


def duplicate_diagnostics(world: w.World, Q: np.ndarray, truth: np.ndarray, cand_ids: np.ndarray) -> dict:
    """Recall by id against recall by distance (a returned duplicate under another id counts as a hit)."""
    K = world.k
    differ, gap = 0, 0.0
    for i in range(len(Q)):
        got = cand_ids[i]
        if (got < 0).any():
            continue
        q = Q[i].astype(np.float64)
        kth = np.linalg.norm(np.asarray(world.db[truth[i][K - 1]], dtype=np.float64) - q)
        d = np.linalg.norm(np.asarray(world.db[got], dtype=np.float64) - q, axis=1)
        by_distance = float((d <= kth * (1 + 1e-9)).mean())
        by_id = float(np.isin(truth[i][:K], got).mean())
        differ += int(abs(by_distance - by_id) > 1e-12)
        gap += by_distance - by_id
    return dict(recall_by_distance_differs_share=differ / max(len(Q), 1), recall_by_distance_minus_id_mean=gap / max(len(Q), 1))


def read_set_recall(world: w.World, Q: np.ndarray, reads: np.ndarray, centers: np.ndarray, m: int) -> float:
    """Share of the exact m nearest centers that the core's HNSW search returned (a diagnostic)."""
    if m <= 0 or m >= len(centers):
        return 1.0
    total = 0.0
    for q, row in zip(Q, reads):
        exact = np.argsort(np.linalg.norm(centers.astype(np.float64) - q.astype(np.float64), axis=1),
                           kind="stable")[:m]
        total += len(np.intersect1d(exact, row[row >= 0])) / m
    return total / len(Q)
