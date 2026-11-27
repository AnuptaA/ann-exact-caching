"""E15: which evidence the intruder-rate feature needs.

v11 answers from the exact top-K of the whole cached pool, so nr, the answer and its recall come from the pool in
every variant. The variants differ in where best fit (offset, size) and the intruder count come from. Shells are the
core's decoded 4-bit shells (pooled vectors excluded, each slot once); distances are float64.

  V0   the core: union answer, features from the m nearest entries (the E1 linear row)
  V1   best fit and intruders from the m nearest entries (the core's m)
  V2   best fit and intruders from the c nearest entries by exact distance, c in C_GRID and all entries
  V3   intruders = exact count over all shell slots (uncapped); best fit from c_fit nearest (32, also 16 and 128)
  V4   V3 censored at cap (2K, 4K, 8K, 16K)
  V5   intruders from an HNSW over the decoded shell vectors: the cap nearest, those closer than nr counted
  V6   oracle (diagnostic only): non-pooled vectors among the stored true top-500 closer than nr

Each variant: OLS on the three standardized features over the training role, lambda certified on the calibration
role with the v10 policy, hit rate and recall of hits on the held-out queries (the E10 sample: unique queries with
request counts as weights, at most e10.MAX_QUERIES of them).
"""

import copy
import time

import numpy as np
from scipy.stats import spearmanr

from recal.micro import e3, e10, fit, intervals as iv, topk, world as w
from recal.micro.e4 import dist64

NAME = "E15"
C_GRID = (8, 16, 32, 64, 128, 256, 512, 1024)
C_FIT = 32
C_FIT_OTHER = (16, 128)
CAP_MULTIPLES = (2, 4, 8, 16)
DEGREES = (8, 16)
EFS = (32, 64, 128, 256)
EF_CONSTRUCTION = 200
ID_BYTES = (2, 4)
RULE_RECOVERY = 0.90
RULE_V5_GAP = 0.01
RULE_V2_GAP = 0.01
TIMING_QUERIES = 500
BLOCK_ELEMENTS = 20_000_000
SHELL_BITS = 4


def clip_c(c: int, n: int) -> int:
    return min(c, n)


class Engine:
    """Per-query features from the pool, the centers and the shell slots of a built cache."""

    def __init__(self, world: w.World, feats: iv.Features):
        self.world, self.k = world, world.k
        self.pool, self.pool_ids = feats.pool, feats.pool_ids
        self.centers, self.radius = feats.centers, feats.radius
        self.shell_vec = feats.shell_vec
        self.n = len(self.centers)
        self.m = clip_c(world.m, self.n)
        slots = feats.slots
        entry = np.concatenate([np.full(len(s), e, np.int64) for e, s in enumerate(slots)]) if self.n else \
            np.zeros(0, np.int64)
        slot = np.concatenate(slots) if self.n else np.zeros(0, np.int64)
        order = np.argsort(slot, kind="stable")
        self.pair_entry = entry[order]
        self.ptr = np.searchsorted(slot[order], np.arange(len(self.shell_vec) + 1))
        if len(self.shell_vec) and (np.diff(self.ptr) < 1).any():
            raise RuntimeError("a shell slot belongs to no entry")
        grid = {clip_c(c, self.n) for c in C_GRID} | {self.n}
        self.count_cs = sorted(grid | {self.m})
        self.fit_cs = sorted(set(self.count_cs) | {clip_c(c, self.n) for c in (C_FIT,) + C_FIT_OTHER})

    def blocks(self, n: int):
        width = max(len(self.pool), self.n, len(self.shell_vec), 1)
        step = max(1, BLOCK_ELEMENTS // width)
        for lo in range(0, n, step):
            yield lo, min(n, lo + step)

    def compute(self, Q: np.ndarray, truth: np.ndarray, top_rows: np.ndarray) -> dict:
        """Per query: nr, recall of the pool answer, offset and size for every c in fit_cs, the intruder count for
        every c in count_cs, the global count and the oracle count (with its censoring flag)."""
        Q = np.ascontiguousarray(Q, dtype=np.float32)
        parts = w.fork_map(lambda lo, hi: self._range(Q[lo:hi], truth[lo:hi], top_rows[lo:hi]), len(Q),
                           self.world.workers)
        return {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}

    def _range(self, Q: np.ndarray, truth: np.ndarray, top_rows: np.ndarray) -> dict:
        n, K = len(Q), self.k
        out = dict(nr=np.zeros(n), y=np.zeros(n), glob=np.zeros(n, np.int64), oracle=np.zeros(n, np.int64),
                   oracle_censored=np.zeros(n, bool))
        for c in self.fit_cs:
            out[f"off{c}"], out[f"siz{c}"] = np.zeros(n), np.zeros(n)
        for c in self.count_cs:
            out[f"cnt{c}"] = np.zeros(n, np.int64)
        pooled = self.pool_ids
        for lo, hi in self.blocks(n):
            Dp, Dc = dist64(Q[lo:hi], self.pool), dist64(Q[lo:hi], self.centers)
            Ds = dist64(Q[lo:hi], self.shell_vec) if len(self.shell_vec) else np.zeros((hi - lo, 0))
            for i in range(hi - lo):
                g = lo + i
                near, nd = topk.exact_topk(self.pool, Q[g], Dp[i], K)
                nr = max(float(nd[-1]), 1e-12)
                out["nr"][g] = nr
                out["y"][g] = np.isin(truth[g][:K], pooled[near]).mean()
                order = np.lexsort((np.arange(self.n), Dc[i]))
                fitv = (Dc[i][order] + nr) / self.radius[order]
                for c in self.fit_cs:
                    b = int(order[int(np.argmin(fitv[:c]))])
                    out[f"off{c}"][g], out[f"siz{c}"][g] = Dc[i][b] / self.radius[b], nr / self.radius[b]
                closer = Ds[i] < nr
                out["glob"][g] = int(closer.sum())
                if closer.any():
                    rank = np.empty(self.n, np.int64)
                    rank[order] = np.arange(self.n)
                    earliest = np.minimum.reduceat(rank[self.pair_entry], self.ptr[:-1])
                    seen = np.sort(earliest[closer])
                    for c in self.count_cs:
                        out[f"cnt{c}"][g] = int(np.searchsorted(seen, c, side="left"))
                ids = top_rows[g]
                dist = np.linalg.norm(self.world.db[ids].astype(np.float64) - Q[g].astype(np.float64), axis=1)
                out["oracle"][g] = int(((dist < nr) & ~np.isin(ids, pooled)).sum())
                out["oracle_censored"][g] = bool(nr > dist.max())
        return out


def build_index(vectors: np.ndarray, degree: int):
    import hnswlib
    index = hnswlib.Index(space="l2", dim=vectors.shape[1])
    index.init_index(max_elements=len(vectors), ef_construction=EF_CONSTRUCTION, M=degree)
    index.set_num_threads(1)     # a threaded build gives a different graph on every run
    index.add_items(vectors.astype(np.float32), np.arange(len(vectors)))
    return index


def index_counts(engine: Engine, index, Q: np.ndarray, nr: np.ndarray, cap: int, ef: int, workers: int) -> dict:
    """Shell vectors closer than nr among the cap nearest returned by the index (float64 distances recomputed)."""
    cap = min(cap, len(engine.shell_vec))
    index.set_ef(max(ef, cap))
    Q = np.ascontiguousarray(Q, dtype=np.float32)

    def part(lo: int, hi: int) -> np.ndarray:
        labels, _ = index.knn_query(Q[lo:hi], k=cap, num_threads=1)
        found = np.zeros(hi - lo, np.int64)
        for i in range(hi - lo):
            d = np.linalg.norm(engine.shell_vec[labels[i].astype(np.int64)] - Q[lo + i].astype(np.float64), axis=1)
            found[i] = int((d < nr[lo + i]).sum())
        return found

    counts = np.concatenate(w.fork_map(part, len(Q), workers))
    return dict(count=counts, cap=cap)


def time_index(index, Q: np.ndarray, cap: int, ef: int) -> float:
    """Microseconds per query of knn_query, one thread, on up to TIMING_QUERIES queries."""
    Q = np.ascontiguousarray(Q[:TIMING_QUERIES], dtype=np.float32)
    index.set_ef(max(ef, cap))
    t0 = time.perf_counter()
    index.knn_query(Q, k=cap, num_threads=1)
    return (time.perf_counter() - t0) / max(len(Q), 1) * 1e6


def weighted_oracle(y: np.ndarray, has: np.ndarray, weight: np.ndarray, target: float) -> float:
    """Weighted share of queries in the largest prefix, by decreasing recall, whose mean recall is >= target."""
    yy, ww = y[has], weight[has]
    if not len(yy):
        return 0.0
    order = np.argsort(-yy, kind="stable")
    num, den = np.cumsum(ww[order] * yy[order]), np.cumsum(ww[order])
    ok = num / den >= target
    return float(den[np.flatnonzero(ok).max()] / weight.sum()) if ok.any() else 0.0


def rho(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return float("nan")
    return float(spearmanr(a[ok], b[ok])[0])


def evaluate(world: w.World, train: dict, cal: dict, held: dict, weight: np.ndarray) -> dict:
    """OLS on train, lambda on cal, hit decisions and rates on held. Each role is dict(F, y, has)."""
    p = world.p
    ok = train["has"] & np.isfinite(train["F"]).all(axis=1)
    nan = float("nan")
    if ok.sum() < 10:
        return dict(certified=False, lam=None, hit_rate=nan, recall_of_hits=nan, oracle_hit_rate=nan, hit=None,
                    score=None)
    model = fit.fit_linear(train["F"][ok], train["y"][ok])
    train_scores = fit.score(model, train["F"][ok])
    okc = cal["has"] & np.isfinite(cal["F"]).all(axis=1)
    cert = fit.certify(train_scores, fit.score(model, cal["F"][okc]), cal["y"][okc], p.T, p.delta, p.min_ess)
    good = held["has"] & np.isfinite(held["F"]).all(axis=1)
    score = np.full(len(weight), -np.inf)
    score[good] = fit.score(model, held["F"][good])
    hit = fit.decide(score, held["has"], cert.lam)
    total = weight.sum()
    return dict(certified=cert.lam is not None, lam=cert.lam,
                hit_rate=float((weight * hit).sum() / total) if cert.lam is not None else nan,
                recall_of_hits=float((weight * hit * held["y"]).sum() / (weight * hit).sum()) if hit.any() else nan,
                oracle_hit_rate=weighted_oracle(held["y"], held["has"], weight, p.T), hit=hit, score=score)


def query_index(world: w.World, vectors: np.ndarray) -> np.ndarray:
    """Index in world.q of each float32 vector (matched by exact bytes), to read its stored top-500."""
    unique = np.unique(world.stream)
    table = {}
    for lo in range(0, len(unique), 4096):
        block = np.ascontiguousarray(world.q[unique[lo:lo + 4096]], dtype=np.float32)
        for j, vec in enumerate(block):
            table[vec.tobytes()] = int(unique[lo + j])
    try:
        return np.asarray([table[np.ascontiguousarray(v, dtype=np.float32).tobytes()] for v in vectors], np.int64)
    except KeyError as error:
        raise RuntimeError("a role query is not a query of the stream") from error


def shell_pool_info(engine: Engine) -> dict:
    d = engine.shell_vec.shape[1] if len(engine.shell_vec) else engine.centers.shape[1]
    slots = len(engine.shell_vec)
    code_bytes = slots * -(-d * SHELL_BITS // 8)
    rows = []
    for degree in DEGREES:
        ids_per_slot = 2 * degree + degree / (degree - 1)
        for id_bytes in ID_BYTES:
            graph = slots * ids_per_slot * id_bytes
            rows.append(dict(degree=degree, id_bytes=id_bytes, ids_per_slot=ids_per_slot, graph_bytes=graph,
                             share_of_shell_pool=graph / code_bytes if code_bytes else float("nan")))
    return dict(shell_slots=slots, shell_pool_bytes=code_bytes, dim=d, index=rows)


def self_check(engine: Engine, feats: iv.Features, data: dict, Q: np.ndarray, n_check: int = 50) -> None:
    """V2(all) offsets, sizes and rates equal the golden Features.exact over every entry."""
    full = copy.copy(feats)
    full.c = engine.n
    m = min(n_check, len(Q))
    ref = full.exact(Q[:m])
    k = engine.k
    mine = np.column_stack([data[f"off{engine.n}"][:m], data[f"siz{engine.n}"][:m],
                            np.log((data[f"cnt{engine.n}"][:m] + 1.0) / k)])
    if not np.allclose(mine, ref, rtol=1e-6, atol=1e-6):
        raise RuntimeError("E15 V2(all) differs from Features.exact over all entries")
    if (data["glob"][:m] != data[f"cnt{engine.n}"][:m]).any():
        raise RuntimeError("E15 global intruder count differs from V2(all)")


def columns(d: dict, c_fit: int, count: np.ndarray, k: int) -> np.ndarray:
    return np.column_stack([d[f"off{c_fit}"], d[f"siz{c_fit}"], np.log((count + 1.0) / k)])


def run(world: w.World, degrees=DEGREES, efs=EFS, log=print) -> tuple[list[dict], dict]:
    if "e15_feats" not in world.cache:
        world.cache["e15_feats"] = world.cache["v11"]["feats"] if "v11" in world.cache else iv.Features(world, c=C_FIT)
    feats = world.cache["e15_feats"]
    engine = Engine(world, feats)
    K, p = world.k, world.p
    nan = float("nan")
    held_vec, held_truth, weight, keep = e10.held_queries(world)
    held_ids = world.heldout()["unique"][keep]
    roles = {}
    for name in ("train", "cal"):
        _, vec = world.role_rows(name)
        roles[name] = (vec, e3.role_truth(world, name))
    ids = {"train": query_index(world, roles["train"][0]), "cal": query_index(world, roles["cal"][0]),
           "held": held_ids}
    queries = {"train": roles["train"][0], "cal": roles["cal"][0], "held": held_vec}
    truths = {"train": roles["train"][1], "cal": roles["cal"][1], "held": held_truth}
    data = {}
    for name in queries:
        data[name] = engine.compute(queries[name], truths[name], np.asarray(world.top[ids[name]], np.int64))
        log(f"[E15] {name}: {len(queries[name])} queries, features done")
    if len(queries["held"]):
        self_check(engine, feats, data["held"], queries["held"])
    v0 = {}
    for name in ("train", "cal"):
        f, _ = world.role_rows(name)
        v0[name] = dict(F=f.F, y=f.recall, has=f.has)
    f0 = world.features(held_vec, held_truth)
    v0["held"] = dict(F=f0.F, y=f0.recall, has=f0.has)
    results = {"V0": evaluate(world, v0["train"], v0["cal"], v0["held"], weight)}
    rows = []
    base = dict(workload=world.name, budget=world.budget, n_train=len(queries["train"]), n_cal=len(queries["cal"]),
                n_held=len(held_vec), n_held_weight=float(weight.sum()), n_entries=engine.n, m=engine.m,
                pool_vectors=len(engine.pool))
    miss = K * (1.0 - data["held"]["y"])
    held_y = data["held"]["y"]
    oracle_count = data["held"]["oracle"]

    def make_row(name, family, res, count=None, y_variant=None, **extra) -> dict:
        row = dict(base, variant=name, family=family, certified=res["certified"], lam=res["lam"],
                   hit_rate=res["hit_rate"], recall_of_hits=res["recall_of_hits"],
                   oracle_hit_rate=res["oracle_hit_rate"],
                   spearman_score_v0=rho(res["score"], results["V0"]["score"])
                   if res["score"] is not None and results["V0"]["score"] is not None else nan,
                   spearman_score_recall=rho(res["score"], held_y if y_variant is None else y_variant)
                   if res["score"] is not None else nan,
                   spearman_count_missing=nan, spearman_count_v6=nan, count_mean=nan, **extra)
        if count is not None:
            row.update(spearman_count_missing=rho(count.astype(float), miss),
                       spearman_count_v6=rho(count.astype(float), oracle_count.astype(float)),
                       count_mean=float(np.average(count, weights=weight)))
        return row

    rows.append(make_row("V0", "V0", results["V0"], y_variant=v0["held"]["y"], c_fit=engine.m, c_count=engine.m))

    def variant(name, family, c_fit, counts, **extra):
        roles_f = {r: dict(F=columns(data[r], c_fit, counts[r], K), y=data[r]["y"],
                           has=np.ones(len(data[r]["y"]), bool)) for r in data}
        res = evaluate(world, roles_f["train"], roles_f["cal"], roles_f["held"], weight)
        results[name] = res
        rows.append(make_row(name, family, res, counts["held"], c_fit=c_fit, **extra))
        log(f"[E15] {name}: hit={res['hit_rate']:.4f} recall_of_hits={res['recall_of_hits']:.4f}")

    def from_c(c):
        return {r: data[r][f"cnt{c}"] for r in data}

    variant("V1", "V1", engine.m, from_c(engine.m), c_count=engine.m)
    for c in engine.count_cs:
        if c in {clip_c(g, engine.n) for g in C_GRID} | {engine.n}:
            variant("V2_all" if c == engine.n else f"V2_c{c}", "V2", c, from_c(c), c_count=c)
    glob = {r: data[r]["glob"] for r in data}
    fit_default = clip_c(C_FIT, engine.n)
    variant("V3", "V3", fit_default, glob, c_count="all")
    for c in C_FIT_OTHER:
        variant(f"V3_cfit{c}", "V3", clip_c(c, engine.n), glob, c_count="all")
    caps = [mult * K for mult in CAP_MULTIPLES]
    for cap in caps:
        capped = {r: np.minimum(glob[r], cap) for r in data}
        variant(f"V4_cap{cap}", "V4", fit_default, capped, c_count="all", cap=cap,
                censored_share=float((weight * (data["held"]["glob"] >= cap)).sum() / weight.sum()))
    shell = shell_pool_info(engine)
    if len(engine.shell_vec):
        for degree in degrees:
            index = build_index(engine.shell_vec, degree)
            log(f"[E15] shell index degree {degree} built over {len(engine.shell_vec)} slots")
            cache = {}
            for cap in caps:
                for ef in efs:
                    eff = max(ef, min(cap, len(engine.shell_vec)))
                    if (cap, eff) not in cache:
                        counts = {r: index_counts(engine, index, queries[r], data[r]["nr"], cap, ef, world.workers)
                                  for r in data}
                        cache[(cap, eff)] = (counts, time_index(index, queries["held"], cap, ef)
                                             if len(queries["held"]) else nan)
                    counts, us = cache[(cap, eff)]
                    name = f"V5_d{degree}_cap{cap}_ef{ef}"
                    variant(name, "V5", fit_default, {r: counts[r]["count"] for r in data}, c_count="index",
                            cap=cap, degree=degree, ef=ef, ef_effective=eff, us_per_query=us,
                            censored_share=float((weight * (counts["held"]["count"] >= counts["held"]["cap"])).sum()
                                                 / weight.sum()))
                    exact_cap = results[f"V4_cap{cap}"]
                    rows[-1].update(flips_vs_v3=flip(results[name], results["V3"], weight),
                                    flips_vs_v4_same_cap=flip(results[name], exact_cap, weight),
                                    count_equal_share_v4=float((weight * (counts["held"]["count"] ==
                                                                          np.minimum(glob["held"], cap))).sum()
                                                               / weight.sum()))
    oracle = {r: data[r]["oracle"] for r in data}
    variant("V6", "V6", fit_default, oracle, c_count="oracle",
            censored_share=float((weight * data["held"]["oracle_censored"]).sum() / weight.sum()))
    summary = dict(shell_pool=shell, rules=rules(world, results, rows), sampling=SAMPLING_NOTE,
                   n_held_unique=len(held_vec), n_held_requests=float(weight.sum()))
    return rows, summary


SAMPLING_NOTE = ("held-out: the E10 sample (unique queries of the scored requests, weighted by request count, at "
                 f"most {e10.MAX_QUERIES} spread evenly); train and calibration roles in full, unit weights")


def flip(a: dict, b: dict, weight: np.ndarray) -> float:
    if a["hit"] is None or b["hit"] is None:
        return float("nan")
    return float((weight * (a["hit"] != b["hit"])).sum() / weight.sum())


def rules(world: w.World, results: dict, rows: list[dict]) -> dict:
    """The success rules, evaluated on this workload."""
    hit = lambda name: results[name]["hit_rate"] if name in results else float("nan")  # noqa: E731
    c32 = f"V2_c{C_FIT}" if f"V2_c{C_FIT}" in results else None
    gap = hit("V0") - hit(c32) if c32 else float("nan")
    recovered = (hit("V3") - hit(c32)) / gap if c32 and gap > 0 else float("nan")
    applies = any(key in world.name.lower() for key in ("laion", "triviaqa"))
    v5 = [r for r in rows if r["family"] == "V5" and r["ef"] <= 128]
    close = [dict(variant=r["variant"], hit_diff_v3=r["hit_rate"] - hit("V3")) for r in v5]
    within = [c["variant"] for c in close if abs(c["hit_diff_v3"]) <= RULE_V5_GAP]
    v2 = [(r["c_count"], r["hit_rate"]) for r in rows if r["family"] == "V2"]
    reach = [c for c, h in v2 if h >= hit("V0") - RULE_V2_GAP]
    return dict(
        gap_v0_minus_v2_c32=gap, v3_recovery=recovered, v3_recovers_90=bool(recovered >= RULE_RECOVERY),
        rule_applies_to_this_workload=applies,
        v5_within_1pt_of_v3=within, v5_any_within_1pt=bool(within), v5_checked=close,
        v2_smallest_c_within_1pt_of_v0=min(reach) if reach else None,
        v2_approaches_v0_only_in_hundreds=bool(reach and min(reach) >= 100),
        v2_hit_by_c={str(c): h for c, h in v2})
