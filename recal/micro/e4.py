"""E4: finding the best-fit sphere.

Entries are read back from the cache (center, radius, list) and their shells are rebuilt exactly from the stored
top-500 answers of the centers, without the cached vectors, which the core never stores as shell vectors. Every
method picks a set of entries for a query and the features follow v10 (nr, best fit, offset, size, intruder
rate); a model is fit and certified per method as in E1, on the same training and calibration queries, and applied
to the same held-out requests. Two things vary:

  where the candidate answer comes from
    union  the K nearest of the union of the lists of the entries read (v10, the core)
    pool   the exact top-K of the whole cached pool (v11: an index over the cached vectors answers)
  how shells are counted
    decoded  the core's 4-bit codes decoded (the default, what the core does)
    exact    full-precision shell vectors (a diagnostic)

Entry searches (all return the entries to read for a query):
  m_core         the core's own HNSW over centers, m entries (union answer)
  m_nearest      the exact m nearest centers (union answer)
  c<N>           the exact N nearest centers, N in 4, 8, 16, 32 (union answer)
  pool_knn       entries listing any of the K nearest cached vectors (the answer is the exact pool top-K: those
                 entries list all K of them)
  v11_m_nearest  the exact m nearest centers, answer from the pool
  v11_c<N>       the exact N nearest centers, N in 4, 8, 16, 32, 64, 128, answer from the pool
The decomposition rows cross {core HNSW, exact m nearest} entry search with {decoded, exact} shells.
"""

import numpy as np

from recal import shell_quant
from recal.micro import fit, topk, world as w

NAME = "E4"
NEAREST = (4, 8, 16, 32)
V11_NEAREST = (4, 8, 16, 32, 64, 128)
SHELL_MODES = ("decoded", "exact")
UNION = "union of the lists of the entries read"
POOL = "exact top-K of the whole cached pool"
SEARCH = {"m_core": "core HNSW over centers (m)", "m_nearest": "exact m nearest centers",
          "pool_knn": "entries listing the K nearest cached vectors", "v11_m_nearest": "exact m nearest centers"}
SPECS = {"pool_knn": POOL, "m_core": UNION, "m_nearest": UNION, **{f"c{c}": UNION for c in NEAREST},
         "v11_m_nearest": POOL, **{f"v11_c{c}": POOL for c in V11_NEAREST}}
METHODS = tuple(SPECS)
DECOMPOSITION = (("m_core", "decoded"), ("m_nearest", "decoded"), ("m_core", "exact"), ("m_nearest", "exact"))
ELEMENTS = 100_000_000      # largest distance block (queries x vectors) held at once


def search_label(name: str) -> str:
    if name in SEARCH:
        return SEARCH[name]
    return f"exact {name.split('c')[-1]} nearest centers"


def dist64(Q: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Float64 Euclidean distances (query x vector)."""
    Q, X = Q.astype(np.float64), X.astype(np.float64)
    return np.sqrt(np.maximum((Q * Q).sum(1)[:, None] + (X * X).sum(1)[None, :] - 2.0 * (Q @ X.T), 0.0))


class EntryTable:
    """Entries of a built cache as arrays: centers, radii, lists (pool slots), shells (zone slots)."""

    def __init__(self, world: w.World, exclude_pool: bool = True):
        """exclude_pool drops cached vectors from the shells, as the core does when it stores them."""
        built = world.built
        self.k = world.k
        self.pool_ids = np.asarray(built.pool_ids, dtype=np.int64)
        entries = world.entries()
        self.n = len(entries)
        self.centers = np.asarray([e["center"] for e in entries], dtype=np.float32).reshape(self.n, -1)
        self.radius = np.maximum(np.asarray([e["radius"] for e in entries], dtype=np.float64), 1e-12)
        self.lists = [np.searchsorted(self.pool_ids, np.asarray(e["response_ids"], dtype=np.int64)) for e in entries]
        shells = self._shells(world, entries)
        if exclude_pool:
            shells = [s[~np.isin(s, self.pool_ids)] for s in shells]
        self.shell_ids = shells
        self.zone_ids = np.unique(np.concatenate(shells)) if shells else np.zeros(0, np.int64)
        self.shells = [np.searchsorted(self.zone_ids, s) for s in shells]
        self.pool = np.asarray(world.db[self.pool_ids], dtype=np.float32)
        self.zone = np.asarray(world.db[self.zone_ids], dtype=np.float32) if len(self.zone_ids) else \
            np.zeros((0, self.centers.shape[1]), np.float32)
        lo, step = world.core.quantizer()
        self.zone_dec = (np.asarray(lo, np.float64) + shell_quant.shell_codes(self.zone, lo, step).astype(np.float64)
                         * np.asarray(step, np.float64)) if len(self.zone_ids) else self.zone.astype(np.float64)
        order = np.concatenate([np.full(len(l), e, np.int64) for e, l in enumerate(self.lists)])
        slots = np.concatenate(self.lists)
        sort = np.argsort(slots, kind="stable")
        self.inv_entries = order[sort]
        self.inv_ptr = np.searchsorted(slots[sort], np.arange(len(self.pool_ids) + 1))

    def _shells(self, world: w.World, entries: list[dict]) -> list[np.ndarray]:
        """Ranks jK+1 .. (j+2)K of the center's exact answer, as the construction proposed them."""
        row = {}
        unique = np.unique(world.stream)
        for i in range(0, len(unique), 4096):
            block = np.asarray(world.q[unique[i:i + 4096]], dtype=np.float32)
            for j, vec in enumerate(block):
                row[hash(vec.tobytes())] = int(unique[i + j])
        out = []
        for e in entries:
            qi = row.get(hash(np.asarray(e["center"], dtype=np.float32).tobytes()))
            if qi is None:
                raise RuntimeError("an entry center is not a query of the stream")
            lo, hi = e["j"] * e["K"], (e["j"] + world.p.shell_width) * e["K"]
            out.append(np.asarray(world.top[qi][lo:hi], dtype=np.int64))
        return out

    def entries_of(self, slots: np.ndarray) -> np.ndarray:
        return np.unique(np.concatenate([self.inv_entries[self.inv_ptr[s]:self.inv_ptr[s + 1]] for s in slots]))


def sq_dist(Q: np.ndarray, X: np.ndarray) -> np.ndarray:
    d2 = (Q * Q).sum(1)[:, None] + (X * X).sum(1)[None, :] - 2.0 * (Q @ X.T)
    return np.sqrt(np.maximum(d2, 0.0))


def compute(table: EntryTable, Q: np.ndarray, truth: np.ndarray, m: int, workers: int = 1, core=None) -> dict:
    """Per method: has, features per shell mode, candidate recall, set size; plus the reference and the shell check
    sizes. core supplies the HNSW entry search of m_core.

    Queries are independent, so with workers > 1 consecutive ranges run in forked processes."""
    parts = w.fork_map(lambda lo, hi: _compute(table, Q[lo:hi], truth[lo:hi], m, core), len(Q), workers)
    if len(parts) == 1:
        return parts[0]
    return dict(methods={name: {key: np.concatenate([p["methods"][name][key] for p in parts])
                                for key in parts[0]["methods"][name]} for name in METHODS},
                ref={key: np.concatenate([p["ref"][key] for p in parts]) for key in parts[0]["ref"]},
                examined=np.concatenate([p["examined"] for p in parts]))


def _compute(table: EntryTable, Q: np.ndarray, truth: np.ndarray, m: int, core=None) -> dict:
    Q = np.ascontiguousarray(Q, dtype=np.float32)
    n, K = len(Q), table.k
    out = {name: dict(has=np.zeros(n, bool), recall=np.zeros(n), size=np.zeros(n, np.int64),
                      contains=np.zeros(n, bool), **{f"F_{mode}": np.zeros((n, 3)) for mode in SHELL_MODES})
           for name in METHODS}
    ref = dict(offset=np.zeros(n), size=np.zeros(n), nr=np.zeros(n), has=np.zeros(n, bool))
    examined = np.zeros(n, np.int64)
    reads = np.asarray(core.entries_read_batch(Q, m)) if core is not None and table.n else None
    block = max(1, ELEMENTS // max(len(table.pool_ids), table.n, 2 * len(table.zone_ids), 1))
    for lo in range(0, n, block):
        hi = min(n, lo + block)
        Dc = sq_dist(Q[lo:hi], table.centers)
        Dp = sq_dist(Q[lo:hi], table.pool)
        Ds = {"decoded": dist64(Q[lo:hi], table.zone_dec) if len(table.zone_ids) else np.zeros((hi - lo, 0)),
              "exact": dist64(Q[lo:hi], table.zone) if len(table.zone_ids) else np.zeros((hi - lo, 0))}
        order = np.argsort(Dc, axis=1)
        for i in range(hi - lo):
            g = lo + i
            if len(table.pool_ids) < K:
                continue
            knn, knn_dist = topk.exact_topk(table.pool, Q[g], Dp[i], K)
            nr_ref = float(knn_dist[-1])
            fit_all = (Dc[i] + nr_ref) / table.radius
            b = int(np.argmin(fit_all))
            ref["offset"][g], ref["size"][g], ref["nr"][g], ref["has"][g] = Dc[i, b] / table.radius[b], \
                nr_ref / table.radius[b], nr_ref, True
            m_set = order[i, :m if m > 0 else table.n]
            sets = {"pool_knn": table.entries_of(knn), "m_nearest": m_set, "v11_m_nearest": m_set}
            if reads is not None:
                sets["m_core"] = reads[g][reads[g] >= 0]
            for c in NEAREST:
                sets[f"c{c}"] = order[i, :c]
            for c in V11_NEAREST:
                sets[f"v11_c{c}"] = order[i, :c]
            for name, S in sets.items():
                answer = (knn, knn_dist) if SPECS[name] == POOL else None
                res = _features_multi(table, S, Dc[i], Dp[i], {k: v[i] for k, v in Ds.items()}, K, Q[g], answer)
                o = out[name]
                o["has"][g], o["size"][g], o["contains"][g] = res["has"], len(S), bool((S == b).any())
                if res["has"]:
                    for mode in SHELL_MODES:
                        o[f"F_{mode}"][g] = res["F"][mode]
                    o["recall"][g] = np.isin(truth[g][:K], table.pool_ids[res["ids"]]).mean()
                if name == "pool_knn":
                    examined[g] = res["examined"]
    return dict(methods=out, ref=ref, examined=examined)


def _features_multi(table: EntryTable, S: np.ndarray, Dc: np.ndarray, Dp: np.ndarray, Ds: dict, K: int,
                    q: np.ndarray, answer=None) -> dict:
    """Features for one entry set under every shell mode. answer = (slots, distances) of the pool's top-K, or None
    for the K nearest of the union of the lists."""
    if not len(S):
        return dict(has=False)
    if answer is None:
        slots = np.unique(np.concatenate([table.lists[e] for e in S]))
        if len(slots) < K:
            return dict(has=False)
        near, near_dist = topk.exact_topk(table.pool, q, Dp[slots], K, slots)
    else:
        near, near_dist = answer
    nr = max(float(near_dist[-1]), 1e-12)
    best = int(S[np.argmin((Dc[S] + nr) / table.radius[S])])
    zone = np.unique(np.concatenate([table.shells[e] for e in S]))
    F = {}
    for mode, D in Ds.items():
        count = int((D[zone] < nr).sum()) if len(zone) else 0
        F[mode] = [Dc[best] / table.radius[best], nr / table.radius[best], np.log((count + 1.0) / K)]
    return dict(has=True, ids=near, examined=len(zone), F=F)


def _features(table: EntryTable, S: np.ndarray, Dc: np.ndarray, Dp: np.ndarray, Ds: np.ndarray, K: int,
              q: np.ndarray) -> dict:
    slots = np.unique(np.concatenate([table.lists[e] for e in S]))
    if len(slots) < K:
        return dict(has=False)
    near, near_dist = topk.exact_topk(table.pool, q, Dp[slots], K, slots)
    nr = max(float(near_dist[-1]), 1e-12)
    fit_set = (Dc[S] + nr) / table.radius[S]
    best = int(S[np.argmin(fit_set)])
    zone = np.unique(np.concatenate([table.shells[e] for e in S])) if len(Ds) else np.zeros(0, np.int64)
    count = int((Ds[zone] < nr).sum()) if len(zone) else 0
    return dict(has=True, ids=near, examined=len(zone),
                F=[Dc[best] / table.radius[best], nr / table.radius[best], np.log((count + 1.0) / K)])


def pct(x: np.ndarray, p: float) -> float:
    return float(np.percentile(x, p)) if len(x) else float("nan")


def method_row(world: w.World, table: EntryTable, name: str, mode: str, tr: dict, ca: dict, he: dict,
               held: dict, m: int) -> dict:
    inverse = held["inverse"]
    t, c, h = tr["methods"][name], ca["methods"][name], he["methods"][name]
    ok = t["has"]
    Ft, Fc, Fh = t[f"F_{mode}"], c[f"F_{mode}"], h[f"F_{mode}"]
    row = dict(workload=world.name, budget=world.budget, section="methods" if mode == "decoded" else "exact_shells",
               method=name, answer_source=SPECS[name], entry_search=search_label(name),
               shells="decoded 4-bit" if mode == "decoded" else "exact", n_entries=table.n, m=m,
               mean_entries_considered=float(h["size"].mean()), has_candidate_share=float(h["has"][inverse].mean()),
               contains_reference_share=float(h["contains"][inverse].mean()))
    miss = h["has"] & ~h["contains"] & he["ref"]["has"]
    for key, col in (("offset", 0), ("size", 1)):
        err = np.abs(Fh[miss, col] - he["ref"][key][miss])
        row[f"{key}_error_mean"], row[f"{key}_error_p95"] = float(err.mean()) if len(err) else float("nan"), \
            pct(err, 95)
    if ok.sum() >= 10:
        model = fit.fit_logit(Ft[ok], t["recall"][ok])
        scores_train = fit.score(model, Ft[ok])
        cert = fit.certify(scores_train, fit.score(model, Fc[c["has"]]), c["recall"][c["has"]], world.p.T,
                           world.p.delta, world.p.min_ess)
        scores = np.where(h["has"], fit.score(model, Fh), -np.inf)
        hit = fit.decide(scores, h["has"], cert.lam)[inverse]
        y = h["recall"][inverse]
        row.update(lam=cert.lam, certified=cert.lam is not None,
                   hit_rate=float(hit.mean()) if cert.lam is not None else float("nan"),
                   recall_of_hits=float(y[hit].mean()) if hit.any() else float("nan"))
    else:
        row.update(lam=None, certified=False, hit_rate=float("nan"), recall_of_hits=float("nan"))
    return row


def run(world: w.World) -> tuple[list[dict], dict]:
    """Rows per method and shell mode, the entry-search by shell decomposition, and diagnostics (shell check size,
    agreement of the core's HNSW entries with the core's own features)."""
    table = EntryTable(world)
    from recal.micro import e3
    hist = world.history()
    train_vec = hist["vec"][hist["role"] == w.ROLE_CODES["train"]]
    train_truth = e3.role_truth(world, "train")
    cal_truth = e3.role_truth(world, "cal")
    cal_vec = world.role_rows("cal")[1]
    held = world.heldout()
    m = world.m
    core = world.core
    tr = compute(table, train_vec, train_truth, m, world.workers, core)
    ca = compute(table, cal_vec, cal_truth, m, world.workers, core)
    he = compute(table, held["vec"], held["truth"], m, world.workers, core)
    n_req = len(held["inverse"])
    rows = [method_row(world, table, name, mode, tr, ca, he, held, m) for mode in SHELL_MODES for name in METHODS]
    by = {(r["method"], r["shells"]): r for r in rows}
    cells = []
    for name, mode in DECOMPOSITION:
        r = dict(by[(name, "decoded 4-bit" if mode == "decoded" else "exact")], section="decomposition")
        r["cell"] = f"{'core HNSW entries' if name == 'm_core' else 'exact m nearest'} + {r['shells']} shells"
        cells.append(r)
    rows += cells
    ex = he["examined"][held["inverse"]]
    diagnostics = dict(workload=world.name, budget=world.budget, n_requests=n_req, n_entries=table.n,
                       intruder_examined=dict(median=pct(ex, 50), p95=pct(ex, 95), p99=pct(ex, 99),
                                              max=int(ex.max()) if len(ex) else 0, unit="shell vectors per query"),
                       core_agreement=core_agreement(he, held),
                       answer_sources={name: SPECS[name] for name in METHODS})
    return rows, diagnostics


def core_agreement(he: dict, held: dict) -> dict:
    """The core's HNSW entries (m_core) against the core's own features: with decoded shells they should agree to
    float noise; with exact shells the intruder rate differs by the quantization."""
    f = held["features"]
    h = he["methods"]["m_core"]
    both = f.has & h["has"]
    if not both.any():
        return dict(n=0)
    out = dict(n=int(both.sum()), has_candidate_agree=float((f.has == h["has"]).mean()))
    for mode in SHELL_MODES:
        F = h[f"F_{mode}"]
        out[mode] = dict(offset_abs_diff_p95=pct(np.abs(f.F[both, 0] - F[both, 0]), 95),
                         size_abs_diff_p95=pct(np.abs(f.F[both, 1] - F[both, 1]), 95),
                         rate_abs_diff_p95=pct(np.abs(f.F[both, 2] - F[both, 2]), 95),
                         rate_equal_share=float((np.abs(f.F[both, 2] - F[both, 2]) < 1e-9).mean()))
    return out
