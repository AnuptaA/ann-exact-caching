"""E4: finding the best-fit sphere.

Entries are read back from the cache (center, radius, list) and their shells are rebuilt exactly from the stored
top-500 answers of the centers (full precision, where the core keeps 4-bit codes). Every method picks a set of
entries for a query; the candidate answer is the K nearest of the union of their lists, the features follow v10
(nr, best fit, offset, size, intruder rate), and a model is fit and certified per method as in E1.

  pool_knn   entries listing any of the query's K nearest cached vectors (exact search over the cache)
  m_nearest  the m nearest centers (v10)
  c<N>       the N nearest centers
"""

import numpy as np

from recal.micro import fit, world as w

NAME = "E4"
NEAREST = (4, 8, 16, 32)
METHODS = ("pool_knn", "m_nearest") + tuple(f"c{c}" for c in NEAREST)
ELEMENTS = 100_000_000      # largest distance block (queries x vectors) held at once


class EntryTable:
    """Entries of a built cache as arrays: centers, radii, lists (pool slots), shells (zone slots)."""

    def __init__(self, world: w.World):
        built = world.built
        self.k = world.k
        self.pool_ids = np.asarray(built.pool_ids, dtype=np.int64)
        entries = world.entries()
        self.n = len(entries)
        self.centers = np.asarray([e["center"] for e in entries], dtype=np.float32).reshape(self.n, -1)
        self.radius = np.maximum(np.asarray([e["radius"] for e in entries], dtype=np.float64), 1e-12)
        self.lists = [np.searchsorted(self.pool_ids, np.asarray(e["response_ids"], dtype=np.int64)) for e in entries]
        shells = self._shells(world, entries)
        self.zone_ids = np.unique(np.concatenate(shells)) if shells else np.zeros(0, np.int64)
        self.shells = [np.searchsorted(self.zone_ids, s) for s in shells]
        self.pool = np.asarray(world.db[self.pool_ids], dtype=np.float32)
        self.zone = np.asarray(world.db[self.zone_ids], dtype=np.float32) if len(self.zone_ids) else \
            np.zeros((0, self.centers.shape[1]), np.float32)
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


def compute(table: EntryTable, Q: np.ndarray, truth: np.ndarray, m: int, workers: int = 1) -> dict:
    """Per method: has, features, candidate recall, set size; plus the reference and the unbounded intruder checks.

    Queries are independent, so with workers > 1 consecutive ranges run in forked processes."""
    parts = w.fork_map(lambda lo, hi: _compute(table, Q[lo:hi], truth[lo:hi], m), len(Q), workers)
    if len(parts) == 1:
        return parts[0]
    return dict(methods={name: {key: np.concatenate([p["methods"][name][key] for p in parts])
                                for key in parts[0]["methods"][name]} for name in METHODS},
                ref={key: np.concatenate([p["ref"][key] for p in parts]) for key in parts[0]["ref"]},
                examined=np.concatenate([p["examined"] for p in parts]))


def _compute(table: EntryTable, Q: np.ndarray, truth: np.ndarray, m: int) -> dict:
    Q = np.ascontiguousarray(Q, dtype=np.float32)
    n, K = len(Q), table.k
    out = {name: dict(has=np.zeros(n, bool), F=np.zeros((n, 3)), recall=np.zeros(n), size=np.zeros(n, np.int64),
                      contains=np.zeros(n, bool)) for name in METHODS}
    ref = dict(offset=np.zeros(n), size=np.zeros(n), nr=np.zeros(n), has=np.zeros(n, bool))
    examined = np.zeros(n, np.int64)
    block = max(1, ELEMENTS // max(len(table.pool_ids), table.n, len(table.zone_ids), 1))
    for lo in range(0, n, block):
        hi = min(n, lo + block)
        Dc = sq_dist(Q[lo:hi], table.centers)
        Dp = sq_dist(Q[lo:hi], table.pool)
        Ds = sq_dist(Q[lo:hi], table.zone) if len(table.zone_ids) else np.zeros((hi - lo, 0))
        order = np.argsort(Dc, axis=1)
        for i in range(hi - lo):
            g = lo + i
            if len(table.pool_ids) < K:
                continue
            knn = np.argpartition(Dp[i], K - 1)[:K]
            nr_ref = float(Dp[i][knn].max())
            fit_all = (Dc[i] + nr_ref) / table.radius
            b = int(np.argmin(fit_all))
            ref["offset"][g], ref["size"][g], ref["nr"][g], ref["has"][g] = Dc[i, b] / table.radius[b], \
                nr_ref / table.radius[b], nr_ref, True
            sets = {"pool_knn": table.entries_of(knn), "m_nearest": order[i, :m if m > 0 else table.n]}
            sets.update({f"c{c}": order[i, :c] for c in NEAREST})
            for name, S in sets.items():
                res = _features(table, S, Dc[i], Dp[i], Ds[i], K)
                o = out[name]
                o["has"][g], o["size"][g], o["contains"][g] = res["has"], len(S), bool((S == b).any())
                if res["has"]:
                    o["F"][g] = res["F"]
                    o["recall"][g] = np.isin(truth[g][:K], table.pool_ids[res["ids"]]).mean()
                if name == "pool_knn":
                    examined[g] = res["examined"]
    return dict(methods=out, ref=ref, examined=examined)


def _features(table: EntryTable, S: np.ndarray, Dc: np.ndarray, Dp: np.ndarray, Ds: np.ndarray, K: int) -> dict:
    slots = np.unique(np.concatenate([table.lists[e] for e in S]))
    if len(slots) < K:
        return dict(has=False)
    near = slots[np.argpartition(Dp[slots], K - 1)[:K]]
    nr = max(float(Dp[near].max()), 1e-12)
    fit_set = (Dc[S] + nr) / table.radius[S]
    best = int(S[np.argmin(fit_set)])
    zone = np.unique(np.concatenate([table.shells[e] for e in S])) if len(Ds) else np.zeros(0, np.int64)
    count = int((Ds[zone] < nr).sum()) if len(zone) else 0
    return dict(has=True, ids=near, examined=len(zone),
                F=[Dc[best] / table.radius[best], nr / table.radius[best], np.log((count + 1.0) / K)])


def pct(x: np.ndarray, p: float) -> float:
    return float(np.percentile(x, p)) if len(x) else float("nan")


def run(world: w.World) -> tuple[list[dict], dict]:
    """Rows per method, and diagnostics (intruder-check size, agreement with the core's own features)."""
    table = EntryTable(world)
    train, _ = world.role_rows("train")
    cal, cal_vec = world.role_rows("cal")
    from recal.micro import e3
    hist = world.history()
    train_vec = hist["vec"][hist["role"] == w.ROLE_CODES["train"]]
    train_truth = e3.role_truth(world, "train")
    cal_truth = e3.role_truth(world, "cal")
    held = world.heldout()
    m = world.m
    tr = compute(table, train_vec, train_truth, m, world.workers)
    ca = compute(table, cal_vec, cal_truth, m, world.workers)
    he = compute(table, held["vec"], held["truth"], m, world.workers)
    inverse, n_req = held["inverse"], len(held["inverse"])
    rows = []
    for name in METHODS:
        t, c, h = tr["methods"][name], ca["methods"][name], he["methods"][name]
        ok = t["has"]
        row = dict(workload=world.name, budget=world.budget, method=name, n_entries=table.n, m=m,
                   mean_entries_considered=float(h["size"].mean()), has_candidate_share=float(h["has"][inverse].mean()),
                   contains_reference_share=float(h["contains"][inverse].mean()))
        miss = h["has"] & ~h["contains"] & he["ref"]["has"]
        for key, col in (("offset", 0), ("size", 1)):
            err = np.abs(h["F"][miss, col] - he["ref"][key][miss])
            row[f"{key}_error_mean"], row[f"{key}_error_p95"] = float(err.mean()) if len(err) else float("nan"), \
                pct(err, 95)
        if ok.sum() >= 10:
            model = fit.fit_logit(t["F"][ok], t["recall"][ok])
            scores_train = fit.score(model, t["F"][ok])
            cert = fit.certify(scores_train, fit.score(model, c["F"][c["has"]]), c["recall"][c["has"]], world.p.T,
                               world.p.delta, world.p.min_ess)
            scores = np.where(h["has"], fit.score(model, h["F"]), -np.inf)
            hit = fit.decide(scores, h["has"], cert.lam)[inverse]
            y = h["recall"][inverse]
            row.update(lam=cert.lam, certified=cert.lam is not None, hit_rate=float(hit.mean()),
                       recall_of_hits=float(y[hit].mean()) if hit.any() else float("nan"))
        else:
            row.update(lam=None, certified=False, hit_rate=0.0, recall_of_hits=float("nan"))
        rows.append(row)
    ex = he["examined"][inverse]
    diagnostics = dict(workload=world.name, budget=world.budget, n_requests=n_req, n_entries=table.n,
                       intruder_examined=dict(median=pct(ex, 50), p95=pct(ex, 95), p99=pct(ex, 99),
                                              max=int(ex.max()) if len(ex) else 0, unit="shell vectors per query"),
                       core_agreement=core_agreement(world, he, held))
    return rows, diagnostics


def core_agreement(world: w.World, he: dict, held: dict) -> dict:
    """Offset and size of the m-nearest method against the core's own features (it searches centers with an HNSW
    and keeps shells at 4 bits, so small differences are expected)."""
    f = held["features"]
    h = he["methods"]["m_nearest"]
    both = f.has & h["has"]
    if not both.any():
        return dict(n=0)
    return dict(n=int(both.sum()), has_candidate_agree=float((f.has == h["has"]).mean()),
                offset_abs_diff_p95=pct(np.abs(f.F[both, 0] - h["F"][both, 0]), 95),
                size_abs_diff_p95=pct(np.abs(f.F[both, 1] - h["F"][both, 1]), 95),
                rate_abs_diff_p95=pct(np.abs(f.F[both, 2] - h["F"][both, 2]), 95))
