"""Construction of the main cache: labels, roles, proposals, selection, shells, m, estimator, lambda."""

from dataclasses import dataclass, field

import numpy as np

from recal import policy
from recal.core import Core
from recal.history import reuse_distances

INDEX_BYTES_PER_ENTRY = 164.0   # HNSW per center (M = 16): links (140) plus 4-byte index, 8-byte label, pointer, level
MODEL_BYTES = 80.0
EPS = 1e-9


# ---- labels --------------------------------------------------------------------------------------

def fetch_labels(snap: dict, log, backend, version: int, batch: int = 4096) -> dict:
    """Fetch the backend responses the history lacks; refetch all if any is older than the data version."""
    stale = bool(snap["has"].any() and (snap["version"][snap["has"]] < version).any())
    need = np.ones(len(snap["k"]), bool) if stale else ~snap["has"]
    fetched = 0
    for K in np.unique(snap["k"][need]):
        rows = np.flatnonzero(need & (snap["k"] == K))
        for s in range(0, len(rows), batch):
            r = rows[s:s + batch]
            ids, dists = backend.search(snap["vec"][r], int(K), kind="label")
            log.store_response(snap["index"][r], ids, dists, version)
            snap["ids"][r, :ids.shape[1]] = ids
            snap["dists"][r, :ids.shape[1]] = dists
            snap["has"][r] = True
            snap["version"][r] = version
            fetched += len(r)
    return {"fetched": fetched, "refetched_all": stale}


# ---- roles ---------------------------------------------------------------------------------------

ROLE_NAMES = ("cand", "sel", "train", "cal")


def split_roles(n: int, fractions, rng, vectors: np.ndarray | None = None, by_query: bool = False) -> dict:
    """Shuffle the history and split it by position (default) or by distinct query vector."""
    if not by_query:
        perm = rng.permutation(n)
        edges = np.round(np.cumsum(fractions) / np.sum(fractions) * n).astype(int)
        parts = np.split(perm, edges[:-1])
        return dict(zip(ROLE_NAMES, parts))
    groups: dict = {}
    for i in range(n):
        groups.setdefault(vectors[i].tobytes(), []).append(i)
    keys = list(groups)
    order = rng.permutation(len(keys))
    targets = np.round(np.cumsum(fractions) / np.sum(fractions) * n).astype(int)
    out = {name: [] for name in ROLE_NAMES}
    role, filled = 0, 0
    for g in order:
        members = groups[keys[g]]
        while role < len(ROLE_NAMES) - 1 and filled >= targets[role]:
            role += 1
        out[ROLE_NAMES[role]].extend(members)
        filled += len(members)
    return {name: np.array(sorted(v), dtype=np.int64) for name, v in out.items()}


# ---- proposals -----------------------------------------------------------------------------------

@dataclass
class Proposals:
    """Candidate entries: five sizes per candidate center, from one deep search per center."""
    centers: np.ndarray            # (n_cand, d)
    K: np.ndarray                  # (n_cand,)
    entry_cand: np.ndarray         # (n_entries,)
    entry_j: np.ndarray
    entry_radius: np.ndarray
    resp: list = field(default_factory=list)    # per entry: ids ranked jK
    zone: list = field(default_factory=list)    # per entry: ids ranked jK+1 .. (j+width)K


def propose(backend, centers: np.ndarray, Ks: np.ndarray, p) -> Proposals:
    keep = _first_of_each_center(centers)
    centers, Ks = centers[keep], Ks[keep]
    entries = {"cand": [], "j": [], "radius": [], "resp": [], "zone": []}
    for K in np.unique(Ks):
        rows = np.flatnonzero(Ks == K)
        ids, dists = backend.search(centers[rows], int(p.deep * K), kind="candidate")
        for local, c in enumerate(rows):
            for j in p.sizes:
                n, hi = int(j * K), int((j + p.shell_width) * K)
                if ids.shape[1] < n:
                    continue
                entries["cand"].append(c)
                entries["j"].append(j)
                entries["radius"].append(float(dists[local, n - 1]))
                entries["resp"].append(ids[local, :n])
                entries["zone"].append(ids[local, n:hi])
    return Proposals(centers, Ks, np.array(entries["cand"]), np.array(entries["j"]), np.array(entries["radius"]),
                     entries["resp"], entries["zone"])


def _first_of_each_center(centers: np.ndarray) -> np.ndarray:
    seen, keep = set(), []
    for i, c in enumerate(centers):
        key = c.tobytes()
        if key not in seen:
            seen.add(key)
            keep.append(i)
    return np.array(keep, dtype=np.int64)


# ---- selection -------------------------------------------------------------------------------

def hits_from_hist(H: np.ndarray, levels: np.ndarray, T: float) -> np.ndarray:
    """Longest prefix of the recall-ranked selection queries whose mean recall is at least T.

    H: (rows, n_levels) query counts per recall level, levels ascending. Returns hits per row."""
    Hd = H[:, ::-1].astype(np.float64)
    r = levels[::-1]
    C = np.cumsum(Hd, axis=1)
    S = np.cumsum(Hd * r, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        fail = (C > 0) & (S / C < T - 1e-12)
    first = np.where(fail.any(axis=1), fail.argmax(axis=1), -1)
    out = C[:, -1].copy()
    rows = np.flatnonzero(first >= 0)
    f = first[rows]
    prev = np.maximum(f - 1, 0)
    c_prev = np.where(f > 0, C[rows, prev], 0.0)
    s_prev = np.where(f > 0, S[rows, prev], 0.0)
    x = np.floor((s_prev - T * c_prev) / (T - r[f]) + 1e-9)
    out[rows] = c_prev + np.clip(x, 0, Hd[rows, f])
    return out


class Greedy:
    """Add the entry that fits and gains the most hits per byte; ties go to more recall per byte.

    The selection queries' recall for a pool is the share of their true neighbors the pool holds (the K
    pooled vectors nearest to a query contain every true neighbor in the pool when the backend is exact).
    """

    def __init__(self, prop: Proposals, truth: np.ndarray, truth_k: np.ndarray, d: int, p):
        self.p, self.d = p, d
        n_e = len(prop.entry_cand)
        self.n_e = n_e
        # compact vector ids over everything the entries mention
        all_ids = np.unique(np.concatenate([np.concatenate(prop.resp), np.concatenate(prop.zone)]
                                           if n_e else [np.array([], np.int64)]))
        self.uv = all_ids
        comp = lambda a: np.searchsorted(all_ids, a)   # noqa: E731
        self.r_len = np.array([len(x) for x in prop.resp])
        self.z_len = np.array([len(x) for x in prop.zone])
        self.r_off = np.r_[0, np.cumsum(self.r_len)]
        self.z_off = np.r_[0, np.cumsum(self.z_len)]
        self.r_v = comp(np.concatenate(prop.resp)).astype(np.int64)
        self.z_v = comp(np.concatenate(prop.zone)).astype(np.int64) if self.z_len.sum() else np.array([], np.int64)
        self.r_ent = np.repeat(np.arange(n_e), self.r_len)
        # selection queries: truth ids as compact ids (-1 if no entry mentions them)
        self.n_s = len(truth)
        levels = sorted({c / int(k) for k in np.unique(truth_k) for c in range(int(k) + 1)})
        self.levels = np.array(levels)
        self.n_lv = len(levels)
        self.level_of = {int(k): np.array([levels.index(c / int(k)) for c in range(int(k) + 1)])
                         for k in np.unique(truth_k)}
        self.q_k = truth_k.astype(np.int64)
        pos = np.clip(comp(truth), 0, len(all_ids) - 1)
        known = (all_ids[pos] == truth) & (np.arange(truth.shape[1])[None, :] < truth_k[:, None])
        q_idx, t_idx = np.nonzero(known)
        v_of_truth = pos[q_idx, t_idx]
        order = np.argsort(v_of_truth, kind="stable")
        self.inv_q = q_idx[order].astype(np.int64)
        v_sorted = v_of_truth[order]
        self.inv_start = np.searchsorted(v_sorted, np.arange(len(all_ids) + 1))
        # pairs (entry, query) with the vector that links them, sorted by (entry, query)
        counts = self.inv_start[self.r_v + 1] - self.inv_start[self.r_v]
        pe = np.repeat(self.r_ent, counts)
        starts = np.repeat(self.inv_start[self.r_v], counts)
        within = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        pq = self.inv_q[starts + within]
        pv = np.repeat(self.r_v, counts)
        o = np.lexsort((pq, pe))
        self.pe, self.pq, self.pv = pe[o], pq[o], pv[o]
        if len(self.pe):
            change = np.r_[True, (self.pe[1:] != self.pe[:-1]) | (self.pq[1:] != self.pq[:-1])]
            self.g_start = np.flatnonzero(change)
            self.g_e, self.g_q = self.pe[self.g_start], self.pq[self.g_start]
        else:
            self.g_start = self.g_e = self.g_q = np.array([], np.int64)
        self.in_pool = np.zeros(len(all_ids), bool)
        self.in_shell = np.zeros(len(all_ids), bool)
        self.count = np.zeros(self.n_s, np.int64)
        self.alive = np.ones(n_e, bool)
        self.H = np.zeros(self.n_lv)
        for k, lv in self.level_of.items():
            self.H[lv[0]] += np.sum(self.q_k == k)
        self.entry_fixed = (4.0 * d + 20.0 + INDEX_BYTES_PER_ENTRY)
        self.slot_bytes = 4.0 * d + 12.0 + 12.0          # data slot plus reverse map entry
        self.shell_bytes = float((d + 1) // 2)

    def _level(self, count: np.ndarray, q: np.ndarray) -> np.ndarray:
        out = np.empty(len(q), np.int64)
        for k, lv in self.level_of.items():
            m = self.q_k[q] == k
            out[m] = lv[np.minimum(count[m], k)]
        return out

    def _evaluate(self):
        n_e, n_lv = self.n_e, self.n_lv
        new_pair = (~self.in_pool[self.pv]) & self.alive[self.pe]
        if len(self.g_start):
            d = np.add.reduceat(new_pair.astype(np.int64), self.g_start)
        else:
            d = np.array([], np.int64)
        sel = d > 0
        ge, gq, dd = self.g_e[sel], self.g_q[sel], d[sel]
        old = self._level(self.count[gq], gq)
        new = self._level(self.count[gq] + dd, gq)
        delta = np.zeros((n_e, n_lv))
        np.add.at(delta, (ge, old), -1.0)
        np.add.at(delta, (ge, new), 1.0)
        base = hits_from_hist(self.H[None, :], self.levels, self.p.T)[0]
        gain = hits_from_hist(self.H[None, :] + delta, self.levels, self.p.T) - base
        recall_gain = delta @ self.levels
        new_resp = np.add.reduceat((~self.in_pool[self.r_v]).astype(np.int64), self.r_off[:-1]) \
            if len(self.r_v) else np.zeros(n_e, np.int64)
        new_resp = np.where(self.r_len > 0, new_resp, 0)
        if self.z_len.sum():
            zone_open = ~self.in_pool[self.z_v]
            zone_new = zone_open & ~self.in_shell[self.z_v]
            n_zone = np.add.reduceat(zone_open.astype(np.int64), np.minimum(self.z_off[:-1], len(self.z_v) - 1))
            n_zone_new = np.add.reduceat(zone_new.astype(np.int64), np.minimum(self.z_off[:-1], len(self.z_v) - 1))
            n_zone = np.where(self.z_len > 0, n_zone, 0)
            n_zone_new = np.where(self.z_len > 0, n_zone_new, 0)
        else:
            n_zone = n_zone_new = np.zeros(n_e, np.int64)
        size = (self.entry_fixed + 4.0 * (self.r_len + n_zone) + new_resp * self.slot_bytes
                + n_zone_new * self.shell_bytes)
        return gain, recall_gain, size, d, ge, gq, dd

    def select(self, budget: float, cand_of_entry: np.ndarray):
        """Run the greedy until nothing fits. Returns selected entry indices in order and the bytes used."""
        chosen, used = [], 0.0
        while True:
            if not self.alive.any():
                break
            gain, rgain, size, d, ge, gq, dd = self._evaluate()
            fits = self.alive & (size <= budget - used)
            if not fits.any():
                break
            per_byte = np.where(fits, gain / size, -np.inf)
            tie = np.where(fits, rgain / size, -np.inf)
            best = per_byte.max()
            top = np.flatnonzero(per_byte >= best - 1e-15 * max(abs(best), 1.0))
            e = int(top[np.argmax(tie[top])])
            if gain[e] <= 0 and rgain[e] <= 0:
                break
            chosen.append(e)
            used += float(size[e])
            # update the pool, the charged shell, the counts and the histogram
            r = self.r_v[self.r_off[e]:self.r_off[e + 1]]
            newly = r[~self.in_pool[r]]
            z = self.z_v[self.z_off[e]:self.z_off[e + 1]]
            self.in_pool[r] = True
            z = z[~self.in_pool[z]]
            self.in_shell[z] = True
            mine = ge == e
            q, dq = gq[mine], dd[mine]
            lv_old = self._level(self.count[q], q)
            self.count[q] += dq
            lv_new = self._level(self.count[q], q)
            np.add.at(self.H, lv_old, -1.0)
            np.add.at(self.H, lv_new, 1.0)
            self.alive[cand_of_entry == cand_of_entry[e]] = False
            if len(newly) == 0 and not mine.any():
                continue
        return chosen, used


# ---- mini cache size -----------------------------------------------------------------------------

def mini_cache_size(hashes: np.ndarray, M: float, p) -> int:
    """R = 95th percentile of the reuse distances, at most 10% of M, at least 256 entries."""
    entry = 40.0 + 8.0 * p.K
    cap = int(p.mini_cap * M // entry)
    dist = reuse_distances(hashes)
    r95 = int(np.percentile(dist, p.mini_percentile, method="higher")) if len(dist) else 0
    return max(p.mini_floor, min(r95, cap))


# ---- the construction -------------------------------------------------------------------------------

@dataclass
class Built:
    core: Core
    model: dict
    m: int
    R: int
    edges: np.ndarray
    window: policy.CalibrationWindow
    lam: float | None
    certified: policy.Certified
    h0: float
    W: int
    margin_unreachable: bool
    train_scores: np.ndarray
    centers: np.ndarray
    center_K: np.ndarray
    n_entries: int
    info: dict


def build(snap: dict, backend, p, M: float, rng, request: int, old_centers: np.ndarray | None = None,
          old_K: np.ndarray | None = None, h0_prev: float = 0.2, hashes: np.ndarray | None = None,
          passes: int = 3) -> Built:
    """One construction from a labeled history snapshot. The caller swaps the result in.

    The window is charged to M by its size W, which depends on the hit rate h0 of the cache being built.
    The first pass uses the previous h0; when the true W makes the total exceed M, the build is repeated
    with the measured h0 (same roles and proposals).
    """
    seed = int(rng.integers(2**31))
    memo: dict = {}
    built = _build_once(snap, backend, p, M, seed, request, old_centers, old_K, h0_prev, hashes, memo)
    for _ in range(passes - 1):
        if not built.info["over_budget"]:
            break
        built = _build_once(snap, backend, p, M, seed, request, old_centers, old_K, max(built.h0, 1e-3), hashes,
                            memo)
        built.info["extra_pass"] = True
    return built


def _build_once(snap, backend, p, M, seed, request, old_centers, old_K, h0_prev, hashes, memo) -> Built:
    d = backend.dim
    vec, K, ids, dists = snap["vec"], snap["k"], snap["ids"], snap["dists"]
    rng = np.random.default_rng(seed)
    roles = split_roles(len(vec), p.roles, rng, vec, p.split_by_query)
    # mini cache size, then the byte budget of the main cache
    R = mini_cache_size(hashes if hashes is not None else np.array([], np.uint64), M, p)
    w_est, _ = policy.window_size(p.H, p.a, max(h0_prev, 1e-3), p.window_items)
    fixed = (p.headroom * M + R * (40.0 + 8.0 * p.K) + w_est * p.item_bytes + 32.0 * (p.n_thresholds + 1)
             + MODEL_BYTES + 8.0 * d)
    budget = M - fixed
    # proposals: candidate queries plus the old centers (refetched by the search)
    cand = roles["cand"]
    centers, Ks = vec[cand], K[cand]
    if old_centers is not None and len(old_centers):
        centers = np.vstack([centers, old_centers])
        Ks = np.r_[Ks, old_K]
    if "prop" not in memo:
        memo["prop"] = propose(backend, centers, Ks, p)
    prop = memo["prop"]
    # selection queries: truth from the history labels
    sel = roles["sel"]
    greedy = Greedy(prop, ids[sel][:, :int(K.max())], K[sel], d, p)
    chosen, used = greedy.select(budget, prop.entry_cand)
    # build the core: entries, then shells, then the center index
    core = Core(d, p.k_max)
    pool_ids = np.unique(np.concatenate([prop.resp[e] for e in chosen])) if chosen else np.array([], np.int64)
    zone_ids = np.unique(np.concatenate([prop.zone[e] for e in chosen])) if chosen else np.array([], np.int64)
    all_ids = np.unique(np.concatenate([pool_ids, zone_ids]))
    all_vec = backend.vectors_of(all_ids) if len(all_ids) else np.zeros((0, d), np.float32)
    lo = all_vec.min(axis=0) if len(all_vec) else np.zeros(d, np.float32)
    hi = all_vec.max(axis=0) if len(all_vec) else np.ones(d, np.float32)
    core.set_quantizer(lo, hi)
    core.set_budget(M)
    at = lambda a: all_vec[np.searchsorted(all_ids, a)]   # noqa: E731
    for e in chosen:
        c = prop.entry_cand[e]
        core.add_entry(prop.centers[c], int(prop.K[c]), int(prop.entry_j[e]), prop.resp[e], at(prop.resp[e]),
                       prop.entry_radius[e])
    for n, e in enumerate(chosen):
        if len(prop.zone[e]):
            core.add_shell(n, prop.zone[e], at(prop.zone[e]))
    core.finish_build(16, 200, 64, 100)
    sel_centers = prop.centers[prop.entry_cand[chosen]] if chosen else np.zeros((0, d), np.float32)
    sel_K = prop.K[prop.entry_cand[chosen]] if chosen else np.zeros(0, np.int64)
    # m from the training queries
    train = roles["train"]
    Kt = int(p.K)
    tq = vec[train]
    keep = K[train] == Kt
    train, tq = train[keep], tq[keep]
    needed = np.asarray(core.entries_needed_batch(tq, ids[train][:, :Kt])) if len(tq) and chosen else np.array([])
    needed = needed[needed >= 0]
    n_entries = len(chosen)
    m = int(min(np.percentile(needed, p.m_percentile, method="higher"), n_entries)) if len(needed) else n_entries
    if p.idealized:
        m = 0
    # train the recall estimator on the training queries' candidate responses
    feat, cand_ids, _ = core.candidates_batch(tq, Kt, m) if len(tq) else (np.zeros((0, 7)), None, None)
    ok = feat[:, 0] == 1
    F = feat[ok][:, [2, 3, 4]]
    y = _recall(cand_ids[ok], ids[train][ok][:, :Kt]) if ok.any() else np.array([])
    model = policy.logistic_model(F, y) if len(y) >= 10 else {"mean": np.zeros(3), "scale": np.ones(3),
                                                                "coef": np.zeros(3), "intercept": 0.0}
    core.set_model(model["mean"], model["scale"], model["coef"], model["intercept"])
    train_scores = policy.score_features(F, model) if len(y) else np.array([0.0])
    edges = policy.threshold_edges(train_scores, p.n_thresholds)
    # seed the window with the calibration queries scored on the new cache, all with weight 1
    cal = roles["cal"]
    cal = cal[K[cal] == Kt]
    cfeat, ccand, _ = core.candidates_batch(vec[cal], Kt, m) if len(cal) and chosen else (np.zeros((0, 7)), None, None)
    cok = cfeat[:, 0] == 1
    cscore = policy.score_features(cfeat[cok][:, [2, 3, 4]], model) if cok.any() else np.array([])
    cy = _recall(ccand[cok], ids[cal][cok][:, :Kt]) if cok.any() else np.array([])
    window = policy.CalibrationWindow(edges, p.H, p.T, p.delta, p.min_ess)
    for s, yy in zip(cscore, cy):
        window.add(request, float(s), float(yy), 1.0)
    cert = window.certify()
    h0 = float((cscore >= cert.lam).sum() / len(cal)) if cert.lam is not None and len(cal) else 0.0
    W, unreachable = policy.window_size(p.H, p.a, max(h0, 1e-9), p.window_items)
    window.W = W
    window.expire(request)
    core.set_external_bytes("calibration_window", float(W * p.item_bytes + 4 * 8 * (len(edges) + 1)))
    mini_bytes = R * (40.0 + 8.0 * p.K)
    total = core.bytes_by_structure()["total"]
    info = {"total_bytes": total, "over_budget": total + mini_bytes > M, "entries": n_entries, "used_bytes": used, "budget": budget, "R": R, "mini_bytes": mini_bytes,
            "m": m, "n_train": int(len(y)), "n_cal": int(len(cscore)), "roles": {k: len(v) for k, v in roles.items()},
            "candidates": int(len(prop.centers)), "proposed_entries": int(len(prop.entry_cand))}
    return Built(core, model, m, R, edges, window, cert.lam, cert, h0, W, unreachable, train_scores, sel_centers,
                 sel_K, n_entries, info)


def _recall(cand_ids: np.ndarray, truth_ids: np.ndarray) -> np.ndarray:
    """Share of the true neighbors each candidate response holds."""
    out = np.empty(len(cand_ids))
    for i in range(len(cand_ids)):
        out[i] = np.isin(truth_ids[i], cand_ids[i]).mean()
    return out
