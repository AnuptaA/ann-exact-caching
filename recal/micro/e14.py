"""E14: a tiered calibration window, precise vectors only where certification can look.

The held-out stream is labeled by the audit rule (misses weigh 1, audited hits 1/a; audit, half and keep decisions
are per-request uniform draws fixed by the seed, independent of anything later). Half A trains and stores no vector;
half B certifies lambda with the worst-case scan of recal.micro.intervals. lambda_floor is the score of the first
failing threshold of the current scan minus a margin; it is taken from the half-B items of the W requests before the
item's block (the history's calibration role while fewer than W/2 requests have passed).
  class P: audited hits and items scoring at least lambda_floor at arrival, kept with a precise copy (16 bits, native)
  class C: the rest of half B, each kept with probability p (weight times 1/p) with a coarse copy (4 or 8 bits)
The reference keeps every half-B item at 16 bits. Each certified lambda is applied to the next W/2 requests.
Cache change is emulated by caches built on histories shifted forward by 10% and 25% of the history length; items
are rescored against them (same model, thresholds and labels; features, recall and intervals recomputed) only when
they arrive after the end of the shifted history.
"""

import shutil
import tempfile
from statistics import NormalDist

import numpy as np

from recal import ksets
from recal.micro import e10, e11, fit, intervals as iv, quantize, world as w
from recal.stored import StoredAnswers
from recal.system import ReCal

NAME = "E14"
ESS_TARGET = 400
SHIFTS = (0.10, 0.25)
SEEDS = 5
P_VALUES = (0.1, 0.25, 0.5)
COARSE_BITS = (4, 8)
PRECISE_BITS = (16, "native")
MARGINS = (0.0, 0.025, 0.05, 0.1)
PRIMARY = dict(precise=16, coarse=4, p=0.25, margin=0.05)
STRIDE_PARTS = 10
UNDER_T_GAP = 0.01
LOOSER_GAP = 0.005
MAX_GRID = 20_000_000
BOOT = 2000
QUANTILES = (5, 25, 50, 75, 95)


def configs() -> list[dict]:
    """Every precise/coarse/p combination at the primary margin, and the margin sweep at the primary p."""
    out = [dict(precise=pb, coarse=cb, p=p, margin=PRIMARY["margin"])
           for pb in PRECISE_BITS for cb in COARSE_BITS for p in P_VALUES]
    out += [dict(precise=16, coarse=cb, p=PRIMARY["p"], margin=m)
            for cb in COARSE_BITS for m in MARGINS if m != PRIMARY["margin"]]
    return out


def config_key(c: dict) -> str:
    return f"P{c['precise']}_C{c['coarse']}_p{c['p']:g}_d{c['margin']:g}"


# ---- certification with any number of weight classes ------------------------------------------------------------

def worst_case_bound(y: np.ndarray, wt: np.ndarray, certain: np.ndarray, ambiguous: np.ndarray, z: float):
    """recal.micro.intervals.worst_case_bound for any number of distinct weights among the ambiguous items (here
    1, 1/a and 1/p): the same bound, minimized over the count of ambiguous items taken from each weight class."""
    yc, wc = y[certain], wt[certain]
    items = np.flatnonzero(ambiguous)
    weights = np.unique(wt[items]) if len(items) else np.zeros(0)
    classes = [(float(c), np.sort(y[items[wt[items] == c]])) for c in weights]
    shape = [len(v) + 1 for _, v in classes]
    if int(np.prod(shape)) > MAX_GRID:
        raise ValueError(f"worst-case grid {shape} exceeds {MAX_GRID} cells")
    ks = [np.arange(n).reshape([-1 if j == i else 1 for j in range(len(shape))]) for i, n in enumerate(shape)]

    def prefix(values: np.ndarray, power: int) -> np.ndarray:
        return np.r_[0.0, np.cumsum(values ** power)]

    W, SW2 = wc.sum() + np.zeros(shape), (wc * wc).sum() + np.zeros(shape)
    S1, S2 = (wc * yc).sum() + np.zeros(shape), (wc * yc * yc).sum() + np.zeros(shape)
    for (c, v), k in zip(classes, ks):
        W, SW2 = W + c * k, SW2 + c * c * k
        S1, S2 = S1 + c * prefix(v, 1)[k], S2 + c * prefix(v, 2)[k]
    nonempty = W > 0
    if not nonempty.any():
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        mean, ess = S1 / W, W * W / SW2
        denom = np.maximum(W - SW2 / W, 1e-12)
        g = np.maximum(S2 - W * mean ** 2, 0.0)
        feasible = np.where(nonempty, mean - z * np.sqrt(g / denom) / np.sqrt(ess), np.inf)
        best = np.unravel_index(np.argmin(feasible), feasible.shape)
        t = float(mean[best])
        R = (wc * (yc - t) ** 2).sum() + np.zeros(shape)
        for (c, v), k in zip(classes, ks):
            R = R + c * prefix(np.sort((v - t) ** 2)[::-1], 1)[k]
        bound = np.where(nonempty, S1 / W - z * np.sqrt(R / denom) / np.sqrt(ess), np.inf)
    return dict(lcb=float(bound.min()), lcb_prefix=float(feasible[best]),
                ess=float(np.where(nonempty, ess, np.inf).min()), n_ambiguous=int(len(items)))


def scan(edges: np.ndarray, low: np.ndarray, high: np.ndarray, y: np.ndarray, wt: np.ndarray, target: float,
         delta: float, min_ess: float) -> dict:
    """certify_worst_case's scan (strictest first, skip low ESS, stop at the first failure); also returns the
    first failing threshold (None when none fails)."""
    z = NormalDist().inv_cdf(1.0 - delta)
    out = dict(lam=None, first_fail=None, ess=0.0, n_ambiguous=0)
    for lam in np.asarray(edges)[::-1]:
        certain = low >= lam
        stats = worst_case_bound(y, wt, certain, (high >= lam) & ~certain, z)
        if stats is None or not np.isfinite(stats["ess"]) or stats["ess"] < min_ess:
            continue
        if stats["lcb"] >= target:
            out.update(lam=float(lam), ess=stats["ess"], n_ambiguous=stats["n_ambiguous"])
        else:
            out["first_fail"] = float(lam)
            break
    return out


def floor_of(first_fail, margin: float) -> float:
    """lambda_floor; no failing threshold means every item may matter (-inf)."""
    return -np.inf if first_fail is None else first_fail - margin


# ---- the stream, labels and classes -----------------------------------------------------------------------------

def draws(n: int, seed: int) -> dict:
    """Per-request uniforms for the audit, the half and the coarse keep decision."""
    u = np.random.default_rng([14, seed]).random((3, n))
    return dict(audit=u[0], half=u[1], keep=u[2])


def labels(hit: np.ndarray, has: np.ndarray, d: dict, a: float) -> dict:
    audited = hit & (d["audit"] < a)
    labeled = has & (~hit | audited)
    half_b = d["half"] < 0.5
    return dict(audited=audited, labeled=labeled, B=labeled & half_b, A=labeled & ~half_b,
                weight=np.where(hit, 1.0 / a, 1.0))


def classes(lab: dict, score: np.ndarray, floor: np.ndarray, keep: np.ndarray, p: float) -> dict:
    """P and C at arrival; C items kept with probability p. Precision is decided once, here."""
    P = lab["B"] & (lab["audited"] | (score >= floor))
    C = lab["B"] & ~P
    return dict(P=P, C=C, kept=C & (keep < p))


def arrival_first_fail(n: int, W: int, block: int, B: np.ndarray, low, high, y, weight, edges, p, cal_fail) -> np.ndarray:
    """First failing threshold in force for each request: from the half-B items of the W requests before its block,
    or the calibration role's while fewer than W/2 requests have passed."""
    out = np.full(n, np.nan)
    for start in range(0, n, block):
        if start < W // 2:
            ff = cal_fail
        else:
            sl = slice(max(0, start - W), start)
            m = B[sl]
            ff = scan(edges, low[sl][m], high[sl][m], y[sl][m], weight[sl][m], p.T, p.delta, p.min_ess)["first_fail"]
        out[start:start + block] = np.nan if ff is None else ff
    return out


def windows(n: int, W: int, eval_len: int, stride: int) -> list[int]:
    return list(range(0, n - W - eval_len + 1, max(stride, 1)))


def applied(lam, score, has, y) -> dict:
    hit = fit.decide(score, has, lam)
    hits = int(hit.sum())
    return dict(hit_rate=float(hit.mean()) if len(hit) else float("nan"), hits=hits,
                recall_of_hits=float(y[hit].sum() / hits) if hits else float("nan"))


# ---- snapshots ---------------------------------------------------------------------------------------------------

def shifted_world(world: w.World, offset: int) -> w.World:
    """The cache built the same way on the history shifted forward by `offset` requests."""
    idx = world.stream[offset:]
    hl = world.history_len
    unique = np.unique(world.stream)
    backend = StoredAnswers(world.db, world.q[unique], world.top[unique])
    work = tempfile.mkdtemp(prefix="micro_e14_")
    try:
        system = ReCal(backend, world.p, world.M, work)
        system.ingest_history(np.asarray(world.q[idx[:hl]], dtype=np.float32), ksets.k_sequence(hl, world.k))
        built = system.built
        snap = {key: np.array(value) for key, value in system.log.snapshot().items()}
    finally:
        shutil.rmtree(work, ignore_errors=True)
    role = w.roles_of(snap, world.p, built)
    out = w.World(world.name, world.budget, world.p, world.M, world.db, world.q, world.top, idx, hl, built, snap,
                  role, world.k)
    out.workers = world.workers
    return out


def snapshot_scores(world: w.World, model: dict, vec: np.ndarray, truth: np.ndarray, feats=None) -> dict:
    """Exact score, candidate, recall and the score interval of each copy (4, 8, 16 bits, native) per unique query."""
    feats = feats if feats is not None else iv.Features(world, c=e10.NEAREST)
    data = e10.labeled_set(world, feats, model, vec, truth, np.ones(len(vec)))
    ctx = dict(feats=feats, model=model)
    out = dict(score=data["score"], has=data["has"], y=data["y"])
    for bits in COARSE_BITS + PRECISE_BITS:
        d = e10.interval_detail(ctx, data, bits)
        out[bits] = (np.where(data["has"], d["low"], -np.inf), np.where(data["has"], d["high"], -np.inf))
    return out


def eligible_rows(n_held: int, history_len: int, offset: int) -> np.ndarray:
    """Held-out positions that arrive after the shifted history [offset, offset + history_len) ends."""
    return np.arange(n_held) + history_len >= offset + history_len


# ---- the run -----------------------------------------------------------------------------------------------------

def run(world: w.World, request_limit: int = 0, seeds: int = SEEDS, shifts=SHIFTS, ess_target: float = ESS_TARGET,
        log=print) -> tuple[list[dict], dict]:
    ctx = e10.context(world)
    if ctx["model"] is None or not ctx["cal"]["has"].any():
        return [], {}
    p, cal, model, edges = world.p, ctx["cal"], ctx["model"], ctx["edges"]
    ok = cal["has"]
    lam0 = fit.certify(ctx["train_scores"], cal["score"][ok], cal["y"][ok], p.T, p.delta, p.min_ess).lam
    cal16 = e10.interval_detail(ctx, cal, 16)
    cal_fail = scan(edges, cal16["low"][ok], cal16["high"][ok], cal["y"][ok], np.ones(int(ok.sum())), p.T, p.delta,
                    p.min_ess)["first_fail"]
    stream = world.full_heldout_stream()
    if request_limit:
        stream = stream[:request_limit]
    n = len(stream)
    uniq, pos = np.unique(stream, return_inverse=True)
    vec = np.asarray(world.q[uniq], dtype=np.float32)
    truth = np.asarray(world.top[uniq][:, :world.k], np.int64)
    snaps = [dict(shift=0.0, offset=0, data=snapshot_scores(world, model, vec, truth, ctx["feats"]))]
    log(f"[E14] snapshot 0 scored: {n} requests, {len(uniq)} unique queries")
    for share in shifts:
        offset = int(round(share * world.history_len))
        sw = shifted_world(world, offset)
        snaps.append(dict(shift=share, offset=offset, data=snapshot_scores(sw, model, vec, truth)))
        log(f"[E14] snapshot shift {share:g} (offset {offset}) built and scored")
    for s in snaps:
        s["req"] = {key: (v[0][pos], v[1][pos]) if isinstance(v, tuple) else v[pos] for key, v in s["data"].items()}
        s["eligible"] = eligible_rows(n, world.history_len, s["offset"])
    base = snaps[0]["req"]
    hit0 = fit.decide(base["score"], base["has"], lam0)
    h = float(hit0.mean())
    W = int(round(ess_target / (p.a * max(h, 1e-9))))
    W = min(W, max(n // 2, 1))
    eval_len = max(W // 2, 1)
    plans = [("W", W, windows(n, W, eval_len, W // STRIDE_PARTS)), ("W/2", W // 2, windows(n, W // 2, eval_len, W // 2))]
    rows, margin_rows = [], []
    for seed in range(seeds):
        d = draws(n, seed)
        lab = labels(hit0, base["has"], d, p.a)
        lo16, hi16 = base[16]
        ff_arrival = arrival_first_fail(n, W, max(W // STRIDE_PARTS, 1), lab["B"], lo16, hi16, base["y"], lab["weight"],
                                        edges, p, cal_fail)
        cls = {}
        for m in MARGINS:
            floor = np.where(np.isnan(ff_arrival), -np.inf, ff_arrival - m)
            for pv in P_VALUES:
                cls[(m, pv)] = classes(lab, base["score"], floor, d["keep"], pv)
            if m == PRIMARY["margin"]:
                arrival_floor = floor
        margin_rows.append(arrival_margins(seed, lab, base, arrival_floor, cls[(PRIMARY["margin"], PRIMARY["p"])]))
        for kind, length, starts in plans:
            for start in starts:
                win = slice(start, start + length)
                nxt = slice(start + length, start + length + eval_len)
                for s in snaps:
                    if start + length < s["offset"]:
                        continue                       # the next stretch would be scored against its own history
                    rows += window_rows(seed, kind, length, start, win, nxt, s, snaps[0], lab, cls, arrival_floor,
                                        edges, p, margin_rows)
        log(f"[E14] seed {seed} done")
    summary = summarize(rows, margin_rows, world, dict(n_requests=n, n_unique=len(uniq), hit_rate_policy=h, lam_policy=lam0,
                                                     W=W, eval_len=eval_len, a=p.a, T=p.T,
                                                     offsets=[s["offset"] for s in snaps],
                                                     n_windows={k: len(v) for k, _, v in plans}))
    return rows, summary


def arrival_margins(seed: int, lab: dict, base: dict, floor: np.ndarray, cl: dict) -> dict:
    """Score minus floor at arrival for half-B items (the primary margin), and the class shares."""
    B = lab["B"]
    gap = base["score"][B] - floor[B]
    gap = gap[np.isfinite(gap)]
    out = dict(kind="arrival", seed=seed, n_B=int(B.sum()), share_P=float(cl["P"][B].mean()) if B.any() else np.nan,
               share_C=float(cl["C"][B].mean()) if B.any() else np.nan,
               share_audited=float(lab["audited"][B].mean()) if B.any() else np.nan,
               score_minus_floor=pct(gap))
    for bits in COARSE_BITS:
        lo, hi = base[bits]
        C = cl["C"]
        half = (hi[C] - lo[C]) / 2.0
        reach = hi[C] >= floor[C]
        out[f"C{bits}_upper_reaches_floor_share"] = float(reach.mean()) if C.any() else np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"C{bits}_floor_minus_upper_halfwidths"] = pct(((floor[C] - hi[C]) / half)[half > 0])
    return out


def pct(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    return {f"p{q}": float(np.percentile(v, q)) for q in QUANTILES} | dict(n=int(len(v))) if len(v) else dict(n=0)


def window_rows(seed, kind, length, start, win, nxt, s, s0, lab, cls, arrival_floor, edges, p, margin_rows) -> list:
    """Reference and every tiered configuration on one window under one snapshot, applied to the next stretch."""
    r = s["req"]
    elig = s["eligible"]
    inwin = lab["B"][win] & elig[win]
    y, wt = r["y"][win], lab["weight"][win]
    lo16, hi16 = r[16][0][win], r[16][1][win]
    ref = scan(edges, lo16[inwin], hi16[inwin], y[inwin], wt[inwin], p.T, p.delta, p.min_ess)
    common = dict(seed=seed, window=kind, W=length, start=start, shift=s["shift"], n_B=int(inwin.sum()),
                  n_A=int((lab["A"][win] & elig[win]).sum()), lam_ref=ref["lam"], ess_ref=ref["ess"],
                  first_fail_ref=ref["first_fail"], **{f"{k}_ref": v for k, v in
                                                       applied(ref["lam"], r["score"][nxt], r["has"][nxt],
                                                               r["y"][nxt]).items()})
    if s["shift"] > 0 and kind == "W":
        b = s0["req"]
        ff0 = scan(edges, b[16][0][win][inwin], b[16][1][win][inwin], b["y"][win][inwin], wt[inwin], p.T, p.delta,
                   p.min_ess)["first_fail"]
        common["first_fail_same_items_shift0"] = ff0
        margin_rows.append(shift_margins(seed, start, s, win, inwin, cls, ref["first_fail"], ff0, arrival_floor))
    out = []
    for c in configs():
        cl = cls[(c["margin"], c["p"])]
        P, kept = cl["P"][win] & inwin, cl["kept"][win] & inwin
        use = P | kept
        plo, phi = r[c["precise"]][0][win], r[c["precise"]][1][win]
        clo, chi = r[c["coarse"]][0][win], r[c["coarse"]][1][win]
        low, high = np.where(P, plo, clo), np.where(P, phi, chi)
        weights = wt * np.where(kept, 1.0 / c["p"], 1.0)
        t = scan(edges, low[use], high[use], y[use], weights[use], p.T, p.delta, p.min_ess)
        res = applied(t["lam"], r["score"][nxt], r["has"][nxt], r["y"][nxt])
        out.append(dict(common, config=config_key(c), **c, n_P=int(P.sum()), n_C=int((cl["C"][win] & inwin).sum()),
                        n_C_kept=int(kept.sum()), lam=t["lam"], ess=t["ess"], n_ambiguous=t["n_ambiguous"],
                        **res))
    return out


def lowest_finite(values: np.ndarray) -> float:
    v = values[np.isfinite(values)]
    return float(v.min()) if len(v) else np.nan


def shift_margins(seed, start, s, win, inwin, cls, ff_new, ff_old, arrival_floor) -> dict:
    """C items (primary margin) under the shifted cache: is the new interval's upper end below the new floor?"""
    r = s["req"]
    m = PRIMARY["margin"]
    C = cls[(m, PRIMARY["p"])]["C"][win] & inwin
    floor = floor_of(ff_new, m)
    out = dict(kind="shift", seed=seed, start=start, shift=s["shift"], n_C=int(C.sum()), first_fail_new=ff_new,
               first_fail_old_same_items=ff_old, floor_new=floor,
               arrival_floor_mean=float(np.mean(arrival_floor[win][inwin])) if inwin.any() else np.nan,
               arrival_floor_min=lowest_finite(arrival_floor[win][inwin]),
               crossed_share=float((r["score"][win][C] >= floor).mean()) if C.any() else np.nan)
    if ff_new is None:      # no failing threshold under the new cache: no floor, every item may matter
        out["crossed_share"] = np.nan
        C = np.zeros_like(C)
    for bits in COARSE_BITS:
        lo, hi = r[bits][0][win][C], r[bits][1][win][C]
        half = (hi - lo) / 2.0
        below = hi < floor
        out[f"C{bits}_below_floor_share"] = float(below.mean()) if C.any() else np.nan
        out[f"C{bits}_crossed_but_below_share"] = float((below & (r["score"][win][C] >= floor)).mean()) if C.any() \
            else np.nan
        out[f"C{bits}_ambiguous_share"] = float((~below).mean()) if C.any() else np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"C{bits}_margin_halfwidths"] = ((floor - hi) / half)[half > 0]
    return out


# ---- summary -----------------------------------------------------------------------------------------------------

def item_bytes(dim: int, k: int) -> dict:
    """Bytes of a half-A item and of a half-B item kept at each precision (E11's layout)."""
    out = {bits: e11.item_bytes(dim, k, bits)[1] for bits in (4, 8, 16, "native")}
    out["A"] = e11.item_bytes(dim, k, 16)[0]
    return out


def block_bootstrap(values: np.ndarray, block: int, rng) -> tuple[float, float]:
    """95% moving-block bootstrap interval of the mean (block windows overlap by construction)."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = len(v)
    if n < 2:
        return float("nan"), float("nan")
    block = max(1, min(block, n))
    starts = np.arange(n - block + 1)
    per = -(-n // block)
    means = [np.concatenate([v[s:s + block] for s in rng.choice(starts, per)])[:n].mean() for _ in range(BOOT)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(rows: list[dict], margin_rows: list[dict], world: w.World, info: dict) -> dict:
    T = info["T"]
    dim, k = world.db.shape[1], world.k
    size = item_bytes(dim, k)
    rng = np.random.default_rng(14)
    out = dict(info, workload=world.name, budget=world.budget, item_bytes={str(b): v for b, v in size.items()},
               native_info=quantize.native_info(world.q))
    out["n_independent_windows"] = info["n_requests"] / (1.5 * info["W"])
    base = [r for r in rows if r["window"] == "W" and r["shift"] == 0.0]
    memory = {}
    for c in configs():
        if c["margin"] != PRIMARY["margin"]:
            continue
        mine = [r for r in base if r["config"] == config_key(c)]
        nA, nB = np.mean([r["n_A"] for r in mine]), np.mean([r["n_B"] for r in mine])
        nP, nC, nK = (np.mean([r[key] for r in mine]) for key in ("n_P", "n_C", "n_C_kept"))
        full16 = nA * size["A"] + nB * size[16]
        tiered = nA * size["A"] + nP * size[c["precise"]] + nK * size[c["coarse"]]
        memory[config_key(c)] = dict(n_A=nA, n_B=nB, n_P=nP, n_C=nC, n_C_kept=nK, bytes_all16=full16,
                                     bytes_all8=nA * size["A"] + nB * size[8], bytes_tiered=tiered,
                                     ratio_all16_over_tiered=full16 / tiered if tiered else float("nan"),
                                     ratio_all16_over_all8=full16 / (nA * size["A"] + nB * size[8]))
    out["memory"] = memory
    arrival = [m for m in margin_rows if m["kind"] == "arrival"]
    out["arrival"] = dict(share_P=float(np.mean([m["share_P"] for m in arrival])),
                          share_C=float(np.mean([m["share_C"] for m in arrival])),
                          share_audited=float(np.mean([m["share_audited"] for m in arrival])),
                          score_minus_floor_seed0=arrival[0]["score_minus_floor"] if arrival else {},
                          **{f"C{b}_upper_reaches_floor_share": float(np.mean([m[f"C{b}_upper_reaches_floor_share"]
                                                                               for m in arrival]))
                             for b in COARSE_BITS},
                          **{f"C{b}_floor_minus_upper_halfwidths_seed0": arrival[0][f"C{b}_floor_minus_upper_halfwidths"]
                             for b in COARSE_BITS if arrival})
    shifted = {}
    for share in sorted({m["shift"] for m in margin_rows if m["kind"] == "shift"}):
        mine = [m for m in margin_rows if m["kind"] == "shift" and m["shift"] == share]
        drift = np.array([np.nan if m["first_fail_new"] is None or m["first_fail_old_same_items"] is None
                          else m["first_fail_new"] - m["first_fail_old_same_items"] for m in mine], dtype=float)
        arr = np.array([np.nan if m["first_fail_new"] is None else m["floor_new"] - m["arrival_floor_min"]
                        for m in mine], dtype=float)
        entry = dict(n_window_seeds=len(mine), no_floor_share=float(np.mean([m["first_fail_new"] is None for m in mine])),
                     crossed_share=wmean(mine, "crossed_share"),
                     floor_drift_same_items=pct(drift),
                     floor_drift_none_share=float(np.isnan(drift).mean()) if len(drift) else float("nan"),
                     implied_margin_p95=float(np.percentile(np.maximum(-drift[np.isfinite(drift)], 0), 95))
                     if np.isfinite(drift).any() else float("nan"),
                     new_floor_minus_lowest_arrival_floor=pct(arr))
        for b in COARSE_BITS:
            entry[f"C{b}_below_floor_share"] = wmean(mine, f"C{b}_below_floor_share")
            entry[f"C{b}_ambiguous_share"] = wmean(mine, f"C{b}_ambiguous_share")
            entry[f"C{b}_crossed_but_below_share"] = wmean(mine, f"C{b}_crossed_but_below_share")
            entry[f"C{b}_margin_halfwidths"] = pct(np.concatenate([m[f"C{b}_margin_halfwidths"] for m in mine])
                                                   if mine else np.zeros(0))
        shifted[f"{share:g}"] = entry
    out["after_shift"] = shifted
    cert = {}
    for kind in ("W", "W/2"):
        for share in sorted({r["shift"] for r in rows}):
            for c in configs():
                key = config_key(c)
                mine = [r for r in rows if r["window"] == kind and r["shift"] == share and r["config"] == key]
                if mine:
                    cert[f"{kind}|{share:g}|{key}"] = compare(mine, T, STRIDE_PARTS if kind == "W" else 1, rng)
    out["certification"] = cert
    out["verdict"] = verdict(out)
    return out


def wmean(rows: list[dict], key: str) -> float:
    """Mean share over windows, weighted by the C items each holds."""
    num = sum(r[key] * r["n_C"] for r in rows if r["n_C"] and np.isfinite(r[key]))
    den = sum(r["n_C"] for r in rows if r["n_C"] and np.isfinite(r[key]))
    return float(num / den) if den else float("nan")


def compare(rows: list[dict], T: float, block: int, rng) -> dict:
    """Reference against tiered over windows: means, paired differences (block bootstrap over windows, the seeds
    averaged per window), under-T shares and how often the tiered lambda is looser."""
    starts = sorted({r["start"] for r in rows})
    lam = lambda v: np.nan if v is None else v  # noqa: E731
    diff_hit, diff_roh = [], []
    for s in starts:
        here = [r for r in rows if r["start"] == s]
        diff_hit.append(np.mean([r["hit_rate"] - r["hit_rate_ref"] for r in here]))
        diff_roh.append(np.nanmean([r["recall_of_hits"] - r["recall_of_hits_ref"] for r in here])
                        if any(r["hits"] and r["hits_ref"] for r in here) else np.nan)
    under = lambda key, hits: float(np.mean([r[hits] > 0 and r[key] < T - UNDER_T_GAP for r in rows]))  # noqa: E731
    both = [r for r in rows if r["lam"] is not None and r["lam_ref"] is not None]
    return dict(
        n_windows=len(starts), n_rows=len(rows),
        certified_ref=float(np.mean([r["lam_ref"] is not None for r in rows])),
        certified_tiered=float(np.mean([r["lam"] is not None for r in rows])),
        lam_ref=float(np.nanmean([lam(r["lam_ref"]) for r in rows])), lam_tiered=float(np.nanmean([lam(r["lam"]) for r in rows])),
        hit_rate_ref=float(np.mean([r["hit_rate_ref"] for r in rows])),
        hit_rate_tiered=float(np.mean([r["hit_rate"] for r in rows])),
        recall_of_hits_ref=float(np.nanmean([r["recall_of_hits_ref"] for r in rows])),
        recall_of_hits_tiered=float(np.nanmean([r["recall_of_hits"] for r in rows])),
        ess_ref=float(np.mean([r["ess_ref"] for r in rows])), ess_tiered=float(np.mean([r["ess"] for r in rows])),
        ambiguous_tiered=float(np.mean([r["n_ambiguous"] for r in rows])),
        hit_diff=float(np.nanmean(diff_hit)), hit_diff_ci=block_bootstrap(np.array(diff_hit), block, rng),
        roh_diff=float(np.nanmean(diff_roh)) if np.isfinite(diff_roh).any() else float("nan"),
        roh_diff_ci=block_bootstrap(np.array(diff_roh), block, rng),
        under_T_ref=under("recall_of_hits_ref", "hits_ref"), under_T_tiered=under("recall_of_hits", "hits"),
        looser_share=float(np.mean([r["lam"] < r["lam_ref"] - LOOSER_GAP for r in both])) if both else float("nan"))


def verdict(out: dict) -> dict:
    """The rule of thumb on the primary configuration (P 16-bit, C 4-bit, p 0.25, margin 0.05)."""
    key = config_key(PRIMARY)
    mem = out["memory"].get(key, {})
    cert = out["certification"].get(f"W|0|{key}", {})
    last = out["after_shift"].get(f"{max(SHIFTS):g}", {})
    amb = last.get(f"C{PRIMARY['coarse']}_ambiguous_share", float("nan"))
    return dict(config=key,
                memory_ratio=mem.get("ratio_all16_over_tiered"), memory_ok=bool(mem.get("ratio_all16_over_tiered", 0) >= 3),
                hit_diff=cert.get("hit_diff"), hit_ok=bool(abs(cert.get("hit_diff", np.inf)) <= 0.005),
                under_T_ref=cert.get("under_T_ref"), under_T_tiered=cert.get("under_T_tiered"),
                independent_windows=out["n_independent_windows"],
                ambiguous_after_largest_shift=amb, ambiguous_ok=bool(amb <= 0.10))
