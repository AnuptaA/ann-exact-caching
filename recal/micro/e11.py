"""E11: window memory against certification, with scores kept at 8 or 16 bits.

A calibration item holds d p / 8 + 4K + 33 bytes (the quantized vector, its error, 4K ids and bookkeeping), a
training item 4K + 31 (no vector). A window share sigma of M holds n items per half; the window spans
W = 2n / (labeled share) main-path requests. As in E8, lambda is certified on the calibration half of consecutive
stretches of the held-out stream (misses weigh 1, audited hits 1/a) and applied to the next stretch; the
certification uses the score intervals of the stored copies (recal.micro.intervals).
"""

import numpy as np

from recal.micro import e10, fit, intervals as iv, quantize, world as w

NAME = "E11"
SHARES = (0.005, 0.01, 0.02, 0.03, 0.05)
PRECISIONS = (8, 16)
DELTAS = (0.05, 0.10)
TRAIN_FIXED_BYTES = 31
CAL_FIXED_BYTES = 33


def pick_budgets(budgets: list[float]) -> list[float]:
    """The smallest and the middle budget."""
    ordered = sorted(budgets)
    return sorted({ordered[0], ordered[len(ordered) // 2]}) if ordered else []


def item_bytes(dim: int, k: int, bits) -> tuple[int, float]:
    """(training item, calibration item) bytes; bits "native" is int8 data kept exactly (8 bits per coordinate)."""
    width = 8 if bits == "native" else bits
    return 4 * k + TRAIN_FIXED_BYTES, dim * width / 8.0 + 4 * k + CAL_FIXED_BYTES


def items_per_half(share: float, M: float, dim: int, k: int, bits) -> int:
    train, cal = item_bytes(dim, k, bits)
    return int(share * M // (train + cal))


def ess_of(weights: np.ndarray) -> float:
    """Kish effective sample size (sum w)^2 / sum w^2 (0 for no items)."""
    return float(weights.sum() ** 2 / (weights ** 2).sum()) if len(weights) else 0.0


def stretch_rows(low, high, score, has, y, idx, W: int, policy_lam, edges, p, delta: float, rng) -> list[dict]:
    n = len(idx)
    size = min(W, n)
    if size < 1:
        return []
    n_stretch = max(n // size, 1)
    out = []
    for j in range(n_stretch):
        sl = idx[j * size:(j + 1) * size]
        hit = fit.decide(score[sl], has[sl], policy_lam)
        audited = hit & (rng.random(len(sl)) < p.a)
        labeled = has[sl] & (~hit | audited)
        use = labeled & (rng.random(len(sl)) < 0.5)
        weights = np.where(hit[use], 1.0 / p.a, 1.0)
        cert = iv.certify_worst_case(edges, low[sl][use], high[sl][use], y[sl][use], p.T, delta, p.min_ess, weights) \
            if use.any() else dict(lam=None)
        blocked = False
        if cert["lam"] is None and use.any():     # would it certify if only the effective sample size were waived?
            relaxed = iv.certify_worst_case(edges, low[sl][use], high[sl][use], y[sl][use], p.T, delta, 1.0, weights)
            blocked = relaxed["lam"] is not None
        row = dict(certified=cert["lam"] is not None, in_sample=n_stretch == 1, labeled=int(use.sum()),
                   audited_hits=int(hit[use].sum()), n_eff_served=ess_of(weights[hit[use]]),
                   n_eff_all_labeled=ess_of(weights), min_ess_blocks=blocked)
        if cert["lam"] is not None:
            row.update(sampling_slack=cert["slack"], quantization_slack=cert["quantization_slack"],
                       bound_gap=cert["bound_gap"], ess_min_certified=cert["ess"],
                       ambiguous=cert["n_ambiguous"] / max(int(use.sum()), 1))
        nxt = idx[(j + 1) * size:(j + 2) * size] if j + 1 < n_stretch else (sl if n_stretch == 1 else None)
        if nxt is not None:
            applied = fit.decide(score[nxt], has[nxt], cert["lam"])
            row.update(hit_rate=float(applied.mean()), hits=int(applied.sum()), recall_sum=float(y[nxt][applied].sum()))
        out.append(row)
    return out


def run(world: w.World, request_limit: int = 0, shares=SHARES, precisions=PRECISIONS, deltas=DELTAS) -> list[dict]:
    """Every request of the held-out stream (or its first request_limit) is scored once, with the exact features of
    its query and the interval of its copy; a stretch is W consecutive requests, so no request is dropped."""
    ctx = e10.context(world)
    if ctx["model"] is None or not ctx["cal"]["has"].any():
        return []
    p, cal = world.p, ctx["cal"]
    dim, k, M = world.db.shape[1], world.k, world.M
    ok = cal["has"]
    full = fit.certify(ctx["train_scores"], cal["score"][ok], cal["y"][ok], p.T, p.delta, p.min_ess)
    stream = world.full_heldout_stream()
    if request_limit:
        stream = stream[:request_limit]
    uniq, pos = np.unique(stream, return_inverse=True)
    vec = np.asarray(world.q[uniq], dtype=np.float32)
    data = e10.labeled_set(world, ctx["feats"], ctx["model"], vec, np.asarray(world.top[uniq][:, :k], np.int64),
                           np.ones(len(uniq)))
    score, has, y = data["score"][pos], data["has"][pos], data["y"][pos]
    idx = np.arange(len(stream))
    h0 = float(fit.decide(score, has, full.lam).mean())
    share_labeled = (1.0 - h0) + p.a * h0
    rows = []
    precisions = tuple(precisions) + (("native",) if "native" not in precisions
                                      and quantize.native_info(world.q)["native_bits"] == 8 else ())
    for bits in precisions:
        low_u, high_u, _ = e10.intervals_of(ctx, data, bits)
        low, high = low_u[pos], high_u[pos]
        for share in shares:
            n_items = items_per_half(share, M, dim, k, bits)
            W = int(2 * n_items / share_labeled) if share_labeled > 0 else 0
            for delta in deltas:
                rng = np.random.default_rng(p.seed + 29)
                stretches = stretch_rows(low, high, score, has, y, idx, W, full.lam, ctx["edges"], p, delta, rng)
                rows.append(summarize_row(world, bits, share, delta, n_items, W, h0, full.lam, stretches, p))
                print(describe(rows[-1]), flush=True)
    return rows


def describe(r: dict) -> str:
    """One line per row: what the window holds and why it does or does not certify."""
    return (f"[E11] {r['workload']} bits={r['bits']} share={r['window_share_of_M']:g} delta={r['delta']:g} "
            f"items/half={r['items_per_half']} W={r['W_requests']} stretches={r['n_stretches']} "
            f"audited_hits/half={r['audited_hits_per_half']:.0f} n_eff_served={r['n_eff_served']:.0f} "
            f"n_eff_all_labeled={r['n_eff_all_labeled']:.0f} certified={r['certified_share']:.2f} "
            f"min_ess_alone_blocks={r['min_ess_blocks_share']:.2f}" + ("  IN_SAMPLE (excluded)" if r["in_sample"] else ""))


def summarize_row(world, bits, share, delta, n_items, W, h0, lam_policy, stretches, p) -> dict:
    done = [s for s in stretches if "hit_rate" in s]
    cert = [s for s in stretches if s["certified"]]
    hits = sum(s["hits"] for s in done)
    mean = lambda key, rows: float(np.mean([s[key] for s in rows])) if rows else float("nan")  # noqa: E731
    return dict(
        workload=world.name, budget=world.budget, bits=bits, window_share_of_M=share, delta=delta,
        items_per_half=n_items, W_requests=W, reference_hit_rate=h0, n_stretches=len(stretches),
        certified_share=float(np.mean([s["certified"] for s in stretches])) if stretches else 0.0,
        hit_rate=mean("hit_rate", done), recall_of_hits=float(sum(s["recall_sum"] for s in done) / hits) if hits
        else float("nan"),
        share_below_target=float(np.mean([s["recall_sum"] / s["hits"] < p.T for s in done if s["hits"]]))
        if any(s["hits"] for s in done) else float("nan"),
        sampling_slack=mean("sampling_slack", cert), quantization_slack=mean("quantization_slack", cert),
        bound_gap_mean=mean("bound_gap", cert),
        ambiguous_share=mean("ambiguous", cert), in_sample=bool(done and done[0]["in_sample"]),
        labeled_per_half=mean("labeled", stretches), audited_hits_per_half=mean("audited_hits", stretches),
        n_eff_served=mean("n_eff_served", stretches), n_eff_all_labeled=mean("n_eff_all_labeled", stretches),
        ess_min_certified=mean("ess_min_certified", cert), min_ess=p.min_ess,
        min_ess_blocks_share=float(np.mean([s["min_ess_blocks"] for s in stretches])) if stretches else 0.0)


def rank(row: dict) -> tuple[float, float]:
    hit = row["hit_rate"]
    return row["certified_share"], hit if hit == hit else -1.0


def summary(rows: list[dict], min_share: float = 0.9) -> list[dict]:
    """Per workload and budget, over the rows that are not in_sample: the smallest window share at which certification
    holds (certified share at or above min_share), per precision and delta present in the rows, and which precision
    does better at each share. Keys for precisions or deltas that were not run are left out."""
    out = []
    keys = sorted({(r["workload"], r["budget"]) for r in rows})
    for workload, budget in keys:
        mine = [r for r in rows if r["workload"] == workload and r["budget"] == budget and not r["in_sample"]]
        precisions = list(dict.fromkeys(r["bits"] for r in mine))
        deltas = sorted({r["delta"] for r in mine})
        smallest = {}
        for bits in precisions:
            for delta in deltas:
                here = [r for r in mine if r["bits"] == bits and r["delta"] == delta]
                if here:
                    ok = [r["window_share_of_M"] for r in here if r["certified_share"] >= min_share]
                    smallest[f"bits{bits}_delta{delta}"] = min(ok) if ok else None
        better = {}
        for share in sorted({r["window_share_of_M"] for r in mine}):
            for delta in deltas:
                pair = {r["bits"]: r for r in mine if r["window_share_of_M"] == share and r["delta"] == delta}
                if len(pair) >= 2:
                    best = max(pair, key=lambda b: rank(pair[b]))
                    better[f"share{share}_delta{delta}"] = best
        out.append(dict(workload=workload, budget=budget, n_rows=len(mine),
                        n_in_sample_excluded=sum(1 for r in rows if r["workload"] == workload
                                                 and r["budget"] == budget and r["in_sample"]),
                        smallest_share_certifying=smallest, better_bits=better))
    return out
