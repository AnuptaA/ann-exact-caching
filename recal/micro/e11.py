"""E11: window memory against certification, with scores kept at 8 or 16 bits.

A calibration item holds d p / 8 + 4K + 33 bytes (the quantized vector, its error, 4K ids and bookkeeping), a
training item 4K + 31 (no vector). A window share sigma of M holds n items per half; the window spans
W = 2n / (labeled share) main-path requests. As in E8, lambda is certified on the calibration half of consecutive
stretches of the held-out stream (misses weigh 1, audited hits 1/a) and applied to the next stretch; the
certification uses the score intervals of the stored copies (recal.micro.intervals).
"""

import numpy as np

from recal.micro import e10, fit, intervals as iv, world as w

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


def item_bytes(dim: int, k: int, bits: int) -> tuple[int, float]:
    return 4 * k + TRAIN_FIXED_BYTES, dim * bits / 8.0 + 4 * k + CAL_FIXED_BYTES


def items_per_half(share: float, M: float, dim: int, k: int, bits: int) -> int:
    train, cal = item_bytes(dim, k, bits)
    return int(share * M // (train + cal))


def request_arrays(world: w.World, held: dict) -> np.ndarray:
    """Position (in the kept unique queries) of every held-out request whose query was kept, in stream order."""
    inverse = world.heldout()["inverse"]
    position = np.full(len(world.heldout()["vec"]), -1, np.int64)
    position[held["keep"]] = np.arange(len(held["keep"]))
    mapped = position[inverse]
    return mapped[mapped >= 0]


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
        row = dict(certified=cert["lam"] is not None, in_sample=n_stretch == 1, labeled=int(use.sum()))
        if cert["lam"] is not None:
            row.update(sampling_slack=cert["slack"], quantization_slack=cert["mean_possible"] - cert["mean"],
                       ambiguous=cert["n_ambiguous"] / max(int(use.sum()), 1))
        nxt = idx[(j + 1) * size:(j + 2) * size] if j + 1 < n_stretch else (sl if n_stretch == 1 else None)
        if nxt is not None:
            applied = fit.decide(score[nxt], has[nxt], cert["lam"])
            row.update(hit_rate=float(applied.mean()), hits=int(applied.sum()), recall_sum=float(y[nxt][applied].sum()))
        out.append(row)
    return out


def run(world: w.World) -> list[dict]:
    ctx = e10.context(world)
    if ctx["model"] is None or not ctx["cal"]["has"].any():
        return []
    p, cal, held = world.p, ctx["cal"], ctx["held"]
    dim, k, M = world.db.shape[1], world.k, world.M
    ok = cal["has"]
    full = fit.certify(ctx["train_scores"], cal["score"][ok], cal["y"][ok], p.T, p.delta, p.min_ess)
    idx = request_arrays(world, held)
    h0 = float(fit.decide(held["score"], held["has"], full.lam)[idx].mean()) if len(idx) else 0.0
    share_labeled = (1.0 - h0) + p.a * h0
    rows = []
    for bits in PRECISIONS:
        low, high, _ = e10.intervals_of(ctx, held, bits)
        for share in SHARES:
            n_items = items_per_half(share, M, dim, k, bits)
            W = int(2 * n_items / share_labeled) if share_labeled > 0 else 0
            for delta in DELTAS:
                rng = np.random.default_rng(p.seed + 29)
                stretches = stretch_rows(low, high, held["score"], held["has"], held["y"], idx, W, full.lam,
                                         ctx["edges"], p, delta, rng)
                rows.append(summarize_row(world, bits, share, delta, n_items, W, h0, full.lam, stretches, p))
    return rows


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
        ambiguous_share=mean("ambiguous", cert), in_sample=bool(done and done[0]["in_sample"]))


def rank(row: dict) -> tuple[float, float]:
    hit = row["hit_rate"]
    return row["certified_share"], hit if hit == hit else -1.0


def summary(rows: list[dict], min_share: float = 0.9) -> list[dict]:
    """Per workload and budget: the smallest window share at which certification holds (certified share at or above
    min_share on a held-out stretch), per precision and delta, and which precision does better at each share."""
    out = []
    keys = sorted({(r["workload"], r["budget"]) for r in rows})
    for workload, budget in keys:
        mine = [r for r in rows if r["workload"] == workload and r["budget"] == budget and not r["in_sample"]]
        smallest = {}
        for bits in PRECISIONS:
            for delta in DELTAS:
                ok = [r["window_share_of_M"] for r in mine if r["bits"] == bits and r["delta"] == delta
                      and r["certified_share"] >= min_share]
                smallest[f"bits{bits}_delta{delta}"] = min(ok) if ok else None
        better = {}
        for share in SHARES:
            for delta in DELTAS:
                pair = {r["bits"]: r for r in mine if r["window_share_of_M"] == share and r["delta"] == delta}
                if len(pair) == 2:
                    first, second = (rank(pair[bits]) for bits in PRECISIONS)
                    better[f"share{share}_delta{delta}"] = PRECISIONS[0] if first >= second else PRECISIONS[1]
        out.append(dict(workload=workload, budget=budget, smallest_share_certifying=smallest, better_bits=better))
    return out
