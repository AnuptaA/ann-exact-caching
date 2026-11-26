"""Intruder check: where the core's intruder rate and E4's differ.

On the same held-out queries and the same entries (the ones the core reads) the intruder count is computed:
  a  the core (candidates_batch, 4-bit shells)
  b  Python, the core's shell membership, full-precision vectors
  c  Python, the core's shell membership, the core's 4-bit codes decoded
  d  E4 as first written (shells keep cached vectors), and d_fixed with cached vectors removed
b against c isolates the quantization, a against c checks this reimplementation, c against d isolates the
definitions. nr is the core's for a, b, c and E4's own for d.
"""

import numpy as np

from recal.micro import e1, e4, fit, quantize, world as w

NAME = "intruder_check"
MAX_QUERIES = 4000
BLOCK_ELEMENTS = 20_000_000
RELATIVE_BANDS = (0.01, 0.02, 0.05)
PAIRS = (("a", "c"), ("a", "b"), ("b", "c"), ("b", "d_fixed"), ("c", "d_fixed"), ("c", "d_asis"), ("b", "d_asis"),
         ("a", "d_asis"), ("a", "d_fixed"))


def decode_codes(codes: np.ndarray, lo: np.ndarray, step: np.ndarray) -> np.ndarray:
    """Vectors of 4-bit codes as the core reads them: lo + code * step, even coordinates in the low nibble."""
    d = len(lo)
    out = np.empty((len(codes), d), np.float64)
    out[:, 0::2] = lo[0::2].astype(np.float64) + (codes & 15).astype(np.float64)[:, :len(lo[0::2])] * \
        step[0::2].astype(np.float64)
    out[:, 1::2] = lo[1::2].astype(np.float64) + (codes >> 4).astype(np.float64)[:, :len(lo[1::2])] * \
        step[1::2].astype(np.float64)
    return out


def shell_slot_ids(world: w.World, table: e4.EntryTable) -> np.ndarray:
    """Vector id of every shell slot of the core, from the entries' slot lists; checked against the codes."""
    core = world.core
    n_slots = core.n_shell_slots()
    ids = np.full(n_slots, -1, np.int64)
    for e, entry in enumerate(world.entries()):
        slots = np.asarray(entry["shell_slots"], dtype=np.int64)
        listed = table.shell_ids[e]
        if len(slots) != len(listed):
            raise RuntimeError(f"entry {e}: the core lists {len(slots)} shell slots, rebuilt shells have "
                               f"{len(listed)} (cached vectors removed)")
        if len(slots) and (ids[slots] >= 0).any() and (ids[slots][ids[slots] >= 0] != listed[ids[slots] >= 0]).any():
            raise RuntimeError(f"entry {e}: a shell slot is shared by two different vectors")
        ids[slots] = listed
    if (ids < 0).any():
        raise RuntimeError("some shell slots of the core belong to no entry")
    lo, step = core.quantizer()
    want = quantize.shell_codes(world.db[ids], lo, step)
    codes = core.shell_codes()
    got = np.zeros_like(want)
    got[:, 0::2] = (codes & 15)[:, :want[:, 0::2].shape[1]]
    got[:, 1::2] = (codes >> 4)[:, :want[:, 1::2].shape[1]]
    wrong = int((got != want).any(axis=1).sum())
    if wrong:
        raise RuntimeError(f"{wrong} of {n_slots} shell slots hold codes that differ from the rebuilt vectors; "
                           "the slot-to-id mapping or the quantizer range is not what was assumed")
    return ids


def pair_stats(x: np.ndarray, y: np.ndarray, K: int) -> dict:
    diff = y.astype(np.int64) - x.astype(np.int64)
    rate = np.abs(np.log((y + 1.0) / K) - np.log((x + 1.0) / K))
    return dict(count_agree_share=float((diff == 0).mean()), count_diff_mean=float(diff.mean()),
                count_abs_diff_p95=e4.pct(np.abs(diff), 95), count_abs_diff_p99=e4.pct(np.abs(diff), 99),
                count_abs_diff_max=int(np.abs(diff).max()), rate_abs_diff_p95=e4.pct(rate, 95),
                rate_abs_diff_mean=float(rate.mean()))


def run(world: w.World) -> tuple[list[dict], dict]:
    K, core, m = world.k, world.core, world.m
    held = world.heldout()
    vec = held["vec"]
    if len(vec) > MAX_QUERIES:
        keep = np.linspace(0, len(vec) - 1, MAX_QUERIES).astype(int)
        vec = vec[keep]
    Q = np.ascontiguousarray(vec, dtype=np.float32)
    feat, _, _ = core.candidates_batch(Q, K, m)
    has = feat[:, 0] == 1
    nr, count_core = feat[:, 1], feat[:, 5].astype(np.int64)
    read = core.entries_read_batch(Q, m)
    table_asis = e4.EntryTable(world, exclude_pool=False)
    table = e4.EntryTable(world, exclude_pool=True)
    slot_ids = shell_slot_ids(world, table)
    lo, step = core.quantizer()
    full = np.asarray(world.db[slot_ids], dtype=np.float64)
    decoded = decode_codes(core.shell_codes(), lo, step)
    entry_slots = [np.asarray(e["shell_slots"], dtype=np.int64) for e in world.entries()]
    n = len(Q)
    counts = {key: np.zeros(n, np.int64) for key in ("a", "b", "c", "d_asis", "d_fixed")}
    counts["a"] = count_core
    band = {b: np.zeros(n) for b in RELATIVE_BANDS}
    examined = np.zeros(n, np.int64)
    block = max(1, BLOCK_ELEMENTS // max(len(full), table_asis.n, len(table_asis.pool_ids), 1))
    for lo_i in range(0, n, block):
        hi_i = min(n, lo_i + block)
        q64 = Q[lo_i:hi_i].astype(np.float64)
        D_full = np.sqrt(np.maximum((q64 ** 2).sum(1)[:, None] + (full ** 2).sum(1)[None, :]
                                    - 2.0 * q64 @ full.T, 0.0)) if len(full) else np.zeros((hi_i - lo_i, 0))
        D_dec = np.sqrt(np.maximum((q64 ** 2).sum(1)[:, None] + (decoded ** 2).sum(1)[None, :]
                                   - 2.0 * q64 @ decoded.T, 0.0)) if len(full) else np.zeros((hi_i - lo_i, 0))
        dists = {name: (e4.sq_dist(Q[lo_i:hi_i], t.centers), e4.sq_dist(Q[lo_i:hi_i], t.pool),
                        e4.sq_dist(Q[lo_i:hi_i], t.zone) if len(t.zone_ids) else np.zeros((hi_i - lo_i, 0)), t)
                 for name, t in (("d_asis", table_asis), ("d_fixed", table))}
        for i in range(hi_i - lo_i):
            g = lo_i + i
            if not has[g]:
                continue
            S = read[g][read[g] >= 0]
            slots = np.unique(np.concatenate([entry_slots[e] for e in S])) if len(S) else np.zeros(0, np.int64)
            examined[g] = len(slots)
            counts["b"][g] = int((D_full[i, slots] < nr[g]).sum())
            counts["c"][g] = int((D_dec[i, slots] < nr[g]).sum())
            if len(slots):
                rel = np.abs(D_full[i, slots] / nr[g] - 1.0)
                for b in RELATIVE_BANDS:
                    band[b][g] = float((rel < b).sum())
            for name, (Dc, Dp, Ds, t) in dists.items():
                res = e4._features(t, S, Dc[i], Dp[i], Ds[i], K, Q[g])
                counts[name][g] = int(round(np.exp(res["F"][2]) * K - 1.0)) if res["has"] else 0
    rows = pair_rows(world, counts, feat, has, K)
    spread = {f"vectors_within_{int(b * 100)}pct_of_nr_mean": float(band[b][has].mean()) for b in RELATIVE_BANDS}
    extra = dict(workload=world.name, budget=world.budget, n_queries=int(has.sum()), n_entries_read=int(m),
                 core_count_mean=float(counts["a"][has].mean()), shell_vectors_examined_mean=float(examined[has].mean()),
                 mean_count={k: float(v[has].mean()) for k, v in counts.items()}, **spread)
    return rows, extra


def hit_decisions(world: w.World, feat: np.ndarray, has: np.ndarray, counts: np.ndarray, K: int,
                  model: dict, lam) -> np.ndarray:
    F = np.column_stack([feat[:, 2], feat[:, 3], np.log((counts + 1.0) / K)])
    scores = np.where(has, fit.score(model, F), -np.inf)
    return fit.decide(scores, has, lam)


def pair_rows(world: w.World, counts: dict, feat: np.ndarray, has: np.ndarray, K: int) -> list[dict]:
    model = e1.fit_models(world)[fit.LOGIT]
    lam = None
    if model is not None:
        cal, _ = world.role_rows("cal")
        lam = e1.lambda_for(world, model, cal.F, cal.recall, cal.has).lam
    hits = {k: hit_decisions(world, feat, has, v, K, model, lam) if model is not None else np.zeros(len(has), bool)
            for k, v in counts.items()}
    rows = []
    for x, y in PAIRS:
        stats = pair_stats(counts[x][has], counts[y][has], K)
        rows.append(dict(workload=world.name, budget=world.budget, pair=f"{x}_vs_{y}", n_queries=int(has.sum()),
                         **stats, decision_flip_share=float((hits[x][has] != hits[y][has]).mean()),
                         hit_rate_x=float(hits[x][has].mean()), hit_rate_y=float(hits[y][has].mean()),
                         lam=lam))
    return rows
