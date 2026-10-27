#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Hit policies HP1 to HP3, cap learning, hindsight ceiling and stream metrics.

All comparisons use integers. With K = 10 every estimated recall is a count divided by K, so in units of 0.05
recall an estimate is est_units = 2 * count and a deficit is 20 - est_units (a deficit of d tenths is 2 d units).
The target T is 16, 18 or 19 units and the per-step slack (1 - T) is 4, 2 or 1 units, so no float tolerance is
needed. The accounting deficit a is clipped at 0 (a negative deficit never creates slack); ranking and the HP3 cap
comparison use the unclipped deficit d."""

import numpy as np

import ee_config as cfg

U = cfg.UNITS_PER_RECALL

#-------------------------------------------------------------------------------
# Units

def counts_to_units(counts: np.ndarray, K: int = cfg.K) -> np.ndarray:
    """Recall counts (out of K, unclipped) to integer units of 0.05 recall."""
    assert U % K == 0, f"K = {K} does not divide {U} recall units"
    return np.asarray(counts, dtype=np.int64) * (U // K)

def deficits(est_units: np.ndarray) -> tuple:
    """(unclipped d, accounting a) in units, d = 1 - est."""
    d = U - np.asarray(est_units, dtype=np.int64)
    return d, np.maximum(d, 0)

def stream_order(seed: int, n: int) -> np.ndarray:
    return np.random.default_rng([seed, 31]).permutation(n)

#-------------------------------------------------------------------------------
# Cap, hindsight

def largest_prefix(a_sorted_units: np.ndarray, total_slack_units: int) -> int:
    """Number of smallest (clipped) deficits whose running sum stays <= the total slack."""
    return int((np.cumsum(a_sorted_units) <= total_slack_units).sum())

def hindsight_count(est_units: np.ndarray, T: float) -> int:
    """Hit count of the hindsight ceiling: sort deficits ascending, largest prefix with sum <= n (1 - T)."""
    d, a = deficits(est_units)
    order = np.argsort(d, kind="stable")
    return largest_prefix(a[order], cfg.slack_units(T, len(d)))

def learn_cap(train_est_units: np.ndarray, T: float) -> int:
    """HP3 cap in units: rank train queries by the unclipped deficit d ascending, accept while the running sum of the
    clipped deficit a = max(0, d) stays within n_train (1 - T), cap = max(0, largest accepted d). The cap is floored
    at 0, also when every accepted query is overestimated (d < 0) or none is accepted."""
    d, a = deficits(train_est_units)
    order = np.argsort(d, kind="stable")
    h = largest_prefix(a[order], cfg.slack_units(T, len(d)))
    return max(0, int(d[order][h - 1])) if h > 0 else 0

#-------------------------------------------------------------------------------
# Policies over a stream (queries already in stream order)

def run_policy(policy: str, est_units: np.ndarray, T: float, cap_units: int = None) -> np.ndarray:
    """Boolean hit array. HP1: est >= T. HP2: cumdef + a <= t (1 - T) with cumdef the accounting deficits of past hits.
    HP3: HP2 and d <= cap."""
    est_units = np.asarray(est_units, dtype=np.int64)
    T_u, step = cfg.target_units(T), cfg.per_step_slack_units(T)
    if policy == "HP1":
        return est_units >= T_u
    assert policy in ("HP2", "HP3")
    if policy == "HP3":
        assert cap_units is not None, "HP3 needs a cap"
    d, a = deficits(est_units)
    hits = np.zeros(len(est_units), dtype=bool)
    cumdef = 0
    for t in range(1, len(est_units) + 1):
        i = t - 1
        if cumdef + a[i] <= t * step and (policy == "HP2" or d[i] <= cap_units):
            hits[i] = True
            cumdef += int(a[i])
    return hits

def delivered_units(hits: np.ndarray, true_counts: np.ndarray, K: int = cfg.K) -> np.ndarray:
    """True recall of a hit (in units), 1 for a miss."""
    return np.where(hits, counts_to_units(true_counts, K), U)

def stream_metrics(hits: np.ndarray, true_counts: np.ndarray, T: float, K: int = cfg.K,
                   burn_in: int = cfg.BURN_IN) -> dict:
    """Hit rate, achieved recall Y, running Y_t summaries and per-hit true recall statistics."""
    n = len(hits)
    deliv = delivered_units(hits, true_counts, K)
    cum = np.cumsum(deliv)
    t = np.arange(1, n + 1)
    below = cum < cfg.target_units(T) * t  # exact: Y_t < T
    hit_recall = np.asarray(true_counts)[hits] / K
    out = dict(n=n, hits=int(hits.sum()), hit_rate=hits.mean(), achieved_recall=cum[-1] / (U * n),
               y_min=(cum / (U * t)).min(), frac_below_T=below.mean(),
               frac_below_T_after_burn_in=below[burn_in:].mean() if n > burn_in else np.nan,
               one_minus_T=1.0 - T)
    if hits.any():
        out.update(mean_hit_recall=hit_recall.mean(), p10_hit_recall=float(np.percentile(hit_recall, 10)),
                   frac_hits_below_half=(hit_recall < 0.5).mean(), frac_hits_zero_recall=(hit_recall == 0).mean())
    else:
        out.update(mean_hit_recall=np.nan, p10_hit_recall=np.nan, frac_hits_below_half=np.nan,
                   frac_hits_zero_recall=np.nan)
    return out

#-------------------------------------------------------------------------------
# One pool, one arm, one T: all three policies

def evaluate_pool(seed: int, T: float, test_est_counts: np.ndarray, test_true_counts: np.ndarray,
                  train_est_counts: np.ndarray, train_true_counts: np.ndarray, K: int = cfg.K,
                  oracle_arm: bool = True) -> list:
    """Rows for HP1, HP2 and HP3 on one pool. `*_est_counts` are the arm's estimated recall counts (unclipped for the
    LSH arm); `*_true_counts` are true recall counts (train: leave-group-out). The train arrays feed the HP3 cap and
    the proxy-gap logging; the policies only ever see estimates, delivered recall is true recall."""
    n_test, n_train = len(test_est_counts), len(train_est_counts)
    t_order, r_order = stream_order(seed, n_test), stream_order(seed, n_train)
    est_u = counts_to_units(test_est_counts, K)[t_order]
    true_c = np.asarray(test_true_counts)[t_order]
    train_est_u = counts_to_units(train_est_counts, K)
    cap = learn_cap(train_est_u, T)
    hs_arm = hindsight_count(counts_to_units(test_est_counts, K), T) / n_test
    hs_oracle = hindsight_count(counts_to_units(test_true_counts, K), T) / n_test
    rows = []
    for policy in ("HP1", "HP2", "HP3"):
        hits = run_policy(policy, est_u, T, cap if policy == "HP3" else None)
        row = stream_metrics(hits, true_c, T, K)
        row.update(policy=policy, T=T, hindsight_hit_rate_oracle=hs_oracle,
                   hindsight_hit_rate_lsh=np.nan if oracle_arm else hs_arm, learned_cap_units=np.nan,
                   train_hit_fraction=np.nan, test_hit_fraction=row["hit_rate"])
        if policy == "HP3":
            train_hits = run_policy("HP3", train_est_u[r_order], T, cap)
            row.update(learned_cap_units=cap, train_hit_fraction=float(train_hits.mean()))
        if not oracle_arm:
            err = (np.asarray(test_est_counts) - np.asarray(test_true_counts)) / K
            row.update(est_signed_error=float(err.mean()), est_mae=float(np.abs(err).mean()))
        rows.append(row)
    return rows
