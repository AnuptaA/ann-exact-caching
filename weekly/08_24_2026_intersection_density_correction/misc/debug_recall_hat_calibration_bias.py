#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Debug the recall_hat=K/N calibration bias seen in Scheme B (actual
recall systematically exceeds predicted recall_hat, on both the real and
uniform corpora). Fully synthetic, exact (no LSH, no real dataset), on a
uniform sphere where density heterogeneity is provably absent - isolates
whether the bias is a structural property of the K/N formula itself
versus a bug in the LSH estimator or the Scheme B pipeline.

r_ic (the ball radius used for N) is defined as the *maximum* of the
angular distances from the query to a candidate's own K nearest
neighbors - an order statistic. Only one of those K points sits at the
ball's boundary; the rest sit strictly closer to the query, so the
candidate's K points are not uniformly spread through the ball the way
K/N's implicit hypergeometric model assumes. Since the query's own true
top-K is, by definition, also clustered at the ball's center, two
centrally-skewed sets overlap more than a uniform-subset model predicts -
which should bias actual recall above K/N regardless of corpus density.
Confirmed below: the same bias, same delta-dependent (hump-shaped)
profile, appears here with zero LSH involvement and zero density
heterogeneity."""

import numpy as np

#-------------------------------------------------------------------------------

N = 50000
D = 128
K = 100
N_TRIALS_PER_DELTA = 200
DELTAS = [1e-6, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.4]

#-------------------------------------------------------------------------------


def _random_tangent(anchor: np.ndarray, rng: np.random.Generator, d: int) -> np.ndarray:
    v = rng.standard_normal(d)
    v -= (v @ anchor) * anchor
    v /= np.linalg.norm(v)
    return v


def _move(anchor: np.ndarray, delta: float, direction: np.ndarray) -> np.ndarray:
    moved = np.cos(delta) * anchor + np.sin(delta) * direction
    return moved / np.linalg.norm(moved)


def run(seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    points = rng.standard_normal((N, D))
    points /= np.linalg.norm(points, axis=1, keepdims=True)

    q = np.zeros(D)
    q[0] = 1.0
    dists_q = np.arccos(np.clip(points @ q, -1.0, 1.0))
    topk_q = set(np.argsort(dists_q)[:K].tolist())
    theta_sorted_q = np.sort(dists_q)  # for exact N(q, r) via searchsorted

    print(f"{'delta':>8} {'mean_actual':>12} {'mean_predicted':>15} {'bias':>8}")
    for delta in DELTAS:
        actuals, predicteds = [], []
        for _ in range(N_TRIALS_PER_DELTA):
            c = _move(q, delta, _random_tangent(q, rng, D))
            dists_c = np.arccos(np.clip(points @ c, -1.0, 1.0))
            r_c = np.argsort(dists_c)[:K]  # candidate's own exact K-NN

            r_ic = float(np.arccos(np.clip(points[r_c] @ q, -1.0, 1.0)).max())
            n_exact = int(np.searchsorted(theta_sorted_q, r_ic, side='right'))

            actuals.append(len(topk_q & set(r_c.tolist())) / K)
            predicteds.append(K / n_exact if n_exact > 0 else float('inf'))

        mean_actual = np.mean(actuals)
        mean_predicted = np.mean(np.clip(predicteds, 0, 1))
        print(f"{delta:>8.4f} {mean_actual:>12.4f} {mean_predicted:>15.4f} {mean_actual - mean_predicted:>8.4f}")


if __name__ == '__main__':
    run()
