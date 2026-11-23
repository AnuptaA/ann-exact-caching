"""Popularity shifts over k-means clusters of the vectors, on the group universe.

Clusters come from k-means (64 centroids, FAISS, seed 0) on a sample of the dataset's vectors. Every group
(through its base query vector) belongs to its nearest centroid. A schedule gives, for each request, the
probability that traffic goes to the new set of clusters; inside a cluster groups follow a Zipf popularity.
  abrupt      the popular clusters swap at one request
  gradual     traffic moves from the old set to the new set over a span of requests
  correlated  the gradual shift, while held-out vectors of the new clusters are inserted
"""

import numpy as np

from workloads.q1.streams import zipf_probabilities
from workloads.q1.universe import Universe

N_CLUSTERS = 64
SEED = 0


def kmeans_centroids(sample: np.ndarray, k: int = N_CLUSTERS, seed: int = SEED, niter: int = 20,
                     allow_fallback: bool = False) -> np.ndarray:
    """FAISS k-means (seed 0 for cluster runs). Without faiss it refuses; the seeded numpy Lloyd is for tests
    and local checks and must be asked for."""
    sample = np.ascontiguousarray(sample, dtype=np.float32)
    try:
        import faiss
    except ImportError:
        if not allow_fallback:
            raise RuntimeError("faiss is required for cluster runs (faiss.Kmeans, seed 0); the numpy fallback is "
                               "for tests: pass allow_fallback=True or --allow-numpy-kmeans")
        return _lloyd(sample, k, seed, niter)
    km = faiss.Kmeans(sample.shape[1], k, niter=niter, seed=seed, verbose=False)
    km.train(sample)
    return km.centroids.copy()


def _lloyd(x: np.ndarray, k: int, seed: int, niter: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = [x[rng.integers(len(x))].astype(np.float64)]
    for _ in range(1, k):
        d2 = np.min([((x - c) ** 2).sum(axis=1) for c in centers], axis=0)
        centers.append(x[rng.choice(len(x), p=d2 / d2.sum())].astype(np.float64))
    centers = np.array(centers)
    for _ in range(niter):
        labels = assign(x, centers)
        for c in range(k):
            members = x[labels == c]
            if len(members):
                centers[c] = members.mean(axis=0)
    return centers.astype(np.float32)


def assign(vectors: np.ndarray, centroids: np.ndarray, block: int = 8192) -> np.ndarray:
    """Nearest centroid of every vector."""
    c = np.asarray(centroids, dtype=np.float64)
    c_sq = (c ** 2).sum(axis=1)
    out = np.empty(len(vectors), dtype=np.int64)
    for s in range(0, len(vectors), block):
        v = np.asarray(vectors[s:s + block], dtype=np.float64)
        out[s:s + block] = np.argmin(c_sq[None, :] - 2.0 * v @ c.T, axis=1)
    return out


def cluster_sets(n_clusters: int = N_CLUSTERS, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Two disjoint halves of the clusters: the initially popular set and the new set."""
    perm = np.random.default_rng([seed, 11]).permutation(n_clusters)
    half = n_clusters // 2
    return np.sort(perm[:half]), np.sort(perm[half:2 * half])


def new_set_probability(n_requests: int, kind: str, start: int, span: int = 0) -> np.ndarray:
    """Probability per request that traffic goes to the new set."""
    t = np.arange(n_requests)
    if kind == "abrupt":
        return (t >= start).astype(np.float64)
    if kind in ("gradual", "correlated"):
        if span <= 0:
            raise ValueError("a gradual shift needs a span")
        return np.clip((t - start) / span, 0.0, 1.0)
    raise ValueError(f"unknown shift {kind}")


def shift_stream(uni: Universe, group_cluster: np.ndarray, kind: str, n_requests: int, start: int, span: int = 0,
                 s: float = 0.99, seed: int = 0, n_clusters: int = N_CLUSTERS) -> dict:
    """Request ids and the phase (0 before the shift, 1 during, 2 after) of every request."""
    old, new = cluster_sets(n_clusters)
    members_of = {c: np.flatnonzero(group_cluster == c) for c in range(n_clusters)}
    old = np.array([c for c in old if len(members_of[c])], dtype=np.int64)
    new = np.array([c for c in new if len(members_of[c])], dtype=np.int64)
    if not len(old) or not len(new):
        raise ValueError("a cluster set has no groups")
    rng = np.random.default_rng([seed, 17])
    p_new = new_set_probability(n_requests, kind, start, span)
    use_new = rng.random(n_requests) < p_new
    pick = rng.random(n_requests)
    cluster = np.where(use_new, new[(pick * len(new)).astype(int) % len(new)],
                       old[(pick * len(old)).astype(int) % len(old)])
    order = {c: np.random.default_rng([seed, 23, c]).permutation(g) for c, g in members_of.items()}
    ids = np.empty(n_requests, dtype=np.int64)
    for c in np.unique(cluster):
        rows = np.flatnonzero(cluster == c)
        groups = order[int(c)]
        ranks = rng.choice(len(groups), size=len(rows), p=zipf_probabilities(len(groups), s))
        ids[rows] = groups[ranks] * uni.group_size + rng.integers(0, uni.group_size, size=len(rows))
    end = start + (span if kind != "abrupt" else 0)
    phase = np.where(np.arange(n_requests) < start, 0, np.where(np.arange(n_requests) < max(end, start + 1), 1, 2))
    return {"ids": ids, "phase": phase.astype(np.int8), "new_set_probability": p_new}


def correlated_inserts(vector_cluster: np.ndarray, held_out: np.ndarray, start: int, span: int, n_inserts: int,
                       n_clusters: int = N_CLUSTERS, seed: int = 0) -> dict:
    """Held-out vectors of the new clusters, inserted evenly over the shift span (request index, vector id)."""
    _, new = cluster_sets(n_clusters)
    pool = held_out[np.isin(vector_cluster[held_out], new)]
    if len(pool) < n_inserts:
        raise ValueError(f"only {len(pool)} held-out vectors in the new clusters, {n_inserts} wanted")
    chosen = np.random.default_rng([seed, 31]).choice(pool, size=n_inserts, replace=False)
    t = start + (np.arange(n_inserts) * span) // max(n_inserts, 1)
    return {"t": t.astype(np.int64), "id": chosen.astype(np.int64)}
