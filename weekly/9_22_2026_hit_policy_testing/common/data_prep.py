#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Data preparation and locality regimes (spec 3): ONE shared Zipf-weighted
topic pool per (dataset, regime), drawn from by TRAIN, CALIB, and TEST alike,
with an independent small angular-jitter perturbation applied to every draw
(repeats must NOT reuse one cached perturbed vector, since that would make
high-locality queries literally identical rather than near-duplicates).
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import numpy as np

from datasets.dataloaders import load_esci_data, load_sift_dataset
from utils.base.perturbation import generate_perturbed_vector

from config import DATASET_CONFIG, N_CALIB, N_TEST, N_TRAIN, PERTURBATION_LEVEL, REGIME_CONFIG, SEED

#-------------------------------------------------------------------------------

def normalize_rows(X: np.ndarray) -> np.ndarray:
    return X / np.linalg.norm(X, axis=1, keepdims=True)

#-------------------------------------------------------------------------------

def load_base_pool(dataset: str) -> np.ndarray:
    cfg = DATASET_CONFIG[dataset]
    X = np.load(cfg["base_vectors_path"]).astype(np.float64)
    if cfg["normalize_base"]:
        X = normalize_rows(X)
    else:
        norms = np.linalg.norm(X, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4), f"{dataset} base expected pre-normalized"
    return X

#-------------------------------------------------------------------------------

def load_official_query_pool(dataset: str) -> np.ndarray:
    if dataset == "sift":
        _, _, query, _ = load_sift_dataset("datasets/sift")
        return normalize_rows(query.astype(np.float64))
    elif dataset == "esci":
        _, query_vectors, _, _ = load_esci_data("datasets/esci", load_embeddings=False)
        query = query_vectors.astype(np.float64)
        norms = np.linalg.norm(query, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4), "esci queries expected pre-normalized"
        return query
    raise ValueError(dataset)

#-------------------------------------------------------------------------------

def zipf_pmf(pool_size: int, exponent: float) -> np.ndarray:
    ranks = np.arange(1, pool_size + 1, dtype=np.float64)
    weights = ranks ** (-exponent)
    return weights / weights.sum()

#-------------------------------------------------------------------------------

def build_topic_pool(query_pool_size: int, pool_size: int, rng: np.random.Generator) -> np.ndarray:
    """Sub-selects pool_size distinct queries from the FULL query pool to
    serve as 'topics', shared by TRAIN/CALIB/TEST alike. Scales down if the
    query pool is smaller than the requested pool_size (spec's 'scale down
    if unavailable')."""
    pool_size = min(pool_size, query_pool_size)
    return rng.choice(query_pool_size, size=pool_size, replace=False)

#-------------------------------------------------------------------------------

def draw_with_repetition_and_perturb(topic_pool_idx: np.ndarray, query_pool: np.ndarray, n_draws: int,
                                      zipf_exponent: float, rng: np.random.Generator, seed_base: int) -> tuple:
    """Draws n_draws topic indices with repetition from a Zipf-weighted
    distribution over the shared topic pool, then applies an INDEPENDENT
    small angular-jitter perturbation to every single draw. Even repeats
    of the same topic get their own fresh perturbation (confirmed
    interpretation), so repeated topics yield near-duplicate, not identical,
    vectors."""
    pmf = zipf_pmf(len(topic_pool_idx), zipf_exponent)
    draw_ranks = rng.choice(len(topic_pool_idx), size=n_draws, p=pmf)
    drawn_topic_idx = topic_pool_idx[draw_ranks]

    out = np.empty((n_draws, query_pool.shape[1]), dtype=np.float64)
    for i in range(n_draws):
        v = query_pool[drawn_topic_idx[i]]
        # generate_perturbed_vector uses np.random.seed() (global legacy state) internally
        # isolate this side effect to one call per draw, with a distinct derived seed each time.
        out[i] = generate_perturbed_vector(v, PERTURBATION_LEVEL, seed=seed_base + i)
    return out, drawn_topic_idx

#-------------------------------------------------------------------------------

def build_regime_query_sets(dataset: str, regime: str, query_pool: np.ndarray, out_dir: Path) -> Path:
    """Builds and persists one (dataset, regime)'s fixed TRAIN=1000/CALIB=500/
    TEST=5000 query sets, all drawn from ONE shared topic pool (disjointness
    at the instance level only. Idempotent."""
    out_path = out_dir / f"{dataset}_{regime}_queries.npz"
    if out_path.exists():
        return out_path

    regime_cfg = REGIME_CONFIG[regime]
    zipf_exponent = regime_cfg["zipf_exponent"]
    # no_locality's pool spans all three splits combined, to stay a "fresh topic,
    # near-zero repeat" baseline even though the pool is now shared across splits
    topic_pool_size = regime_cfg["topic_pool_size"] if regime_cfg["topic_pool_size"] is not None else (N_TRAIN + N_CALIB + N_TEST)

    rng = np.random.default_rng(SEED)
    topic_pool_idx = build_topic_pool(len(query_pool), topic_pool_size, rng)
    seed_cursor = SEED * 1000

    splits = {}
    topic_ids = {}
    for split_name, n_draws in [("train", N_TRAIN), ("calib", N_CALIB), ("test", N_TEST)]:
        vectors, drawn_topic_idx = draw_with_repetition_and_perturb(
            topic_pool_idx, query_pool, n_draws, zipf_exponent, rng, seed_cursor,
        )
        seed_cursor += n_draws
        splits[split_name] = vectors
        topic_ids[split_name] = drawn_topic_idx

    np.savez(
        out_path,
        train=splits["train"].astype(np.float32),
        calib=splits["calib"].astype(np.float32),
        test=splits["test"].astype(np.float32),
        train_topic_idx=topic_ids["train"], calib_topic_idx=topic_ids["calib"], test_topic_idx=topic_ids["test"],
    )
    return out_path

#-------------------------------------------------------------------------------

def load_regime_query_sets(path: Path) -> tuple:
    data = np.load(path)
    return data["train"], data["calib"], data["test"]

#-------------------------------------------------------------------------------

def load_regime_topic_ids(path: Path) -> tuple:
    """(train_topic_idx, calib_topic_idx, test_topic_idx) -- which shared-pool
    topic each draw came from, for the topic-level bootstrap LCB."""
    data = np.load(path)
    return data["train_topic_idx"], data["calib_topic_idx"], data["test_topic_idx"]

#-------------------------------------------------------------------------------

def prepare_all(out_dir: Path) -> dict:
    """Orchestrates section 3 end-to-end for both datasets x 3 regimes.
    Returns {(dataset, regime): path}."""
    from config import DATASETS, REGIMES

    paths = {}
    for dataset in DATASETS:
        query_pool = load_official_query_pool(dataset)
        for regime in REGIMES:
            path = build_regime_query_sets(dataset, regime, query_pool, out_dir)
            paths[(dataset, regime)] = path
    return paths
