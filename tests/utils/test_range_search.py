#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from utils.base import range_search, angular_distance, euclidean_distance

#-------------------------------------------------------------------------------

def _make_base(n: int = 20, d: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, d)).astype(np.float32)
    norms = np.linalg.norm(base, axis=1, keepdims=True)
    return base / norms

def _make_query(d: int = 8, seed: int = 99) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(d).astype(np.float32)
    return q / np.linalg.norm(q)

#-------------------------------------------------------------------------------

def test_range_search_angular_correctness():
    """All returned indices satisfy the radius condition; no outside index does."""
    print("Test: angular correctness")
    base = _make_base()
    query = _make_query()

    for radius in [0.05, 0.15, 0.30]:
        result = range_search(query, radius, base, metric="angular")

        # every returned index must be within radius
        for idx in result:
            d = angular_distance(query, base[idx])
            assert d <= radius + 1e-6, f"idx {idx} has dist {d} > radius {radius}"

        # every index outside result must be beyond radius
        outside = set(range(len(base))) - set(result.tolist())
        for idx in outside:
            d = angular_distance(query, base[idx])
            assert d > radius - 1e-6, f"idx {idx} has dist {d} <= radius {radius} but was excluded"

    print("  Passed.")

#-------------------------------------------------------------------------------

def test_range_search_euclidean_correctness():
    """Same correctness test for euclidean metric on normalized vectors."""
    print("Test: euclidean correctness")
    base = _make_base()
    query = _make_query()

    for radius in [0.10, 0.30, 0.60]:
        result = range_search(query, radius, base, metric="euclidean")

        for idx in result:
            d = euclidean_distance(query, base[idx])
            assert d <= radius + 1e-6, f"idx {idx} has dist {d} > radius {radius}"

        outside = set(range(len(base))) - set(result.tolist())
        for idx in outside:
            d = euclidean_distance(query, base[idx])
            assert d > radius - 1e-6, f"idx {idx} has dist {d} <= radius {radius} but was excluded"

    print("  Passed.")

#-------------------------------------------------------------------------------

def test_range_search_zero_radius():
    """Zero radius: only exact-match vectors (distance 0) returned."""
    print("Test: zero radius")
    base = _make_base()
    query = _make_query()

    for metric in ("angular", "euclidean"):
        result = range_search(query, 0.0, base, metric=metric)
        # query is not in base, so result should be empty or contain only dist-0 entries
        for idx in result:
            d = angular_distance(query, base[idx]) if metric == "angular" else \
                euclidean_distance(query, base[idx])
            assert d <= 1e-6, f"distance {d} > 0 at zero radius"

    print("  Passed.")

#-------------------------------------------------------------------------------

def test_range_search_large_radius():
    """Radius larger than any possible distance returns all indices."""
    print("Test: large radius returns all")
    base = _make_base()
    query = _make_query()

    for metric in ("angular", "euclidean"):
        # angular max is 1.0, euclidean (normalized) max is 2.0
        large_radius = 2.0
        result = range_search(query, large_radius, base, metric=metric)
        assert len(result) == len(base), \
            f"Expected all {len(base)} indices, got {len(result)} for metric={metric}"

    print("  Passed.")

#-------------------------------------------------------------------------------

def test_range_search_zero_norm_query():
    """Zero-norm query vector returns empty array without crashing."""
    print("Test: zero-norm query")
    base = _make_base()
    zero_query = np.zeros(8, dtype=np.float32)

    for metric in ("angular", "euclidean"):
        result = range_search(zero_query, 0.5, base, metric=metric)
        assert isinstance(result, np.ndarray), "should return ndarray"
        assert len(result) == 0, f"Expected empty result for zero query, got {len(result)}"

    print("  Passed.")

#-------------------------------------------------------------------------------

def test_range_search_zero_norm_base_rows():
    """Zero-norm base rows do not crash; they get a safe fallback distance."""
    print("Test: zero-norm base rows")
    d = 8
    base = _make_base(n=10, d=d)
    base[3] = np.zeros(d, dtype=np.float32)  # inject a zero-norm row
    query = _make_query(d=d)

    for metric in ("angular", "euclidean"):
        result = range_search(query, 0.5, base, metric=metric)
        assert isinstance(result, np.ndarray)
        # index 3 (zero-norm) should not crash; its distance depends on fallback
        # angular: arccos(0)/pi = 0.5, so it appears at exactly 0.5

    print("  Passed.")

#-------------------------------------------------------------------------------

def test_range_search_returns_int64():
    """Result dtype is int64."""
    print("Test: returns int64")
    base = _make_base()
    query = _make_query()

    for metric in ("angular", "euclidean"):
        result = range_search(query, 0.3, base, metric=metric)
        assert result.dtype == np.int64, f"Expected int64, got {result.dtype}"

    print("  Passed.")

#-------------------------------------------------------------------------------

def test_range_search_content_not_order():
    """Result is a set of correct indices regardless of order."""
    print("Test: content (not order)")
    base = _make_base()
    query = _make_query()
    radius = 0.2

    for metric in ("angular", "euclidean"):
        result = range_search(query, radius, base, metric=metric)
        result_set = set(result.tolist())
        # recompute expected set brute-force
        dist_fn = angular_distance if metric == "angular" else euclidean_distance
        expected = {i for i in range(len(base)) if dist_fn(query, base[i]) <= radius}
        assert result_set == expected, \
            f"Result set mismatch for metric={metric}: got {result_set}, expected {expected}"

    print("  Passed.")

#-------------------------------------------------------------------------------

def test_range_search_invalid_metric():
    """Unsupported metric raises ValueError."""
    print("Test: invalid metric raises ValueError")
    base = _make_base()
    query = _make_query()
    try:
        range_search(query, 0.5, base, metric="cosine")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("  Passed.")

#-------------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Testing range_search")
    print("=" * 70)

    test_range_search_angular_correctness()
    test_range_search_euclidean_correctness()
    test_range_search_zero_radius()
    test_range_search_large_radius()
    test_range_search_zero_norm_query()
    test_range_search_zero_norm_base_rows()
    test_range_search_returns_int64()
    test_range_search_content_not_order()
    test_range_search_invalid_metric()

    print("\n" + "=" * 70)
    print("All tests passed.")
    print("=" * 70)

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
