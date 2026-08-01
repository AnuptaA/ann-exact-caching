#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from utils.base import MainMemory, get_distance_function, range_search

#-------------------------------------------------------------------------------

def test_random_initialization():
    """Test generating random vectors."""
    print("Test 1: Random initialization")
    mm = MainMemory(M=100, d=64, seed=42)
    
    assert mm.M == 100, "Wrong M"
    assert mm.d == 64, "Wrong d"
    assert len(mm.vectors) == 100, "Wrong number of vectors"
    assert mm.vectors[0].shape[0] == 64, "Wrong dimension"
    print("Random initialization works.")

#-------------------------------------------------------------------------------

def test_from_vectors():
    """Test initializing from existing vectors."""
    print("\nTest 2: Initialize from existing vectors")
    
    # create some vectors
    existing_vecs = [np.random.randn(32) for _ in range(50)]
    
    mm = MainMemory(vectors=existing_vecs)
    
    assert mm.M == 50, "Wrong M"
    assert mm.d == 32, "Wrong d"
    assert len(mm.vectors) == 50, "Wrong number of vectors"
    assert np.array_equal(mm.vectors[0], existing_vecs[0]), "Vectors don't match"
    print("Initialization from vectors works.")

#-------------------------------------------------------------------------------

def test_top_k_search():
    """Test top-k search."""
    print("\nTest 3: Top-k search")
    
    mm = MainMemory(M=1000, d=128, seed=42)
    query = np.random.randn(128)
    
    top_k_vecs, top_k_dists, gap = mm.top_k_search(query, k=10, metric="euclidean")
    
    assert len(top_k_vecs) == 10, "Wrong number of results"
    assert len(top_k_dists) == 10, "Wrong number of distances"
    assert gap >= 0, "Gap should be non-negative"
    
    # check distances are sorted
    for i in range(len(top_k_dists) - 1):
        assert top_k_dists[i] <= top_k_dists[i+1], "Distances not sorted"
    
    print("Top-k search works.")

#-------------------------------------------------------------------------------

def test_missing_args():
    """Test that missing arguments raise error."""
    print("\nTest 4: Missing arguments")

    try:
        mm = MainMemory()  # should fail
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"Correctly raised error: {e}")

#-------------------------------------------------------------------------------

def test_top_k_search_tracked_matches_untracked():
    """The untracked (vectorized matmul) and tracked (Python loop, every call
    counted) code paths must return identical results -- a future performance
    optimization to either path should not silently diverge from the other."""
    print("\nTest 5: tracked vs untracked top_k_search equivalence")

    for metric in ("euclidean", "angular"):
        mm = MainMemory(M=300, d=32, seed=7)
        query = np.random.RandomState(1).randn(32)
        k = 15

        untracked = mm.top_k_search(query, k, metric=metric, return_indices=True)
        distance_func = get_distance_function(metric)
        tracked = mm.top_k_search(
            query, k, metric=metric, distance_tracker=distance_func, return_indices=True
        )

        vecs_u, dists_u, gap_u, idx_u = untracked
        vecs_t, dists_t, gap_t, idx_t = tracked

        assert idx_u == idx_t, f"{metric}: index mismatch\n{idx_u}\nvs\n{idx_t}"
        assert np.allclose(dists_u, dists_t, atol=1e-5), f"{metric}: distance mismatch"
        assert abs(gap_u - gap_t) < 1e-5, f"{metric}: gap mismatch {gap_u} vs {gap_t}"
        for a, b in zip(vecs_u, vecs_t):
            assert np.allclose(a, b), f"{metric}: vector mismatch"

        print(f"  {metric}: tracked and untracked paths agree")

#-------------------------------------------------------------------------------

def test_range_search_matches_top_k_search():
    """range_search(query, kth_dist, base, metric) must return the same index
    set as top_k_search(query, k, metric) -- both independently implement
    vectorized normalize+distance logic and should agree."""
    print("\nTest 6: range_search vs top_k_search equivalence")

    rng = np.random.RandomState(3)
    base = rng.randn(200, 24).astype(np.float32)
    mm = MainMemory(vectors=list(base))
    query = rng.randn(24).astype(np.float32)
    k = 10

    for metric in ("euclidean", "angular"):
        _, top_k_dists, _, top_k_indices = mm.top_k_search(
            query, k, metric=metric, return_indices=True
        )
        kth_dist = top_k_dists[-1]

        range_indices = set(range_search(query, kth_dist, base, metric).tolist())
        assert range_indices == set(top_k_indices), (
            f"{metric}: range_search {range_indices} != top_k_search {set(top_k_indices)}"
        )
        print(f"  {metric}: range_search matches top_k_search's index set")

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    print("="*80)
    print("Testing MainMemory")
    print("="*80)

    test_random_initialization()
    test_from_vectors()
    test_top_k_search()
    test_missing_args()
    test_top_k_search_tracked_matches_untracked()
    test_range_search_matches_top_k_search()

    print("\n" + "="*80)
    print("All tests passed.")
    print("="*80)
#-------------------------------------------------------------------------------