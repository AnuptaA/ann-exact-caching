#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from utils.base import MainMemory, KVCache

#-------------------------------------------------------------------------------

def test_cache_population():
    """Test populating cache from queries."""
    mm = MainMemory(M=1000, d=64, seed=42)
    cache = KVCache(metric="euclidean")
    queries = np.random.randn(10, 64).astype(np.float32)

    cache.populate_from_queries(queries, mm, K=20)

    assert cache.size() == 10, f"Expected 10 entries, got {cache.size()}"

    entries = cache.get_all_entries()
    assert len(entries) == 10, "Wrong number of entries"
    assert len(entries[0].top_k_vectors) == 20, "Wrong K"

#-------------------------------------------------------------------------------

def test_cache_entry():
    """Test cache entry properties."""
    mm = MainMemory(M=100, d=32, seed=42)
    query = np.random.randn(32)

    top_k_vecs, top_k_dists, gap = mm.top_k_search(query, k=5, metric="euclidean")

    cache = KVCache(metric="euclidean")
    cache.add_entry(query, top_k_vecs, float(top_k_dists[-1]), gap / 2.0)

    entry = cache.get_all_entries()[0]

    assert len(entry.top_k_vectors) == 5, "Wrong k"
    assert entry.get_radius() == top_k_dists[-1], "Wrong radius"
    assert entry.get_half_gap() == gap / 2, "Wrong half gap"

#-------------------------------------------------------------------------------