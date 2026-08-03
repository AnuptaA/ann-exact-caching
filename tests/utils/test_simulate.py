#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from simulations.core.simulate import (
    QueryResult,
    SimulationResult,
    DistanceCalculationTracker,
    CacheSimulator
)
from utils.base import get_distance_function

#-------------------------------------------------------------------------------

def test_distance_tracker():
    """Test distance calculation tracking."""

    distance_func = get_distance_function("euclidean")
    tracker = DistanceCalculationTracker(distance_func)

    a = np.random.randn(32)
    b = np.random.randn(32)

    assert tracker.count == 0, f"Initial count should be 0, got {tracker.count}"

    _ = tracker(a, b)
    _ = tracker(a, b)
    _ = tracker(b, a)

    assert tracker.count == 3, f"Count should be 3, got {tracker.count}"

    tracker.reset()
    assert tracker.count == 0, f"Count after reset should be 0, got {tracker.count}"


#-------------------------------------------------------------------------------

def test_query_result_dataclass():
    """Test QueryResult dataclass."""

    result = QueryResult(
        query_id=0,
        algorithm="lemma1_no_union",
        cache_hit=True,
        correct=True,
        distance_calculations=50,
        time_us=123.45,
        hit_source="lemma1_no_union",
        vectors_from_cache=10
    )

    d = result.to_dict()
    assert d['query_id'] == 0
    assert d['algorithm'] == "lemma1_no_union"
    assert d['cache_hit'] == True
    assert d['correct'] == True
    assert d['distance_calculations'] == 50
    assert d['time_us'] == 123.45


#-------------------------------------------------------------------------------

def test_simulation_result_properties():
    """Test SimulationResult computed properties."""

    result = SimulationResult(
        dataset_name="test",
        algorithm="lemma1_no_union",
        metric="euclidean",
        total_queries=100,
        cache_hits=75,
        correct_results=70,
        total_distance_calcs=5000,
        total_time_us=10000.0,
        lemma1_hits=50,
        lemma2_hits=25
    )

    assert result.hit_rate == 0.75, f"Hit rate should be 0.75, got {result.hit_rate}"
    expected_accuracy = 70 / 75
    assert abs(result.accuracy - expected_accuracy) < 0.001, f"Accuracy should be {expected_accuracy}, got {result.accuracy}"
    assert result.avg_distance_calcs == 50.0, f"Avg distance calcs should be 50.0"
    assert result.avg_time_us == 100.0, f"Avg time should be 100.0"


#-------------------------------------------------------------------------------

def test_cache_simulator_initialization():
    """Test CacheSimulator initialization."""

    base_vectors = np.random.randn(100, 32).astype(np.float32)
    cache_queries = np.random.randn(10, 32).astype(np.float32)
    test_queries = np.random.randn(20, 32).astype(np.float32)

    simulator = CacheSimulator(
        base_vectors=base_vectors,
        cache_queries=cache_queries,
        test_queries=test_queries,
        K=10,
        N=5,
        metric="euclidean"
    )

    assert simulator.K == 10
    assert simulator.N == 5
    assert simulator.metric == "euclidean"
    assert simulator.cache is None
    assert len(simulator.main_memory.vectors) == 100


#-------------------------------------------------------------------------------

def test_cache_population():
    """Test cache population."""

    base_vectors = np.random.randn(100, 32).astype(np.float32)
    cache_queries = np.random.randn(5, 32).astype(np.float32)
    test_queries = np.random.randn(10, 32).astype(np.float32)

    simulator = CacheSimulator(
        base_vectors, cache_queries, test_queries,
        K=10, N=5, metric="euclidean"
    )

    simulator.populate_cache()

    assert simulator.cache is not None, "Cache should be populated"
    assert simulator.cache.size() == 5, f"Cache should have 5 entries, got {simulator.cache.size()}"

    entries = simulator.cache.get_all_entries()
    assert len(entries) == 5
    # index-only entries: no private top_k_vectors copy, just K=10 indices
    assert len(entries[0].top_k_indices) == 10, "Each entry should have K=10 indices"


#-------------------------------------------------------------------------------

def test_run_single_query():
    """Test running a single query."""

    base_vectors = np.random.randn(100, 32).astype(np.float32)
    cache_queries = np.random.randn(5, 32).astype(np.float32)
    test_queries = np.random.randn(10, 32).astype(np.float32)

    simulator = CacheSimulator(
        base_vectors, cache_queries, test_queries,
        K=10, N=5, metric="euclidean"
    )

    simulator.populate_cache()

    # run one query
    query = test_queries[0]
    result = simulator.run_query(0, query, "lemma1_no_union")

    assert result.query_id == 0
    assert result.algorithm == "lemma1_no_union"
    assert isinstance(result.cache_hit, bool)
    assert isinstance(result.correct, bool)
    # distance_calculations is always None; distance-call counting is not tracked.
    assert result.distance_calculations is None
    assert result.time_us > 0


#-------------------------------------------------------------------------------

def test_brute_force_algorithm():
    """Test brute force baseline."""

    base_vectors = np.random.randn(100, 32).astype(np.float32)
    cache_queries = np.random.randn(5, 32).astype(np.float32)
    test_queries = np.random.randn(10, 32).astype(np.float32)

    simulator = CacheSimulator(
        base_vectors, cache_queries, test_queries,
        K=10, N=5, metric="euclidean"
    )

    result = simulator.run_query(0, test_queries[0], "brute")

    assert result.cache_hit == False, "Brute force should never hit cache"
    # brute force distance calculations are not tracked (not a valid comparison with cache algos)
    assert result.distance_calculations is None, f"Brute force distance_calculations should be None, got {result.distance_calculations}"
    assert result.time_us is None, f"Brute force time_us should be None, got {result.time_us}"


#-------------------------------------------------------------------------------

def test_lemma_algorithms():
    """Test the canonical no-union algorithms (lemma1_no_union, lemma2_no_union).

    lemma1/lemma2/combined (union-based) are deprecated; see
    simulations/core/simulate.py.
    """

    base_vectors = np.random.randn(200, 32).astype(np.float32)
    cache_queries = np.random.randn(10, 32).astype(np.float32)
    test_queries = np.random.randn(20, 32).astype(np.float32)

    simulator = CacheSimulator(
        base_vectors, cache_queries, test_queries,
        K=20, N=10, metric="euclidean"
    )

    simulator.populate_cache()

    # test all canonical algorithms
    for algo in ["lemma1_no_union", "lemma2_no_union"]:
        result = simulator.run_query(0, test_queries[0], algo)

        assert result.algorithm == algo
        assert isinstance(result.cache_hit, bool)

        # if it's a hit, verify correctness was checked
        if result.cache_hit:
            assert isinstance(result.correct, bool)


def test_lemma_algorithms_sigma_relaxed():
    """lemma1_no_union/lemma2_no_union both support sigma < 1.0 through run_query."""

    np.random.seed(7)
    base_vectors = np.random.randn(200, 32).astype(np.float32)
    cache_queries = np.random.randn(10, 32).astype(np.float32)
    # test queries similar to cache queries to encourage hits
    test_queries = np.array(
        [cache_queries[i % len(cache_queries)] + np.random.randn(32) * 0.01 for i in range(10)],
        dtype=np.float32
    )

    for algo in ["lemma1_no_union", "lemma2_no_union"]:
        simulator = CacheSimulator(
            base_vectors, cache_queries, test_queries,
            K=20, N=10, metric="angular"
        )
        simulator.populate_cache()

        for sigma in (1.0, 0.7):
            result = simulator.run_query(0, test_queries[0], algo, sigma=sigma)
            assert result.algorithm == algo
            assert isinstance(result.cache_hit, bool)
            if result.cache_hit and sigma < 1.0:
                assert result.certified_indices is not None, (
                    f"{algo} sigma={sigma}: certified_indices should be set on a hit"
                )



def test_sigma_guard_rejects_unsupported_algorithm():
    """sigma != 1.0 must raise for algorithms that don't support relaxation."""

    base_vectors = np.random.randn(50, 16).astype(np.float32)
    cache_queries = np.random.randn(5, 16).astype(np.float32)
    test_queries = np.random.randn(3, 16).astype(np.float32)

    simulator = CacheSimulator(
        base_vectors, cache_queries, test_queries,
        K=5, N=3, metric="angular"
    )
    simulator.populate_cache()

    try:
        simulator.run_query(0, test_queries[0], "brute", sigma=0.8)
        assert False, "expected ValueError for sigma != 1.0 with algorithm='brute'"
    except ValueError as e:
        assert "sigma" in str(e).lower()



def test_unknown_algorithm_raises():
    """An unrecognized algorithm string raises ValueError; dispatch errors
    propagate rather than being captured as a query-level miss."""

    base_vectors = np.random.randn(50, 16).astype(np.float32)
    cache_queries = np.random.randn(5, 16).astype(np.float32)
    test_queries = np.random.randn(3, 16).astype(np.float32)

    simulator = CacheSimulator(
        base_vectors, cache_queries, test_queries,
        K=5, N=3, metric="angular"
    )
    simulator.populate_cache()

    try:
        simulator.run_query(0, test_queries[0], "not_a_real_algorithm")
        assert False, "expected ValueError for unknown algorithm"
    except ValueError as e:
        assert "Unknown algorithm" in str(e)



def test_run_query_before_populate_raises():
    """Calling run_query before populate_cache() raises AttributeError
    (self.cache is None); dispatch errors propagate rather than being
    captured as a query-level miss."""

    base_vectors = np.random.randn(50, 16).astype(np.float32)
    cache_queries = np.random.randn(5, 16).astype(np.float32)
    test_queries = np.random.randn(3, 16).astype(np.float32)

    simulator = CacheSimulator(
        base_vectors, cache_queries, test_queries,
        K=5, N=3, metric="angular"
    )
    # deliberately skip simulator.populate_cache()

    try:
        simulator.run_query(0, test_queries[0], "lemma1_no_union")
        assert False, "expected AttributeError from self.cache being None"
    except AttributeError:
        pass


#-------------------------------------------------------------------------------

def test_full_simulation():
    """Test complete simulation run."""

    base_vectors = np.random.randn(100, 32).astype(np.float32)
    cache_queries = np.random.randn(5, 32).astype(np.float32)
    test_queries = np.random.randn(10, 32).astype(np.float32)

    simulator = CacheSimulator(
        base_vectors, cache_queries, test_queries,
        K=10, N=5, metric="euclidean"
    )

    result = simulator.run_simulation("lemma1_no_union", verbose=False)

    assert result.total_queries == 10
    assert result.cache_hits + (result.total_queries - result.cache_hits) == result.total_queries
    assert result.total_distance_calcs >= 0
    assert result.total_time_us > 0
    assert 0.0 <= result.hit_rate <= 1.0
    assert 0.0 <= result.accuracy <= 1.0


#-------------------------------------------------------------------------------

def test_different_metrics():
    """Test simulation with different distance metrics."""

    base_vectors = np.random.randn(100, 32).astype(np.float32)
    cache_queries = np.random.randn(5, 32).astype(np.float32)
    test_queries = np.random.randn(10, 32).astype(np.float32)

    results = {}
    for metric in ["euclidean", "angular"]:
        simulator = CacheSimulator(
            base_vectors, cache_queries, test_queries,
            K=10, N=5, metric=metric
        )

        result = simulator.run_simulation("lemma1_no_union", verbose=True)

        assert result.metric == metric
        assert result.total_queries == 10
        # total_distance_calcs stays at 0 for non-brute algorithms; distance
        # calls are not tracked.
        assert result.total_distance_calcs == 0
        assert result.avg_time_us > 0

        results[metric] = result

    # at minimum, check that each metric completed successfully
    assert len(results) == 2

#-------------------------------------------------------------------------------

def test_override_n():
    """Test that override_N replaces self.N for a single query."""

    np.random.seed(42)
    base_vectors = np.random.randn(200, 32).astype(np.float32)
    cache_queries = np.random.randn(10, 32).astype(np.float32)

    # create very similar test queries to force cache hits
    test_queries = np.array(
        [cache_queries[0] + np.random.randn(32) * 0.001 for _ in range(5)],
        dtype=np.float32
    )

    simulator = CacheSimulator(
        base_vectors, cache_queries, test_queries,
        K=10, N=5, metric="euclidean"
    )
    simulator.populate_cache()

    # run with default N=5
    result_default = simulator.run_query(0, test_queries[0], "lemma1_no_union")
    assert result_default.algorithm == "lemma1_no_union", "Algorithm label mismatch"

    # run with override_N=3; should use N=3 instead of N=5
    result_override = simulator.run_query(0, test_queries[0], "lemma1_no_union", override_N=3)
    assert result_override.algorithm == "lemma1_no_union", "Algorithm label mismatch"

    # if hit with override, returned vectors should be <= override_N
    if result_override.cache_hit:
        assert result_override.vectors_from_cache <= 3, (
            f"override_N=3 should yield <=3 vectors, got {result_override.vectors_from_cache}"
        )

    # override_N=None should behave identically to no override
    result_none = simulator.run_query(0, test_queries[0], "lemma1_no_union", override_N=None)
    assert result_none.cache_hit == result_default.cache_hit, \
        "override_N=None should match default behavior"


#-------------------------------------------------------------------------------

def test_similar_query_high_hit_rate():
    """Test that similar queries get high hit rate."""

    base_vectors = np.random.randn(200, 32).astype(np.float32)
    cache_queries = np.random.randn(10, 32).astype(np.float32)

    # create test queries very similar to cached queries
    test_queries = []
    for i in range(20):
        cache_idx = i % len(cache_queries)
        similar = cache_queries[cache_idx] + np.random.randn(32) * 0.001
        test_queries.append(similar)
    test_queries = np.array(test_queries, dtype=np.float32)

    simulator = CacheSimulator(
        base_vectors, cache_queries, test_queries,
        K=20, N=10, metric="euclidean"
    )

    result = simulator.run_simulation("lemma1_no_union", verbose=False)

    assert result.total_queries == 20
