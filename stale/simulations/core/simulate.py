#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""
Shared functionality for simulating cache algorithms on a dataset.
"""

import numpy as np
import time
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.base import (
    MainMemory,
    KVCache,
    DistanceMetric,
    get_distance_function
)
from utils.base.lemmas_sigma import (
    lemma1_no_union_sigma, lemma2_no_union_sigma, overlap_threshold_hit_policy,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

#-------------------------------------------------------------------------------

@dataclass
class QueryResult:
    """Results from a single query."""
    query_id: int
    algorithm: str
    cache_hit: bool
    correct: bool
    distance_calculations: Optional[int] # calls to distance function during cache lookup only
    time_us: Optional[float]
    hit_source: Optional[str] = None
    vectors_from_cache: int = 0
    error: Optional[str] = None
    result_indices: Optional[List[int]] = None  # indices of result vectors for validation
    certified_indices: Optional[List[int]] = None

    def to_dict(self) -> Dict:
        return asdict(self)

#-------------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Aggregate simulation results."""
    dataset_name: str
    algorithm: str
    metric: str
    total_queries: int
    cache_hits: int
    correct_results: int
    total_distance_calcs: Optional[int] # none for brute force
    total_time_us: Optional[float] # none for brute force
    lemma1_hits: int = 0
    lemma2_hits: int = 0
    sigma: float = 1.0
    mean_recall: float = 0.0  # avg fraction of true top-K certified on a hit; lemma1 sigma runs only
    query_results: List[QueryResult] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return self.cache_hits / self.total_queries if self.total_queries > 0 else 0.0
    
    @property
    def accuracy(self) -> float:
        """Accuracy of cache hits, no-hit runs are conventionally perfect."""
        return self.correct_results / self.cache_hits if self.cache_hits > 0 else 1
    
    @property
    def avg_distance_calcs(self) -> Optional[float]:
        if self.total_distance_calcs is None:
            return None
        return self.total_distance_calcs / self.total_queries if self.total_queries > 0 else 0.0

    @property
    def avg_time_us(self) -> Optional[float]:
        if self.total_time_us is None:
            return None
        return self.total_time_us / self.total_queries if self.total_queries > 0 else 0.0
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['hit_rate'] = self.hit_rate
        d['accuracy'] = self.accuracy
        d['avg_distance_calcs'] = self.avg_distance_calcs
        d['avg_time_us'] = self.avg_time_us
        return d

#-------------------------------------------------------------------------------

class DistanceCalculationTracker:
    """Wrapper to count distance calculations."""
    
    def __init__(self, distance_func):
        self.distance_func = distance_func
        self.count = 0
    
    def __call__(self, a, b):
        self.count += 1
        return self.distance_func(a, b)
    
    def reset(self):
        self.count = 0

#-------------------------------------------------------------------------------

class CacheSimulator:
    """Simulate cache algorithms on a dataset."""
    
    def __init__(
        self,
        base_vectors: np.ndarray,
        cache_queries: np.ndarray,
        test_queries: np.ndarray,
        K: int,
        N: int,
        metric: DistanceMetric = "euclidean",
        r_t_hat: Optional[float] = None,
        overlap_threshold: float = 0.9,
    ):
        """
        Initialize the cache simulation.

        Args:
            base_vectors: Database vectors
            cache_queries: Queries to populate the cache
            test_queries: Queries to test the algorithms on
            K: Number of vectors to cache per query
            N: Number of vectors to retrieve per query
            metric: Distance metric
            r_t_hat: Workload-calibrated top-K radius estimate, required
                only by algorithm="overlap_threshold"
            overlap_threshold: Intersection-fraction threshold for
                algorithm="overlap_threshold"
        """
        self.base_vectors = base_vectors
        self.cache_queries = cache_queries
        self.test_queries = test_queries
        self.K = K
        self.N = N
        self.metric = metric
        self.r_t_hat = r_t_hat
        self.overlap_threshold = overlap_threshold

        # create main memory from base vectors
        self.main_memory = MainMemory(vectors=list(base_vectors))
        self._vector_id_to_index = {
            id(vec): idx for idx, vec in enumerate(self.main_memory.vectors)
        }

        self.distance_func = get_distance_function(metric)

        # shared pre-normalized base matrix to eliminate duplication
        base_norms = np.linalg.norm(base_vectors, axis=1, keepdims=True)
        base_norms[base_norms == 0.0] = 1.0
        self.base_hat = (base_vectors / base_norms).astype(np.float32)

        # initialize cache
        self.cache = None

    def populate_cache(self):
        """Populate cache with cache queries."""
        logger.info(f"Populating cache with {len(self.cache_queries)} queries...")
        self.cache = KVCache(metric=self.metric)

        for idx, query in enumerate(self.cache_queries):
            # compute top-K for this query
            top_k_vecs, top_k_dists, gap, top_k_indices = self.main_memory.top_k_search(
                query,
                self.K,
                self.metric,
                return_indices=True
            )
            # eliminate duplication by storing only the indices
            self.cache.add_entry(query, [], float(top_k_dists[-1]), gap / 2.0, top_k_indices)

            if idx < 3:
                logger.debug(f"Cache entry {idx}: K={len(top_k_indices)}, gap={gap:.6f}, "
                           f"first_dist={top_k_dists[0]:.6f}, last_dist={top_k_dists[-1]:.6f}")

        logger.info(f"Cache populated with {self.cache.size()} entries")
    
    def run_query(
        self,
        query_id: int,
        query: np.ndarray,
        algorithm: str,
        ground_truth_indices: Optional[List[int]] = None,
        override_N: Optional[int] = None,
        sigma: float = 1.0
    ) -> QueryResult:
        """
        Run a single query through the cache algorithm.

        Args:
            query_id: Query identifier
            query: Query vector
            algorithm: Algorithm to use
            ground_truth_indices: Optional precomputed ground truth indices for validation
            sigma: Recall threshold in (0, 1]. Supported by "lemma1_no_union" and
                "lemma2_no_union"; other algorithms raise if sigma != 1.0. sigma=1.0
                (default) reproduces exact, non-relaxed behavior.

        Returns:
            QueryResult with metrics
        """
        # time_us covers only algorithm execution
        # index lookup and validation happen after time_us is recorded
        start_time = time.perf_counter()

        if query_id < 3:
            logger.debug(f"Query {query_id} ({algorithm}): query_shape={query.shape}")
            if ground_truth_indices is not None:
                logger.debug(f"  Ground truth: {len(ground_truth_indices)} indices")

        N = override_N if override_N is not None else self.N

        if sigma != 1.0 and algorithm not in ("lemma1_no_union", "lemma2_no_union"):
            raise ValueError(
                f"sigma={sigma} != 1.0 is only supported for algorithm='lemma1_no_union' "
                f"or 'lemma2_no_union' (got algorithm={algorithm!r})"
            )

        metadata = {}
        if algorithm == "brute":
            # brute force: compute actual top-N from main memory
            # performance (distance calculations, time) is not tracked
            result, _, _, brute_indices = self.main_memory.top_k_search(
                query, N, self.metric, return_indices=True
            )
            metadata["result_indices"] = brute_indices
            is_hit = False
        elif algorithm == "lemma1_no_union":
            d_Q_batch = self.cache.get_d_Q_batch(query)
            entry_radii = self.cache.get_entry_radii()
            result, is_hit, metadata = lemma1_no_union_sigma(
                query, self.cache.get_all_entries(), N, sigma, self.metric,
                precomputed_d_Q=d_Q_batch, entry_radii=entry_radii,
                base_hat=self.base_hat, base_vectors=self.base_vectors,
            )
        elif algorithm == "lemma2_no_union":
            d_Q_batch = self.cache.get_d_Q_batch(query)
            result, is_hit, metadata = lemma2_no_union_sigma(
                query, self.cache.get_all_entries(), N, sigma, self.metric,
                precomputed_d_Q=d_Q_batch,
                base_hat=self.base_hat, base_vectors=self.base_vectors,
            )
        elif algorithm == "overlap_threshold":
            d_Q_batch = self.cache.get_d_Q_batch(query)
            entry_radii = self.cache.get_entry_radii()
            result, is_hit, metadata = overlap_threshold_hit_policy(
                query, self.cache.get_all_entries(), N, sigma, self.metric,
                precomputed_d_Q=d_Q_batch, precomputed_r_c=entry_radii,
                base_hat=self.base_hat, base_vectors=self.base_vectors,
                r_t_hat=self.r_t_hat, threshold=self.overlap_threshold,
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        elapsed_us = None if algorithm == "brute" else (time.perf_counter() - start_time) * 1e6

        # convert result vectors to stable indices for validation
        result_indices = metadata.get("result_indices")
        if result_indices is not None:
            result_indices = [int(idx) for idx in result_indices]
        elif result is not None:
            result_indices = [
                self._vector_id_to_index[id(vec)]
                for vec in result
                if id(vec) in self._vector_id_to_index
            ]
        
        # determine correctness
        is_correct = False
        
        if algorithm == "brute":
            # brute force is always "correct" by definition
            if result is not None and ground_truth_indices is not None and result_indices is not None:
                is_correct = self._compare_results_by_indices(result_indices, ground_truth_indices)
            elif result is not None:
                is_correct = True  # assume brute force is correct by default
        elif is_hit and result is not None:
            # for cache algorithms, verify if hit was correct
            if ground_truth_indices is not None and result_indices is not None:
                is_correct = self._compare_results_by_indices(result_indices, ground_truth_indices)
            else:
                is_correct = False  # no ground truth provided, treat as unverified
        
        if query_id < 3:
            logger.debug(f"  Result: hit={is_hit}, correct={is_correct}, "
                       f"result_len={len(result) if result else 0}")

        certified_indices = metadata.get('certified_indices')
        if certified_indices is not None:
            certified_indices = [int(idx) for idx in certified_indices]

        return QueryResult(
            query_id=query_id,
            algorithm=algorithm,
            cache_hit=is_hit,
            correct=is_correct,
            distance_calculations=None,
            time_us=elapsed_us,
            hit_source=metadata.get('hit_lemma') or metadata.get('hit_source'),
            vectors_from_cache=metadata.get('vectors_added', 0),
            error=metadata.get('error'),
            result_indices=result_indices,
            certified_indices=certified_indices,
        )
    
    def _compare_results_by_indices(self, result_indices: List[int], ground_truth_indices: List[int]) -> bool:
        """
        Compare two result sets by indices.
        
        Args:
            result_indices: Result vector indices
            ground_truth_indices: Ground truth vector indices
            
        Returns:
            True if results match the ground truth indices, False otherwise
        """
        if len(result_indices) != len(ground_truth_indices):
            logger.debug(f"Length mismatch: result={len(result_indices)}, ground_truth={len(ground_truth_indices)}")
            return False
        
        result_set = set(result_indices)
        gt_set = set(ground_truth_indices)
        match = result_set == gt_set
        
        if not match:
            logger.debug(f"Index set mismatch: {len(result_set)} vs {len(gt_set)} unique indices")
            only_in_result = result_set - gt_set
            only_in_gt = gt_set - result_set
            if only_in_result:
                logger.debug(f"Only in result: {sorted(only_in_result)[:5]}")
            if only_in_gt:
                logger.debug(f"Only in GT: {sorted(only_in_gt)[:5]}")
        
        return match
    
    def run_simulation(
        self,
        algorithm: str,
        verbose: bool = False,
        populate: bool = True
    ) -> SimulationResult:
        """
        Run complete simulation with all test queries.
        
        Args:
            algorithm: Algorithm to use
            verbose: Print progress
            
        Returns:
            SimulationResult with aggregate metrics
        """
        print(f"\n{'='*80}")
        print(f"Running simulation: {algorithm} on {self.metric} distance")
        print(f"{'='*80}")
        
        if algorithm != "brute" and populate:
            self.populate_cache()
        
        query_results = []
        cache_hits = 0
        correct_results = 0
        total_distance_calcs = None if algorithm == "brute" else 0
        total_time_us = None if algorithm == "brute" else 0.0
        lemma1_hits = 0
        lemma2_hits = 0

        for i, query in enumerate(self.test_queries):
            if verbose and (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(self.test_queries)} queries...")

            result = self.run_query(i, query, algorithm)
            query_results.append(result)

            if result.cache_hit:
                cache_hits += 1
            if result.correct:
                correct_results += 1

            if result.distance_calculations is not None:
                total_distance_calcs += result.distance_calculations
            if result.time_us is not None:
                total_time_us += result.time_us

            if result.hit_source == "lemma1":
                lemma1_hits += 1
            elif result.hit_source == "lemma2":
                lemma2_hits += 1

        sim_result = SimulationResult(
            dataset_name="",
            algorithm=algorithm,
            metric=self.metric,
            total_queries=len(self.test_queries),
            cache_hits=cache_hits,
            correct_results=correct_results,
            total_distance_calcs=total_distance_calcs,
            total_time_us=total_time_us,
            lemma1_hits=lemma1_hits,
            lemma2_hits=lemma2_hits,
            query_results=query_results
        )
        
        self._print_summary(sim_result)
        
        return sim_result
    
    def _print_summary(self, result: SimulationResult):
        """Print simulation summary."""
        print(f"\n{'='*80}")
        print("Simulation summary")
        print(f"{'='*80}")
        print(f"Algorithm: {result.algorithm}")
        print(f"Metric: {result.metric}")
        print(f"Total Queries: {result.total_queries}")
        print(f"Cache Hits: {result.cache_hits} ({result.hit_rate:.1%})")
        print(f"  - Lemma 1 Hits: {result.lemma1_hits}")
        print(f"  - Lemma 2 Hits: {result.lemma2_hits}")
        print(f"Correct Results: {result.correct_results} ({result.accuracy:.1%})")
        dc = f"{result.avg_distance_calcs:.1f}" if result.avg_distance_calcs is not None else "N/A (brute)"
        t = f"{result.avg_time_us:.1f} us" if result.avg_time_us is not None else "N/A (brute)"
        print(f"Avg Distance Calcs: {dc}")
        print(f"Avg Time: {t}")
        print(f"{'='*80}\n")
#-------------------------------------------------------------------------------