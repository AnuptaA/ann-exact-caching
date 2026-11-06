#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import json
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass, asdict

from utils.base.perturbation import generate_perturbed_vector

#-------------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    """Configuration for synthetic dataset generation."""
    name: str
    num_base_vectors: int
    num_cache_queries: int
    num_test_queries: int
    dimension: int
    K: int  # vectors cached per query
    N: int  # vectors requested per test query
    seed: Optional[int] = None
    
    # generate queries in different ways
    test_query_strategy: str = "similar"        # "similar", "union_clustered"
    num_clusters: int = 5                       # number of base vectors for Set 5 union_clustered strategy
    perturbation_level: Optional[int] = None    # for angular perturbation: 0 = large, 1 = medium, 2 = small
    N_range: Optional[Tuple[int, int]] = None   # for variable N tests (Set 4)
    test_N_values: Optional[List[int]] = None   # actual N values per query for variable N
    test_perturbation_angles_deg: Optional[List[float]] = None  # sampled perturbation angle per test query (degrees)
    num_test_centers: Optional[int] = None      # if set, use only cache_queries[:num_test_centers] as center pool
    test_query_seed: Optional[int] = None       # if set, reset RNG before test query generation
    shared_test_queries_path: Optional[str] = None  # if set, load test queries from this path instead of generating
    
    def to_dict(self) -> Dict:
        """Serialize config to a dictionary."""
        return asdict(self)

#-------------------------------------------------------------------------------

class SyntheticDatasetGenerator:
    """Generate synthetic datasets for lemma verification."""
    
    def __init__(self, config: DatasetConfig):
        """
        Initialize generator with configuration.
        
        Args:
            config: Dataset configuration
        """
        self.config = config
        if config.seed is not None:
            np.random.seed(config.seed)
        
        self.base_vectors: Optional[np.ndarray] = None
        self.cache_queries: Optional[np.ndarray] = None
        self.test_queries: Optional[np.ndarray] = None
    
    def generate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate complete dataset.
        
        Returns:
            Tuple of (base_vectors, cache_queries, test_queries)
        """
        print(f"Generating dataset: {self.config.name}")
        print(f"Base vectors: {self.config.num_base_vectors}")
        print(f"Cache queries: {self.config.num_cache_queries}")
        print(f"Test queries: {self.config.num_test_queries}")
        print(f"Dimension: {self.config.dimension}")
        print(f"K={self.config.K}, N={self.config.N}")
        
        # generate database vectors
        self.base_vectors = self._generate_base_vectors()
        
        # generate pre-populated cache queries
        self.cache_queries = self._generate_cache_queries()
        
        # generate test queries based on strategy
        self.test_queries = self._generate_test_queries()
        
        # for variable-N datasets, assign a per-query N sampled from N_range
        if self.config.N_range is not None:
            lo, hi = self.config.N_range
            self.config.test_N_values = [
                int(np.random.randint(lo, hi + 1))
                for _ in range(self.config.num_test_queries)
            ]
            print(f"Assigned per-query N values: range [{lo}, {hi}], "
                  f"mean={np.mean(self.config.test_N_values):.1f}")
        
        return self.base_vectors, self.cache_queries, self.test_queries
    
    def _generate_base_vectors(self) -> np.ndarray:
        """Generate base database vectors."""
        print("Generating base vectors...")
        vectors = np.random.randn(
            self.config.num_base_vectors, 
            self.config.dimension
        ).astype(np.float32)
        return vectors
    
    def _generate_cache_queries(self) -> np.ndarray:
        """Generate queries for cache population."""
        print("Generating cache queries...")
        
        # special handling for union_clustered strategy in Set 5
        if self.config.test_query_strategy == "union_clustered":
            return self._generate_clustered_cache_queries()
        
        # default: random cache queries
        queries = np.random.randn(
            self.config.num_cache_queries,
            self.config.dimension
        ).astype(np.float32)
        
        return queries
    
    def _generate_clustered_cache_queries(self) -> np.ndarray:
        """
        Generate clustered cache queries for union effectiveness test in Set 5.
        
        Creates X base vectors, then fills cache to 1024 by generating small 
        perturbations of these base vectors.
        """
        X = self.config.num_clusters  # number of base vectors
        
        base_vectors = np.random.randn(X, self.config.dimension).astype(np.float32)
        base_vectors = base_vectors / (np.linalg.norm(base_vectors, axis=1, keepdims=True) + 1e-10)
        
        # store for later use in test query generation
        self.base_cache_vectors = base_vectors
        
        queries = []
        queries_per_base = self.config.num_cache_queries // X
        
        for base_vec in base_vectors:
            for _ in range(queries_per_base):
                queries.append(generate_perturbed_vector(base_vec, self.config.perturbation_level))
        
        # fill remaining slots if needed
        while len(queries) < self.config.num_cache_queries:
            base_idx = np.random.randint(0, X)
            queries.append(generate_perturbed_vector(base_vectors[base_idx], self.config.perturbation_level))
        
        return np.array(queries[:self.config.num_cache_queries], dtype=np.float32)
    
    def _generate_test_queries(self) -> np.ndarray:
        """Generate test queries based on strategy."""
        if self.config.shared_test_queries_path is not None:
            path = Path(self.config.shared_test_queries_path)
            print(f"Loading shared test queries from {path}...")
            return np.load(path)
        if self.config.test_query_seed is not None:
            np.random.seed(self.config.test_query_seed)
        print(f"Generating test queries (strategy: {self.config.test_query_strategy})...")
        
        strategy = self.config.test_query_strategy
        
        if strategy == "similar":
            return self._generate_similar_test_queries()
        elif strategy == "union_clustered":
            return self._generate_union_clustered_test_queries()
        else:
            raise ValueError(f"Unknown test query strategy: {strategy}")
    
    def _generate_similar_test_queries(self) -> np.ndarray:
        """
        Generate test queries clustered around randomly selected cached queries.
        
        Repeatedly picks a cache query (without replacement, reshuffling when
        exhausted), draws a random cluster size in [8, 10], generates that many
        angular perturbations, and subtracts from the remaining budget until
        the full num_test_queries is reached.
        """
        queries = []
        remaining = self.config.num_test_queries
        angles = []
        
        # shuffle cache indices for without-replacement sampling
        n_centers = min(
            self.config.num_test_centers or self.config.num_cache_queries,
            len(self.cache_queries)
        )
        cache_indices = np.random.permutation(n_centers)
        idx_pos = 0
        
        while remaining > 0:
            # reshuffle when all cache queries have been used as centers
            if idx_pos >= len(cache_indices):
                cache_indices = np.random.permutation(self.config.num_cache_queries)
                idx_pos = 0
            
            center = self.cache_queries[cache_indices[idx_pos]]
            idx_pos += 1
            center_norm = center / (np.linalg.norm(center) + 1e-10)
            
            # randomly choose cluster size in [8, 10], capped by remaining budget
            n = min(remaining, int(np.random.randint(8, 11)))
            for _ in range(n):
                vec, angle = generate_perturbed_vector(center_norm, self.config.perturbation_level, return_angle=True)
                queries.append(vec)
                angles.append(angle)
            remaining -= n
        
        self.config.test_perturbation_angles_deg = angles
        return np.array(queries, dtype=np.float32)
    
    def _generate_union_clustered_test_queries(self) -> np.ndarray:
        """
        Generate test queries as small perturbations of the X base cache vectors
        (Set 5).
        
        Requires _generate_clustered_cache_queries() to have been called first
        so that self.base_cache_vectors is populated.
        """
        if not hasattr(self, 'base_cache_vectors'):
            raise RuntimeError("base_cache_vectors not set")
        
        queries = []
        angles = []
        X = len(self.base_cache_vectors)
        queries_per_base = max(1, self.config.num_test_queries // X)
        
        for base_vec in self.base_cache_vectors:
            for _ in range(queries_per_base):
                if len(queries) >= self.config.num_test_queries:
                    break
                vec, angle = generate_perturbed_vector(base_vec, self.config.perturbation_level, return_angle=True)
                queries.append(vec)
                angles.append(angle)
            if len(queries) >= self.config.num_test_queries:
                break
        
        # fill any remaining slots round-robin
        i = 0
        while len(queries) < self.config.num_test_queries:
            vec, angle = generate_perturbed_vector(self.base_cache_vectors[i % X], self.config.perturbation_level, return_angle=True)
            queries.append(vec)
            angles.append(angle)
            i += 1
        
        self.config.test_perturbation_angles_deg = angles
        return np.array(queries[:self.config.num_test_queries], dtype=np.float32)
    
    def save(self, output_dir: str):
        """
        Save generated dataset to disk.
        
        Args:
            output_dir: Directory to save dataset files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        dataset_path = output_path / self.config.name
        dataset_path.mkdir(exist_ok=True)
        
        print(f"\nSaving dataset to: {dataset_path}")
        
        # save vectors
        np.save(dataset_path / "base_vectors.npy", self.base_vectors)
        np.save(dataset_path / "cache_queries.npy", self.cache_queries)
        np.save(dataset_path / "test_queries.npy", self.test_queries)
        
        # save configuration
        with open(dataset_path / "config.json", 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        print(f"Saved base_vectors.npy: {self.base_vectors.shape}")
        print(f"Saved cache_queries.npy: {self.cache_queries.shape}")
        print(f"Saved test_queries.npy: {self.test_queries.shape}")
        print(f"Saved config.json")

#-------------------------------------------------------------------------------

def load_synthetic_dataset(dataset_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Load a previously generated synthetic dataset.
    
    Args:
        dataset_path: Path to dataset directory
        
    Returns:
        Tuple of (base_vectors, cache_queries, test_queries, config)
    """
    path = Path(dataset_path)
    
    base_vectors = np.load(path / "base_vectors.npy")
    cache_queries = np.load(path / "cache_queries.npy")
    test_queries = np.load(path / "test_queries.npy")
    
    with open(path / "config.json", 'r') as f:
        config = json.load(f)
    
    return base_vectors, cache_queries, test_queries, config

#-------------------------------------------------------------------------------

def create_test_set_1_baseline(seed: int = 42) -> List[DatasetConfig]:
    # Test Set 1: Baseline datasets with fixed parameters.
    configs = []
    for s in [seed, seed + 1, seed + 2]:
        configs.append(DatasetConfig(
            name=f"set1_baseline_seed{s}",
            num_base_vectors=10000,
            num_cache_queries=1024,
            num_test_queries=512,
            dimension=128,
            K=100,
            N=20,
            perturbation_level=2,
            test_query_strategy="similar",
            seed=s
        ))
    return configs

#-------------------------------------------------------------------------------

def create_test_set_2_perturbation(seed: int = 42) -> List[DatasetConfig]:
    # Test Set 2: Varying perturbation levels for similar queries.
    configs = []
    for s in [seed, seed + 1, seed + 2]:
        for level, name in [(2, "small"), (1, "medium"), (0, "large")]:
            configs.append(DatasetConfig(
                name=f"set2_pert_{name}_seed{s}",
                num_base_vectors=10000,
                num_cache_queries=1024,
                num_test_queries=128,
                dimension=128,
                K=100,
                N=20,
                perturbation_level=level,
                test_query_strategy="similar",
                seed=s
            ))
    return configs

#-------------------------------------------------------------------------------

def create_test_set_3_kn_ratio(seed: int = 42) -> List[DatasetConfig]:
    # Test Set 3: Varying K/N ratios.
    configs = []
    for s in [seed, seed + 1, seed + 2]:
        for K in [10, 20, 30, 50, 100]:
            for N in [10, 20, 30, 50, 100]:
                configs.append(DatasetConfig(
                    name=f"set3_kn_K{K}_N{N}_seed{s}",
                    num_base_vectors=10000,
                    num_cache_queries=1024,
                    num_test_queries=128,
                    dimension=128,
                    K=K,
                    N=N,
                    perturbation_level=2,
                    test_query_strategy="similar",
                    seed=s
                ))
    return configs

#-------------------------------------------------------------------------------

def create_test_set_4_variable_n(seed: int = 42) -> List[DatasetConfig]:
    # Test Set 4: Variability of N
    configs = []
    for s in [seed, seed + 1, seed + 2]:
        for var_name, n_range in [("low", (30, 40)), ("medium", (25, 45)), ("high", (10, 60))]:
            configs.append(DatasetConfig(
                name=f"set4_varn_{var_name}_seed{s}",
                num_base_vectors=10000,
                num_cache_queries=1024,
                num_test_queries=128,
                dimension=128,
                K=100,
                N=0,
                N_range=n_range,
                perturbation_level=2,
                test_query_strategy="similar",
                seed=s
            ))
    return configs

#-------------------------------------------------------------------------------

def create_test_set_5_union_effectiveness(seed: int = 42) -> List[DatasetConfig]:
    # Test Set 5: Union effectiveness -- sweep K/N ratios with K=50 fixed.
    # N in {25, 35, 45, 50, 65, 75} gives K/N in {2.0, 1.43, 1.11, 1.0, 0.77, 0.67}.
    # N > K (N=65, N=75) forces union as the only path to a cache hit.
    # X in {10, 20, 50, 75} controls cluster count (number of base cache vectors).
    configs = []
    for N in [25, 35, 45, 50, 65, 75]:  # K/N ratios 2.0, 1.43, 1.11, 1.0, 0.77, 0.67
        for s in [seed, seed + 1, seed + 2]:
            for X in [10, 20, 50, 75]:
                configs.append(DatasetConfig(
                    name=f"set5_K50N{N}_X{X}_seed{s}",
                    num_base_vectors=10000,
                    num_cache_queries=1024,
                    num_test_queries=128,
                    dimension=128,
                    K=50,
                    N=N,
                    perturbation_level=2,
                    test_query_strategy="union_clustered",
                    num_clusters=X,
                    seed=s
                ))
    return configs

#-------------------------------------------------------------------------------

def create_test_set_6_cache_size(seed: int = 42) -> List[DatasetConfig]:
    # Test Set 6: Varying cache sizes with a fixed test query set.
    # For each seed, C=128 is generated first with its own test queries.
    # All other cache sizes load those same test queries from disk, so hit rate
    # depends on how many of the sampled anchors fall within the cache for each C.
    base_data_dir = Path("datasets/synthetic/data")
    configs = []
    for s in [seed, seed + 1, seed + 2]:
        ref_name = f"set6_cache_128_seed{s}"
        ref_test_path = str(base_data_dir / ref_name / "test_queries.npy")
        # C=128 first so its test_queries.npy exists before other configs load it
        configs.append(DatasetConfig(
            name=ref_name,
            num_base_vectors=10000,
            num_cache_queries=128,
            num_test_queries=128,
            dimension=128,
            K=100,
            N=20,
            perturbation_level=2,
            test_query_strategy="similar",
            seed=s,
            test_query_seed=s,
        ))
        for cache_size in [4, 8, 16, 32, 64, 256, 512]:
            configs.append(DatasetConfig(
                name=f"set6_cache_{cache_size}_seed{s}",
                num_base_vectors=10000,
                num_cache_queries=cache_size,
                num_test_queries=128,
                dimension=128,
                K=100,
                N=20,
                perturbation_level=2,
                test_query_strategy="similar",
                seed=s,
                test_query_seed=s,
                shared_test_queries_path=ref_test_path,
            ))
    return configs

#-------------------------------------------------------------------------------

def generate_all_test_sets(output_dir: str = "datasets/synthetic/data"):
    all_configs = []
    all_configs.extend(create_test_set_1_baseline())
    all_configs.extend(create_test_set_2_perturbation())
    all_configs.extend(create_test_set_3_kn_ratio())
    all_configs.extend(create_test_set_4_variable_n())
    all_configs.extend(create_test_set_5_union_effectiveness())
    all_configs.extend(create_test_set_6_cache_size())

    print(f"Generating {len(all_configs)} datasets...")
    for i, config in enumerate(all_configs, 1):
        print(f"[{i}/{len(all_configs)}] {config.name}")
        generator = SyntheticDatasetGenerator(config)
        generator.generate()
        generator.save(output_dir)

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-set', choices=[
        'set1',
        'set2',
        'set3',
        'set4',
        'set5',
        'set6',
        'all'
    ], default='all')
    args = parser.parse_args()

    test_set = args.test_set

    if test_set == 'all':
        generate_all_test_sets()
    else:
        if test_set == 'set1':
            configs = create_test_set_1_baseline()
        elif test_set == 'set2':
            configs = create_test_set_2_perturbation()
        elif test_set == 'set3':
            configs = create_test_set_3_kn_ratio()
        elif test_set == 'set4':
            configs = create_test_set_4_variable_n()
        elif test_set == 'set5':
            configs = create_test_set_5_union_effectiveness()
        elif test_set == 'set6':
            configs = create_test_set_6_cache_size()

        print(f"Generating {len(configs)} datasets for {test_set}...")
        for i, config in enumerate(configs, 1):
            print(f"[{i}/{len(configs)}] {config.name}")
            generator = SyntheticDatasetGenerator(config)
            generator.generate()
            generator.save("datasets/synthetic/data")