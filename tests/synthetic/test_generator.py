#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import shutil
from datasets.synthetic.generator import (
    DatasetConfig,
    SyntheticDatasetGenerator,
    load_synthetic_dataset
)

#-------------------------------------------------------------------------------

SMALL_ANGLE_MAX_DEG = 5.0

#-------------------------------------------------------------------------------

def test_similar_generation():
    """Test 'similar' strategy: shapes and per-query angle metadata."""
    config = DatasetConfig(
        name="test_similar",
        num_base_vectors=100,
        num_cache_queries=20,
        num_test_queries=50,
        dimension=32,
        K=5,
        N=5,
        test_query_strategy="similar",
        perturbation_level=2,
        seed=42
    )

    gen = SyntheticDatasetGenerator(config)
    base, cache_q, test_q = gen.generate()

    # check shapes
    assert base.shape == (100, 32), f"Wrong base shape: {base.shape}"
    assert cache_q.shape == (20, 32), f"Wrong cache shape: {cache_q.shape}"
    assert test_q.shape == (50, 32), f"Wrong test shape: {test_q.shape}"

    # per-query angles must be collected
    angles = config.test_perturbation_angles_deg
    assert angles is not None, "test_perturbation_angles_deg should be populated"
    assert len(angles) == 50, f"Expected 50 angles, got {len(angles)}"

    # all angles should be within the small perturbation range [0, 5] degrees
    for i, a in enumerate(angles):
        assert 0.0 <= a <= SMALL_ANGLE_MAX_DEG + 1e-6, \
            f"Angle {i}={a:.4f}deg out of range [0, {SMALL_ANGLE_MAX_DEG}]"

#-------------------------------------------------------------------------------

def test_union_clustered_generation():
    """Test 'union_clustered' strategy: shapes and per-query angle metadata."""
    num_test = 40
    config = DatasetConfig(
        name="test_union_clustered",
        num_base_vectors=100,
        num_cache_queries=20,
        num_test_queries=num_test,
        dimension=32,
        K=5,
        N=5,
        test_query_strategy="union_clustered",
        num_clusters=5,
        perturbation_level=2,   # small perturbations
        seed=42
    )

    gen = SyntheticDatasetGenerator(config)
    base, cache_q, test_q = gen.generate()

    assert base.shape == (100, 32), f"Wrong base shape: {base.shape}"
    assert cache_q.shape == (20, 32), f"Wrong cache shape: {cache_q.shape}"
    assert test_q.shape == (num_test, 32), f"Wrong test shape: {test_q.shape}"

    angles = config.test_perturbation_angles_deg
    assert angles is not None, "test_perturbation_angles_deg should be populated"
    assert len(angles) == num_test, f"Expected {num_test} angles, got {len(angles)}"

    # all angles must be in small range
    for i, a in enumerate(angles):
        assert 0.0 <= a <= SMALL_ANGLE_MAX_DEG + 1e-6, \
            f"Angle {i}={a:.4f}deg out of range [0, {SMALL_ANGLE_MAX_DEG}]"

#-------------------------------------------------------------------------------

def test_variable_n_generation():
    """Test 'similar' strategy with N_range: test_N_values populated."""
    lo, hi = 10, 20
    num_test = 60
    config = DatasetConfig(
        name="test_variable_n",
        num_base_vectors=100,
        num_cache_queries=20,
        num_test_queries=num_test,
        dimension=32,
        K=25,
        N=0,                        # N=0 signals variable-N mode
        N_range=(lo, hi),
        test_query_strategy="similar",
        perturbation_level=2,
        seed=42
    )

    gen = SyntheticDatasetGenerator(config)
    gen.generate()

    n_vals = config.test_N_values
    assert n_vals is not None, "test_N_values should be populated for N_range datasets"
    assert len(n_vals) == num_test, f"Expected {num_test} N values, got {len(n_vals)}"

    for i, n in enumerate(n_vals):
        assert lo <= n <= hi, f"N_values[{i}]={n} out of range [{lo}, {hi}]"

    # angles should also be populated
    assert config.test_perturbation_angles_deg is not None, \
        "test_perturbation_angles_deg should also be populated in variable-N mode"
    assert len(config.test_perturbation_angles_deg) == num_test

#-------------------------------------------------------------------------------

def test_save_and_load():
    """Test save/load round-trip: arrays and config metadata survive serialization."""
    test_dir = "./tests/generation/tmp"

    config = DatasetConfig(
        name="test_save",
        num_base_vectors=50,
        num_cache_queries=10,
        num_test_queries=24,
        dimension=16,
        K=3,
        N=3,
        test_query_strategy="similar",
        perturbation_level=2,
        seed=42
    )

    gen = SyntheticDatasetGenerator(config)
    base_orig, cache_orig, test_orig = gen.generate()
    gen.save(test_dir)

    # load back
    dataset_path = Path(test_dir) / "test_save"
    base_loaded, cache_loaded, test_loaded, cfg_loaded = load_synthetic_dataset(str(dataset_path))

    # shapes match
    assert base_loaded.shape == base_orig.shape, "Base vectors shape mismatch"
    assert cache_loaded.shape == cache_orig.shape, "Cache queries shape mismatch"
    assert test_loaded.shape == test_orig.shape, "Test queries shape mismatch"

    # values match
    assert np.allclose(base_loaded, base_orig), "Base vector values mismatch"
    assert np.allclose(cache_loaded, cache_orig), "Cache query values mismatch"
    assert np.allclose(test_loaded, test_orig), "Test query values mismatch"

    # per-query angles survive JSON round-trip
    assert "test_perturbation_angles_deg" in cfg_loaded, \
        "test_perturbation_angles_deg missing from saved config"
    saved_angles = cfg_loaded["test_perturbation_angles_deg"]
    assert saved_angles is not None, "Saved angles should not be None"
    assert len(saved_angles) == 24, f"Expected 24 saved angles, got {len(saved_angles)}"
    assert np.allclose(saved_angles, config.test_perturbation_angles_deg, atol=1e-6), \
        "Saved angles don't match original"

    # cleanup
    shutil.rmtree(test_dir)

#-------------------------------------------------------------------------------

def test_reproducibility():
    """Test that the same seed produces identical outputs."""
    def make(seed):
        config = DatasetConfig(
            name="test_repro",
            num_base_vectors=80,
            num_cache_queries=16,
            num_test_queries=30,
            dimension=32,
            K=5,
            N=5,
            test_query_strategy="similar",
            perturbation_level=2,
            seed=seed
        )
        gen = SyntheticDatasetGenerator(config)
        base, cache_q, test_q = gen.generate()
        angles = list(config.test_perturbation_angles_deg)
        return base, cache_q, test_q, angles

    base1, cache1, test1, angles1 = make(42)
    base2, cache2, test2, angles2 = make(42)
    base3, cache3, test3, angles3 = make(99)   # different seed

    assert np.array_equal(base1, base2), "Same seed should give same base vectors"
    assert np.array_equal(cache1, cache2), "Same seed should give same cache queries"
    assert np.array_equal(test1, test2), "Same seed should give same test queries"
    assert np.allclose(angles1, angles2), "Same seed should give same angles"

    # different seed should (very likely) give different test queries
    assert not np.array_equal(test1, test3), \
        "Different seeds should give different test queries"

#-------------------------------------------------------------------------------