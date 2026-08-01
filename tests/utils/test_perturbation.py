#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from utils.base.perturbation import (
    sample_perturbation_angle,
    generate_perturbed_vector,
    euclidean_distance_from_angle,
    cosine_distance_from_angle,
    verify_perturbation,
    ALPHA,
    BETA
)

#-------------------------------------------------------------------------------

def test_sample_perturbation_angle():
    """Test angular perturbation sampling function."""
    print("Test 1: sample_perturbation_angle ranges")
    
    np.random.seed(42)
    
    # test large perturbation (lambda=0)
    angles_large = [sample_perturbation_angle(0) for _ in range(100)]
    assert all(BETA <= a <= np.pi for a in angles_large), "Large angles should be in [30, 180]"

    # test medium perturbation (lambda=1)
    angles_medium = [sample_perturbation_angle(1) for _ in range(100)]
    assert all(ALPHA <= a <= BETA for a in angles_medium), "Medium angles should be in [5, 30]"

    # test small perturbation (lambda=2)
    angles_small = [sample_perturbation_angle(2) for _ in range(100)]
    assert all(0 <= a <= ALPHA for a in angles_small), "Small angles should be in [0, 5]"
    
    print("sample_perturbation_angle works")

#-------------------------------------------------------------------------------

def test_generate_perturbed_vector():
    """Test vector perturbation generation."""
    print("\nTest 2: Generate perturbed vector")
    
    np.random.seed(42)
    v = np.random.randn(128)
    v_norm = v / np.linalg.norm(v)
    
    # test with small perturbation
    v_prime = generate_perturbed_vector(v, perturbation_level="small", seed=42)
    
    assert v_prime.shape == v.shape, "Shape should match"
    assert np.abs(np.linalg.norm(v_prime) - 1.0) < 1e-6, "Should be normalized"
    
    cos_theta = np.dot(v_norm, v_prime)
    actual_angle = np.arccos(np.clip(cos_theta, -1, 1))
    
    assert 0 <= actual_angle <= ALPHA, f"Small perturbation angle {actual_angle} should be <= {ALPHA}"
    
    print("Perturbed vector generation works.")

#-------------------------------------------------------------------------------

def test_perturbation_levels():
    """Test all three perturbation levels."""
    print("\nTest 3: All perturbation levels")
    
    v = np.random.randn(64)
    v_norm = v / np.linalg.norm(v)
    
    for level_name, level_val in [("small", 2), ("medium", 1), ("large", 0)]:
        v_prime = generate_perturbed_vector(v, perturbation_level=level_val, seed=42)
        
        cos_theta = np.dot(v_norm, v_prime / np.linalg.norm(v_prime))
        angle = np.arccos(np.clip(cos_theta, -1, 1))
        
        if level_name == "small":
            assert 0 <= angle <= ALPHA, f"Small: {angle} not in [0, {ALPHA}]"
        elif level_name == "medium":
            assert ALPHA <= angle <= BETA, f"Medium: {angle} not in [{ALPHA}, {BETA}]"
        else:  # large
            assert BETA <= angle <= np.pi, f"Large: {angle} not in [{BETA}, {np.pi}]"
    
    print("All perturbation levels work.")

#-------------------------------------------------------------------------------

def test_distance_conversions():
    """Test distance conversion functions."""
    print("\nTest 4: Distance conversions")
    
    # test at 90 degrees
    theta_90 = np.pi / 2
    euc_dist = euclidean_distance_from_angle(theta_90)
    cos_dist = cosine_distance_from_angle(theta_90)
    
    # at 90 degrees: cos(90) = 0
    # euclidean: sqrt(2*(1-0)) = sqrt(2) ~= 1.414
    # cosine: 1 - 0 = 1.0
    assert np.isclose(euc_dist, np.sqrt(2), atol=1e-6), f"Expected {np.sqrt(2)}, got {euc_dist}"
    assert np.isclose(cos_dist, 1.0, atol=1e-6), f"Expected 1.0, got {cos_dist}"
    
    # test at 0 degrees
    theta_0 = 0.0
    euc_dist_0 = euclidean_distance_from_angle(theta_0)
    cos_dist_0 = cosine_distance_from_angle(theta_0)
    
    assert np.isclose(euc_dist_0, 0.0, atol=1e-6), f"Expected 0, got {euc_dist_0}"
    assert np.isclose(cos_dist_0, 0.0, atol=1e-6), f"Expected 0, got {cos_dist_0}"
    
    print("Distance conversions work.")

#-------------------------------------------------------------------------------

def test_verify_perturbation():
    """Test perturbation verification."""
    print("\nTest 5: Verify perturbation")
    
    v = np.random.randn(32)
    expected_angle = np.pi / 4
    
    v_prime = generate_perturbed_vector(v, perturbation_level="medium", seed=42)
    result = verify_perturbation(v, v_prime, expected_angle, tolerance=np.pi)
    
    assert 'actual_angle_deg' in result
    assert 'actual_euclidean' in result
    assert 'actual_cosine' in result
    
    print("Perturbation verification works.")

#-------------------------------------------------------------------------------

def test_reproducibility():
    """Test that perturbation is reproducible with seed."""
    print("\nTest 6: Reproducibility with seed")
    
    v = np.random.randn(64)
    
    v1 = generate_perturbed_vector(v, perturbation_level="medium", seed=123)
    v2 = generate_perturbed_vector(v, perturbation_level="medium", seed=123)
    
    assert np.allclose(v1, v2), "Same seed should produce same perturbation"
    
    v3 = generate_perturbed_vector(v, perturbation_level="medium", seed=456)
    assert not np.allclose(v1, v3), "Different seed should produce different perturbation"
    
    print("Reproducibility works.")

#-------------------------------------------------------------------------------

def test_return_angle():
    """Test that return_angle=True returns (v', angle_deg) with correct angle range."""
    print("\nTest 7: return_angle parameter")

    ALPHA_DEG = np.degrees(ALPHA)   # 5.0
    BETA_DEG = np.degrees(BETA)    # 30.0

    np.random.seed(42)
    v = np.random.randn(64)

    for level, lo, hi in [(2, 0.0, ALPHA_DEG), (1, ALPHA_DEG, BETA_DEG), (0, BETA_DEG, 180.0)]:
        result = generate_perturbed_vector(v, perturbation_level=level, return_angle=True)

        # must return a 2-tuple
        assert isinstance(result, tuple) and len(result) == 2, \
            f"level {level}: expected (v', float), got {type(result)}"

        v_prime, angle_deg = result

        # v' is an ndarray matching the original shape
        assert isinstance(v_prime, np.ndarray), "v' should be ndarray"
        assert v_prime.shape == v.shape, f"shape mismatch: {v_prime.shape} vs {v.shape}"

        # angle_deg is a plain float
        assert isinstance(angle_deg, float), f"angle should be float, got {type(angle_deg)}"

        # angle is within the expected range for this level
        assert lo - 1e-6 <= angle_deg <= hi + 1e-6, \
            f"level {level}: angle {angle_deg:.4f}deg not in [{lo}, {hi}]"

        # verify the geometric angle between v and v' matches angle_deg
        v_norm = v / np.linalg.norm(v)
        vp_norm = v_prime / np.linalg.norm(v_prime)
        cos_theta = np.clip(np.dot(v_norm, vp_norm), -1.0, 1.0)
        geometric_deg = np.degrees(np.arccos(cos_theta))

        assert abs(geometric_deg - angle_deg) < 1e-4, \
            f"level {level}: returned angle {angle_deg:.6f}deg != geometric angle {geometric_deg:.6f}deg"

    # return_angle=False (default) must still return a plain ndarray
    v_plain = generate_perturbed_vector(v, perturbation_level=2, return_angle=False)
    assert isinstance(v_plain, np.ndarray), "return_angle=False should return ndarray"

    print("return_angle works.")

#-------------------------------------------------------------------------------

def test_distance_from_angle_matches_direct_distance_metrics():
    """euclidean_distance_from_angle/cosine_distance_from_angle (trig formulas
    on a known angle) must agree with utils.base.distance_metrics' direct
    vector-based computation on an actually-generated pair of vectors at that
    angle -- two independent implementations of the same identity."""
    print("\nTest 8: distance-from-angle matches direct distance_metrics")

    from utils.base.distance_metrics import euclidean_distance, cosine_distance

    np.random.seed(99)
    v = np.random.randn(32)

    for level in (0, 1, 2):
        v_prime, angle_deg = generate_perturbed_vector(v, perturbation_level=level, return_angle=True)
        theta = np.radians(angle_deg)

        v_hat = v / np.linalg.norm(v)
        vp_hat = v_prime / np.linalg.norm(v_prime)

        expected_euclidean = euclidean_distance_from_angle(theta)
        actual_euclidean = euclidean_distance(v_hat, vp_hat)
        assert abs(expected_euclidean - actual_euclidean) < 1e-4, (
            f"level {level}: euclidean_distance_from_angle={expected_euclidean:.6f} "
            f"vs direct euclidean_distance={actual_euclidean:.6f}"
        )

        expected_cosine = cosine_distance_from_angle(theta)
        actual_cosine = cosine_distance(v_hat, vp_hat)
        assert abs(expected_cosine - actual_cosine) < 1e-4, (
            f"level {level}: cosine_distance_from_angle={expected_cosine:.6f} "
            f"vs direct cosine_distance={actual_cosine:.6f}"
        )

    print("distance-from-angle formulas agree with direct distance_metrics computation.")

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    print("="*80)
    print("Testing Perturbation Utilities")
    print("="*80)
    
    test_sample_perturbation_angle()
    test_generate_perturbed_vector()
    test_perturbation_levels()
    test_distance_conversions()
    test_verify_perturbation()
    test_reproducibility()
    test_return_angle()
    test_distance_from_angle_matches_direct_distance_metrics()

    print("\n" + "="*80)
    print("All tests passed.")
    print("="*80)

#-------------------------------------------------------------------------------
