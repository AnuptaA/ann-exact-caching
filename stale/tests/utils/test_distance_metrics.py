#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from utils.base import (
    euclidean_distance,
    cosine_distance,
    angular_distance,
    get_distance_function
)

#-------------------------------------------------------------------------------

def test_euclidean_distance():
    """Test Euclidean distance calculation."""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])

    dist = euclidean_distance(a, b)
    expected = np.sqrt(2.0)

    assert np.isclose(dist, expected), f"Expected {expected}, got {dist}"

    dist_same = euclidean_distance(a, a)
    assert np.isclose(dist_same, 0.0), f"Same vector should have 0 distance, got {dist_same}"

#-------------------------------------------------------------------------------

def test_cosine_distance():
    """Test cosine distance calculation."""
    # parallel vectors should have distance 0
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([2.0, 0.0, 0.0])

    dist = cosine_distance(a, b)
    assert np.isclose(dist, 0.0), f"Parallel vectors should have 0 distance, got {dist}"

    # orthogonal vectors should have distance 1
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])

    dist = cosine_distance(a, b)
    assert np.isclose(dist, 1.0), f"Orthogonal vectors should have distance 1, got {dist}"

    # opposite vectors should have distance 2
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([-1.0, 0.0, 0.0])

    dist = cosine_distance(a, b)
    assert np.isclose(dist, 2.0), f"Opposite vectors should have distance 2, got {dist}"

#-------------------------------------------------------------------------------

def test_angular_distance():
    """Test angular distance calculation."""
    # parallel vectors should have distance 0
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([2.0, 0.0, 0.0])

    dist = angular_distance(a, b)
    assert np.isclose(dist, 0.0), f"Parallel vectors should have 0 angular distance, got {dist}"

    # orthogonal vectors should have distance 0.5
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])

    dist = angular_distance(a, b)
    expected = 0.5
    assert np.isclose(dist, expected), f"Orthogonal vectors should have distance pi/2, got {dist}"

    # opposite vectors should have distance 1.0
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([-1.0, 0.0, 0.0])

    dist = angular_distance(a, b)
    expected = 1.0
    assert np.isclose(dist, expected), f"Opposite vectors should have distance 1.0, got {dist}"

#-------------------------------------------------------------------------------

def test_get_distance_function():
    """Test distance function factory."""
    euclidean_func = get_distance_function("euclidean")
    cosine_func = get_distance_function("cosine")
    angular_func = get_distance_function("angular")

    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])

    # check they run without error
    _ = euclidean_func(a, b)
    _ = cosine_func(a, b)
    _ = angular_func(a, b)

    try:
        get_distance_function("invalid_metric")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

#-------------------------------------------------------------------------------

def test_triangle_inequality():
    """Test that Euclidean and Angular satisfy triangle inequality."""
    # generate random vectors
    np.random.seed(42)
    a = np.random.randn(10)
    b = np.random.randn(10)
    c = np.random.randn(10)

    # test Euclidean
    d_ab = euclidean_distance(a, b)
    d_bc = euclidean_distance(b, c)
    d_ac = euclidean_distance(a, c)

    assert d_ac <= d_ab + d_bc + 1e-10, "Euclidean violates triangle inequality"

    # test Angular
    d_ab = angular_distance(a, b)
    d_bc = angular_distance(b, c)
    d_ac = angular_distance(a, c)

    assert d_ac <= d_ab + d_bc + 1e-10, "Angular violates triangle inequality"

#-------------------------------------------------------------------------------
