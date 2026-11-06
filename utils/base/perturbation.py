#!/usr/bin/env python3
#-------------------------------------------------------------------------------

import numpy as np
from typing import Literal

#-------------------------------------------------------------------------------

PerturbationLevel = Literal[0, 1, 2, "small", "medium", "large"]

# thresholds in radians
ALPHA = np.pi / 36  # 5 degrees
BETA = np.pi / 6    # 30 degrees

#-------------------------------------------------------------------------------

def sample_perturbation_angle(lambda_val: Literal[0, 1, 2]) -> float:
    """
    Generate random angular perturbation based on perturbation level.

    Args:
        lambda_val: Perturbation level (0 = large, 1 = medium, 2 = small)

    Returns:
        Angular perturbation in radians.
    """
    if lambda_val == 0:  # large: (30, 180]
        return np.random.uniform(BETA, np.pi)
    elif lambda_val == 1:  # medium: (5, 30]
        return np.random.uniform(ALPHA, BETA)
    elif lambda_val == 2:  # small: [0, 5]
        return np.random.uniform(0, ALPHA)
    else:
        raise ValueError(f"Invalid perturbation level: {lambda_val}. Must be 0, 1, or 2.")

#-------------------------------------------------------------------------------

def generate_perturbed_vector(
    v: np.ndarray,
    perturbation_level: PerturbationLevel,
    seed: int = None,
    return_angle: bool = False
) -> np.ndarray:
    """
    Generate a perturbed vector v' from base vector v using Rodrigues' rotation formula.
    
    Args:
        v: Base vector (will be normalized)
        perturbation_level: 0/large, 1/medium, or 2/small
        seed: Optional random seed
        return_angle: If True, return (v', theta_deg) instead of just v'
        
    Returns:
        Perturbed vector v' (normalized), or (v', theta_deg) if return_angle=True
    """
    if seed is not None:
        np.random.seed(seed)
    
    # normalize v
    v_norm = v / np.linalg.norm(v)

    if perturbation_level == "large":
        lambda_val = 0
    elif perturbation_level == "medium":
        lambda_val = 1
    elif perturbation_level == "small":
        lambda_val = 2
    elif perturbation_level in (0, 1, 2):
        lambda_val = perturbation_level
    else:
        raise ValueError(
            f"Invalid perturbation_level: {perturbation_level!r}. "
            "Must be one of 'large'/'medium'/'small' or 0/1/2."
        )
    
    # generate angular perturbation
    theta = sample_perturbation_angle(lambda_val)
    
    # step 1: generate random direction
    u = np.random.randn(len(v_norm))
    u = u / np.linalg.norm(u)
    
    # step 2: orthogonalize to v
    v_parallel = np.dot(u, v_norm) * v_norm
    v_perp = u - v_parallel
    v_perp_norm = np.linalg.norm(v_perp)
    
    if v_perp_norm < 1e-10:
        # u and v are nearly parallel, try again
        u = np.random.randn(len(v_norm))
        u = u / np.linalg.norm(u)
        v_parallel = np.dot(u, v_norm) * v_norm
        v_perp = u - v_parallel
        v_perp_norm = np.linalg.norm(v_perp)
    
    v_perp = v_perp / v_perp_norm
    
    # step 3: rotate by theta
    v_prime = v_norm * np.cos(theta) + v_perp * np.sin(theta)
    
    # ensure normalization
    v_prime = v_prime / np.linalg.norm(v_prime)
    
    if return_angle:
        return v_prime, float(np.degrees(theta))
    return v_prime

#-------------------------------------------------------------------------------

def euclidean_distance_from_angle(theta: float) -> float:
    """
    Compute Euclidean distance from angular distance for normalized vectors.
    
    Args:
        theta: Angular distance in radians
        
    Returns:
        Euclidean distance
    """
    return np.sqrt(2 * (1 - np.cos(theta)))

#-------------------------------------------------------------------------------

def cosine_distance_from_angle(theta: float) -> float:
    """
    Compute cosine distance from angular distance for normalized vectors.
    
    Args:
        theta: Angular distance in radians
        
    Returns:
        Cosine distance
    """
    return 1 - np.cos(theta)

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------

