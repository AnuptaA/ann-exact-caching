#!/usr/bin/env python3
#-------------------------------------------------------------------------------

from .distance_metrics import (
    DistanceMetric,
    euclidean_distance,
    cosine_distance,
    angular_distance,
    get_distance_function
)
from .main_memory import MainMemory, range_search
from .kv_cache import KVCache, CacheEntry
from .lemmas import (
    lemma1_circular_inclusion,
    lemma2_half_gap,
    combined_algorithm,
    binary_search_last_index,
    lemma1_no_union,
    lemma2_no_union
)
from .perturbation import (
    generate_perturbed_vector,
    sample_perturbation_angle,
    euclidean_distance_from_angle,
    cosine_distance_from_angle,
    verify_perturbation,
    PerturbationLevel,
    ALPHA,
    BETA
)

__all__ = [
    'DistanceMetric',
    'euclidean_distance',
    'cosine_distance',
    'angular_distance',
    'get_distance_function',
    'MainMemory',
    'range_search',
    'KVCache',
    'CacheEntry',
    'lemma1_circular_inclusion',
    'lemma2_half_gap',
    'combined_algorithm',
    'binary_search_last_index',
    'lemma1_no_union',
    'lemma2_no_union',
    'generate_perturbed_vector',
    'sample_perturbation_angle',
    'euclidean_distance_from_angle',
    'cosine_distance_from_angle',
    'verify_perturbation',
    'PerturbationLevel',
    'ALPHA',
    'BETA',
]