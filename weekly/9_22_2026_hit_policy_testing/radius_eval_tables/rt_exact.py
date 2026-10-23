#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Exact-rows pass: theta and distance covariates for query vectors against a database."""

import numpy as np
import pandas as pd

import rt_config
from utils.base.distance_metrics import pairwise_angular_distance

COVARIATE_RANKS = (1, 10, 50)  # nearest, 10th and 50th nearest database vector

def exact_rows_chunked(queries: np.ndarray, database: np.ndarray, chunk_size: int = rt_config.CHUNK_SIZE):
    """Yields (start, end, rows) with rows the full (chunk, N) angular distance matrix."""
    for s in range(0, len(queries), chunk_size):
        e = min(len(queries), s + chunk_size)
        yield s, e, pairwise_angular_distance(queries[s:e], database)

def query_covariates(queries: np.ndarray, database: np.ndarray, K: int = rt_config.K_TOPK,
                     chunk_size: int = rt_config.CHUNK_SIZE) -> pd.DataFrame:
    """theta (K-th nearest), nn1, d10, d50 per query."""
    ranks = [r - 1 for r in COVARIATE_RANKS] + [K - 1]
    out = np.empty((len(queries), len(ranks)))
    for s, e, rows in exact_rows_chunked(queries, database, chunk_size):
        out[s:e] = np.partition(rows, ranks, axis=1)[:, ranks]
    return pd.DataFrame(dict(theta=out[:, 3], nn1=out[:, 0], d10=out[:, 1], d50=out[:, 2]))
