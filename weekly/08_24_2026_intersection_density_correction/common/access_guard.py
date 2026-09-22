#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Runtime guard for the query-time corpus-access constraint: at query time,
only the incoming query vector, a cache entry's own stored point IDs (and
their coordinates via the shared pool), and precomputed global summaries
(LSH codes, LUTs) may be touched - no other corpus coordinate may be read.

This has always been enforced by convention in this project (functions
simply never receiving corpus-wide arrays), never checked at runtime. That
gap is exactly what let the median-halving-tree/LSH-HT line get invalidated
undetected (see ../POSTMORTEM.md). GuardedCoordinatePool makes the boundary
an enforced runtime check for the new, unreviewed estimators (M3, M4) in
estimator_comparison_run_experiment.py."""

import numpy as np

#-------------------------------------------------------------------------------


class AccessViolation(Exception):
    """Raised when a query-time read touches a corpus coordinate outside the
    currently-permitted ID set."""


class GuardedCoordinatePool:
    """Wraps a shared (N, D) coordinate array. At construction time (or
    whenever unrestricted), any index is allowed. Inside `restrict_to`, only
    indices within the given ID set may be read; anything else raises
    AccessViolation. Global-summary arrays (LSH codes, LUTs) are not
    coordinates and are never routed through this class - they are exempt by
    construction, not by a bypass flag."""

    def __init__(self, coords: np.ndarray):
        self._coords = coords
        self._allowed_ids = None  # None means unrestricted (construction-time use)

    def restrict_to(self, allowed_ids: np.ndarray) -> "_RestrictionScope":
        return _RestrictionScope(self, np.asarray(allowed_ids))

    def __getitem__(self, idx):
        idx_arr = np.atleast_1d(np.asarray(idx))
        if self._allowed_ids is not None:
            if not np.isin(idx_arr, self._allowed_ids).all():
                bad = idx_arr[~np.isin(idx_arr, self._allowed_ids)]
                raise AccessViolation(
                    f"query-time read of corpus id(s) {bad.tolist()} outside the "
                    f"permitted set (size {len(self._allowed_ids)})"
                )
        return self._coords[idx]

    @property
    def full(self) -> np.ndarray:
        """Unrestricted access - construction-time use only. Never call this
        from a query-time code path."""
        return self._coords


class _RestrictionScope:
    def __init__(self, pool: GuardedCoordinatePool, allowed_ids: np.ndarray):
        self._pool = pool
        self._allowed_ids = allowed_ids
        self._previous = None

    def __enter__(self) -> GuardedCoordinatePool:
        self._previous = self._pool._allowed_ids
        self._pool._allowed_ids = self._allowed_ids
        return self._pool

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._pool._allowed_ids = self._previous
