"""Tests for the fvecs reader."""

import numpy as np

from utils.base.io import load_fvecs_fast


def _write_fvecs(path, vectors):
    n, d = vectors.shape
    rows = np.concatenate([np.full((n, 1), d, dtype=np.int32), vectors.astype(np.float32).view(np.int32)], axis=1)
    rows.astype(np.int32).tofile(path)


def test_load_fvecs_fast_reads_all_or_a_prefix(tmp_path):
    vectors = np.random.default_rng(0).normal(size=(20, 5)).astype(np.float32)
    _write_fvecs(tmp_path / "x.fvecs", vectors)
    assert np.array_equal(load_fvecs_fast(str(tmp_path / "x.fvecs")), vectors)
    assert np.array_equal(load_fvecs_fast(str(tmp_path / "x.fvecs"), max_n=7), vectors[:7])
