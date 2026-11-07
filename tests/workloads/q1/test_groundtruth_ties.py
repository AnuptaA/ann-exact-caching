"""Ground truth with heavy ties and duplicate vectors: no repeated or missing ids, exact distances."""

import numpy as np
import pytest

from workloads.q1 import groundtruth


def tied_base(copies: int = 300, points: int = 4, dim: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.repeat(rng.integers(-3, 4, size=(points, dim)).astype(np.float32), copies, axis=0)


@pytest.mark.parametrize("base_chunk", [37, 128, 10_000])
def test_top_k_has_no_duplicate_or_missing_ids_under_heavy_ties(base_chunk):
    base = tied_base()
    queries = base[::300] + 0.25
    ids, dist = groundtruth.exact_topk_chunk(queries, base, k=100, base_chunk=base_chunk)
    for row_ids, row_dist in zip(ids, dist):
        assert len(set(row_ids.tolist())) == 100 and row_ids.min() >= 0
        assert (np.diff(row_dist) >= 0).all()
    exact = np.sort(np.linalg.norm(base[None] - queries[:, None], axis=2), axis=1)[:, :100]
    assert np.allclose(dist, exact, atol=1e-4)


def test_result_is_independent_of_the_base_chunking_when_ties_fit_the_margin():
    base = tied_base(copies=20)
    queries = base[::20] + 0.1
    first = groundtruth.exact_topk_chunk(queries, base, k=10, base_chunk=17)
    second = groundtruth.exact_topk_chunk(queries, base, k=10, base_chunk=1000)
    assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])
