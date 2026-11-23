import numpy as np
import pytest

from harness.backend.answer_table import load_answer_table
from harness.backend.diskann_search import DiskannBackend
from recal.stored import StoredAnswers
from scripts import precompute_answers as pre
from workloads.q1.groundtruth import load_ground_truth


def small_world(n=400, d=6, nq=50, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(n, d)).astype(np.float32), rng.normal(size=(nq, d)).astype(np.float32))


def test_answer_depth_is_seven_times_k_max():
    assert pre.answer_depth(100) == 700 and pre.answer_depth(10) == 70


def test_exact_answer_table_reaches_depth_and_feeds_stored_answers(tmp_path):
    base, queries = small_world()
    from harness.backend.exact_table import ExactTableBackend
    depth = pre.precompute_answers(ExactTableBackend(base), queries, base, tmp_path / "a", 5, 40, block=20)
    assert depth == 35
    ids, dist = load_answer_table(tmp_path / "a")
    assert ids.shape == (50, 35) and (np.diff(dist, axis=1) >= -1e-6).all()
    stored = StoredAnswers(base, queries, ids)
    got, _ = stored.search(queries[7], 35)
    assert got.tolist() == ids[7].tolist()


def test_ground_truth_depth_200_supports_liveness_filtering(tmp_path):
    base, queries = small_world(n=500)
    pre.precompute_ground_truth(queries, base, tmp_path / "gt", device="numpy")
    ids, dist = load_ground_truth(tmp_path / "gt")
    assert ids.shape == (50, 200) and np.isfinite(dist).all()
    dead = set(ids[0, :30].tolist())
    live = [i for i in ids[0] if i not in dead][:100]
    assert len(live) == 100


def test_depth_beyond_the_pgvector_limit_is_rejected(tmp_path):
    base, queries = small_world()
    from harness.backend.exact_table import ExactTableBackend
    with pytest.raises(ValueError):
        pre.precompute_answers(ExactTableBackend(base), queries, base, tmp_path / "b", 150, 1000)


class FakeIndex:
    def __init__(self, base):
        self.base = base.astype(np.float32)

    def search(self, q, k_neighbors, complexity, beam_width):
        d = ((self.base - q.astype(np.float32)) ** 2).sum(axis=1)
        return np.argsort(d)[:k_neighbors], np.sort(d)[:k_neighbors]


def test_diskann_backend_wrapper_checks_search_list(tmp_path):
    base, queries = small_world()
    backend = DiskannBackend(FakeIndex(base), np.float32)
    ids = backend.search(queries[:3], 10, 20)
    assert ids.shape == (3, 10)
    with pytest.raises(ValueError):
        backend.search(queries[:3], 10, 5)
