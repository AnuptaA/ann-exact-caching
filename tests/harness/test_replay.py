"""End to end on tiny fake data: prepare, replay through the real Aker drivers, score.

Skipped when the drivers are not built (build/replay/ak_replay_*).
"""

import os
from pathlib import Path

import numpy as np
import pytest

from harness import config
from harness.runner import manifest, replay, results
from harness.runner.prepare import prepare_dataset

DRIVERS = Path(os.environ.get("HARNESS_REPLAY_DIR", Path(__file__).resolve().parents[2] / "build" / "replay"))
needs_driver = pytest.mark.skipif(
    not (DRIVERS / "ak_replay_standard").exists() or not (DRIVERS / "ak_replay_proximity").exists(),
    reason="replay drivers not built")


class ExactBackend:
    name = "exact-fake"

    def __init__(self, base):
        self.base = base

    def search(self, queries, list_size, ef_search):
        d = ((queries[:, None, :] - self.base[None]) ** 2).sum(2)
        return np.argsort(d, axis=1, kind="stable")[:, :list_size]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(12, 8)).astype(np.float32) * 4
    base = (centers[rng.integers(0, 12, 3000)] + rng.normal(size=(3000, 8)).astype(np.float32) * 0.5)
    official = (centers[rng.integers(0, 12, 60)] + rng.normal(size=(60, 8)).astype(np.float32) * 0.5)
    root = tmp_path_factory.mktemp("q1")
    prepare_dataset(root, "toy", base, official, ExactBackend(base), ef_search=50, list_size=20,
                    n_groups=30, group_size=4, n_requests=600, zipf=(0.99,), no_locality=True)
    return root


def test_prepare_wrote_the_expected_layout(dataset):
    ds = replay.DatasetDir(dataset, "toy")
    assert ds.answer_ids.shape[1] == 20 and len(ds.gt_ids) == ds.universe.n_ids
    assert set(ds.streams) == {"zipf_0.99", "uniform", "no_locality"}
    assert ds.backend["backend"] == "exact-fake" and ds.history_fraction == 0.3


@needs_driver
@pytest.mark.parametrize("cell", [
    manifest.Cell("toy", "zipf_0.99", "aker", "paper", 10),
    manifest.Cell("toy", "uniform", "proximity", "p1", 5),
    manifest.Cell("toy", "no_locality", "aker", "paper", 1),
])
def test_cell_runs_end_to_end_and_is_resumable(dataset, tmp_path, cell):
    ds = replay.DatasetDir(dataset, "toy")
    path = replay.run_cell(cell, ds, DRIVERS, tmp_path / "results", tmp_path / "work", timeout=120)
    header, arrays = results.read_result(path)
    s = header["summary"]
    assert results.is_done(tmp_path / "results", cell.cell_id)
    assert 0.0 <= s["hit_rate"] <= 1.0 and 0.0 <= s["delivered_recall"] <= 1.0
    assert len(arrays["recall"]) == len(ds.streams[cell.stream])
    assert header["aker_revision"].startswith("e7639a5") and header["ef_search"] == 50
    assert header["settings"]["mode"] == ("standard" if cell.system == "aker" else "proximity")
    assert "mode=" in header["driver_summary"]
    if cell.stream == "uniform":
        assert s["hit_rate"] > 0.3                      # 30 bases repeat constantly


@needs_driver
def test_exact_repeats_are_equality_hits_and_misses_have_backend_recall(dataset, tmp_path):
    ds = replay.DatasetDir(dataset, "toy")
    cell = manifest.Cell("toy", "uniform", "aker", "paper", 10)
    header, arrays = results.read_result(replay.run_cell(cell, ds, DRIVERS, tmp_path / "r", tmp_path / "w"))
    miss = arrays["outcome"] == 0
    assert (arrays["recall"][miss] == 1.0).all()        # exact backend: misses are perfect
    assert header["summary"]["equality_hit_rate"] > 0.5  # 600 requests over 30 base queries
    assert header["kind_mismatch"] == 0


@needs_driver
def test_budget_limits_the_cache(dataset, tmp_path):
    ds = replay.DatasetDir(dataset, "toy")
    big = replay.run_cell(manifest.Cell("toy", "zipf_0.99", "aker", "paper", 10), ds, DRIVERS,
                          tmp_path / "r", tmp_path / "w")
    assert results.read_result(big)[0]["settings"]["pool_size"] == config.pool_size_for(3000, 10)


def test_grid_ids_sharding_and_budget_rule():
    grid = manifest.build_grid(["esci"], seeds=(0,))
    ids = [c.cell_id for c in grid]
    assert len(ids) == len(set(ids)) == 5 * 3 * (1 + 2)     # streams x budgets x (aker + 2 proximity)
    shards = [manifest.shard(grid, i, 4) for i in range(4)]
    assert sum(len(s) for s in shards) == len(grid)
    assert {c.cell_id for s in shards for c in s} == set(ids)
    assert "no_locality" in manifest.default_streams("sphere") and "no_locality" not in manifest.default_streams("spacev")
    with pytest.raises(ValueError):
        manifest.shard(grid, 4, 4)
    with pytest.raises(ValueError):
        manifest.Cell("d", "s", "aker", "paper", 1).percentile()
    with pytest.raises(ValueError):
        results.write_result(Path("."), "x", {"cell": 1})
