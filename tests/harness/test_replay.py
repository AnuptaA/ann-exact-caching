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
    # strict equality: served by an entry stored under this exact id; others are approximate
    n_hits = (header["equality_hits_in_window"] + header["approximate_filter_hits_in_window"]
              + header["linked_hits_in_window"])
    assert n_hits == round(header["summary"]["hit_rate"] * header["summary"]["requests_measured"])
    assert header["equality_hits_in_window"] <= header["seen_before_hits_in_window"]
    assert 0 < header["peak_pool_vectors_sampled"] <= header["settings"]["pool_size"]


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


@needs_driver
def test_cli_run_resumes_and_shards(dataset, tmp_path, capsys):
    from harness import cli
    base_args = ["--datasets", "toy", "--streams", "uniform", "--shard", "0/6"]
    cli.main(["grid", *base_args])
    listed = capsys.readouterr().out.split()
    assert len(listed) >= 1
    run_args = ["run", *base_args, "--data-dir", str(dataset), "--out-dir", str(tmp_path / "r"),
                "--work-dir", str(tmp_path / "w"), "--driver-dir", str(DRIVERS)]
    cli.main(run_args)
    done = sorted(p.name for p in (tmp_path / "r").glob("*.npz"))
    assert len(done) == len(listed)
    stamp = [p.stat().st_mtime_ns for p in (tmp_path / "r").glob("*.npz")]
    cli.main(run_args)                                      # resume: nothing is rerun
    assert stamp == [p.stat().st_mtime_ns for p in (tmp_path / "r").glob("*.npz")]


def test_peak_pool_parsing():
    log = ("100 ANNSCache Status\n  Vector pool: VectorSlotPool Status: 10 inserted out of total 40, "
           "observed elements: 10\n200 x\n  Vector pool: VectorSlotPool Status: 37 inserted out of total 40, "
           "observed elements: 37\n")
    assert replay.peak_pool_vectors(log) == 37 and replay.peak_pool_vectors("") == 0


@needs_driver
def test_proximity_thresholds_are_cached_and_budgets_share_them(dataset, tmp_path):
    ds = replay.DatasetDir(dataset, "toy")
    work = tmp_path / "w"
    for budget in (1, 5):
        cell = manifest.Cell("toy", "zipf_0.99", "proximity", "p1", budget)
        header, _ = results.read_result(replay.run_cell(cell, ds, DRIVERS, tmp_path / "r", work))
        assert header["settings"]["global_thresh"] == header["thresholds"]["history"]["1.0"]
    cached = list((work / "thresholds").glob("toy-zipf_0.99-s0.json"))
    assert len(cached) == 1
    info = header["thresholds"]
    assert info["universe_wide"]["1.0"] > 0 and info["distinct_history_queries"] > 1


@needs_driver
def test_collect_table_and_csv(dataset, tmp_path):
    from harness import collect
    ds = replay.DatasetDir(dataset, "toy")
    for cell in (manifest.Cell("toy", "uniform", "aker", "paper", 10),
                 manifest.Cell("toy", "zipf_0.99", "proximity", "p0.1", 5)):
        replay.run_cell(cell, ds, DRIVERS, tmp_path / "r", tmp_path / "w")
    table = collect.rows(tmp_path / "r")
    assert len(table) == 2 and {r["system"] for r in table} == {"aker", "proximity"}
    assert collect.write_csv(tmp_path / "r", tmp_path / "out.csv") == 2
    assert "hit_rate" in (tmp_path / "out.csv").read_text().splitlines()[0]
    assert "uniform" in collect.format_table(table)


def test_missing_dataset_files_give_actionable_errors(tmp_path):
    (tmp_path / "nope").mkdir()
    with pytest.raises(FileNotFoundError, match=r"make q1-generate DATASET=nope"):
        replay.DatasetDir(tmp_path, "nope")


def test_driver_crash_and_timeout_report_the_log_tail(tmp_path):
    crash = tmp_path / "crash.sh"
    crash.write_text("#!/bin/sh\necho 'about to fail: bad input' >&2\nexit 3\n")
    crash.chmod(0o755)
    with pytest.raises(RuntimeError, match="(?s)exited with code 3.*about to fail"):
        replay.run_driver(crash, [], tmp_path, None)
    slow = tmp_path / "slow.sh"
    slow.write_text("#!/bin/sh\necho started\nsleep 5\n")
    slow.chmod(0o755)
    with pytest.raises(RuntimeError, match="(?s)timed out after 0.5s.*started"):
        replay.run_driver(slow, [], tmp_path, 0.5)


def test_is_done_ignores_results_from_other_commits(tmp_path):
    header = {"cell": {}, "config_hash": "h", "commit": "aaaa1111", "backend": "b", "ef_search": 1,
              "aker_revision": "r"}
    results.write_result(tmp_path, "c1", header, x=np.zeros(1))
    assert results.is_done(tmp_path, "c1") and results.is_done(tmp_path, "c1", "aaaa1111")
    assert not results.is_done(tmp_path, "c1", "bbbb2222") and not results.is_done(tmp_path, "c2")


@needs_driver
def test_stream_inputs_are_built_once_and_shared_by_configs(dataset, tmp_path):
    ds = replay.DatasetDir(dataset, "toy")
    work = tmp_path / "w"
    for cell in (manifest.Cell("toy", "uniform", "aker", "paper", 10),
                 manifest.Cell("toy", "uniform", "proximity", "p1", 5)):
        replay.run_cell(cell, ds, DRIVERS, tmp_path / "r", work)
    shared = list((work / "inputs").iterdir())
    assert [p.name for p in shared] == ["toy-uniform-s0"] and (shared[0] / "READY").exists()
    assert not list((work / "inputs").glob("*.tmp"))
    with pytest.raises(FileNotFoundError, match="build the drivers"):
        replay.run_cell(manifest.Cell("toy", "uniform", "aker", "paper", 10), ds, tmp_path / "nodrivers",
                        tmp_path / "r2", work)
