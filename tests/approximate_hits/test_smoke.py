#!/usr/bin/env python3
"""Wiring smoke test: every driver runs end to end on tiny fake inputs.

Fake ESCI vectors, SIFT fvecs files and an Aker trace are written to tmp
directories; the test checks that paths, argument names and file hand-offs
work for all three workloads, not that the numbers are right.
"""

import json

import numpy as np
import pytest

from simulations.approximate_hits import cli
from utils.cache import workloads


def evaluate(name, run_dir, *argv):
    cli.main(["eval", name, str(run_dir), *map(str, argv)])


def populate(*argv):
    cli.main(["populate", *map(str, argv)])


def _write_fvecs(path, arr):
    arr = np.asarray(arr, dtype=np.float32)
    rows = np.hstack([np.full((len(arr), 1), arr.shape[1], dtype=np.int32).view(np.float32), arr])
    path.write_bytes(rows.tobytes())


@pytest.fixture(scope="module")
def esci_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("esci")
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((6, 8)) * 2
    np.save(root / "esci_db2000_s42.npy",
            (centers[rng.integers(0, 6, 2000)] + rng.standard_normal((2000, 8))).astype(np.float32))
    np.save(root / "esci_queries.npy",
            (centers[rng.integers(0, 6, 1500)] + rng.standard_normal((1500, 8))).astype(np.float32))
    out = root / "run"
    populate("esci", "--esci-dir", root, "--n-db", 2000, "--n-cand-queries", 120,
             "--n-score-queries", 120, "--n-test", 100, "--k", 5, "--k-values", 5, 50, 500,
             "--targets", 0.7, 0.9, "--checkpoints", 0.05, 0.1, "--max-frac", 0.2,
             "--workers", 1, "--out", out)
    return out


@pytest.fixture(scope="module")
def synthetic_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("sift")
    rng = np.random.default_rng(1)
    for name, n in (("sift_base", 400), ("sift_learn", 200), ("sift_query", 60)):
        _write_fvecs(root / f"{name}.fvecs", rng.standard_normal((n, 8)))
    old = workloads.PATHS.sift
    workloads.PATHS.sift = root
    out = root / "run"
    try:
        populate("synthetic", "--n-db", 300, "--group-size", 5, "--n-cand-requests", 60,
                 "--n-score-requests", 120, "--n-test-requests", 120, "--k", 3,
                 "--k-values", 5, 20, 50, "--targets", 0.7, 0.9, "--checkpoints", 0.1, 0.2,
                 "--max-frac", 0.3, "--workers", 1, "--out", out)
        populate("sift", "--n-db", 300, "--n-cand-queries", 60, "--n-score-queries", 60,
                 "--n-test", 40, "--k", 3, "--k-values", 5, 20, "--targets", 0.8,
                 "--checkpoints", 0.1, "--max-frac", 0.3, "--out", root / "run_sift")
    except BaseException:
        workloads.PATHS.sift = old
        raise
    yield out
    workloads.PATHS.sift = old


@pytest.fixture(scope="module")
def trace_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("aker")
    rng = np.random.default_rng(2)
    np.save(root / "spacev-10m.npy", rng.integers(-50, 50, size=(300, 6)).astype(np.int8))
    pool = rng.integers(-50, 50, size=(60, 6)).astype(np.int8)
    np.save(root / "spacev-sim-100k-0.3.npy", pool[rng.integers(0, 60, 800)])
    old = workloads.PATHS.aker
    workloads.PATHS.aker = root
    out = root / "run"
    try:
        populate("trace", "--trace", "spacev-aker-0.3", "--k", 3, "--k-values", 5, 20,
                 "--targets", 0.7, 0.9, "--checkpoints", 0.1, 0.2, "--max-frac", 0.3,
                 "--workers", 1, "--out", out)
    except BaseException:
        workloads.PATHS.aker = old
        raise
    yield out
    workloads.PATHS.aker = old


def test_populate_wrote_results_for_every_workload(esci_run, synthetic_run, trace_run):
    for d, extra in ((esci_run, None), (synthetic_run, "workload_stats"), (trace_run, None)):
        data = json.loads((d / "results.json").read_text())
        assert {r["target"] for r in data["results"]} == {0.7, 0.9}
        assert all(r["points"] for r in data["results"])
        assert (d / "hit_rate_vs_cache_size_linear.png").exists()
        assert extra is None or extra in data
    assert (trace_run / "topk.npz").exists()
    assert (synthetic_run.parent / "run_sift" / "results.json").exists()


@pytest.mark.parametrize("name,main,args,output", [
    ("core", "core", ["--sizes", "0.1", "--n-each", "60", "--workers", "1"], "core.json"),
    ("simple", "simple", ["--sizes", "0.1", "--n-each", "60", "--R", "500", "--workers", "1"],
     "simple.json"),
    ("final", "final", ["--n-each", "60", "--R", "500", "--cal-sizes", "40", "--draws", "3",
                                "--workers", "1"], "final.json"),
    ("shell_ablation", "shell-ablation", ["--sizes", "0.1", "--n-each", "60", "--R", "500",
                                                   "--workers", "1"], "shell_ablation.json"),
    ("shell_size", "shell-size", ["--sizes", "0.1", "--ps", "0.5", "--Rs", "500",
                                          "--cal-sizes", "40", "--draws", "3", "--n-each", "60",
                                          "--workers", "1"], "shell_size.json"),
    ("shell_budget", "shell-budget", ["--budgets", "0.1", "--fs", "0.85", "--Rs", "500",
                                              "--n-each", "60", "--workers", "1"], "shell_budget.json"),
    ("gen_caches", "generalize", ["--mode", "caches", "--sizes", "0.1", "--base-size", "0.1",
                                          "--other-target", "0.7", "--n-each", "60"],
     "generalize_caches.json"),
    ("gen_k", "generalize", ["--mode", "k", "--ks", "1", "5", "--source-k", "5",
                                     "--n-each", "60"], "generalize_k.json"),
    ("certificates", "certificates", ["--profile-sizes", "10", "--workers", "1"],
     "certificates.json"),
    ("calibrated", "calibrated", ["--n-fit", "60", "--n-calib", "60", "--workers", "1",
                                          "--targets", "0.9"], "calibrated.json"),
    ("predictors", "predictors", ["--targets", "0.9", "--sizes", "0.1", "--n-each", "60",
                                          "--workers", "1"], "predictors.json"),
    ("transfer", "transfer", ["--sizes", "0.05", "--base-size", "0.1",
                                                "--other-targets", "0.7", "--perturb", "0.3",
                                                "--n-each", "60", "--workers", "1"],
     "threshold_transfer.json"),
    ("calibration_methods", "calibration",
     ["--targets", "0.9", "--sizes", "0.1", "--n-fit", "60", "--n-cal", "40", "--trials", "2",
      "--n-pool-requests", "200", "--workers", "1"], "calibration_methods.json"),
])
def test_esci_run_through_every_evaluator(esci_run, name, main, args, output):
    evaluate(main, esci_run, *args)
    assert json.loads((esci_run / output).read_text())


@pytest.mark.parametrize("name,main,args,output", [
    ("simple", "simple", ["--sizes", "0.2", "--n-each", "40", "--R", "50", "--workers", "1"],
     "simple.json"),
    ("certificates", "certificates", ["--profile-sizes", "10", "--workers", "1"],
     "certificates.json"),
    ("calibrated", "calibrated", ["--n-fit", "40", "--n-calib", "40", "--workers", "1",
                                          "--targets", "0.9"], "calibrated.json"),
    ("predictors", "predictors", ["--targets", "0.9", "--sizes", "0.2", "--n-each", "40",
                                          "--workers", "1"], "predictors.json"),
    ("calibration_methods", "calibration",
     ["--targets", "0.9", "--sizes", "0.2", "--n-fit", "40", "--n-cal", "30", "--trials", "2",
      "--n-pool-requests", "100", "--workers", "1"], "calibration_methods.json"),
])
def test_synthetic_run_through_the_zipf_branches(synthetic_run, name, main, args, output):
    evaluate(main, synthetic_run, *args)
    assert json.loads((synthetic_run / output).read_text())


@pytest.mark.parametrize("name,main,args,output", [
    ("simple", "simple", ["--sizes", "0.2", "--R", "20", "--workers", "1"], "simple.json"),
    ("calibrated", "calibrated", ["--workers", "1", "--targets", "0.9"], "calibrated.json"),
    ("predictors", "predictors", ["--targets", "0.9", "--sizes", "0.2", "--workers", "1"],
     "predictors.json"),
    ("aker", "aker", ["--target", "0.9"], "aker_thresholds.json"),
])
def test_trace_run_through_the_trace_branches(trace_run, name, main, args, output):
    evaluate(main, trace_run, *args)
    assert json.loads((trace_run / output).read_text())


def test_every_command_group_has_help():
    for group in cli.GROUPS:
        with pytest.raises(SystemExit) as e:
            cli.main([group, "--help"])
        assert e.value.code == 0
