#!/usr/bin/env python3
"""Wiring smoke test: every driver runs end to end on a tiny fake ESCI directory.

Checks that paths, argument names and file hand-offs work; it does not check numbers.
"""

import json
import sys

import numpy as np
import pytest

from simulations.approximate_hits import (
    eval_calibrated, eval_calibration_methods, eval_certificates, eval_core, eval_final,
    eval_generalize, eval_predictors, eval_shell_ablation, eval_shell_budget, eval_shell_size,
    eval_simple, eval_threshold_transfer, run_e2e_oracle_sift,
)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("esci")
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((6, 8)) * 2
    np.save(root / "esci_db2000_s42.npy",
            (centers[rng.integers(0, 6, 2000)] + rng.standard_normal((2000, 8))).astype(np.float32))
    np.save(root / "esci_queries.npy",
            (centers[rng.integers(0, 6, 1500)] + rng.standard_normal((1500, 8))).astype(np.float32))
    out = root / "run"
    run(run_e2e_oracle_sift.main, "--dataset", "esci", "--esci-dir", str(root), "--n-db", "2000",
        "--n-cand-queries", "120", "--n-score-queries", "120", "--n-test", "100", "--k", "5",
        "--k-values", "5", "50", "500", "--targets", "0.7", "0.9", "--checkpoints", "0.05", "0.1",
        "--max-frac", "0.2", "--workers", "1", "--out", str(out))
    return out


def run(main, *argv):
    old = sys.argv
    sys.argv = ["prog", *map(str, argv)]
    try:
        main()
    finally:
        sys.argv = old


def test_populate_wrote_results(run_dir):
    data = json.loads((run_dir / "results.json").read_text())
    assert {r["target"] for r in data["results"]} == {0.7, 0.9}
    assert all(r["points"] for r in data["results"])


@pytest.mark.parametrize("name,main,args,output", [
    ("core", eval_core.main, ["--sizes", "0.1", "--n-each", "60", "--workers", "1"], "core.json"),
    ("simple", eval_simple.main, ["--sizes", "0.1", "--n-each", "60", "--R", "500", "--workers", "1"],
     "simple.json"),
    ("final", eval_final.main, ["--n-each", "60", "--R", "500", "--cal-sizes", "40", "--draws", "3",
                                "--workers", "1"], "final.json"),
    ("shell_ablation", eval_shell_ablation.main, ["--sizes", "0.1", "--n-each", "60", "--R", "500",
                                                   "--workers", "1"], "shell_ablation.json"),
    ("shell_size", eval_shell_size.main, ["--sizes", "0.1", "--ps", "0.5", "--Rs", "500",
                                          "--cal-sizes", "40", "--draws", "3", "--n-each", "60",
                                          "--workers", "1"], "shell_size.json"),
    ("shell_budget", eval_shell_budget.main, ["--budgets", "0.1", "--fs", "0.85", "--Rs", "500",
                                              "--n-each", "60", "--workers", "1"], "shell_budget.json"),
    ("gen_caches", eval_generalize.main, ["--mode", "caches", "--sizes", "0.1", "--base-size", "0.1",
                                          "--other-target", "0.7", "--n-each", "60"],
     "generalize_caches.json"),
    ("gen_k", eval_generalize.main, ["--mode", "k", "--ks", "1", "5", "--source-k", "5", "--n-each", "60"],
     "generalize_k.json"),
    ("certificates", eval_certificates.main, ["--profile-sizes", "10", "--workers", "1"],
     "certificates.json"),
    ("calibrated", eval_calibrated.main, ["--n-fit", "60", "--n-calib", "60", "--workers", "1",
                                          "--targets", "0.9"], "calibrated.json"),
    ("predictors", eval_predictors.main, ["--targets", "0.9", "--sizes", "0.1", "--n-each", "60",
                                          "--workers", "1"], "predictors.json"),
    ("transfer", eval_threshold_transfer.main, ["--sizes", "0.05", "--base-size", "0.1",
                                                "--other-targets", "0.7", "--perturb", "0.3",
                                                "--n-each", "60", "--workers", "1"],
     "threshold_transfer.json"),
    ("calibration_methods", eval_calibration_methods.main,
     ["--targets", "0.9", "--sizes", "0.1", "--n-fit", "60", "--n-cal", "40", "--trials", "2",
      "--n-pool-requests", "200", "--workers", "1"], "calibration_methods.json"),
])
def test_driver_runs_and_writes_its_output(run_dir, name, main, args, output):
    run(main, run_dir, *args)
    assert json.loads((run_dir / output).read_text())
