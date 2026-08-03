#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""End-to-end tests for the proposal-based population pipeline."""

import json
import functools
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from simulations.population.population_runner import run_population_benchmark as _run_benchmark
from utils.base.population import (
    propose_candidates,
    propose_candidates_ball,
)
from tests.helpers import create_mini_benchmark_with_population_data
from datasets.dataloaders import load_esci_benchmark, load_sift_benchmark

#-------------------------------------------------------------------------------

STRATEGY = 'greedy_cost'
METRIC = 'angular'
ALGORITHMS = ['lemma1_no_union']
SEED = 42

LOAD_BENCHMARK_FN = {'sift': load_sift_benchmark, 'esci': load_esci_benchmark}

#-------------------------------------------------------------------------------

def _make_mini(dataset: str, tmpdir: Path):
    bm_dir = tmpdir / 'test_benchmark'
    create_mini_benchmark_with_population_data(dataset, bm_dir)
    return bm_dir


def _run(dataset: str, tmpdir: Path, propose_fn, output_base: Path, budget: int = 50,
         override_K: int = 1, k_test_override=None) -> Path:
    _run_benchmark(
        benchmark_name='test_benchmark',
        load_benchmark_fn=LOAD_BENCHMARK_FN[dataset],
        metric=METRIC,
        output_dir=output_base,
        benchmark_base_dir=tmpdir,
        propose_fn=propose_fn,
        budget_values=[budget],
        strategies=[STRATEGY],
        algorithms=ALGORITHMS,
        seed=SEED,
        run_ts='run',
        override_K=override_K,
        k_test_override=k_test_override,
    )
    return output_base / 'run' / 'test_benchmark'

#-------------------------------------------------------------------------------


@pytest.mark.parametrize('dataset', ['sift', 'esci'])
def test_e2e_point_proposal_runner(dataset):
    tmpdir = Path(tempfile.mkdtemp(prefix=f'{dataset}_prop_point_'))
    try:
        _make_mini(dataset, tmpdir)
        out_base = tmpdir / 'out'
        propose_fn = functools.partial(propose_candidates_ball, n_max=5, K=0)
        bm_dir = _run(dataset, tmpdir, propose_fn, out_base, budget=50, override_K=1)

        summary_path = bm_dir / 'B50' / 'non_oracle' / STRATEGY / 'summary.json'
        assert summary_path.exists(), f"Missing summary.json: {summary_path}"
        summary = json.loads(summary_path.read_text())
        result = next(r for r in summary['results'] if r['algorithm'] == 'lemma1_no_union')
        assert 0.0 <= result['hit_rate'] <= 1.0, f"hit_rate={result['hit_rate']}"

        traj_path = bm_dir / 'B50' / f'{STRATEGY}_trajectory.json'
        assert traj_path.exists()
        traj = json.loads(traj_path.read_text())
        assert isinstance(traj, list)
        for step in traj:
            assert 'vectors_cached' in step
            assert step['vectors_cached'] <= 50
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.parametrize('dataset', ['sift', 'esci'])
def test_e2e_ball_proposal_runner(dataset):
    tmpdir = Path(tempfile.mkdtemp(prefix=f'{dataset}_prop_ball_'))
    try:
        _make_mini(dataset, tmpdir)
        out_base = tmpdir / 'out'
        propose_fn = functools.partial(propose_candidates_ball, n_max=5, K=5)
        bm_dir = _run(dataset, tmpdir, propose_fn, out_base, budget=50, override_K=1)

        summary_path = bm_dir / 'B50' / 'non_oracle' / STRATEGY / 'summary.json'
        assert summary_path.exists(), "Missing summary.json for ball proposal"
        summary = json.loads(summary_path.read_text())
        result = next(r for r in summary['results'] if r['algorithm'] == 'lemma1_no_union')
        assert 0.0 <= result['hit_rate'] <= 1.0

        traj = json.loads((bm_dir / 'B50' / f'{STRATEGY}_trajectory.json').read_text())
        for step in traj:
            assert step['vectors_cached'] <= 50
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.parametrize('dataset', ['sift', 'esci'])
def test_e2e_k_test_zero(dataset):
    """K_test=0: cache entries still populated at K=1, but obs_r_Q/train_r_t
    and test_r_t are both zeroed, so coverage and simulate-time hit
    certification reduce to pure distance containment. sim_K is not
    clamped, avoiding a division by zero, and mean_recall scores 1.0 for
    every hit, since containment alone is the full task with no K-sized
    target to fall short of."""
    tmpdir = Path(tempfile.mkdtemp(prefix=f'{dataset}_prop_ktest0_'))
    try:
        _make_mini(dataset, tmpdir)
        out_base = tmpdir / 'out'
        propose_fn = functools.partial(propose_candidates_ball, n_max=5, K=0)
        bm_dir = _run(dataset, tmpdir, propose_fn, out_base, budget=50, k_test_override=0)

        summary_path = bm_dir / 'B50' / 'non_oracle' / STRATEGY / 'summary.json'
        assert summary_path.exists(), "Missing summary.json for K_test=0 run"
        summary = json.loads(summary_path.read_text())
        result = next(r for r in summary['results'] if r['algorithm'] == 'lemma1_no_union')
        assert isinstance(result['hit_rate'], float)
        assert 0.0 <= result['hit_rate'] <= 1.0
        if result['hit_rate'] > 0.0:
            assert result['mean_recall'] == 1.0, (
                "K=0 hits must score recall=1.0 (containment alone is the "
                "full task at K=0)"
            )
        else:
            assert result['mean_recall'] == 0.0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.parametrize('dataset', ['sift', 'esci'])
def test_e2e_point_various_n_max_values(dataset):
    """A single call proposes every sphere in [1, n_max] at once; n_max is
    not swept externally. This just confirms the runner produces valid
    output across a range of n_max sizes.

    Mini benchmark has 20 cache queries, so n_max must be <= 19.
    """
    tmpdir = Path(tempfile.mkdtemp(prefix=f'{dataset}_prop_nmax_'))
    try:
        _make_mini(dataset, tmpdir)
        for n_max in [1, 2, 5, 10]:
            out_base = tmpdir / f'out_nmax{n_max}'
            propose_fn = functools.partial(propose_candidates_ball, n_max=n_max, K=0)
            bm_dir = _run(dataset, tmpdir, propose_fn, out_base, budget=30, override_K=1)
            summary_path = bm_dir / 'B30' / 'non_oracle' / STRATEGY / 'summary.json'
            assert summary_path.exists(), f"Missing output for n_max={n_max}"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.parametrize('dataset', ['sift', 'esci'])
def test_e2e_ball_runner_no_k_cache_constraint(dataset):
    """K_cache = K_test = K, so there is no separate K_cache to skip
    against; _run_ball must run for any k_test >= 1 without special-casing."""
    from simulations.population.run_population import _run_ball

    tmpdir = Path(tempfile.mkdtemp(prefix=f'{dataset}_prop_ball_run_'))
    try:
        _make_mini(dataset, tmpdir)
        out_base = tmpdir / 'out'
        _run_ball(
            benchmark_name='test_benchmark',
            n_max=5, k_test=5,
            strategies=[STRATEGY], seed=SEED, run_ts='ts',
            output_dir=out_base, benchmark_base_dir=tmpdir,
            load_benchmark_fn=LOAD_BENCHMARK_FN[dataset],
        )
        # Budgets are derived as fractions of n_base, so just confirm some
        # summary.json was produced under the ball output tree.
        ball_root = out_base / 'ts' / 'ball' / 'nmax5_k5'
        assert ball_root.exists(), f"Missing ball output dir: {ball_root}"
        assert list(ball_root.rglob('summary.json')), "No summary.json produced"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.parametrize('dataset', ['sift', 'esci'])
def test_e2e_ball_runner_k_test_zero_containment(dataset):
    """_run_ball's K=0 (containment mode) means "hit iff contained"
    end-to-end: propose_candidates_ball's margin is uniformly 0 and
    population_runner.py's sim_K is not clamped. Runs the driver end-to-end
    at k_test=0 and confirms mean_recall scores 1.0 per hit (see
    test_e2e_k_test_zero)."""
    from simulations.population.run_population import _run_ball

    tmpdir = Path(tempfile.mkdtemp(prefix=f'{dataset}_prop_ball_k0_'))
    try:
        _make_mini(dataset, tmpdir)
        out_base = tmpdir / 'out'
        _run_ball(
            benchmark_name='test_benchmark',
            n_max=5, k_test=0,
            strategies=[STRATEGY], seed=SEED, run_ts='ts',
            output_dir=out_base, benchmark_base_dir=tmpdir,
            load_benchmark_fn=LOAD_BENCHMARK_FN[dataset],
        )
        ball_root = out_base / 'ts' / 'ball' / 'nmax5_k0'
        assert ball_root.exists(), f"Missing ball output dir: {ball_root}"
        summaries = list(ball_root.rglob('summary.json'))
        assert summaries, "No summary.json produced for k_test=0"
        summary = json.loads(summaries[0].read_text())
        result = next(r for r in summary['results'] if r['algorithm'] == 'lemma1_no_union')
        assert isinstance(result['hit_rate'], float)
        assert 0.0 <= result['hit_rate'] <= 1.0
        if result['hit_rate'] > 0.0:
            assert result['mean_recall'] == 1.0
        else:
            assert result['mean_recall'] == 0.0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.parametrize('dataset', ['sift', 'esci'])
def test_e2e_budget_utilization_range(dataset):
    """Memory utilization = vectors_cached / n_base must be in [0, 1]."""
    tmpdir = Path(tempfile.mkdtemp(prefix=f'{dataset}_prop_util_'))
    try:
        bm_dir = _make_mini(dataset, tmpdir)
        n_base = np.load(bm_dir / 'base_vectors.npy').shape[0]
        budgets = [10, 30, 60]
        out_base = tmpdir / 'out'
        _run_benchmark(
            benchmark_name='test_benchmark',
            load_benchmark_fn=LOAD_BENCHMARK_FN[dataset],
            metric=METRIC,
            output_dir=out_base,
            benchmark_base_dir=tmpdir,
            propose_fn=functools.partial(propose_candidates_ball, n_max=5, K=0),
            budget_values=budgets,
            strategies=[STRATEGY],
            algorithms=ALGORITHMS,
            seed=SEED,
            run_ts='run',
            override_K=1,
        )
        for budget in budgets:
            traj_path = (
                out_base / 'run' / 'test_benchmark' / f'B{budget}'
                / f'{STRATEGY}_trajectory.json'
            )
            if not traj_path.exists():
                continue
            traj = json.loads(traj_path.read_text())
            if traj:
                util = traj[-1]['vectors_cached'] / n_base
                assert 0.0 <= util <= 1.0 + 1e-6, f"Utilization out of range: {util:.4f}"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.parametrize('dataset', ['sift', 'esci'])
def test_e2e_sigma_output_structure(dataset):
    """Sigma sweep: B{budget}/sigma_*/ directory structure, summary.json's
    sigma/hit_rate/mean_recall fields, and a trajectory written once per
    budget/strategy, not once per sigma, since sigma only affects the hit
    test, not the proposal/caching trajectory."""
    sigma_values = [0.70, 0.90]
    tmpdir = Path(tempfile.mkdtemp(prefix=f'{dataset}_prop_sigma_'))
    try:
        _make_mini(dataset, tmpdir)
        out_base = tmpdir / 'out'
        propose_fn = functools.partial(propose_candidates_ball, n_max=5, K=5)
        budget = 50
        _run_benchmark(
            benchmark_name='test_benchmark',
            load_benchmark_fn=LOAD_BENCHMARK_FN[dataset],
            metric=METRIC,
            output_dir=out_base,
            benchmark_base_dir=tmpdir,
            propose_fn=propose_fn,
            budget_values=[budget],
            strategies=[STRATEGY],
            algorithms=ALGORITHMS,
            seed=SEED,
            run_ts='run',
            override_K=1,
            sigma_values=sigma_values,
        )
        b_dir = out_base / 'run' / 'test_benchmark' / f'B{budget}'

        # trajectory: sigma-independent, written once per budget/strategy
        traj_path = b_dir / f'{STRATEGY}_trajectory.json'
        assert traj_path.exists(), f"trajectory missing: {traj_path}"
        traj = json.loads(traj_path.read_text())
        assert isinstance(traj, list)

        for sigma in sigma_values:
            sigma_dir = b_dir / f'sigma_{sigma:.2f}'
            summary_path = sigma_dir / 'non_oracle' / STRATEGY / 'summary.json'
            assert summary_path.exists(), f"summary missing: {summary_path}"
            summary = json.loads(summary_path.read_text())
            result = next(r for r in summary['results'] if r['algorithm'] == 'lemma1_no_union')

            for field in ('sigma', 'hit_rate', 'mean_recall'):
                assert field in result, f"summary.json missing field '{field}' for sigma={sigma}"
            assert abs(result['sigma'] - sigma) < 1e-6, (
                f"sigma mismatch: stored={result['sigma']} expected={sigma}"
            )
            assert 0.0 <= result['hit_rate'] <= 1.0
            assert 0.0 <= result['mean_recall'] <= 1.0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.parametrize('dataset', ['sift', 'esci'])
def test_e2e_runner_alpha_backward_compat(dataset):
    """Alpha-based propose_candidates must still work through run_population_benchmark."""
    tmpdir = Path(tempfile.mkdtemp(prefix=f'{dataset}_alpha_compat_'))
    try:
        _make_mini(dataset, tmpdir)
        out_base = tmpdir / 'out'
        propose_fn = functools.partial(propose_candidates, alpha_values=[5.0])
        _run_benchmark(
            benchmark_name='test_benchmark',
            load_benchmark_fn=LOAD_BENCHMARK_FN[dataset],
            metric=METRIC,
            output_dir=out_base,
            benchmark_base_dir=tmpdir,
            propose_fn=propose_fn,
            budget_values=[50],
            strategies=['random'],
            algorithms=['lemma1_no_union'],
            seed=SEED,
            run_ts='run',
            override_K=1,
        )
        summary_path = (
            out_base / 'run' / 'test_benchmark' / 'B50' / 'non_oracle' / 'random' / 'summary.json'
        )
        assert summary_path.exists(), "Alpha-based runner broke after refactor"
        summary = json.loads(summary_path.read_text())
        result = next(r for r in summary['results'] if r['algorithm'] == 'lemma1_no_union')
        assert 0.0 <= result['hit_rate'] <= 1.0
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
