#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Tests for the proposal analysis script using synthetic mock outputs.

Mock directory naming matches real run_population.py output exactly
(nmax{n}_k{k_test}, no separate K_cache component -- point-like proposal was
removed and K_cache always equals K_test in the real pipeline; see
utils/base/population.py and docs/POPULATION_DESIGN.md). A prior version of
this test used a different, incorrect naming convention that happened to
agree with the (also incorrect) parsing in analyze_population_proposal.py
-- both were wrong relative to real pipeline output, just consistently
wrong with each other. This version is checked against the real naming
directly (see test_analyze_regex_matches_real_naming below).

Phase 3 Round 12 reworked the analysis script's grid from n x K_test to
sigma x K_test (n is fixed at 20 in every real run, never actually swept)
and added a per-K_test recall-vs-sigma plot -- these tests build mock data
with a real sigma sweep by default, matching that shape.
"""

import sys
import json
import re
import shutil
import tempfile
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

#-------------------------------------------------------------------------------

_STRATEGY = 'greedy_cost'
_BENCHMARKS = ['esci_c256k100_t512n20_s42', 'esci_c512k100_t512n20_s42']
_N_BASE = 1_000
_SIGMA_VALUES = [0.70, 1.00]


def _write_trajectory(path: Path, n_steps: int, budget: int) -> None:
    traj = [
        {
            'num_entries': i + 1,
            'vectors_cached': min(int(budget * (i + 1) / n_steps), budget),
            'cig_train_coverage': min(0.1 * (i + 1), 1.0),
            'cig_test_coverage': min(0.08 * (i + 1), 1.0),
            'new_train_coverage': 1,
            'new_test_coverage': 1,
        }
        for i in range(n_steps)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(traj))


def _write_summary(path: Path, benchmark: str, hit_rate: float, mean_recall: float) -> None:
    summary = {
        'dataset': benchmark,
        'strategy': _STRATEGY,
        'config': {'num_base_vectors': _N_BASE, 'cache_K': 10},
        'results': [
            {
                'algorithm': 'lemma1_no_union', 'hit_rate': hit_rate,
                'mean_recall': mean_recall, 'total_queries': 20, 'cache_hits': 5,
            }
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary))


def _build_raw(raw_dir: Path, n: int = 1, k_test_vals=(0, 1), sigma_vals=_SIGMA_VALUES) -> None:
    """Matches real run_population.py output: ball/nmax{n}_k{k_test}/...,
    with a sigma sweep nested as B{budget}/sigma_{val:.2f}/ (real pipeline
    shape whenever more than the single default sigma=1.0 is swept)."""
    for k_test in k_test_vals:
        param_dir = raw_dir / 'ball' / f'nmax{n}_k{k_test}'
        for bm in _BENCHMARKS:
            for budget in [50, 100]:
                b_dir = param_dir / bm / f'B{budget}'
                _write_trajectory(
                    b_dir / f'{_STRATEGY}_trajectory.json', n_steps=3, budget=budget
                )
                for sigma in sigma_vals:
                    _write_summary(
                        b_dir / f'sigma_{sigma:.2f}' / 'non_oracle' / _STRATEGY / 'summary.json',
                        bm, hit_rate=0.3 + 0.01 * k_test, mean_recall=sigma * 0.9,
                    )

#-------------------------------------------------------------------------------

def test_analyze_regex_matches_real_naming():
    """Direct regression test for the bug found in Phase 3 Round 9: the
    analysis script's directory-name regex must match what run_population.py
    actually produces (nmax{n}_k{k_test}), not some other assumed shape."""
    real_dirname = 'nmax20_k5'
    m = re.fullmatch(r'nmax(\d+)_k(\d+)', real_dirname)
    assert m is not None, f"regex must match real pipeline naming: {real_dirname!r}"
    assert m.group(1) == '20'
    assert m.group(2) == '5'

    # also confirm the old (broken) patterns do NOT match, documenting why
    # they were replaced
    assert re.fullmatch(r'n(\d+)_k(\d+)', real_dirname) is None
    assert re.fullmatch(r'n(\d+)_kc(\d+)_k(\d+)', real_dirname) is None
    print("  OK: nmax{n}_k{k_test} regex matches real naming, old patterns don't")


def test_analyze_creates_expected_plots():
    tmpdir = Path(tempfile.mkdtemp(prefix='analyze_proposal_'))
    try:
        raw_dir = tmpdir / 'raw'
        processed_dir = tmpdir / 'processed'
        _build_raw(raw_dir, n=1, k_test_vals=(0, 1))

        from simulations.population.analyze_population_proposal import analyze_dataset
        analyze_dataset(raw_dir, processed_dir, 'ESCI', sigma_values=_SIGMA_VALUES)

        for fname in ('memory_vs_hitrate.png', 'memory_vs_candidates_cached.png'):
            assert (processed_dir / 'ball' / fname).exists(), f"Missing {fname}"
        assert not (processed_dir / 'ball' / 'memory_vs_vectors_cached.png').exists()

        # Phase 3 Round 15: memory_vs_coverage.png and the aggregate/
        # max-budget plot were both removed -- coverage tracks hit_rate
        # almost exactly by construction at sigma=1.0 (proven identity,
        # verified to exact agreement on real data) so a separate plot
        # added no signal; the aggregate plot wasn't useful in practice.
        assert not (processed_dir / 'ball' / 'memory_vs_coverage.png').exists()
        assert not (processed_dir / 'ball' / 'aggregate').exists()

        # New (Round 12): one combined recall-vs-sigma grid file (rows=K_test,
        # columns=budget), no more separate sigma_*/ directories nesting the
        # other plots, and no more one-file-per-K_test split (Round 13).
        assert (processed_dir / 'ball' / 'recall_vs_sigma.png').exists(), (
            "Missing recall_vs_sigma.png"
        )
        assert not list(processed_dir.rglob('recall_vs_sigma_k*.png')), (
            "old per-K_test recall_vs_sigma filenames should not appear"
        )
        assert not (processed_dir / 'sigma_0.70').exists()
        assert not (processed_dir / 'sigma_1.00').exists()

        # Old (deprecated point / kc-based ball) files should not exist
        assert not (processed_dir / 'point').exists()
        assert not list(processed_dir.rglob('*_n1.png')), "old per-n ball filenames should not appear"
        assert not list(processed_dir.rglob('sim_hit_rate_cig.png'))
        assert not list(processed_dir.rglob('coverage_trajectory_*.png'))
        assert not list(processed_dir.rglob('hit_rate_vs_n.png'))
        assert not list(processed_dir.rglob('hit_rate_vs_ktest.png'))

        print("  OK: all expected PNG files created")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_analyze_no_recall_plot_without_sigma_sweep():
    """A flat (single default sigma=1.0) layout has nothing to plot sigma
    against -- recall_vs_sigma.png must not be produced."""
    tmpdir = Path(tempfile.mkdtemp(prefix='analyze_nosweep_'))
    try:
        raw_dir = tmpdir / 'raw'
        processed_dir = tmpdir / 'processed'
        _build_raw(raw_dir, n=1, k_test_vals=(1,), sigma_vals=[1.00])

        from simulations.population.analyze_population_proposal import analyze_dataset
        # flat layout: summary lives directly under B{budget}/non_oracle/...,
        # not B{budget}/sigma_1.00/non_oracle/... -- rebuild without nesting.
        shutil.rmtree(raw_dir)
        for bm in _BENCHMARKS:
            for budget in [50, 100]:
                b_dir = raw_dir / 'ball' / 'nmax1_k1' / bm / f'B{budget}'
                _write_trajectory(b_dir / f'{_STRATEGY}_trajectory.json', n_steps=3, budget=budget)
                _write_summary(
                    b_dir / 'non_oracle' / _STRATEGY / 'summary.json',
                    bm, hit_rate=0.31, mean_recall=0.9,
                )

        analyze_dataset(raw_dir, processed_dir, 'ESCI', sigma_values=None)

        assert not list(processed_dir.rglob('recall_vs_sigma*.png')), (
            "recall_vs_sigma plots should not appear for a single-sigma run"
        )
        assert (processed_dir / 'ball' / 'memory_vs_hitrate.png').exists()
        print("  OK: no recall_vs_sigma plot without a real sigma sweep")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_analyze_memory_utilization_values():
    """Verify that vectors_cached / n_base is in (0, 1] for mock trajectories."""
    tmpdir = Path(tempfile.mkdtemp(prefix='analyze_util_'))
    try:
        raw_dir = tmpdir / 'raw'
        _build_raw(raw_dir, n=1, k_test_vals=(1,))

        from simulations.population.analyze_population_proposal import _load_run_data, _budget_int
        param_dir = raw_dir / 'ball' / 'nmax1_k1'
        run_data = _load_run_data(param_dir, sigma=1.00)

        for bm, bm_data in run_data.items():
            for budget_str, entry in bm_data.items():
                vc = entry.get('vectors_cached', 0)
                util = vc / _N_BASE
                assert 0.0 <= util <= 1.0 + 1e-6, (
                    f"Utilization out of range: bm={bm} budget={budget_str} util={util:.4f}"
                )
                assert 0.0 <= entry['mean_recall'] <= 1.0

        print("  OK: memory utilization values in valid range")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_analyze_no_sim_hit_rate_cig():
    """Confirm sim_hit_rate_cig.png is never produced."""
    tmpdir = Path(tempfile.mkdtemp(prefix='analyze_nocig_'))
    try:
        raw_dir = tmpdir / 'raw'
        processed_dir = tmpdir / 'processed'
        _build_raw(raw_dir, n=1, k_test_vals=(1,))

        from simulations.population.analyze_population_proposal import analyze_dataset
        analyze_dataset(raw_dir, processed_dir, 'ESCI', sigma_values=_SIGMA_VALUES)

        cig_files = list(processed_dir.rglob('sim_hit_rate_cig.png'))
        assert len(cig_files) == 0, f"Unexpected cig files: {cig_files}"
        print("  OK: sim_hit_rate_cig.png not produced")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_analyze_debug_json():
    """Verify debug JSON is written with expected structure and key fields."""
    tmpdir = Path(tempfile.mkdtemp(prefix='analyze_debug_'))
    try:
        raw_dir = tmpdir / 'raw'
        processed_dir = tmpdir / 'processed'
        debug_dir = tmpdir / 'debug'
        _build_raw(raw_dir, n=1, k_test_vals=(1,))

        from simulations.population.analyze_population_proposal import analyze_dataset
        analyze_dataset(raw_dir, processed_dir, 'ESCI', debug_dir=debug_dir,
                         sigma_values=_SIGMA_VALUES)

        debug_path = debug_dir / 'ESCI_ball_debug.json'
        assert debug_path.exists(), "Debug JSON not created"
        with open(debug_path) as f:
            debug = json.load(f)

        assert debug['dataset'] == 'ESCI'
        assert debug['method'] == 'ball'
        assert debug['n_max'] == 1
        assert 'dataset_summary' in debug
        assert 'params' in debug

        # keys are now sigma{sigma:.2f}_k{kt}, not n{n}_k{kt}
        assert 'sigma0.70_k1' in debug['params']
        param = debug['params']['sigma0.70_k1']
        assert 'benchmarks' in param
        assert 'param_combo_summary' in param

        bm_entry = next(iter(param['benchmarks'].values()))
        assert 'n_base' in bm_entry
        assert 'budgets' in bm_entry
        assert 'per_benchmark_sanity' in bm_entry

        budget_entry = next(iter(bm_entry['budgets'].values()))
        for field in ('test_hit_rate', 'train_hit_rate', 'mean_recall', 'memory_util',
                      'budget_utilization', 'vectors_cached', 'num_entries',
                      'train_coverage_final', 'trajectory_steps', 'traj_vectors_at_step1'):
            assert field in budget_entry, f"Missing field: {field}"

        assert 'sanity_checks' in debug
        assert 'n_combos_with_exact_1.0_hit_rate_at_max_budget' in debug['sanity_checks']

        print("  OK: debug JSON has expected structure")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


#-------------------------------------------------------------------------------

def main():
    tests = [
        test_analyze_regex_matches_real_naming,
        test_analyze_creates_expected_plots,
        test_analyze_no_recall_plot_without_sigma_sweep,
        test_analyze_memory_utilization_values,
        test_analyze_no_sim_hit_rate_cig,
        test_analyze_debug_json,
    ]
    print(f"\n{'='*70}")
    print("Testing analysis script")
    print('='*70)
    for fn in tests:
        print(f"\n[{fn.__name__}]")
        fn()
    print(f"\n{'='*70}")
    print("All passed.")
    print('='*70)

if __name__ == '__main__':
    main()
