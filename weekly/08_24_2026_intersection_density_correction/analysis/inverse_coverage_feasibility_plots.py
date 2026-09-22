#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Feasibility report for inverse-coverage counting: the min_coverage-
stratified error table (the single most important result - it decides
go/no-go per POSTMORTEM.md's successor direction), plus the hit-rate-vs-
threshold table. Reads E2 output from
schemes/inverse_coverage_feasibility_run_experiment.py."""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent.parent / 'results' / 'inverse_coverage_feasibility'
OUT_DIR = Path(__file__).parent.parent / 'plots' / 'inverse_coverage_feasibility'
COVERAGE_BINS = [0.0, 1e-10, 1e-5, 1e-2, 0.5, 1.0]
COVERAGE_LABELS = ['[0,1e-10)', '[1e-10,1e-5)', '[1e-5,1e-2)', '[1e-2,0.5)', '[0.5,1.0]']

#-------------------------------------------------------------------------------


def stratified_table(dataset: str, d: dict) -> None:
    print(f"\n=== {dataset}: min_coverage-stratified error table ===")
    err = np.abs(d['true_recall'] - d['recall_hat'])
    idx = np.digitize(d['min_coverage'], COVERAGE_BINS) - 1
    idx = np.clip(idx, 0, len(COVERAGE_LABELS) - 1)
    print(f"{'coverage bin':>16} {'n':>6} {'frac':>7} {'mean_abs_err':>13}")
    for b, label in enumerate(COVERAGE_LABELS):
        mask = idx == b
        if mask.sum() == 0:
            continue
        print(f"{label:>16} {mask.sum():>6} {mask.mean():>7.4f} {err[mask].mean():>13.4f}")
    trust = d['min_coverage'] >= 0.5
    print(f"  trustworthy fraction (min_coverage>=0.5): {trust.mean():.4f} ({trust.sum()}/{len(trust)})")
    print(f"  correlation(min_coverage, abs_error): {np.corrcoef(d['min_coverage'], err)[0, 1]:+.4f}")
    print(f"  mean_abs_err recall_hat: {err.mean():.4f}, mean_abs_err baseline: "
          f"{np.abs(d['true_recall'] - d['baseline_recall']).mean():.4f}")


def hit_rate_table(dataset: str, d: dict, thresholds=(0.3, 0.5, 0.7)) -> None:
    print(f"\n=== {dataset}: hit-rate / precision at recall_hat thresholds ===")
    print(f"{'threshold':>10} {'hit_rate':>9} {'precision(mean true_recall|serve)':>36}")
    for th in thresholds:
        serve = d['recall_hat'] >= th
        hit_rate = serve.mean()
        precision = d['true_recall'][serve].mean() if serve.any() else float('nan')
        print(f"{th:>10.2f} {hit_rate:>9.4f} {precision:>36.4f}")


def plot_stratified(dataset: str, d: dict, out_dir: Path) -> None:
    err = np.abs(d['true_recall'] - d['recall_hat'])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(d['min_coverage'], err, s=8, alpha=0.4)
    ax.set_xscale('symlog', linthresh=1e-10)
    ax.set_xlabel('min_coverage')
    ax.set_ylabel('|true_recall - recall_hat|')
    ax.set_title(f'{dataset}: error vs. min_coverage confidence signal')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / f'min_coverage_vs_error_{dataset}.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {out_path}")

#-------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', default=['sift', 'esci'])
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        path = RESULTS_DIR / f'e2_{dataset}.npz'
        if not path.exists():
            print(f"skipping {dataset}: {path} not found")
            continue
        d = np.load(path)
        stratified_table(dataset, d)
        hit_rate_table(dataset, d)
        plot_stratified(dataset, d, OUT_DIR)


if __name__ == '__main__':
    main()
