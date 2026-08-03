#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Shared plotting/labeling helpers for hit-policy and synthetic analysis scripts."""

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

#-------------------------------------------------------------------------------

ALGO_ORDER = [
    'lemma1', 'lemma1_no_union',
    'lemma2', 'lemma2_no_union',
    'combined',
    'brute',
]

ALGO_LABELS = {
    'lemma1':            'CIG',
    'lemma1_no_union':   'CIG (no union)',
    'lemma2':            'HGG',
    'lemma2_no_union':   'HGG (no union)',
    'combined':          'Combined',
    'brute':             'Brute Force',
}

#-------------------------------------------------------------------------------

def lbl(algo: str) -> str:
    return ALGO_LABELS.get(algo, algo)

def lbls(algos) -> List[str]:
    return [lbl(a) for a in algos]

def cap_metric(m: str) -> str:
    return m.title()

def cap_metrics(metrics) -> List[str]:
    return [cap_metric(m) for m in metrics]

def save_fig(fig: plt.Figure, path: Path, name: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / name, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  {name}")

def pct_fmt(ax, axis: str = 'y') -> None:
    formatter = mticker.FuncFormatter(lambda v, _: f'{v:.0%}')
    if axis == 'y':
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_formatter(formatter)

def load_summaries(raw_dir: str, kind: str = 'summaries') -> List[Dict]:
    raw_path = Path(raw_dir)
    summaries = []
    for p in sorted(raw_path.glob('*/summary.json')):
        with open(p) as f:
            summaries.append(json.load(f))
    print(f"Loaded {len(summaries)} {kind}")
    return summaries
