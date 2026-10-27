#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Constants for the end-to-end cache pipeline tests (population + hit policy, oracle and LSH radius arms).

Notation: K is the top-K size (10 here, unlike common/config.py), T the target recall, n the number of train
(and test) queries, B the pool budget in vectors, lambda the local fraction of queries."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE.parent
THESIS_ROOT = HERE.parents[2]
HIT_POLICY_DIR = THESIS_ROOT / "weekly" / "9_22_2026_hit_policy_testing"

# Imported without edits: dataset loaders, ground truth helpers, logging, perturbation, plot style, rt_* helpers.
for _p in (THESIS_ROOT, HIT_POLICY_DIR / "common", HIT_POLICY_DIR / "radius_eval_tables"):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

#-------------------------------------------------------------------------------
# Paths (results, plots and pdf are covered by the repo .gitignore patterns weekly/*/results|plots|pdf)

RESULTS_DIR = OUTPUT_ROOT / "results"
GROUND_TRUTH_DIR = RESULTS_DIR / "ground_truth"
LOGS_DIR = RESULTS_DIR / "logs"
PLOTS_DIR = OUTPUT_ROOT / "plots"
PDF_DIR = OUTPUT_ROOT / "pdf"

for _d in (RESULTS_DIR, GROUND_TRUTH_DIR, LOGS_DIR, PLOTS_DIR, PDF_DIR):
    _d.mkdir(parents=True, exist_ok=True)

#-------------------------------------------------------------------------------
# Fixed experimental design

DATASETS = ["sift", "esci"]
K = 10
N_DB = 50_000
N_QUERIES = 2500  # train queries, and also test queries
BUDGET_FRACTIONS = (0.01, 0.05, 0.10)
BUDGETS = tuple(int(round(f * N_DB)) for f in BUDGET_FRACTIONS)  # 500, 2500, 5000
TARGETS = (0.8, 0.9, 0.95)
LAMBDAS = (0.0, 0.5, 1.0)
LAMBDA_NAMES = {0.0: "no", 0.5: "some", 1.0: "high"}
SEEDS = (0, 1, 2, 3, 4)
BENCH_SEEDS = (100, 101, 102)  # LSH mini-benchmark only, never reported results

# LSH arm: (L tables, m bits per table); the first three use rows [0, 512) of the 2048-row hyperplane matrix
CONFIGS = [(1, 512), (4, 128), (8, 64), (1, 2048), (4, 512), (8, 256), (16, 128)]
NU_GRID = (1, 2, 4, 8, 16, 32)

# Regimes: the lambda mixture (50 train and 50 test copies per anchor) and the diffuse regime outside it
# (500 anchors, 5 copies each), identified by (lambda, copies per anchor).
REGIME_SPECS = {"lam0": (0.0, 50), "lam0.5": (0.5, 50), "lam1": (1.0, 50), "diffuse": (1.0, 5)}

QUERIES_PER_ANCHOR = 50
MAX_ANGLE_DEG = 10.0
CANDIDATE_MAX_SET = 5000  # candidates with more than this many vectors are dropped
PREFIX_LEN = CANDIDATE_MAX_SET + 1  # sorted nearest-vector prefix kept per train query
FLOOR_UNITS = 10  # HP3f floor: estimated recall >= 0.5, i.e. 10 units of 0.05 (est count >= 5 of 10)
GROUP_UNION_MAX_I = 20
BALL_MAX_I = 20
BURN_IN = 100

# Hit policy arithmetic is exact: recall units of 0.05 (K = 10 makes every estimate a multiple of 0.1).
UNITS_PER_RECALL = 20
GT_CHUNK = 250

#-------------------------------------------------------------------------------
# Safety limits (same rules as the earlier long runs)

RSS_LIMIT_MB = 5500.0
MIN_FREE_DISK_GB = 2.0
EXIT_ABORT = 3

#-------------------------------------------------------------------------------

def n_anchors(lam: float, n: int = N_QUERIES) -> int:
    return int(round(lam * n / QUERIES_PER_ANCHOR))

def target_units(T: float) -> int:
    """Target recall in units of 0.05 (16, 18, 19 for T = 0.8, 0.9, 0.95)."""
    return int(round(T * UNITS_PER_RECALL))

def slack_units(T: float, n: int) -> int:
    """Total slack n (1 - T) in units of 0.05 recall."""
    return int(round(n * (1.0 - T) * UNITS_PER_RECALL))

def per_step_slack_units(T: float) -> int:
    """Slack per stream step (1 - T) in units of 0.05 recall (4, 2, 1)."""
    return UNITS_PER_RECALL - target_units(T)

def regime_tag(lam: float, per_anchor: int = QUERIES_PER_ANCHOR) -> str:
    """Regime name used in file names: lam0, lam0.5, lam1 for the mixture, diffuse (500 x 5) outside it."""
    if per_anchor == QUERIES_PER_ANCHOR:
        return f"lam{lam:g}"
    return "diffuse" if (lam, per_anchor) == REGIME_SPECS["diffuse"] else f"lam{lam:g}_pa{per_anchor}"
