#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Single source of truth for every fixed constant in the real-data benchmark
spec."""

from pathlib import Path

#-------------------------------------------------------------------------------

SEED = 42  # the spec's "1 seed", matching this project's existing MODEL_SEED convention

N = 50_000  # subsampled DB pool size per dataset
K = 100
K_EXTENDED = 200  # item G: CALIB/TEST ground truth depth (headroom for diagnostics); K=100 unchanged for recall itself

BUDGET_FRACTIONS = [0.01, 0.05, 0.10]
BUDGETS = {frac: round(frac * N) for frac in BUDGET_FRACTIONS}  # {0.01: 500, 0.05: 2500, 0.1: 5000}

N_TRAIN = 1000
N_CALIB = 500
N_TEST = 5000

DATASETS = ["sift", "esci"]

DATASET_CONFIG = {
    "sift": dict(
        base_vectors_path=Path("datasets/sift/base_vectors_50k.npy"),
        normalize_base=True,
    ),
    "esci": dict(
        base_vectors_path=Path("datasets/esci/base_vectors_50k.npy"),
        normalize_base=False,
    ),
}

#-------------------------------------------------------------------------------
# Section 3: locality regimes

REGIMES = ["no_locality", "mid_locality", "high_locality"]

# topic pool size is resolved per-dataset at runtime (workload-size default for
# no_locality is data-dependent); zipf_exponent is fixed here.
REGIME_CONFIG = {
    "no_locality": dict(topic_pool_size=None, zipf_exponent=0.0),  # None -> workload size at runtime
    "mid_locality": dict(topic_pool_size=3000, zipf_exponent=0.6),  # midpoint of 1000-5000
    "high_locality": dict(topic_pool_size=250, zipf_exponent=1.1),
}

PERTURBATION_LEVEL = "small"  # utils/base/perturbation.py's small band (0-5 degrees)

#-------------------------------------------------------------------------------
# Section 5/7: entry selection and hit policies

M_NEAREST_ANCHORS = 50
EPSILON = 1e-9

ESTIMATOR_NAMES = ["thetaK", "lshraw", "combo"]
WEIGHTINGS = [True, False]  # True = soft, False = hard, must be bool, consumers do `if soft:`
HIT_POLICIES = [(est, w) for est in ESTIMATOR_NAMES for w in WEIGHTINGS]  # 6 total

#-------------------------------------------------------------------------------
# Section 6: LSH construction

LSH_M_HYPERPLANES = 16
LSH_N_TABLES = 16
LSH_C_MIN = 15
LSH_S_SWEEP = [12_000, 25_000, 50_000]
LSH_COMBINE_WEIGHTINGS = ["precision_weighted", "uniform"]

#-------------------------------------------------------------------------------
# Section 8: population policies

POPULATION_POLICIES = ["threshold_baseline", "balls_agg", "sets_agg"]

T_POP = 0.8  # fixed cache-construction target, distinct from T_eval, never swept
N_NEIGHBORS_PER_ANCHOR = 20  # baseline/balls-agg: top-20 smallest-R neighbors per anchor

BALLS_AGG_F_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SETS_AGG_L_GRID = [50, 60, 70, 80, 90, 100, 125]

LAMBDA_GRID = [0.5, 1.0, 2.0, 4.0, 8.0]  # balls-agg/sets-agg calibration (8.4), grid not continuous

LCB_Z = 1.645  # one-sided 95% lower confidence bound

#-------------------------------------------------------------------------------
# Section 7/10: alpha and T_eval

ALPHA_GRID = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]  # soft-variant only, per estimator
T_EVAL_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
TAU_GRID_SIZE = 201  # np.linspace(0, 1, TAU_GRID_SIZE) candidate decision thresholds
