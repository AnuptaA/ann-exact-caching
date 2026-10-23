#!/usr/bin/env python3
#-------------------------------------------------------------------------------
"""Constants for the multi-table LSH radius experiments (radius_eval_tables).

Notation: K is the top-K size only (100), L the number of tables, m the bits per
table, nu the nearest labeled points taken per table, S the number of sampled
database vectors. theta(x) is the distance to the K-th nearest database vector.
MEMORY_LIMIT_MB is local; logging_utils.py does not define it."""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parents[2]))  # thesis root, for utils.base
sys.path.append(str(HERE.parent / "common"))  # config, data_prep, ground_truth, logging_utils, perturbation, plot_style

from config import K as K_TOPK

#-------------------------------------------------------------------------------

OUTPUT_ROOT = HERE.parent
RESULTS_DIR = OUTPUT_ROOT / "results"
DRY_RUN_DIR = RESULTS_DIR / "dry_run"
PLOTS_DIR = OUTPUT_ROOT / "plots"
PDF_DIR = OUTPUT_ROOT / "pdf"
LOGS_DIR = RESULTS_DIR / "logs"
THETA_CACHE_DIR = RESULTS_DIR / "theta_cache"

for _d in (RESULTS_DIR, PLOTS_DIR, LOGS_DIR, THETA_CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

MEMORY_LIMIT_MB = 6000
RSS_LIMIT_MB = 5500.0
MIN_FREE_DISK_GB = 2.0
EXIT_ABORT = 3

#-------------------------------------------------------------------------------
# Grid

DATASETS = ["sift", "esci"]
STAGES = [0, 1, 2, 3, 4]
DIM = {"sift": 128, "esci": 384}
N_DB = 50_000

BITS_MAX = 2048
BLOCK_BITS = 64
CONFIGS = [(1, 512), (4, 128), (8, 64), (1, 2048), (4, 512), (8, 256), (16, 128)]
NU_GRID = [1, 2, 4, 8, 16, 32]
S_GRID = [2500, 10_000, 50_000]
REF_S = 10_000
SEEDS = [0, 1, 2, 3, 4]
POP_KINDS = ["sample_train", "sample_only"]
# Exact union counts: every seed at REF_S, seed 0 only at the other S values.
UNION_S_ALL_SEEDS = [REF_S]

CHUNK_SIZE = 250

#-------------------------------------------------------------------------------
# Data

DATA_SEED = 42
N_TRAIN = 2500
N_TEST = 2500
N_ANCHORS = 50
MAX_ANGLE_DEG = 10.0
DUP_TOL = 1e-6  # distance at or below this counts as an identical vector

DUMP_STAGES = [3, 4]
DUMP_N_QUERIES = 1000
DUMP_NU_MAX = 32
DUMP_TIE_CAP = 64

#-------------------------------------------------------------------------------

CONFIG_COLORS = {
    "L1_m512": "#2a78d6",
    "L4_m128": "#eb6834",
    "L8_m64": "#2a9d5c",
    "L1_m2048": "#8e5bd6",
    "L4_m512": "#c9a227",
    "L8_m256": "#d6457a",
    "L16_m128": "#4a4a4a",
}

def config_name(L: int, m: int) -> str:
    return f"L{L}_m{m}"

CONFIG_NAMES = [config_name(L, m) for L, m in CONFIGS]

def config_tables(L: int, m: int) -> list:
    """Table t of a configuration uses rows [t*m, (t+1)*m) of W: returns (row_lo, m)."""
    return [(t * m, m) for t in range(L)]

def unique_tables(configs: list = None) -> list:
    seen = []
    for L, m in (configs or CONFIGS):
        for tb in config_tables(L, m):
            if tb not in seen:
                seen.append(tb)
    return seen

def stage_db_key(dataset: str, stage: int) -> str:
    """Key of the database used by a stage: synthetic U in stages 0 to 2, the real base in 3 and 4."""
    return f"U{DIM[dataset]}" if stage <= 2 else f"{dataset}_base"

def stage_kind(stage: int) -> str:
    return {0: "uniform", 1: "real", 2: "locality", 3: "real", 4: "locality"}[stage]
