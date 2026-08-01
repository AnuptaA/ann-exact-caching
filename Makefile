VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
# The population sweep is memory-bound, not CPU-bound: a single work item's
# propose_candidates stage alone was measured peaking at ~2.3GB (see
# docs/POPULATION_DESIGN.md), and this is an 8GB machine -- 3+ simultaneous
# workers risks exceeding available RAM and thrashing (slower, not faster),
# even with other applications closed. 2 is the configuration actually
# measured giving a real ~1.38x speedup; it is not a guess or the CPU count
# (8 cores are available, but irrelevant here).
WORKERS ?= 2

SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c
LOGDIR = logs
RUN_LOG_TS := $(shell date +%Y%m%d_%H%M%S)

.PHONY: all help
.PHONY: synthetic-generate synthetic-run synthetic-analyze
.PHONY: sift-prepare sift-hit-policy-run sift-hit-policy-analyze
.PHONY: sift-population-prepare
.PHONY: esci-prepare esci-hit-policy-run esci-hit-policy-analyze
.PHONY: esci-population-prepare
.PHONY: microbenchmark
.PHONY: population
.PHONY: esci-population-ball sift-population-ball
.PHONY: population-proposal-analyze population-proposal
.PHONY: esci-population-ball-exact sift-population-ball-exact
.PHONY: population-proposal-exact
.PHONY: debug-ball-geometry debug-trajectory debug-sigma-sensitivity debug-distance-concentration debug-vector-coverage debug-population
.PHONY: analyze-hit-policy
.PHONY: dataset-characterize full-pipeline
.PHONY: test test-utils test-synthetic test-sift test-esci
.PHONY: test-hit-policy test-population test-containment test-proposal
.PHONY: test-dataloaders test-generation test-sigma
.PHONY: count-datasets
.PHONY: clobber clobber-synthetic clobber-sift clobber-esci clean-cache

all: help

help:
	@echo "Synthetic:    synthetic-generate | synthetic-run | synthetic-analyze"
	@echo "SIFT hit-policy: sift-prepare | sift-hit-policy-run | sift-hit-policy-analyze"
	@echo "ESCI hit-policy: esci-prepare | esci-hit-policy-run | esci-hit-policy-analyze"
	@echo "Population:   population  (microbenchmark + collected results, both datasets)"
	@echo "              microbenchmark        (containment microbenchmark, both datasets)"
	@echo "              {sift,esci}-population-prepare"
	@echo "              population-proposal"
	@echo "              population-proposal-exact  (sigma=1.0 only)"
	@echo "              {esci,sift}-population-ball[-exact]"
	@echo "              debug-population       (ball-geometry + trajectory + sigma-sensitivity + distance-concentration diagnostics)"
	@echo "              debug-ball-geometry | debug-trajectory | debug-sigma-sensitivity | debug-distance-concentration"
	@echo "              debug-vector-coverage  (standalone: coverage ceiling vs actual hit rate, sigma=1.0 only)"
	@echo "Characterize: dataset-characterize  (query-clustering ratio benchmark)"
	@echo "              full-pipeline         (dataset-characterize then population)"
	@echo "Tests:        test  (full suite)  |  test-utils | test-synthetic | test-sift | test-esci"
	@echo "              test-hit-policy | test-population | test-containment | test-proposal"
	@echo "              test-dataloaders | test-generation | test-sigma"
	@echo "Cleanup:      clobber[-synthetic|-sift|-esci]"

# ── Synthetic workflow ────────────────────────────────────────────────────────

synthetic-generate:
	$(PYTHON) datasets/synthetic/generator.py --test-set all

synthetic-run:
	$(PYTHON) simulations/synthetic/run_synthetic.py

synthetic-analyze:
	$(PYTHON) simulations/synthetic/analyze_synthetic.py
	$(PYTHON) simulations/pdf/compile.py --dataset synthetic

# ── SIFT hit-policy workflow ──────────────────────────────────────────────────

sift-prepare:
	$(PYTHON) datasets/sift/prepare_benchmark.py --all

sift-hit-policy-run:
	$(PYTHON) simulations/hit_policy/run_hit_policy.py --dataset sift

sift-hit-policy-analyze:
	$(PYTHON) simulations/hit_policy/analyze_hit_policy.py --dataset sift
	$(PYTHON) simulations/pdf/compile.py --dataset sift

# ── SIFT population workflow ──────────────────────────────────────────────────

sift-population-prepare:
	$(PYTHON) datasets/sift/prepare_benchmark.py --all
	$(PYTHON) datasets/sift/prepare_benchmark.py --extend-population

# ── ESCI hit-policy workflow ──────────────────────────────────────────────────

esci-prepare:
	$(PYTHON) datasets/esci/prepare_benchmark.py --all

esci-hit-policy-run:
	$(PYTHON) simulations/hit_policy/run_hit_policy.py --dataset esci

esci-hit-policy-analyze:
	$(PYTHON) simulations/hit_policy/analyze_hit_policy.py --dataset esci
	$(PYTHON) simulations/pdf/compile.py --dataset esci

# ── ESCI population workflow ──────────────────────────────────────────────────

esci-population-prepare:
	$(PYTHON) datasets/esci/prepare_benchmark.py --preset population
	$(PYTHON) datasets/esci/prepare_benchmark.py --extend-population

# ── Microbenchmark ────────────────────────────────────────────────────────────

microbenchmark: sift-population-prepare esci-population-prepare
	@mkdir -p $(LOGDIR)
	set -o pipefail; $(PYTHON) simulations/population/microbenchmarks/microbenchmark_containment.py --dataset sift 2>&1 | tee $(LOGDIR)/microbenchmark-sift_$(RUN_LOG_TS).log
	set -o pipefail; $(PYTHON) simulations/population/microbenchmarks/microbenchmark_containment.py --dataset esci 2>&1 | tee $(LOGDIR)/microbenchmark-esci_$(RUN_LOG_TS).log

# ── Combined population ───────────────────────────────────────────────────────

population: microbenchmark
	$(PYTHON) simulations/pdf/collect_results_population.py

# ── Ball-like population runs ──────────────────────────────────────────────────

SIGMA_VALUES ?= 0.70 0.80 0.90 1.00

esci-population-ball: esci-population-prepare
	@mkdir -p $(LOGDIR)
	set -o pipefail; $(PYTHON) simulations/population/run_population.py --dataset esci --workers $(WORKERS) --sigma $(SIGMA_VALUES) 2>&1 | tee $(LOGDIR)/esci-population-ball_$(RUN_LOG_TS).log

sift-population-ball: sift-population-prepare
	@mkdir -p $(LOGDIR)
	set -o pipefail; $(PYTHON) simulations/population/run_population.py --dataset sift --workers $(WORKERS) --sigma $(SIGMA_VALUES) 2>&1 | tee $(LOGDIR)/sift-population-ball_$(RUN_LOG_TS).log

population-proposal-analyze: debug-distance-concentration
	@mkdir -p $(LOGDIR)
	set -o pipefail; $(PYTHON) simulations/population/analyze_population_proposal.py 2>&1 | tee $(LOGDIR)/population-proposal-analyze_$(RUN_LOG_TS).log
	$(PYTHON) simulations/pdf/compile_proposal.py

population-proposal: esci-population-ball sift-population-ball \
                     debug-ball-geometry debug-trajectory \
                     population-proposal-analyze

# ── Exact (sigma=1.0, no sweep) population runs ───────────────────────────────

esci-population-ball-exact: esci-population-prepare
	$(PYTHON) simulations/population/run_population.py --dataset esci --workers $(WORKERS) --sigma 1.0

sift-population-ball-exact: sift-population-prepare
	$(PYTHON) simulations/population/run_population.py --dataset sift --workers $(WORKERS) --sigma 1.0

population-proposal-exact: esci-population-ball-exact sift-population-ball-exact \
                           population-proposal-analyze

# ── Population debug diagnostics (Phase 3 Round 14) ───────────────────────────
# Standalone: call propose_candidates_ball/select_candidates directly against
# already-prepared benchmark data, no changes to population_runner.py/
# selection.py. See docs/POPULATION_DESIGN.md.

debug-ball-geometry:
	@mkdir -p $(LOGDIR)
	set -o pipefail; $(PYTHON) simulations/population/debug_ball_geometry.py 2>&1 | tee $(LOGDIR)/debug-ball-geometry_$(RUN_LOG_TS).log

debug-trajectory:
	@mkdir -p $(LOGDIR)
	set -o pipefail; $(PYTHON) simulations/population/debug_trajectory.py 2>&1 | tee $(LOGDIR)/debug-trajectory_$(RUN_LOG_TS).log

debug-sigma-sensitivity:
	@mkdir -p $(LOGDIR)
	set -o pipefail; $(PYTHON) simulations/population/debug_sigma_sensitivity.py 2>&1 | tee $(LOGDIR)/debug-sigma-sensitivity_$(RUN_LOG_TS).log

debug-distance-concentration:
	@mkdir -p $(LOGDIR)
	set -o pipefail; $(PYTHON) simulations/population/debug_distance_concentration.py 2>&1 | tee $(LOGDIR)/debug-distance-concentration_$(RUN_LOG_TS).log

debug-vector-coverage:
	@mkdir -p $(LOGDIR)
	set -o pipefail; $(PYTHON) simulations/population/debug_vector_coverage.py 2>&1 | tee $(LOGDIR)/debug-vector-coverage_$(RUN_LOG_TS).log

debug-population: debug-ball-geometry debug-trajectory debug-sigma-sensitivity debug-distance-concentration debug-vector-coverage

# ── Combined hit-policy analysis ──────────────────────────────────────────────

analyze-hit-policy: synthetic-analyze sift-hit-policy-analyze esci-hit-policy-analyze
	$(PYTHON) simulations/pdf/collect_results.py

# ── Characterization ──────────────────────────────────────────────────────────

dataset-characterize:
	$(PYTHON) simulations/analysis/query_clustering_ratio.py

full-pipeline: dataset-characterize population

# ── Tests ─────────────────────────────────────────────────────────────────────

test:
	$(PYTHON) -m pytest tests/ -q

test-utils:
	$(PYTHON) -m pytest tests/utils/ -q

test-synthetic:
	$(PYTHON) -m pytest tests/synthetic/ -q

test-sift:
	$(PYTHON) -m pytest tests/ -k "sift" -q

test-esci:
	$(PYTHON) -m pytest tests/ -k "esci" -q

test-hit-policy:
	$(PYTHON) -m pytest tests/hit_policy/ -q

test-population:
	$(PYTHON) -m pytest tests/population/ -q

test-containment:
	$(PYTHON) -m pytest tests/population/microbenchmarks/test_microbenchmark_containment.py -q

test-proposal:
	$(PYTHON) -m pytest tests/utils/test_proposals.py tests/utils/test_selection_with_new_candidates.py \
		tests/population/test_e2e_population_proposal.py tests/population/test_analyze_proposal.py -q

test-dataloaders:
	$(PYTHON) -m pytest tests/dataloaders/ -q

test-generation:
	$(PYTHON) -m pytest tests/generation/ -q

test-sigma:
	$(PYTHON) -m pytest tests/ -k "sigma" -q

# ── Utilities ─────────────────────────────────────────────────────────────────

count-datasets:
	@ls -1d datasets/synthetic/data/set*_* 2>/dev/null | wc -l

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean-cache:
	find datasets/synthetic/data -name "ground_truth_cache" -type d -exec rm -rf {} + 2>/dev/null || true

clobber: clobber-synthetic clobber-sift clobber-esci
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

clobber-synthetic: clean-cache
	rm -rf datasets/synthetic/data/*

clobber-sift:
	find datasets/sift -maxdepth 1 -type d -name "sift_b*" -exec rm -rf {} +

clobber-esci:
	find datasets/esci -maxdepth 1 -type d -name "esci_c*" -exec rm -rf {} +
