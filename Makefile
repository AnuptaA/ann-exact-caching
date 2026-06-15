VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

K ?=
_K_FLAG = $(if $(K),--K $(K),)

.PHONY: all help
.PHONY: synthetic-generate synthetic-run synthetic-analyze
.PHONY: sift-prepare sift-run sift-analyze
.PHONY: sift-population-prepare sift-population-run sift-population-run-all sift-population-analyze sift-population
.PHONY: sift-population-run-K sift-population-run-all-K
.PHONY: sift-alpha-microbenchmark sift-k-microbenchmark
.PHONY: esci-prepare esci-run esci-analyze
.PHONY: esci-population-prepare esci-population-run esci-population-run-all esci-population-analyze esci-population
.PHONY: esci-population-run-K esci-population-run-all-K
.PHONY: esci-alpha-microbenchmark esci-k-microbenchmark
.PHONY: population
.PHONY: characterize
.PHONY: test \
        test-utils test-distance-metrics test-kv-cache test-lemmas test-main-memory \
        test-perturbation test-simulate \
        test-range-search test-population test-selection \
        test-synthetic test-generator test-e2e-synthetic test-analysis \
        test-sift test-sift-edge-cases test-sift-e2e test-dataloaders \
        test-sift-population-e2e test-sift-microbenchmark test-sift-microbenchmark-k \
        test-esci test-esci-edge-cases test-esci-e2e \
        test-esci-population-e2e test-esci-microbenchmark test-esci-microbenchmark-k \
        test-population-all
.PHONY: count-datasets
.PHONY: clobber clobber-synthetic clobber-sift clobber-esci clean-cache

all: help

help:
	@echo "Synthetic:  synthetic-generate | synthetic-run | synthetic-analyze"
	@echo "SIFT:       sift-prepare | sift-run | sift-analyze"
	@echo "Population: sift-population-prepare | sift-population-run | sift-population-analyze"
	@echo "            sift-population  (all three in sequence)"
	@echo "            sift-population-run K=<n>  (quick run with K override)"
	@echo "            sift-population-run-K      (quick run for all K=[1,5,10,20])"
	@echo "            sift-population-run-all-K  (full run for all K=[1,5,10,20])"
	@echo "ESCI:       esci-prepare | esci-run | esci-analyze"
	@echo "            esci-population-prepare | esci-population-run | esci-population-analyze"
	@echo "            esci-population  (all three in sequence)"
	@echo "            esci-population-run K=<n>  (quick run with K override)"
	@echo "            esci-population-run-K      (quick run for all K=[1,5,10,20])"
	@echo "            esci-population-run-all-K  (full run for all K=[1,5,10,20])"
	@echo "Combined:   population  (sift-population + esci-population)"
	@echo "Microbenchmark: sift-alpha-microbenchmark | sift-k-microbenchmark"
	@echo "                esci-alpha-microbenchmark | esci-k-microbenchmark"
	@echo "All:        analyze  (runs all workloads + collects results)"
	@echo "Characterize: characterize (all workloads, cross-workload plots)"
	@echo "Tests:      test | test-utils | test-synthetic | test-sift | test-esci"
	@echo "            test-population-all  (all population tests, both datasets)"
	@echo "            utils:     test-distance-metrics test-kv-cache test-lemmas"
	@echo "                       test-main-memory test-perturbation test-simulate"
	@echo "                       test-range-search test-population test-selection"
	@echo "            synthetic: test-generator test-e2e-synthetic test-analysis"
	@echo "            sift:      test-sift-edge-cases test-sift-e2e test-dataloaders"
	@echo "                       test-sift-population-e2e"
	@echo "                       test-sift-microbenchmark test-sift-microbenchmark-k"
	@echo "            esci:      test-esci-edge-cases test-esci-e2e"
	@echo "                       test-esci-population-e2e"
	@echo "                       test-esci-microbenchmark test-esci-microbenchmark-k"
	@echo "Utilities:  count-datasets"
	@echo "Cleanup:    clobber[-synthetic|-sift|-esci]"

# ── Synthetic workflow ────────────────────────────────────────────────────────

synthetic-generate:
	$(PYTHON) datasets/synthetic/generator.py --test-set all

synthetic-run:
	$(PYTHON) simulations/synthetic/run_synthetic.py

synthetic-analyze:
	$(PYTHON) simulations/synthetic/analyze_synthetic.py
	$(PYTHON) simulations/synthetic/compile_pdf.py

# ── SIFT workflow ─────────────────────────────────────────────────────────────

sift-prepare:
	$(PYTHON) datasets/sift/prepare_benchmark.py --all

sift-run:
	$(PYTHON) simulations/sift/run_sift.py

sift-analyze:
	$(PYTHON) simulations/sift/analyze_sift.py
	$(PYTHON) simulations/sift/compile_pdf.py

# ── SIFT population workflow ──────────────────────────────────────────────────

datasets/sift/.population_prepared:
	$(PYTHON) datasets/sift/prepare_benchmark.py --all
	$(PYTHON) datasets/sift/prepare_benchmark.py --extend-population
	touch $@

sift-population-prepare: datasets/sift/.population_prepared

sift-population-run: datasets/sift/.population_prepared
	$(PYTHON) simulations/sift/run_population_sift.py --quick $(_K_FLAG)

sift-population-run-all: datasets/sift/.population_prepared
	$(PYTHON) simulations/sift/run_population_sift.py $(_K_FLAG)

sift-population-run-K: datasets/sift/.population_prepared
	for k in 1 5 10 20; do \
		$(PYTHON) simulations/sift/run_population_sift.py --quick --K $$k; \
	done

sift-population-run-all-K: datasets/sift/.population_prepared
	for k in 1 5 10 20; do \
		$(PYTHON) simulations/sift/run_population_sift.py --K $$k; \
	done

sift-population-analyze:
	$(PYTHON) simulations/sift/analyze_population_sift.py

sift-population: sift-population-prepare sift-population-run sift-population-analyze

sift-alpha-microbenchmark: datasets/sift/.population_prepared
	$(PYTHON) simulations/sift/microbenchmark_candidates.py

sift-k-microbenchmark: datasets/sift/.population_prepared
	$(PYTHON) simulations/sift/microbenchmark_k.py

# ── ESCI workflow ─────────────────────────────────────────────────────────────

esci-prepare:
	$(PYTHON) datasets/esci/prepare_benchmark.py --all

esci-run:
	$(PYTHON) simulations/esci/run_esci.py

esci-analyze:
	$(PYTHON) simulations/esci/analyze_esci.py
	$(PYTHON) simulations/esci/compile_pdf.py

# ── ESCI population workflow ──────────────────────────────────────────────────

datasets/esci/.population_prepared:
	$(PYTHON) datasets/esci/prepare_benchmark.py --all
	$(PYTHON) datasets/esci/prepare_benchmark.py --extend-population
	touch $@

esci-population-prepare: datasets/esci/.population_prepared

esci-population-run: datasets/esci/.population_prepared
	$(PYTHON) simulations/esci/run_population_esci.py --quick $(_K_FLAG)

esci-population-run-all: datasets/esci/.population_prepared
	$(PYTHON) simulations/esci/run_population_esci.py $(_K_FLAG)

esci-population-run-K: datasets/esci/.population_prepared
	for k in 1 5 10 20; do \
		$(PYTHON) simulations/esci/run_population_esci.py --quick --K $$k; \
	done

esci-population-run-all-K: datasets/esci/.population_prepared
	for k in 1 5 10 20; do \
		$(PYTHON) simulations/esci/run_population_esci.py --K $$k; \
	done

esci-population-analyze:
	$(PYTHON) simulations/esci/analyze_population_esci.py

esci-population: esci-population-prepare esci-population-run esci-population-analyze

esci-alpha-microbenchmark: datasets/esci/.population_prepared
	$(PYTHON) simulations/esci/microbenchmark_candidates.py

esci-k-microbenchmark: datasets/esci/.population_prepared
	$(PYTHON) simulations/esci/microbenchmark_k.py

# ── Combined population ───────────────────────────────────────────────────────

population: sift-population esci-population

# ── Combined analysis ─────────────────────────────────────────────────────────

analyze: synthetic-analyze sift-analyze esci-analyze
	$(PYTHON) simulations/collect_results.py

# ── Characterization ──────────────────────────────────────────────────────────

characterize:
	$(PYTHON) simulations/characterize.py

# ── Tests ─────────────────────────────────────────────────────────────────────

test: test-utils test-synthetic test-sift test-esci

# utils tests
test-utils: test-distance-metrics test-kv-cache test-lemmas test-main-memory \
            test-perturbation test-simulate \
            test-range-search test-population test-selection

test-distance-metrics:
	$(PYTHON) tests/utils/test_distance_metrics.py

test-kv-cache:
	$(PYTHON) tests/utils/test_kv_cache.py

test-lemmas:
	$(PYTHON) tests/utils/test_lemmas.py

test-main-memory:
	$(PYTHON) tests/utils/test_main_memory.py

test-perturbation:
	$(PYTHON) tests/utils/test_perturbation.py

test-simulate:
	$(PYTHON) tests/utils/test_simulate.py

test-range-search:
	$(PYTHON) tests/utils/test_range_search.py

test-population:
	$(PYTHON) tests/utils/test_population.py

test-selection:
	$(PYTHON) tests/utils/test_selection.py

# synthetic tests
test-synthetic: test-generator test-e2e-synthetic test-analysis

test-generator:
	$(PYTHON) tests/synthetic/test_generator.py

test-e2e-synthetic:
	$(PYTHON) tests/synthetic/test_e2e_mini.py

test-analysis:
	$(PYTHON) tests/synthetic/test_analysis.py

# sift tests
test-sift: test-sift-edge-cases test-sift-e2e test-dataloaders test-sift-population-e2e test-sift-microbenchmark test-sift-microbenchmark-k

test-sift-edge-cases:
	$(PYTHON) tests/sift/test_edge_cases.py

test-sift-e2e:
	$(PYTHON) tests/sift/test_e2e_mini.py

test-dataloaders:
	$(PYTHON) tests/sift/test_dataloaders.py

test-sift-population-e2e:
	$(PYTHON) tests/sift/test_e2e_population.py

test-sift-microbenchmark:
	$(PYTHON) tests/sift/test_microbenchmark.py

test-sift-microbenchmark-k:
	$(PYTHON) tests/sift/test_microbenchmark_k.py

# esci tests
test-esci: test-esci-edge-cases test-esci-e2e test-esci-population-e2e test-esci-microbenchmark test-esci-microbenchmark-k

test-esci-edge-cases:
	$(PYTHON) tests/esci/test_edge_cases.py

test-esci-e2e:
	$(PYTHON) tests/esci/test_e2e_mini.py

test-esci-population-e2e:
	$(PYTHON) tests/esci/test_e2e_population.py

test-esci-microbenchmark:
	$(PYTHON) tests/esci/test_microbenchmark.py

test-esci-microbenchmark-k:
	$(PYTHON) tests/esci/test_microbenchmark_k.py

# population tests (both datasets)
test-population-all: test-sift-population-e2e test-sift-microbenchmark test-sift-microbenchmark-k \
                     test-esci-population-e2e test-esci-microbenchmark test-esci-microbenchmark-k

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
