VENV = venv
PYTHON = $(VENV)/bin/python
PY = PYTHONPATH=. $(PYTHON)
AH = simulations/approximate_hits

# Run directory used by the evaluation targets (written by populate-*).
RUN ?= $(AH)/results/esci_score20k
STREAM ?= shift
METHOD ?= triggered

SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c

.PHONY: all help
.PHONY: sift-prepare esci-prepare
.PHONY: populate-esci populate-sift eval-core eval-final eval-shell
.PHONY: online-prepare online-run online-summary figures
.PHONY: test test-utils test-dataloaders test-generation test-cache test-approximate
.PHONY: clobber clobber-sift clobber-esci

all: help

help:
	@echo "Data:      sift-prepare | esci-prepare"
	@echo "Populate:  populate-esci | populate-sift      (writes results.json into RUN)"
	@echo "Evaluate:  eval-core | eval-final | eval-shell   (RUN=<run dir>)"
	@echo "Online:    online-prepare | online-run METHOD=<static|periodic|triggered|lru|aker|qvcache> | online-summary"
	@echo "Figures:   figures"
	@echo "Tests:     test | test-utils | test-cache | test-approximate | test-dataloaders | test-generation"
	@echo "Cleanup:   clobber | clobber-sift | clobber-esci"
	@echo "Retired exact-caching pipelines live in stale/ (see stale/README.md)."

# -- Data -------------------------------------------------------------------

sift-prepare:
	$(PY) datasets/sift/prepare_benchmark.py --all

esci-prepare:
	$(PY) datasets/esci/prepare_benchmark.py --all

# -- Populate and evaluate --------------------------------------------------

populate-esci:
	$(PY) $(AH)/run_e2e_oracle_sift.py --dataset esci --out $(RUN)

populate-sift:
	$(PY) $(AH)/run_e2e_oracle_sift.py --dataset sift --out $(RUN)

eval-core:
	$(PY) $(AH)/eval_core.py $(RUN)

eval-final:
	$(PY) $(AH)/eval_final.py $(RUN)

eval-shell:
	$(PY) $(AH)/eval_simple.py $(RUN)
	$(PY) $(AH)/eval_shell_ablation.py $(RUN)

# -- Online simulation ------------------------------------------------------

online-prepare:
	$(PY) $(AH)/online_prepare.py

online-run:
	$(PY) $(AH)/online_sim.py --method $(METHOD) --stream $(STREAM)

online-summary:
	$(PY) $(AH)/online_summary.py $(AH)/results/online/esci1m/sim_$(STREAM)_b0.1

# -- Figures ----------------------------------------------------------------

figures:
	$(PY) $(AH)/make_summary_figures.py

# -- Tests ------------------------------------------------------------------

test:
	$(PYTHON) -m pytest tests/ -q

test-utils:
	$(PYTHON) -m pytest tests/utils/ tests/base/ -q

test-cache:
	$(PYTHON) -m pytest tests/cache/ -q

test-approximate:
	$(PYTHON) -m pytest tests/approximate_hits/ -q

test-dataloaders:
	$(PYTHON) -m pytest tests/dataloaders/ -q

test-generation:
	$(PYTHON) -m pytest tests/generation/ -q

# -- Cleanup ----------------------------------------------------------------

clobber: clobber-sift clobber-esci
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

clobber-sift:
	find datasets/sift -maxdepth 1 -type d -name "sift_b*" -exec rm -rf {} +

clobber-esci:
	find datasets/esci -maxdepth 1 -type d -name "esci_c*" -exec rm -rf {} +
