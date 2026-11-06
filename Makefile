VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c

.PHONY: all help
.PHONY: sift-prepare esci-prepare
.PHONY: test test-utils test-dataloaders test-generation
.PHONY: clobber clobber-sift clobber-esci

all: help

help:
	@echo "Targets:"
	@echo "  Datasets:  sift-prepare | esci-prepare"
	@echo "  Tests:     test | test-utils | test-dataloaders | test-generation"
	@echo "  Cleanup:   clobber | clobber-sift | clobber-esci"
	@echo "Retired exact-caching pipelines live in stale/ (see stale/README.md)."

# -- Datasets ---------------------------------------------------------------

sift-prepare:
	$(PYTHON) datasets/sift/prepare_benchmark.py --all

esci-prepare:
	$(PYTHON) datasets/esci/prepare_benchmark.py --all

# -- Tests ------------------------------------------------------------------

test:
	$(PYTHON) -m pytest tests/ -q

test-utils:
	$(PYTHON) -m pytest tests/utils/ -q

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
