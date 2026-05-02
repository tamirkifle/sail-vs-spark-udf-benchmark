# Project configuration
SAIL_REPO_DIR ?= third_party/sail
VENV ?= .venv
PY    = $(VENV)/bin/python
PIP   = $(VENV)/bin/pip
RESULTS_DIR ?= results/cpu

.PHONY: help install setup-mock setup-cpu setup-gpu setup-sail test test-fast prep-mock prep-cpu run-mock run-cpu run-gpu run-laptop run-w0-all plot clean

help:
	@echo "sail-vs-spark benchmark — make targets"
	@echo "  install         install this package + deps into local venv"
	@echo "  setup-mock      create/update .venv for mock smoke runs"
	@echo "  setup-cpu       create/update .venv with CPU model deps"
	@echo "  setup-gpu       create/update .venv_gpu with vLLM deps"
	@echo "  setup-sail      clone sail, checkout v0.6.0, and build pysail into local venv"
	@echo "  test            run the full test suite"
	@echo "  test-fast       run only fast tests (skip slow)"
	@echo "  run-mock        run fast mock smoke benchmark and write report"
	@echo "  run-cpu         run CPU/macOS benchmark and write report"
	@echo "  run-gpu         run GPU/vLLM benchmark and write report"
	@echo "  run-laptop      compatibility alias for run-cpu"
	@echo "  run-w0-all      run just W0 depth-1/2/3 × 4 configs"
	@echo "  plot            aggregate + draw all charts"

$(VENV):
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel maturin

install: $(VENV)
	$(PIP) install -e .
	$(PIP) install -r requirements.txt

setup-mock:
	cd $(CURDIR) && scripts/setup_env.sh --mode mock --venv $(VENV)

setup-cpu:
	cd $(CURDIR) && scripts/setup_env.sh --mode cpu --venv $(VENV)

setup-gpu:
	cd $(CURDIR) && scripts/setup_env.sh --mode gpu --venv .venv_gpu

setup-sail: $(VENV)
	@if [ ! -d "$(SAIL_REPO_DIR)" ]; then \
		echo "Cloning Sail into $(SAIL_REPO_DIR)..."; \
		git clone https://github.com/lakehq/sail.git $(SAIL_REPO_DIR); \
	fi
	@echo "Setting up Sail v0.6.0 in $(SAIL_REPO_DIR)"
	cd $(SAIL_REPO_DIR) && git fetch origin --tags --force
	cd $(SAIL_REPO_DIR) && git checkout v0.6.0
	cd $(SAIL_REPO_DIR) && $(abspath $(VENV)/bin/maturin) develop --release

test:
	cd $(CURDIR) && $(PY) -m pytest tests/ -v

test-fast:
	cd $(CURDIR) && $(PY) -m pytest tests/ -v -m "not slow"

prep-mock:
	cd $(CURDIR) && $(PY) scripts/prep_dataset.py --config config/mock.yaml --force-synthetic

prep-cpu:
	cd $(CURDIR) && $(PY) scripts/prep_dataset.py --config config/cpu.yaml

prep-laptop: prep-cpu

run-mock:
	cd $(CURDIR) && scripts/run_benchmark.sh --mode mock --venv $(VENV)

run-cpu:
	cd $(CURDIR) && scripts/run_benchmark.sh --mode cpu --venv $(VENV)

run-gpu:
	cd $(CURDIR) && scripts/run_benchmark.sh --mode gpu --venv .venv_gpu

run-laptop: run-cpu

run-w0-all:
	cd $(CURDIR) && WORKLOADS=w0 scripts/run_benchmark.sh --mode cpu --venv $(VENV)

plot:
	cd $(CURDIR) && $(PY) analysis/aggregate_results.py --results_dir $(RESULTS_DIR)

clean:
	rm -rf results/*/report results/*/*/report
	rm -rf /tmp/nvidia_dmon_*.log
