# Project configuration
SAIL_REPO_DIR ?= third_party/sail
VENV ?= .venv
PY    = $(VENV)/bin/python
PIP   = $(VENV)/bin/pip

.PHONY: help install setup-sail test test-fast prep-laptop run-laptop run-w0-all plot clean

help:
	@echo "sail-vs-spark benchmark — make targets"
	@echo "  install         install this package + deps into local venv"
	@echo "  setup-sail      clone sail, checkout v0.6.0, and build pysail into local venv"
	@echo "  test            run the full test suite"
	@echo "  test-fast       run only fast tests (skip slow)"
	@echo "  prep-laptop     prep UltraFeedback laptop split (100 rows)"
	@echo "  run-laptop      run all 4 workloads × 4 configs on laptop"
	@echo "  run-w0-all      run just W0 depth-1/2/3 × 4 configs"
	@echo "  plot            aggregate + draw all charts"

$(VENV):
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel maturin

install: $(VENV)
	$(PIP) install -e .
	$(PIP) install -r requirements.txt

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

prep-laptop:
	cd $(CURDIR) && $(PY) scripts/prep_dataset.py --config config/laptop.yaml

run-laptop:
	cd $(CURDIR) && bash scripts/run_all_laptop.sh

run-laptop-live:
	cd $(CURDIR) && bash scripts/run_all_laptop_live.sh

run-w0-all:
	cd $(CURDIR) && bash scripts/run_w0_all.sh

plot:
	cd $(CURDIR) && $(PY) analysis/aggregate_results.py --results_dir results/
	cd $(CURDIR) && $(PY) analysis/plot_depth_runtime.py --results_dir results/
	cd $(CURDIR) && $(PY) analysis/plot_serialization.py --results_dir results/
	cd $(CURDIR) && $(PY) analysis/plot_memory.py --results_dir results/
	cd $(CURDIR) && $(PY) analysis/plot_disk_io.py --results_dir results/
	cd $(CURDIR) && $(PY) analysis/plot_gpu_timeline.py --results_dir results/

clean:
	rm -rf results/*.json results/*.parquet results/*.png
	rm -rf /tmp/nvidia_dmon_*.log
	rm -rf /tmp/nvidia_dmon_*.log
