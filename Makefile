VENV ?= /Users/tamir/Documents/MyCode/LakeSail/sail/.venvs/default
PY    = $(VENV)/bin/python
PIP   = $(VENV)/bin/pip

.PHONY: help install test test-fast prep-laptop run-laptop run-w0-all plot clean

help:
	@echo "sail-vs-spark benchmark — make targets"
	@echo "  install         install this package + deps into the sail venv"
	@echo "  test            run the full test suite"
	@echo "  test-fast       run only fast tests (skip slow)"
	@echo "  prep-laptop     prep UltraFeedback laptop split (100 rows)"
	@echo "  run-laptop      run all 4 workloads × 4 configs on laptop"
	@echo "  run-w0-all      run just W0 depth-1/2/3 × 4 configs"
	@echo "  plot            aggregate + draw all charts"

install:
	$(PIP) install -e .
	$(PIP) install -r requirements.txt

test:
	cd $(CURDIR) && $(PY) -m pytest tests/ -v

test-fast:
	cd $(CURDIR) && $(PY) -m pytest tests/ -v -m "not slow"

prep-laptop:
	cd $(CURDIR) && $(PY) scripts/prep_dataset.py --config config/laptop.yaml

run-laptop:
	cd $(CURDIR) && bash scripts/run_all_laptop.sh

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
