# ⛵ Sail vs 🎇 Spark: Explorer Cluster Guide

This folder contains scripts for running benchmarks on the Northeastern Explorer cluster.

## 1. Quick Start

### Step 1: Sync code to the cluster
From your **local machine**, run:
```bash
./slurm/sync.sh push
```

### Step 2: Prepare the environment (Login Node)
SSH into the cluster and run the prep script. This will download models, build Sail, and setup venvs while you still have internet access:
```bash
ssh explorer
cd /scratch/yirga.t/sail_vs_spark_benchmark
bash slurm/prep_download.sh
bash slurm/prep_install.sh
```

### Step 3: Run Benchmarks (Compute Nodes)
Now you can submit the jobs to the offline compute nodes:
```bash
# For laptop-scale CPU benchmarks (short partition)
sbatch slurm/submit_cpu.sh

# For CPU real-model benchmarks with managed CPU vLLM
sbatch slurm/submit_cpu_real_models.sh

# For full-scale GPU benchmarks (H200)
sbatch slurm/submit_gpu.sh

# For a quick GPU verification (e.g. on V100)
sbatch slurm/submit_gpu_smoke.sh
```

### Step 4: Retrieve Results
Once the jobs are finished, run this on your **local machine**:
```bash
./slurm/sync.sh pull
```

## 2. Script Details

- **`sync.sh`**: Handles `rsync` operations. It automatically excludes local virtual environments and previous results to keep the transfer fast.
- **`submit_cpu.sh`**: Uses `.venv`, builds Sail v0.6.0, and runs the mock-model CPU benchmark matrix.
- **`submit_cpu_real_models.sh`**: Uses `.venv`, starts Sail and CPU vLLM through `run_benchmark.sh --mode cpu_real`, and runs the real-model CPU smoke matrix.
- **`submit_gpu.sh`**: Similar to CPU, but requests an H200 GPU and installs heavy AI dependencies (torch, vllm, etc.) before running.
- **`submit_gpu_smoke.sh`**: A faster version that requests any GPU (e.g. V100) and runs a tiny 100-row benchmark using a 0.5B parameter model. Use this to verify that the CUDA code path and dependencies are correct without waiting for an H200.

## 3. Environment
All scripts are configured to use:
- **Project Scratch**: `/scratch/yirga.t/sail_vs_spark_benchmark`
- **Local Model Cache**: `.cache/huggingface/` (within the project folder)
- **Host Alias**: `explorer` (assumes you have this configured in your `~/.ssh/config`)
