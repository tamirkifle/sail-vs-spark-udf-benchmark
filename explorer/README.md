# ⛵ Sail vs 🎇 Spark: Explorer Cluster Guide

This folder contains scripts for running benchmarks on the Northeastern Explorer cluster.

## 1. Quick Start

### Step 1: Sync code to the cluster
From your **local machine**, run:
```bash
./explorer/sync.sh push
```

### Step 2: Prepare the environment (Login Node)
SSH into the cluster and run the prep script. This will download models, build Sail, and setup venvs while you still have internet access:
```bash
ssh explorer
cd /scratch/yirga.t/sail_vs_spark_benchmark
bash explorer/prep_node.sh
```

### Step 3: Run Benchmarks (Compute Nodes)
Now you can submit the jobs to the offline compute nodes:
```bash
# For laptop-scale CPU benchmarks (short partition)
sbatch explorer/submit_cpu.sh

# For full-scale GPU benchmarks (H200)
sbatch explorer/submit_gpu.sh

# For a quick GPU verification (e.g. on V100)
sbatch explorer/submit_gpu_smoke.sh
```

### Step 4: Retrieve Results
Once the jobs are finished, run this on your **local machine**:
```bash
./explorer/sync.sh pull
```

## 2. Script Details

- **`sync.sh`**: Handles `rsync` operations. It automatically excludes local virtual environments and previous results to keep the transfer fast.
- **`submit_cpu.sh`**: Creates a local `.venv` on the cluster node, builds Sail v0.6.0, and runs the "Live Mode" benchmark matrix.
- **`submit_gpu.sh`**: Similar to CPU, but requests an H200 GPU and installs heavy AI dependencies (torch, vllm, etc.) before running.
- **`submit_gpu_smoke.sh`**: A faster version that requests any GPU (e.g. V100) and runs a tiny 100-row benchmark using a 0.5B parameter model. Use this to verify that the CUDA code path and dependencies are correct without waiting for an H200.

## 3. Environment
All scripts are configured to use:
- **Project Scratch**: `/scratch/yirga.t/sail_vs_spark_benchmark`
- **Local Model Cache**: `.cache/huggingface/` (within the project folder)
- **Host Alias**: `explorer` (assumes you have this configured in your `~/.ssh/config`)
