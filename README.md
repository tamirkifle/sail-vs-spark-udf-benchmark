# Sail vs Spark — AI Workload Benchmark Suite

**Purpose.** Prove, with reproducible measurements on real AI workloads, that
**Sail** eliminates the data-movement bottleneck that dominates AI pipelines in
**Spark**. As model size and GPU speed increase, Spark's fixed IPC overhead
becomes a larger fraction of total time — Sail's zero-copy Arrow path makes
the overhead vanish.

## Core hypothesis

> AI pipelines are not compute-bound in Spark — they are data-movement bound.
> As GPUs get faster, Spark becomes **more** inefficient, while Sail scales.

The benchmark quantifies this with 5 workloads × 4 execution configurations
on the same data and same models, differing **only** in the runtime/UDF path.

## Workload matrix (5 × 4 = 20 runs per hardware tier)

| Code | Workload                    | Pattern                                                      |
| ---- | --------------------------- | ------------------------------------------------------------ |
| W0   | Chained-UDF baseline        | `input → UDF1 → UDF2 → UDF3 → output` (depth 1, 2, 3)        |
| W1   | Best-of-N LLM (PRIMARY)     | `prompt → N candidates → score → argmax`                     |
| W2   | Batched LLM inference       | `batch(prompts) → generate`                                  |
| W3   | Embedding pipeline (RAG)    | `text → embedding → similarity score`                        |
| W4   | Agentic refinement loop     | `prompt → generate/score loop → stop on threshold`           |

| Code | Engine | Execution path          | Serialization cost expected |
| ---- | ------ | ----------------------- | --------------------------- |
| A    | Spark  | Row UDF (cloudpickle)   | Highest                     |
| B    | Spark  | Pandas UDF (Arrow IPC)  | Medium                      |
| C    | Sail   | `mapInArrow` zero-copy  | ≈ 0                         |
| D    | Sail   | UDTF `LATERAL`          | ≈ 0                         |

## Hardware tiers

**Milestone 1 — Laptop (CPU / Apple Silicon MPS).** Dataset = 100 rows. Small
models. Proves wiring works and shapes the expected overhead curve before the
GPU run.

**Milestone 2 — GPU (H200 141 GB).** Dataset = 10 000 rows. FP8 large models.
Real numbers for the write-up.

## Quickstart

```bash
# one-time: install the package into the sail venv
source /Users/tamir/Documents/MyCode/LakeSail/sail/.venvs/default/bin/activate
pip install -e .

# install environment-specific runtime deps as needed
# for example: pytest pyarrow pyspark pyyaml sentence-transformers transformers

# 1) prep dataset
python scripts/prep_dataset.py --config config/laptop.yaml

# 2) run one config (fastest smoke test)
python -m sail_vs_spark.runner.cli --config config/laptop.yaml \
       --workload w0 --execution A --depth 1

# 3) run ALL 16 runs on laptop
bash scripts/run_all_laptop.sh

# 4) aggregate + plot
python analysis/aggregate_results.py --results_dir results/
python analysis/plot_gpu_timeline.py --results_dir results/
```

## Key technical constraints (from prior learnings)

1. **In-process inference only.** Models load inside the Python worker.
   No HTTP, no Ollama, no separate servers.
2. **Lazy `torch` imports** inside UDF closures — PySpark cloudpickles every
   closure even under Sail (because it flows over Spark Connect gRPC).
3. **Sail UDTF syntax.** Use `LATERAL fn(col1, col2)`, NOT
   `LATERAL TABLE(fn(...))` and NOT `useArrow=True` in `@udtf`.
4. `BoundaryTimer` implements `__getstate__`/`__setstate__` to survive cloudpickle
   (the `threading.Lock` is dropped and rebuilt on the worker side).

## Layout

See `config/`, `src/sail_vs_spark/`, `scripts/`, `analysis/`, `tests/`.

## Measured metrics per run

- **Boundary timing** (phase breakdown per batch, `results/*_boundary.json`):
  `DATA_TRANSFER_IN`, `MODEL_LOAD`, `TOKENIZE`, `INFERENCE`, `SCORE`,
  `DETOKENIZE`, `DATA_TRANSFER_OUT`.
- **System telemetry** (`results/*_stats.json`):
  wall-clock, CPU %, RSS, peak RAM, GPU util/mem, cumulative bytes written,
  `nvidia-smi dmon` log.
- **Derived** (`results/*_summary.json`):
  `serialization_tax_pct = (transfer_in + transfer_out)/wall`,
  `pipeline_continuity = gpu_active_time / wall`.

## Visualisations produced

1. GPU utilisation timeline (Spark sawtooth vs Sail flatline).
2. Runtime vs pipeline depth (W0).
3. Host RSS over time.
4. Cumulative disk writes over time.
5. Serialization-vs-compute pie per config.
