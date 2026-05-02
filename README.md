# Sail vs Spark — AI Workload Benchmark Suite

**Purpose.** Compare how different execution paths affect performance on real
AI workloads, using reproducible measurements. The benchmark keeps the data
and models fixed while varying only the runtime/UDF path.

## Core hypothesis

> AI pipeline performance is strongly shaped by data movement and runtime
> boundary costs, not just model compute.

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

## Run Modes

Use one setup command and one run command. Every run prepares data, starts
Sail and/or vLLM when needed, executes the matrix, aggregates results, and writes an
HTML report.

| Mode | Config | Runs | Use when |
| ---- | ------ | ---- | -------- |
| `mock` | `config/mock.yaml` | Spark A/B only, synthetic prompts, all models mocked | First install check on macOS/Linux |
| `cpu` | `config/cpu.yaml` | A/B/C/D, local Sail server, HF dataset, mock models on CPU | Laptop or CPU host benchmarking |
| `cpu_real` | `config/cpu_real.yaml` | A/B/C/D, local Sail server, Transformers CPU generation, real scorer/embedder | Real-model CPU smoke benchmark |
| `gpu` | `config/gpu_h200.yaml` | A/B/C/D, local Sail server, vLLM generation | CUDA host / H200 benchmark |

## Quickstart

```bash
# 1) Install the lightest environment and run a full smoke benchmark.
scripts/setup_env.sh --mode mock --venv .venv
scripts/run_benchmark.sh --mode mock --venv .venv

# 2) CPU/macOS run with A/B/C/D. This starts a local Sail server automatically.
scripts/setup_env.sh --mode cpu --venv .venv
scripts/run_benchmark.sh --mode cpu --venv .venv

# 3) Real-model CPU smoke run. This starts Sail automatically; generation runs in-process with Transformers on CPU.
scripts/setup_env.sh --mode cpu_real --venv .venv
scripts/run_benchmark.sh --mode cpu_real --venv .venv

# 4) GPU run. This starts Sail and vLLM automatically.
scripts/setup_env.sh --mode gpu --venv .venv_gpu
scripts/run_benchmark.sh --mode gpu --venv .venv_gpu
```

Reports are written to `results/<mode>/<timestamp>/report/aggregate.html`.
Raw per-run artifacts are under `results/<mode>/<timestamp>/runs/<run_id>/`.
Hugging Face, Transformers, and SentenceTransformers models/cache files are
kept repo-local under `./models` by default.

Useful overrides:

```bash
WORKLOADS="w0 w1" EXECUTIONS="A B" ITERATIONS=1 scripts/run_benchmark.sh --mode mock
RESULTS_DIR=results/demo FORCE_SYNTHETIC=1 scripts/run_benchmark.sh --mode cpu
CONFIG=config/gpu_v100_smoke.yaml scripts/run_benchmark.sh --mode gpu --venv .venv_gpu
MODELS_DIR=/mnt/fast/models scripts/run_benchmark.sh --mode gpu --venv .venv_gpu
```

`cpu_real` caveats:

- `scripts/setup_env.sh --mode cpu_real` installs the standard CPU model stack: `torch`, `transformers`, `sentence-transformers`, and `accelerate`. It does not install vLLM.
- `scripts/run_benchmark.sh --mode cpu_real` does not start vLLM by default. `config/cpu_real.yaml` sets `models.generator.provider: transformers`, so generation happens inside the Python worker process on CPU.
- Real CPU generation is slow. Keep `config/cpu_real.yaml` small for smoke tests, or override `WORKLOADS`, `EXECUTIONS`, and `ITERATIONS` when iterating.
- GPU pipeline continuity and vLLM scheduler metrics are reported as `N/A` when those telemetry sources are unavailable.

## Key technical constraints (from prior learnings)

1. **Generation provider.** GPU real generation is served by vLLM. Mock and
   `cpu` modes use deterministic mock generation so they run on any machine;
   `cpu_real` uses in-process Hugging Face Transformers generation on CPU.
2. **Lazy `torch` imports** inside UDF closures — PySpark cloudpickles every
   closure even under Sail (because it flows over Spark Connect gRPC).
3. **Sail UDTF syntax.** Use `LATERAL fn(col1, col2)`, NOT
   `LATERAL TABLE(fn(...))` and NOT `useArrow=True` in `@udtf`.
4. `BoundaryTimer` implements `__getstate__`/`__setstate__` to survive cloudpickle
   (the `threading.Lock` is dropped and rebuilt on the worker side).

## Layout

See `config/`, `src/sail_vs_spark/`, `scripts/`, `analysis/`, `tests/`.

## Measured metrics per run

- **Trace timing** (Chrome trace events, one file per run under `runs/<run_id>/trace.json`):
  `DATA_TRANSFER_IN`, `MODEL_LOAD`, `TOKENIZE`, `INFERENCE`, `SCORE`,
  `DETOKENIZE`, `DATA_TRANSFER_OUT`.
- **System telemetry** (one file per run under `runs/<run_id>/stats.json`):
  wall-clock, process and process-tree CPU/RSS, peak RAM, optional GPU
  util/mem, optional vLLM metrics, cumulative bytes written, `nvidia-smi dmon`
  log when available.
- **Derived report** (`report/aggregate_summary.json`):
  `serialization_tax_pct = (transfer_in + transfer_out)/wall`,
  `pipeline_continuity = gpu_active_samples / gpu_samples` when GPU telemetry
  is available, otherwise `N/A`.

## Visualisations produced

1. GPU utilisation timeline.
2. Runtime vs pipeline depth (W0).
3. Peak RSS by workload/config.
4. Disk writes over time.
5. Interactive HTML report.
