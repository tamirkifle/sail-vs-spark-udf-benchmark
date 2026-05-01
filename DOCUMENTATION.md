# DOCUMENTATION — Sail vs Spark AI Workload Benchmark

This is the "how it works and how to run it" guide. It is written to be
read top-to-bottom by someone who has never opened the repo before.

Sections:

1. [What the benchmark actually measures](#1-what-the-benchmark-actually-measures)
2. [Mental model of the four configs](#2-mental-model-of-the-four-configs)
3. [Running it on a laptop](#3-running-it-on-a-laptop)
4. [Running it on the cluster (H200)](#4-running-it-on-the-cluster-h200)
5. [Interpreting the results](#5-interpreting-the-results)
6. [File-by-file walkthrough](#6-file-by-file-walkthrough)
7. [Testing + extending the suite](#7-testing--extending-the-suite)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. What the benchmark actually measures

The hypothesis we want to prove, in one sentence:

> AI pipelines are not compute-bound in Spark — they are data-movement bound.
> As GPUs get faster, Spark becomes **more** inefficient; Sail scales.

Every AI pipeline has three kinds of work: moving data to Python, running
compute on it (model.generate, scoring, embedding), and moving results
back. Spark's UDF mechanism makes you pay the moving cost twice per UDF
call, per row or per batch, through pickle or Arrow IPC over a socket.
Sail shares the Arrow buffer via a pointer, so the moving cost goes to
roughly zero.


To quantify that, every run produces three numbers that a single-line
grep should surface:

- **wall_clock_sec** — how long the whole run took.
- **serialization_tax_pct** — `(DATA_TRANSFER_IN + DATA_TRANSFER_OUT) / wall`.
  The lower, the more of your wall time was spent on actual model work.
  Spark A is usually 30–50 %, Sail C/D is ~0 %.
- **pipeline_continuity** — fraction of samples where the GPU was
  actually busy. Spark is a sawtooth (GPU idle while data marshals
  between JVM and Python), Sail is close to a flat line.

The four configs (A, B, C, D) differ **only** in the UDF mechanism. Same
models, same data, same batch sizes. Any difference you see is pure
plumbing.

---

## 2. Mental model of the four configs

All four configs run the same workload on the same data. They differ in
exactly one dimension: how a batch of rows reaches the Python process.

| Config | Engine | Mechanism          | What crosses the boundary?                      |
| ------ | ------ | ------------------ | ----------------------------------------------- |
| A      | Spark  | `@udf` (pickle)    | One row at a time, cloudpickled over a socket.  |
| B      | Spark  | `@pandas_udf`      | `pandas.Series` batches over Arrow IPC socket.  |
| C      | Sail   | `mapInArrow`       | `pa.RecordBatch` whose buffer is owned by Rust. |
| D      | Sail   | `@udtf` + `LATERAL`| Individual scalars from a UDTF routed via SQL.  |


**Why this matters for Config C vs B.** In Config B, the driver serializes
a pandas batch to Arrow IPC bytes, writes them to a socket, Python reads
them back, deserializes them into a DataFrame. That is two memory copies
plus the socket itself. In Config C, the Sail engine hands Python the
*same memory* the Rust engine already owns, via the Arrow C Data
Interface. Look at `config_c_sail_arrow.py`:

```python
def process(batch_iter):
    for batch in batch_iter:
        ids   = batch.column("prompt_id").to_pylist()
        texts = batch.column("prompt_text").to_pylist()
        ...
        yield pa.RecordBatch.from_arrays([...], names=[...])
```

`batch.column(...).to_pylist()` dereferences the Rust-owned buffer; it
never `memcpy`s it. The returned `RecordBatch` is adopted by Rust the
same way. No copies, no socket, no IPC.

**Why Config D needs its own explanation.** Sail's UDTF path has three
interacting footguns that prior experiments surfaced. Config D uses the
one syntax that works:

```python
@udtf(returnType="prompt_id long, best_response string, ...")  # NO useArrow=True
class BestOfNUDTF:
    def eval(self, prompt_id: int, prompt_text: str):
        ...

spark.sql("SELECT u.* FROM prompts, "
          "LATERAL best_of_n(prompt_id, prompt_text) u")
```


The three things you must **not** do with this UDTF:

- **Don't** add `useArrow=True` in the decorator. That switches Sail to
  `PySparkArrowTableUdf` which runs `sum(1 for _ in args2)` to check if
  the input is empty — consuming the streaming iterator and deadlocking
  against Sail's streaming engine.
- **Don't** wrap the call with `TABLE(...)`. That routes through the
  expression resolver (`sail-plan/src/resolver/expression/udf.rs`) which
  has an explicit `Err(...)` branch for `ArrowTable`.
- **Don't** call it as `LATERAL TABLE(fn(...))`. Same rejection path.

The only correct form is `LATERAL fn(col1, col2)` with a plain `@udtf`
decorator. That goes through `udtf.rs` (query resolver), which supports
`ArrowTable`.

### Workload reminder

Four workloads run against all four configs:

| Code | Workload            | Pattern                                        |
| ---- | ------------------- | ---------------------------------------------- |
| W0   | Chained trivial UDF | `prompt_id → +1 → +1 → +1` at depth 1/2/3.     |
| W1   | Best-of-N (primary) | `prompt → N candidates → score → argmax`.      |
| W2   | Batched inference   | `batch(prompts) → generate`.                   |
| W3   | Embedding + RAG sim | `text → embedding → cosine vs ref bank`.       |

W0 is the foundation. It isolates *pure* UDF overhead (no ML at all),
so the runtime-vs-depth plot is effectively a ruler for the per-stage
IPC cost of each config.


---

## 3. Running it on a laptop

**Laptop goal**: prove the wiring works and get the *shape* of the
serialization curve on a small dataset. 100 rows, mock or small models,
CPU or MPS.

### 3.1 One-time setup

Create a local virtual environment and install the benchmark package along with its dependencies:

```bash
cd sail_vs_spark_benchmark

# 1. Create .venv and install base dependencies (pyspark, etc.)
make install

# 2. Setup Sail v0.6.0 from your local Sail repository
# This will checkout the v0.6.0 tag and build pysail into the local .venv
make setup-sail SAIL_REPO_DIR=~/Documents/MyCode/LakeSail/sail
```

You don't need `torch` or `transformers` to exercise the scaffolding —
the loaders fall back to mocks automatically. If you want real models,
install them separately:

```bash
.venv/bin/pip install torch transformers sentence-transformers
```

### 3.2 Prepare the dataset

```bash
python scripts/prep_dataset.py --config config/laptop.yaml --force-synthetic
# → data/laptop/prompts.parquet (100 rows, schema: prompt_id int64, prompt_text string)
```


`--force-synthetic` skips the HuggingFace download and generates
deterministic prompts with the exact same schema. Drop it if you have
network + HF credentials and want the real UltraFeedback prompts. Either
way, `data/laptop/prompts_meta.json` records which source was used.

### 3.3 Run a single cell (fastest smoke test)

```bash
python -m sail_vs_spark.runner.cli \
    --config config/laptop.yaml \
    --workload w0 --execution A --depth 1 \
    --run-id smoke_w0_A_d1 \
    --results-dir results/laptop
```

That writes three files under `results/laptop/`:

- `smoke_w0_A_d1_output.parquet/` — the actual W0 output.
- `smoke_w0_A_d1_stats.json` — wall_clock, RSS, GPU util, disk I/O.
- `smoke_w0_A_d1_manifest.json` — run metadata (host, device, depth, …).

`--workload` accepts `w0|w1|w2|w3`, `--execution` accepts `A|B|C|D`.
For W0 you also pass `--depth N`.

### 3.4 Run the full laptop matrix

```bash
bash scripts/run_all_laptop.sh
```


That executes all **18 cells** (W0 depth 1/2/3 × 4 configs + W1/W2/W3
× 4 configs) in sequence. Two knobs:

- `VENV=/path/to/your/venv bash scripts/run_all_laptop.sh` — override
  the venv.
- `CONFIG=config/my_custom.yaml bash ...` — use a different config file.

The script picks the right invocation for each config:

- A/B — plain `python -m sail_vs_spark.runner.cli ...`
- C/D — `sail spark run -f /tmp/driver.py` (a tiny generated driver
  script that imports the CLI and calls `main()`). `sail spark run`
  starts an ephemeral Sail server on a random port and pre-injects a
  `SparkSession` into the script scope; `build_sail_session` notices
  that via `SparkSession.getActiveSession()` and reuses it.

### 3.5 Aggregate + plot

```bash
python analysis/aggregate_results.py --results_dir results/laptop
python analysis/plot_depth_runtime.py --results_dir results/laptop
python analysis/plot_memory.py        --results_dir results/laptop
python analysis/plot_disk_io.py       --results_dir results/laptop
python analysis/plot_gpu_timeline.py  --results_dir results/laptop
python analysis/plot_serialization.py --results_dir results/laptop
```

Or, in one go:

```bash
make plot
```


Each plot writes a PNG alongside the aggregate:

```
results/laptop/
├── aggregate.csv / .md / .json
├── depth_runtime.png
├── gpu_timeline.png
├── memory.png
├── disk_io.png
└── serialization_pies.png
```

On a laptop without a GPU, `gpu_timeline.png` will be blank — that's
expected. The memory/disk plots still work.

---

## 4. Running it on the cluster (H200)

**Cluster goal**: real numbers for the writeup. 10 000 rows, FP8
large models on an H200 141 GB.

### 4.1 Install on the cluster

On Discovery (Northeastern):

```bash
ssh yirga.t@discovery.northeastern.edu
cd /scratch/yirga.t
git clone <repo> sail_vs_spark_benchmark
cd sail_vs_spark_benchmark

module load anaconda3
conda activate sail    # already has pyspark 4.1, pysail 0.5.3

pip install -e .
pip install -r requirements.txt
pip install torch transformers sentence-transformers vllm
```


### 4.2 Submit via SLURM

```bash
sbatch scripts/slurm_benchmark_all.sh
```

`slurm_benchmark_all.sh` requests a single 5-hour H200 slot and runs all
four configs serially in the **same** job. This is important: running
them as separate jobs risks them landing on different hardware/driver
state, which ruins cross-config comparability. The exact preamble:

```bash
#SBATCH --job-name=sail_vs_spark
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
```

Inside, it `cd`s to `/scratch/yirga.t/sail_vs_spark_benchmark`, sets
`HF_HOME`, runs `bash scripts/run_all_gpu.sh` (which dispatches the 16
cells — same pattern as the laptop script but pointed at
`config/gpu_h200.yaml`), and then runs `analysis/aggregate_results.py`
automatically.

### 4.3 Pulling results back

When SLURM finishes:

```bash
scp -r yirga.t@discovery.northeastern.edu:\
       /scratch/yirga.t/sail_vs_spark_benchmark/results/gpu \
       ~/Documents/MyCode/LakeSail/benchmark_results/
```

Then re-run the plotting scripts locally against that directory.


### 4.4 Iterating on just one config

Rerunning the whole matrix is expensive. To run a single cell on a
GPU node interactively:

```bash
srun --gres=gpu:h200:1 --time=01:00:00 --pty bash
cd /scratch/yirga.t/sail_vs_spark_benchmark
conda activate sail

# Config A/B — plain python
python -m sail_vs_spark.runner.cli \
    --config config/gpu_h200.yaml \
    --workload w1 --execution A \
    --run-id iter_w1_A --results-dir results/gpu

# Config C/D — sail spark run
sail spark run -f /tmp/driver.py
#   ^^^ where driver.py imports the CLI and calls main([...])
```

Prior learnings §9 recommends running order `A → B → C → D` so the
slowest runs finish first. If the job times out, C and D (the
interesting ones) will still have completed.

---

## 5. Interpreting the results

Every run writes one `*_stats.json` + one `*_manifest.json`. The
aggregator collates them into a single table.

### 5.1 What's in `_stats.json`

Key fields (from `MetricsCollector.report()` in `metrics_collector.py`):


```json
{
  "config": "w1_C_laptop",
  "wall_clock_sec": 14.26,
  "avg_gpu_util_pct": 82.1,
  "peak_gpu_util_pct": 95.0,
  "peak_rss_mb": 2548.3,
  "pipeline_continuity": 0.93,
  "bytes_written_delta": 1048576,
  "mb_written_delta": 1.0,
  "write_throughput_mb_s": 0.07,
  "samples": [ {"t_sec": 0.0, "gpu_util_pct": 0, ...}, ... ]
}
```

The `samples` array is what the GPU-timeline, memory, and disk-I/O
plots consume — each entry is one 0.5-second snapshot of CPU, RSS,
GPU util, and cumulative disk I/O.

`pipeline_continuity` is the fraction of samples where
`gpu_util_pct ≥ 10 %` (the threshold is `GPU_UTIL_ACTIVE_THRESHOLD` in
`metrics_collector.py`). It's the headline number for "is the GPU
actually busy?".

### 5.2 What's in `_manifest.json`

Small and boring — run metadata only. Host, python version, device,
dataset size, pointer to the output parquet. It exists so the aggregator
can tag each stats row with its `(workload, execution, depth)`
without parsing filenames.


### 5.3 The aggregate table

`aggregate.md` groups runs by workload and is the fastest way to read
the headline numbers:

```markdown
## Workload w1

| Cfg | Depth | Wall (s) | Rows | GPU util% | Peak RSS (MB) | MB written | Continuity |
|-----|-------|----------|------|-----------|---------------|------------|------------|
| A   |       | 87.4     | 10000| 41.2      | 11234.1       | 2048.3     | 0.48       |
| B   |       | 46.1     | 10000| 62.4      | 8945.8        | 512.7      | 0.71       |
| C   |       | 21.3     | 10000| 86.1      | 7122.0        | 48.2       | 0.93       |
| D   |       | 22.8     | 10000| 85.7      | 7201.5        | 51.7       | 0.92       |
```

What to look for:

- **Wall (s)**: should drop A → B (Arrow helps) → C → D. Expected ratio
  on the H200 run is ~4–5× A-to-C.
- **GPU util%**: should climb monotonically from A to C/D. If A shows
  high GPU util, something else is wrong (likely the model isn't
  actually loaded).
- **Peak RSS**: Sail configs will typically be *lower* because there's
  no intermediate Arrow-IPC buffer sitting in memory.
- **MB written**: Spark spills intermediates to disk when memory
  pressure gets high. Sail streams batches through Python without
  staging. Spark A/B should show noticeably more bytes written than C/D.
- **Continuity**: Spark sawtooth ≈ 0.4–0.7, Sail flatline ≈ 0.85+.


### 5.4 The plots

- **`depth_runtime.png` (W0)** — runtime on the Y axis, pipeline depth
  on the X axis, one line per config. The **slope** of each line is the
  per-stage IPC cost. A and B have steep slopes, C and D should be
  near flat.
- **`gpu_timeline.png`** — GPU utilisation over time, one line per run.
  Spark traces will sawtooth (GPU idle while a batch marshals between
  JVM and Python, then spikes during model.generate, then idle again).
  Sail traces should hug the top of the chart.
- **`memory.png`** — process RSS over time. Useful to see the model
  load spike at the start and to spot memory growth (a real leak would
  show as monotonic growth).
- **`disk_io.png`** — cumulative bytes written, each run normalised to
  start at 0. Spark configs should go up-and-to-the-right when they
  spill; Sail should stay near zero except for the final output write.
- **`serialization_pies.png`** — one pie per `(workload, execution)`
  showing the ratio of serialization/compute/idle from the boundary
  timer JSON. Only produced if boundary JSONs are present (they are
  optional; the minimum benchmark only produces stats + manifest).

### 5.5 The `_boundary.json` files (optional)

If a run is wrapped in a `BoundaryTimer`, it additionally produces
`<run_id>_boundary.json` with per-phase timings:

```json
{
  "config": "w1_config_c",
  "total_wall_sec": 14.26,
  "serialization_tax_pct": 0.0,
  "compute_pct": 96.2,
  "phases": {
    "DATA_TRANSFER_IN":  {"avg_ms": 0.28, "pct_of_wall": 0.0},
    "INFERENCE":         {"avg_ms": 6766.8, "pct_of_wall": 83.7},
    "SCORE":             {"avg_ms": 1010.0, "pct_of_wall": 12.5},
    "DATA_TRANSFER_OUT": {"avg_ms": 11.9,  "pct_of_wall": 0.1}
  }
}
```


Three lines of that JSON are the whole argument of the benchmark:
`DATA_TRANSFER_IN avg_ms` under 1 ms confirms zero-copy. `compute_pct`
above 80 % confirms the GPU was actually doing useful work. The Spark
equivalent will have `DATA_TRANSFER_IN avg_ms ≈ 15–50` and
`serialization_tax_pct ≈ 30–50`.

---

## 6. File-by-file walkthrough

The repo is structured so each file has one job. Sizes are approximate.

```
sail_vs_spark_benchmark/
├── README.md                    (high-level intro)
├── DOCUMENTATION.md             (this file)
├── Makefile                     (targets: install/test/prep/run/plot/clean)
├── pyproject.toml               (pytest config; package find rule)
├── requirements.txt             (runtime + dev deps; torch kept optional)
├── config/
│   ├── laptop.yaml              (laptop tier — 100 rows, mock-friendly)
│   └── gpu_h200.yaml            (cluster tier — 10k rows, FP8 Qwen)
├── data/                        (generated — the prompts parquet lives here)
├── results/                     (generated — per-run artefacts)
├── scripts/                     (entrypoints: prep, laptop, gpu, SLURM)
├── src/sail_vs_spark/           (the package)
├── analysis/                    (aggregator + plotting)
└── tests/                       (pytest suite; 48 passing)
```

### 6.1 `src/sail_vs_spark/profiling/boundary_timer.py`

Measures where time goes at a phase granularity. The important design
choice is that it survives cloudpickle. PySpark pickles every UDF
closure even when the worker is Sail, because the closure travels over
the Spark Connect gRPC protocol before Sail ever sees it.
`threading.Lock` is not picklable, so:


```python
def __getstate__(self) -> dict:
    state = self.__dict__.copy()
    state.pop("_lock", None)           # drop the lock before pickling
    return state

def __setstate__(self, state: dict) -> None:
    self.__dict__.update(state)
    self._lock = threading.Lock()      # rebuild it on the worker side
```

Without those two methods, Config A/B runs die instantly with
`TypeError: cannot pickle '_thread.lock' object`. This was Bug 1 in the
prior debugging history.

The phases are a fixed whitelist (`PHASES`) and `measure()` rejects
unknown names — that prevents typos from silently absorbing time into
an unnamed bucket. `serialization_tax_pct` and `compute_pct` are
computed in `report()`:

```python
serial_sec = (self._totals["DATA_TRANSFER_IN"]
              + self._totals["DATA_TRANSFER_OUT"])
compute_sec = sum(self._totals[p] for p in
                  ("INFERENCE","SCORE","EMBED","SIMILARITY",
                   "TOKENIZE","DETOKENIZE","TRIVIAL_COMPUTE"))
result["serialization_tax_pct"] = serial_sec / total_wall * 100
result["compute_pct"] = compute_sec / total_wall * 100
```

### 6.2 `src/sail_vs_spark/profiling/metrics_collector.py`

A background sampler thread that snapshots CPU %, RSS, GPU util/mem/
power, and disk I/O deltas every 0.5 s. Three things worth pointing out:


**1) No GPU means no crash.** `_launch_nvidia_dmon` wraps `Popen` in a
`FileNotFoundError` catch — on a MacBook this just sets `self._nvidia_proc
= None` and moves on. `_sample_gpu` wraps the nvidia-smi subprocess in a
broad `except` that returns an empty dict. Result: the benchmark runs
unchanged on any machine, the GPU fields are just zero.

**2) It also spawns `nvidia-smi dmon` in parallel.** That produces a
higher-resolution timeline at `/tmp/nvidia_dmon_<config>.log` which the
plotting script can consume separately from the 0.5 s in-process samples.

**3) `pipeline_continuity` is the ratio you want.**

```python
gpu_active = sum(1 for u in gpu_utils
                 if u >= self.GPU_UTIL_ACTIVE_THRESHOLD)   # 10 %
pipeline_continuity = gpu_active / len(gpu_utils)
```

That's the single number that distinguishes "Spark sawtooth" from "Sail
flatline". Not a percentage — a fraction in `[0, 1]`.

### 6.3 `src/sail_vs_spark/dataset/prep.py`

Takes a `(source, split, n_rows, out_dir)` and writes a parquet at
`out_dir/prompts.parquet` plus a tiny `prompts_meta.json`. The parquet
schema is frozen:

```python
table = pa.Table.from_pylist(rows, schema=pa.schema([
    ("prompt_id",   pa.int64()),
    ("prompt_text", pa.string()),
]))
```


If the HuggingFace Hub is reachable and `datasets` is installed, it
loads UltraFeedback and picks the first prompt column it finds
(`prompt`, `instruction`, or `question`). If anything in that chain
fails, it falls back to `_synthetic_rows`, which builds deterministic
prompts from a fixed template/topic grid. That fallback is what keeps
unit tests and laptop smoke runs hermetic — no network required.

### 6.4 `src/sail_vs_spark/models/mock.py`

Three pure-Python "models" with no torch dependency:

```python
class MockGenerator:
    def generate(self, prompts, n=1, max_new_tokens=32): ...
class MockScorer:
    def score(self, prompts, responses): ...
class MockEmbedder:
    def encode(self, texts): ...
```

All three use `_seeded_rng(tag, seed)` internally — a deterministic PRNG
keyed by a `sha256("<seed>|<tag>")`. So `MockScorer.score(prompt,
response)` always returns the same reward for the same pair, which
makes Best-of-N results reproducible across runs. `MockEmbedder.encode`
produces unit-normalised Gaussian vectors so cosine similarity is
well-defined.

Mocks matter for two reasons: (1) tests stay fast and deterministic,
(2) the laptop tier can run the whole 18-cell matrix end-to-end without
torch. Real hardware numbers come from the GPU tier; the laptop tier
proves wiring.


### 6.5 `src/sail_vs_spark/models/loaders.py`

Factory functions: `get_generator(cfg)`, `get_scorer(cfg)`,
`get_embedder(cfg)`. Three design choices worth knowing:

**Lazy torch imports.** No `import torch` at module level. It's inside
`_resolve_device`, `_HFGenerator.__init__`, `_HFScorer.score`. That
matters because:

1. The file must import on machines with no torch installed (otherwise
   the whole package breaks).
2. The file is imported in UDF closures. If torch were captured at the
   module level, cloudpickle would try to pickle torch state and fail.

**Process-local singletons.**

```python
_GENERATOR: Any = None

def get_generator(cfg: dict) -> Any:
    global _GENERATOR
    if _GENERATOR is not None:
        return _GENERATOR
    ...
```

Every Python worker in Spark/Sail loads the model exactly once. Spark's
`spark.python.worker.reuse = true` (set in `build_spark_session`) keeps
that worker alive across tasks, so the model-load cost is paid once and
amortised.


**Automatic mock fallback.** If transformers isn't installed, or the
model name doesn't resolve, or `from_pretrained` raises, the loader
falls back to the mock — and logs a one-line notice so you know:

```python
try:
    _GENERATOR = _HFGenerator(...)
    return _GENERATOR
except Exception as e:
    if not allow_mock:
        raise
    print(f"[loaders] Generator fell back to mock ({e})")
_GENERATOR = MockGenerator(seed=cfg.get("seed", 0))
```

To force the mock (e.g. for unit tests), set `cfg["prefer_mock"] = True`.
To require the real model (and fail loudly if unavailable), set
`cfg["allow_mock"] = False`.

### 6.6 `src/sail_vs_spark/workloads/base.py` + the W0–W3 modules

`Workload` is the minimal contract:

```python
class Workload:
    code: str = "WX"
    name: str = "base"
    result: WorkloadResult

    def init(self, cfg): ...
    def apply(self, prompt_id, prompt_text) -> tuple: ...
    def apply_batch(self, prompt_ids, prompt_texts) -> dict[str, list]: ...
```


Why both `apply` (single row) and `apply_batch` (batched)? Because the
four configs call the workload differently:

- Config A (`@udf`) calls `apply` once per row.
- Configs B/C call `apply_batch` once per batch.
- Config D calls `apply` once per row (the UDTF `eval` is scalar).

`apply_batch`'s base implementation just loops `apply`, which is fine
for W0 (no real batching benefit) but gets overridden in W1/W2/W3 to
exploit true batch APIs — e.g. W1 flattens `(prompt × N_candidates)`
into a single `score()` call.

**W0 — `w0_chained.py`**

```python
class W0Chained(Workload):
    def apply(self, prompt_id, prompt_text):
        x = int(prompt_id)
        for _ in range(self.depth):
            x = x + 1
        return (int(prompt_id), int(x))
```

Pure `x + 1`, depth times. The whole point of this workload is to
expose nothing but UDF overhead.

**W1 — `w1_best_of_n.py`** is the primary workload. `apply_batch` does
the fused-batch trick that makes Sail shine:


```python
# 1) Generate N candidates per prompt in one generator call
all_cands = self._gen.generate(texts, n=self.n_candidates)

# 2) Flatten (prompt × candidate) pairs and score them in ONE call
flat_prompts, flat_cands = [], []
for p, cands in zip(texts, all_cands):
    for c in cands:
        flat_prompts.append(p); flat_cands.append(c)
flat_rewards = self._sc.score(flat_prompts, flat_cands)

# 3) Re-bucket and argmax per prompt
for pid, cands in zip(ids, all_cands):
    sub_rewards = flat_rewards[offset: offset + len(cands)]
    best_local = max(range(len(cands)), key=lambda i: sub_rewards[i])
    ...
```

In Spark A, those three stages would be three separate UDF calls,
paying the IPC cost three times. In Sail C, the whole thing runs inside
one closure, so generation → scoring → argmax never crosses an engine
boundary.

**W2 — `w2_batched.py`** is the simplest real workload: a batched
`generate()` with no scoring. Useful for isolating batch throughput
differences.

**W3 — `w3_embedding.py`** is the RAG-style workload. The reference
queries live as a module-level constant:

```python
_REFERENCE_QUERIES = [
    "What is machine learning?",
    "Describe quantum computing.",
    ...
]
```


`init(cfg)` embeds them once per worker. Then every row just computes
its own embedding and does `n_queries` cosine dots.

**`registry.py`** maps short codes (`w0`/`w1`/`w2`/`w3`) to the classes
and reads the right kwargs out of the config:

```python
def make_workload(code, cfg):
    ...
    if code == "w1":
        n = int(cfg["workloads"]["w1_best_of_n"]["n_candidates"])
        wl = W1BestOfN(n_candidates=n)
    ...
    wl.init(cfg)
    return wl
```

### 6.7 `src/sail_vs_spark/engines/spark_session.py` + `sail_session.py`

Very short. `spark_session.build_spark_session(cfg)` returns a local
`SparkSession.builder...getOrCreate()` with:

```python
.master(f"local[{npart}]")
.config("spark.python.worker.reuse", "true")
.config("spark.sql.execution.arrow.pyspark.enabled", "true")
.config("spark.driver.memory", "2g")
```

`worker.reuse=true` is important: it keeps the Python worker alive
across tasks so the model-load singleton survives. `spark.driver.memory
= 2g` is intentionally tight on the laptop so that memory pressure
shows up in the disk-write metric when it happens.


`sail_session.build_sail_session(cfg)` has one trick:

```python
def build_sail_session(cfg):
    from pyspark.sql import SparkSession
    existing = SparkSession.getActiveSession()
    if existing is not None:
        return existing        # running under `sail spark run -f`
    remote_url = cfg["runner"]["sail_remote_url"]
    return SparkSession.builder.remote(remote_url).getOrCreate()
```

`sail spark run -f script.py` pre-injects a `SparkSession` into the
exec scope. `getActiveSession()` returns that session. If we're not in
that mode (e.g. running against an already-running
`sail spark server --port 50051`), we open a Spark Connect connection
instead. The CLI never has to know the difference.

### 6.8 `src/sail_vs_spark/configs/config_*.py`

Each of the four files implements `run_w0`, `run_w1`, `run_w2`, `run_w3`
with an identical signature:

```python
def run_w1(spark, parquet_path, cfg, output_parquet) -> int:
    ...
    out.write.mode("overwrite").parquet(output_parquet)
    return spark.read.parquet(output_parquet).count()
```

So the CLI can dispatch to them without knowing anything about the
specific config. The only difference between the four files is *which*
UDF mechanism they use.


**`config_a_spark_row.py`** — row UDF baseline.

```python
@udf(returnType=schema)
def best_of_n(pid, text):
    from sail_vs_spark.workloads.w1_best_of_n import W1BestOfN
    wl = W1BestOfN(n_candidates=ncands)
    wl.init(closure_cfg)
    pid2, resp, reward, n = wl.apply(int(pid), str(text))
    return (pid2, resp, float(reward), int(n))
```

Notice the *lazy* import — `W1BestOfN` is only imported on the worker.
That keeps the pickled closure small and avoids dragging torch through
the Spark Connect gRPC pipe.

**`config_b_spark_pandas.py`** — `@pandas_udf` with struct returns.

```python
@pandas_udf(schema)
def best_of_n(pid_s: pd.Series, text_s: pd.Series) -> pd.DataFrame:
    from sail_vs_spark.workloads.w1_best_of_n import W1BestOfN
    wl = W1BestOfN(n_candidates=ncands)
    wl.init(closure_cfg)
    out = wl.apply_batch(pid_s.tolist(), text_s.tolist())
    return pd.DataFrame(out)
```

Same workload, batched form. Spark handles the pandas↔Arrow bridging;
the socket is still there, it's just sending columnar bytes now.


**`config_c_sail_arrow.py`** — `mapInArrow`, the zero-copy path.

```python
def process(batch_iter):
    from sail_vs_spark.workloads.w1_best_of_n import W1BestOfN
    wl = W1BestOfN(n_candidates=ncands); wl.init(closure_cfg)
    for batch in batch_iter:
        ids   = batch.column("prompt_id").to_pylist()   # zero-copy deref
        texts = batch.column("prompt_text").to_pylist()
        out = wl.apply_batch(ids, texts)
        yield pa.RecordBatch.from_arrays(
            [pa.array(out["prompt_id"],     type=pa.int64()),
             pa.array(out["best_response"], type=pa.string()),
             pa.array(out["best_reward"],   type=pa.float32()),
             pa.array(out["n_candidates"],  type=pa.int32())],
            names=["prompt_id","best_response","best_reward","n_candidates"],
        )

df.mapInArrow(process, schema="prompt_id long, best_response string, ...")
```

`batch.column("prompt_id")` is a `pyarrow.Array` whose underlying buffer
is *owned by Rust*. `.to_pylist()` reads through the Arrow C Data
Interface pointer without copying. The yielded `RecordBatch` is adopted
back by Rust without a copy. The phrase "zero-copy" is literal, not
marketing.

**`config_d_sail_udtf.py`** — SQL-native `LATERAL`. This is the file
that encodes all four of the prior-learnings footguns.


```python
# NO useArrow=True here — that would activate PySparkArrowTableUdf
# which drains the streaming iterator and deadlocks against Sail.
@udtf(returnType=schema)
class BestOfNUDTF:
    def __init__(self):
        self._wl = None                 # lazy: not loaded until first eval
    def eval(self, prompt_id: int, prompt_text: str):
        if self._wl is None:
            from sail_vs_spark.workloads.w1_best_of_n import W1BestOfN
            self._wl = W1BestOfN(n_candidates=ncands)
            self._wl.init(closure_cfg)
        pid, resp, reward, n = self._wl.apply(int(prompt_id), str(prompt_text))
        yield (int(pid), str(resp), float(reward), int(n))

spark.udtf.register("best_of_n", BestOfNUDTF)
df.createOrReplaceTempView("prompts")
spark.sql(
    "SELECT u.* FROM prompts, "
    "LATERAL best_of_n(prompt_id, prompt_text) u"         # NO TABLE() wrapper
).write.mode("overwrite").parquet(output_parquet)
```

If you change any of those "NO" lines, you reproduce a previously-solved
bug. See `SAIL_UDF_EXPERIMENT_DEEP_LEARNINGS.md §10` for the four-step
debug history.

### 6.9 `src/sail_vs_spark/runner/cli.py`

Two callable entry points:

- **`run_one(cfg, workload, execution, ...)`** — programmatic API used
  by tests and drivers.
- **`main(argv)`** — the `argparse` CLI used from the shell.


The dispatch is table-driven:

```python
def _dispatch(execution, workload):
    if execution == "A": from ...configs import config_a_spark_row   as m
    elif execution == "B": from ...configs import config_b_spark_pandas as m
    elif execution == "C": from ...configs import config_c_sail_arrow   as m
    elif execution == "D": from ...configs import config_d_sail_udtf    as m
    return getattr(m, f"run_{workload}")
```

The `run_one` body is the whole orchestrator. Read it from top to
bottom and you know everything the CLI does:

```python
run_fn = _dispatch(execution, workload)
spark  = _make_session(execution, cfg)
col    = MetricsCollector(run_id, sample_interval_sec=...)
col.start()
t0 = time.perf_counter()
try:
    if workload == "w0":
        n_rows = run_fn(spark, parquet_path, depth, output_parquet)
    else:
        n_rows = run_fn(spark, parquet_path, cfg, output_parquet)
finally:
    wall = time.perf_counter() - t0
    col.stop()
    col.save(stats_json, extra={"wall_clock_sec": wall, ...})
save_manifest(make_manifest(...), manifest_json)
```

So every run, regardless of workload or config, always produces the
same pair of JSON artefacts — which is why the aggregator can be
config-agnostic.


### 6.10 `scripts/`

Four files, in order of usefulness:

- **`prep_dataset.py`** — thin `argparse` wrapper around
  `dataset.prep.prepare()`. Nothing clever; use it as the dataset
  preparation step in both laptop and cluster flows.
- **`run_all_laptop.sh`** — loops over
  `{W0 depth 1/2/3, W1, W2, W3} × {A, B, C, D}`. A/B dispatch via
  `python -m sail_vs_spark.runner.cli ...`, C/D dispatch via
  `sail spark run -f /tmp/driver.py` where the driver is generated
  on-the-fly:

```bash
cat >"$driver" <<PYEOF
import sys; sys.path.insert(0, "$REPO_DIR/src")
from sail_vs_spark.runner.cli import main
raise SystemExit(main([
    "--config", "$CONFIG",
    "--workload", "$workload", "--execution", "$execution",
    ...
]))
PYEOF
"$SAIL" spark run -f "$driver"
```

This pattern works because `sail spark run` pre-injects a
`SparkSession` into the exec scope, and `build_sail_session` picks it
up via `getActiveSession()`.

- **`run_all_gpu.sh`** — same pattern but points at
  `config/gpu_h200.yaml` and chooses `conda activate sail` + the cluster
  venv path by default. Order is `A → B → C → D` per prior learnings
  (slowest first, so if the job times out, the interesting configs are
  more likely to have completed).
- **`slurm_benchmark_all.sh`** — SLURM wrapper that requests a single
  H200 slot for 5 hours, sources the sail env, runs `run_all_gpu.sh`,
  and auto-aggregates at the end.


### 6.11 `analysis/`

Six scripts, one responsibility each. All accept `--results_dir`.

- **`aggregate_results.py`** — glob every `*_stats.json` under
  `results_dir`, join with the matching `*_manifest.json`, write
  `aggregate.csv`, `aggregate.md`, `aggregate.json`. The markdown file
  is what you paste into the report.
- **`plot_depth_runtime.py`** — W0 only. For each execution config,
  builds `[(depth, wall), ...]` from manifests and plots a line. The
  slope is the per-stage IPC cost.
- **`plot_gpu_timeline.py`** — overlays `samples[*].gpu_util_pct` vs
  `t_sec` across all runs. This is where the "Spark sawtooth / Sail
  flatline" picture comes from.
- **`plot_memory.py`** — `samples[*].rss_mb` vs time. Spot model-load
  spikes and (absence of) leaks.
- **`plot_disk_io.py`** — `samples[*].write_bytes`, normalised to start
  at 0, in MB. If a Spark config shows a ramp and Sail stays flat, that
  ramp is the Spark spill.
- **`plot_serialization.py`** — if boundary JSONs are present, draws
  one pie per `(workload, execution)` of
  `serialization / compute / idle`. If no boundary JSONs are present,
  writes a figure with a helpful "no data yet" placeholder so the
  pipeline doesn't fail.

### 6.12 `config/laptop.yaml` and `config/gpu_h200.yaml`

The two knobs-files. Important fields:


```yaml
# config/laptop.yaml
hardware:
  device: auto                   # cpu | mps | cuda | auto
  num_partitions: 2
dataset:
  n_rows: 100
  out_dir: data/laptop
models:
  generator: { name: "Qwen/Qwen2.5-0.5B-Instruct", allow_mock: true }
  scorer:    { prefer_mock: true }
  embedder:  { name: "sentence-transformers/all-MiniLM-L6-v2", allow_mock: true }
workloads:
  w0_chained:   { depths: [1,2,3], batch_size: 32 }
  w1_best_of_n: { n_candidates: 4, batch_size: 8 }
runner:
  sail_remote_url: "sc://localhost:50051"
  sample_interval_sec: 0.5
  results_dir: results/laptop
```

```yaml
# config/gpu_h200.yaml  (differences only)
hardware:
  device: cuda
  num_partitions: 4
dataset:
  n_rows: 10000
models:
  generator:
    name: "Qwen/Qwen3.5-122B-A10B-FP8"
    fallback_name: "Qwen/Qwen2.5-7B-Instruct"
    dtype: float8_e4m3fn
    use_vllm: true
    allow_mock: false
```

`allow_mock: false` on the GPU config means if the real model can't
load the job fails loudly — no silent downgrade to mock results.


`--device` and `--n-rows` on the CLI override the YAML values at
runtime if you need to tweak a single run without editing the file.

---

## 7. Testing + extending the suite

### 7.1 Running the tests

```bash
make test           # all 48 tests
make test-fast      # skips tests marked @pytest.mark.slow
```

Results expected:

```
48 passed, 1 skipped in ~15s
```

The skipped test is `tests/test_configs.py::test_config_c_w0` — it
requires a live Sail server. To include it, start one and set the env
var:

```bash
sail spark server --ip 0.0.0.0 --port 50051 &
SAIL_REMOTE_URL=sc://localhost:50051 make test
```

### 7.2 Test layout

- `test_boundary_timer.py` (8) — pickle roundtrip, phase arithmetic,
  JSON persistence, merge, p95 on tiny samples.
- `test_metrics_collector.py` (4) — sampling lifecycle, pipeline
  continuity, disk I/O, JSON save.


- `test_mock_model.py` (9) — determinism of gen/score/embed, singleton
  caching, mock fallback paths.
- `test_workloads.py` (13) — each workload's per-row vs batched
  equivalence, registry dispatch.
- `test_dataset.py` (3) — synthetic-fallback schema, determinism, dense
  `prompt_id` column.
- `test_configs.py` (9 + 1 skip) — A/B on W0/W1/W2/W3 against a real
  local `SparkSession`, cross-config row-count equivalence for W0,
  C/D gated on `SAIL_REMOTE_URL`.
- `test_cli.py` (2) — `run_one` end-to-end on a temp dir, verifies
  manifest + stats JSONs land on disk.

### 7.3 Adding a new workload

Four steps, all local to `src/sail_vs_spark/`:

1. Subclass `Workload` in `workloads/w4_yours.py`. Implement `init`,
   `apply`, optionally `apply_batch`. Set `self.result` with the output
   schema.
2. Add the class to `workloads/registry.py`'s `REGISTRY` dict.
3. Add `run_w4` to each of the four `configs/config_*.py` files. For
   A/B use `@udf` / `@pandas_udf`; for C use `mapInArrow`; for D use
   `@udtf` + `LATERAL` (no `useArrow`, no `TABLE()`).
4. Append to `scripts/run_all_{laptop,gpu}.sh`:

```bash
for cfg in A B; do run_spark_cli "w4" "$cfg"; done
for cfg in C D; do run_sail_cli  "w4" "$cfg"; done
```

Also add a workload config section to both YAML files if needed, and
write tests in `test_workloads.py` + `test_configs.py`.


### 7.4 Adding a new hardware tier

Copy `config/gpu_h200.yaml` to `config/my_tier.yaml`, tweak
`dataset.n_rows`, `models.*.name`, and `hardware.device`. Copy
`scripts/run_all_gpu.sh` to `scripts/run_all_my_tier.sh` and change
`CONFIG=` at the top. Everything else just works — the CLI doesn't
know or care which config it's running under.

### 7.5 Adding a new execution config (E, F, …)

1. Write `src/sail_vs_spark/configs/config_e_*.py` with `run_w0`,
   `run_w1`, `run_w2`, `run_w3`.
2. Extend `_dispatch` in `runner/cli.py`:

```python
elif execution == "E":
    from sail_vs_spark.configs import config_e_my_new_thing as m
```

3. Add `"E"` to the `--execution` choices in `main()`.
4. Add `"E"` to the loops in `scripts/run_all_*.sh`.

---

## 8. Troubleshooting

**`TypeError: cannot pickle '_thread.lock'`**
You're capturing something with a `threading.Lock` in a UDF closure.
See `BoundaryTimer.__getstate__` for the pattern — drop the lock in
`__getstate__`, rebuild it in `__setstate__`.

**`UDTF_EVAL_METHOD_ARGUMENTS_DO_NOT_MATCH_SIGNATURE`**
You've called a UDTF with `TABLE(SELECT ... FROM t)` — that delivers
one `Row` object per `eval()` call, not individual scalars. Use
`LATERAL fn(col1, col2)` instead.


**`unsupported Python UDF type for common inline UDF: ArrowTable`**
You've wrapped a UDTF with `LATERAL TABLE(fn(...))`. Remove the
`TABLE()`. The correct form is `LATERAL fn(col1, col2)`.

**Config D hangs forever after session creation**
You added `useArrow=True` to the `@udtf` decorator. That activates
`PySparkArrowTableUdf.__iter_output` which calls
`sum(1 for _ in args2)` to check the iterator — consuming the stream
and deadlocking Sail's producer. Delete `useArrow=True`.

**`ModuleNotFoundError: No module named 'torch'`**
Expected on a bare laptop install. All loaders fall back to mocks when
torch isn't available. To silence it, either install torch or set
`prefer_mock: true` in the YAML so the loaders don't even try.

**Mock fallback when you didn't expect it**
Look for `[loaders] <Thing> fell back to mock (...)` in stdout. Set
`allow_mock: false` in the YAML for that model to make it fail loudly
instead. On the GPU config `allow_mock` is already `false` for exactly
this reason.

**`nvidia-smi: command not found` on the laptop**
Expected. The collector catches `FileNotFoundError` and just skips GPU
samples — the run still completes and produces non-GPU stats.

**Output parquet is empty / has zero rows**
Check the manifest — `output_rows` should match `dataset.n_rows`. If
it's 0, the UDF probably threw an exception inside the partition.
Spark swallows worker exceptions, so re-run with
`--conf spark.sql.pyspark.udf.verbose=true` (already set on recent
PySpark) or run the workload directly from Python on the prompts
parquet to surface the traceback.


**SLURM job hits the wall time before D completes**
Either reduce `dataset.n_rows` for that submission, or run the configs
as a per-config SLURM array (but remember the prior-learnings rule:
running A/B/C/D as separate jobs risks landing on different hardware
state, so include a note in the writeup if you do).

**`sail spark run -f script.py` complains about no Sail binary**
`$SAIL` in the scripts defaults to `$VENV/bin/sail`. Override:
`SAIL=/path/to/sail bash scripts/run_all_laptop.sh`. On the cluster,
`conda activate sail` puts the right binary on `PATH`.

**`pipeline_continuity` is 0.0 even on a real GPU**
Make sure you didn't forget to set `CUDA_VISIBLE_DEVICES` inside the
SLURM script. The `MetricsCollector` launches `nvidia-smi dmon` which
will silently see no GPUs in an empty CUDA namespace.

---

## Quick reference card

```bash
# laptop: one-off cell
python -m sail_vs_spark.runner.cli --config config/laptop.yaml \
       --workload w1 --execution C --run-id demo

# laptop: full matrix + plots
bash scripts/run_all_laptop.sh && make plot

# cluster: submit full benchmark
sbatch scripts/slurm_benchmark_all.sh

# tests
make test

# clean
make clean
```

Everything else is in the source or in `SAIL_UDF_EXPERIMENT_DEEP_LEARNINGS.md`.
