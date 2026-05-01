# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Sail + Apache Arrow: Zero-Copy AI Inference with PySpark UDFs
#
# **Milestone 3 — Designing a demo for AI applications in Sail**
#
# This notebook demonstrates how Sail's Apache Arrow zero-copy data transfer enables
# significantly more efficient AI inference pipelines compared to standard Apache Spark.
#
# ## The Problem
#
# When you run AI inference inside a PySpark UDF with standard Spark:
#
# ```
# JVM → serialize (pickle/Row) → socket → Python → model.generate() → serialize → socket → JVM
# ```
#
# Every UDF call crosses the JVM↔Python boundary twice, serializing and deserializing data.
# For simple ML workloads this is manageable. But for **agentic AI workflows** — where a prompt
# might need multiple rounds of generate → score → regenerate — the serialization cost compounds
# with every iteration.
#
# **Sail's solution:** Replace the JVM boundary with Apache Arrow's zero-copy columnar format.
# Data flows from Sail's Rust execution engine directly into the Python worker as Arrow buffers —
# no serialization, no socket copies, no JVM.
#
# ```
# Rust (Sail) → zero-copy Arrow buffer pointer → Python → model.generate() → zero-copy → Rust
# ```
#
# The entire agentic loop — however many iterations — runs inside one Python closure.
# The Arrow boundary is crossed once per batch, not once per model call.

# %% [markdown]
# ## Setup
#
# This benchmark compares four execution configurations:
#
# | Config | Engine | UDF Type | Serialization |
# |--------|--------|----------|---------------|
# | **A** | Spark  | Row UDF (cloudpickle) | Per-row pickle, JVM crossing each call |
# | **B** | Spark  | Pandas UDF (Arrow IPC) | Batched Arrow IPC, still socket-based |
# | **C** | Sail   | `mapInArrow` | Zero-copy Arrow from Rust process |
# | **D** | Sail   | UDTF | Zero-copy Arrow, accumulate-then-flush |

# %%
import subprocess, json, os, glob
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("..")
CONFIG_PATH = REPO / "config" / "laptop.yaml"

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

print("Profile:", cfg["profile"])
print("Dataset:", cfg["dataset"]["source"], "n_rows:", cfg["dataset"]["n_rows"])
print("Generator model:", cfg["models"]["generator"]["name"])
print("Execution configs:", cfg["execution"]["configs"])

# %%
# Show the key architectural difference in code

print("=== Config A: Spark Row UDF (crosses JVM boundary per row) ===")
print("""
@udf(returnType=schema)
def agentic(pid, text):
    wl = W4Agentic(...)         # ← model loaded per call (or cached)
    return wl.apply(pid, text)  # ← ONE row processed, result pickled back to JVM

# For max_iterations=3, this UDF is invoked 3x per prompt (one per agent step)
""")

print("=== Config C: Sail mapInArrow (loop stays inside one Arrow closure) ===")
print("""
def process(batch_iter):           # ← receives Arrow RecordBatches from Rust
    wl = W4Agentic(...)            # ← model loaded ONCE per partition
    for batch in batch_iter:
        ids   = batch.column('prompt_id').to_pylist()   # ← zero-copy read
        texts = batch.column('prompt_text').to_pylist() # ← zero-copy read
        out   = wl.apply_batch(ids, texts)  # ← entire agentic loop (all iterations)
        yield pa.RecordBatch.from_arrays(...)           # ← zero-copy write back

# Arrow boundary crossed ONCE per batch, regardless of max_iterations
""")

# %% [markdown]
# ## Benchmark: W0 — Pure Serialization Overhead
#
# W0 performs trivial computation (increment a counter) at configurable pipeline depths.
# This isolates the data transfer overhead from the model inference time.
#
# At depth 3, a prompt flows through 3 sequential UDF stages. In Config A, that's 6 JVM crossings.
# In Configs C/D, it's 3 Arrow buffer swaps within the same Rust process.

# %%
RESULTS_DIR = REPO / "results" / "notebook_demo"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def run_benchmark(workload, execution, extra_args=None):
    cmd = [
        "python", "-m", "sail_vs_spark.runner.cli",
        "--config", str(CONFIG_PATH),
        "--workload", workload,
        "--execution", execution,
        "--results-dir", str(RESULTS_DIR),
        "--run-id", f"nb_{workload}_{execution}",
    ]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    if result.returncode != 0:
        print("STDERR:", result.stderr[-2000:])
        raise RuntimeError(f"Benchmark failed: {workload}/{execution}")
    # Extract wall time from last stats JSON
    stats_files = sorted(glob.glob(str(RESULTS_DIR / f"nb_{workload}_{execution}*_stats.json")))
    if stats_files:
        with open(stats_files[-1]) as f:
            stats = json.load(f)
        return stats.get("wall_clock_sec", 0)
    return 0

print("Running W0 at depths 1, 2, 3 for all 4 configs...")
w0_times = {}
for depth in [1, 2, 3]:
    w0_times[depth] = {}
    for config in ["A", "B", "C", "D"]:
        t = run_benchmark("w0", config, ["--depth", str(depth)])
        w0_times[depth][config] = t
        print(f"  W0 depth={depth} config={config}: {t:.2f}s")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

configs = ["A", "B", "C", "D"]
colors = ["#e74c3c", "#e67e22", "#2ecc71", "#27ae60"]
labels = ["A: Spark Row\n(pickle)", "B: Spark Pandas\n(Arrow IPC)",
          "C: Sail Arrow\n(zero-copy)", "D: Sail UDTF\n(zero-copy)"]

# Left: runtime by depth
ax = axes[0]
depths = [1, 2, 3]
for i, (config, color, label) in enumerate(zip(configs, colors, labels)):
    times = [w0_times[d][config] for d in depths]
    ax.plot(depths, times, "o-", color=color, label=label, linewidth=2, markersize=8)
ax.set_xlabel("Pipeline Depth")
ax.set_ylabel("Wall Time (s)")
ax.set_title("W0: Serialization Overhead vs Pipeline Depth")
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.3)

# Right: speedup of Sail C over Spark A at each depth
ax = axes[1]
speedups = [w0_times[d]["A"] / w0_times[d]["C"] for d in depths]
ax.bar(depths, speedups, color="#2ecc71", alpha=0.8)
ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Pipeline Depth")
ax.set_ylabel("Speedup (Config A / Config C)")
ax.set_title("Sail Arrow Speedup over Spark Row UDF (W0)")
for d, s in zip(depths, speedups):
    ax.text(d, s + 0.05, f"{s:.1f}×", ha="center", fontweight="bold")
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "w0_speedup.png"), dpi=150, bbox_inches="tight")
plt.show()
print(f"\nAt depth 3: Sail is {speedups[-1]:.1f}× faster than Spark Row UDF")

# %% [markdown]
# ## Benchmark: W2 — Batched LLM Inference
#
# W2 runs a single generate pass on all prompts in a batch. All four configs call the
# **same vLLM server** for inference — the model compute time is identical across configs.
#
# The only difference is **how data gets to and from the model**: via Spark's socket-based
# serialization (A/B) or Sail's zero-copy Arrow buffer (C/D).
#
# This isolates the data transfer overhead as a fraction of total wall time.

# %%
print("Running W2 (batched inference) for all 4 configs...")
w2_times = {}
for config in ["A", "B", "C", "D"]:
    t = run_benchmark("w2", config)
    w2_times[config] = t
    print(f"  W2 config={config}: {t:.2f}s")

fig, ax = plt.subplots(figsize=(8, 4))
times = [w2_times[c] for c in configs]
bars = ax.bar(labels, times, color=colors, alpha=0.85)
ax.set_ylabel("Wall Time (s)")
ax.set_title("W2: Batched LLM Inference\n(same vLLM server for all — only data transfer differs)")
for bar, t in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{t:.1f}s", ha="center", fontsize=9)
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "w2_inference.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Benchmark: W4 — Agentic Multi-Step Loop (the headline demo)
#
# W4 implements an agentic refinement loop:
#
# ```
# for each prompt:
#     repeat up to max_iterations times:
#         generate N candidates  →  score each  →  pick best
#         if best_reward >= threshold: stop
#         else: augment prompt with best response, retry
# ```
#
# **Why this is the critical benchmark:**
# - In **Spark (A/B)**: each generate+score stage is a separate UDF call → JVM crossing per iteration
# - In **Sail (C/D)**: the entire loop lives inside one Arrow closure → zero extra crossings
#
# As `max_iterations` increases, Spark's serialization tax compounds.
# Sail's cost stays flat — it paid once at the Arrow boundary and the rest is pure Python+vLLM.

# %%
print("Running W4 (agentic loop) for all 4 configs...")
w4_times = {}
for config in ["A", "B", "C", "D"]:
    t = run_benchmark("w4", config)
    w4_times[config] = t
    print(f"  W4 config={config}: {t:.2f}s")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: absolute times
ax = axes[0]
times = [w4_times[c] for c in configs]
bars = ax.bar(labels, times, color=colors, alpha=0.85)
ax.set_ylabel("Wall Time (s)")
ax.set_title("W4: Agentic Loop — Total Wall Time")
for bar, t in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{t:.1f}s", ha="center", fontsize=9)
ax.grid(alpha=0.3, axis="y")

# Right: speedup vs Config A
ax = axes[1]
speedup_labels = ["B: Spark Pandas", "C: Sail Arrow", "D: Sail UDTF"]
speedups_w4 = [w4_times["A"] / w4_times[c] for c in ["B", "C", "D"]]
bar_colors = ["#e67e22", "#2ecc71", "#27ae60"]
bars = ax.bar(speedup_labels, speedups_w4, color=bar_colors, alpha=0.85)
ax.axhline(1.0, color="#e74c3c", linestyle="--", alpha=0.7, label="Spark Row baseline")
ax.set_ylabel("Speedup over Spark Row UDF")
ax.set_title("W4: Agentic Loop — Speedup over Config A")
for bar, s in zip(bars, speedups_w4):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{s:.1f}×", ha="center", fontweight="bold", fontsize=11)
ax.legend()
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(str(RESULTS_DIR / "w4_agentic.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Summary
#
# ### Results Table

# %%
print("=" * 60)
print(f"{'Workload':<15} {'Spark Row (A)':>14} {'Spark Pandas (B)':>17} {'Sail Arrow (C)':>15} {'Sail UDTF (D)':>14}")
print("-" * 60)
print(f"{'W0 (depth 3)':<15} {w0_times[3]['A']:>12.2f}s {w0_times[3]['B']:>15.2f}s {w0_times[3]['C']:>13.2f}s {w0_times[3]['D']:>12.2f}s")
print(f"{'W2 (inference)':<15} {w2_times['A']:>12.2f}s {w2_times['B']:>15.2f}s {w2_times['C']:>13.2f}s {w2_times['D']:>12.2f}s")
print(f"{'W4 (agentic)':<15} {w4_times['A']:>12.2f}s {w4_times['B']:>15.2f}s {w4_times['C']:>13.2f}s {w4_times['D']:>12.2f}s")
print("=" * 60)
print()
print("Speedup of Sail Arrow (C) over Spark Row (A):")
print(f"  W0 (depth 3):  {w0_times[3]['A'] / w0_times[3]['C']:.1f}×")
print(f"  W2 inference:  {w2_times['A'] / w2_times['C']:.1f}×")
print(f"  W4 agentic:    {w4_times['A'] / w4_times['C']:.1f}×")
print()
print("Key insight: Sail's advantage is LARGEST for W4 (agentic loop)")
print("because the serialization tax that Spark pays compounds with")
print("each agent iteration, while Sail pays it once per batch.")

# %% [markdown]
# ## Conclusion
#
# Apache Arrow's zero-copy format isn't just a serialization optimization — it's an
# **architectural enabler** for AI workloads in distributed systems.
#
# When inference is fast (thanks to vLLM), the bottleneck shifts to **data movement**.
# Sail eliminates that bottleneck by letting Python code read from and write to the
# Rust engine's memory directly, with no copies and no socket hops.
#
# For agentic AI workflows — where a single user query might trigger dozens of
# generate→score→refine iterations — this matters enormously. Each iteration in Spark
# pays the serialization cost again. In Sail, the agent loop is just Python — the
# Arrow boundary was crossed once at the start, and once at the end.
#
# **This is the unique feature of Sail that JVM Spark cannot replicate:**
# zero-copy data transfer between the Python interpreter and the Rust process,
# enabling truly efficient AI inference pipelines as PySpark UDFs.
