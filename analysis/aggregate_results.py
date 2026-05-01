import json
import pandas as pd
from pathlib import Path
import argparse
import re

def get_label(cfg):
    labels = {
        "A": "Spark (Row/Pickle)",
        "B": "Spark (Pandas/Arrow)",
        "C": "Sail (Zero-Copy)",
        "D": "Sail (SQL-Native)"
    }
    return labels.get(cfg, cfg)

def aggregate(results_dir):
    data = []
    path = Path(results_dir)
    
    for manifest_path in path.glob("*_manifest.json"):
        stats_path = manifest_path.parent / manifest_path.name.replace("_manifest.json", "_stats.json")
        trace_path = manifest_path.parent / manifest_path.name.replace("_manifest.json", "_trace.json")
        if not stats_path.exists():
            continue
            
        with open(manifest_path) as f:
            m = json.load(f)
        with open(stats_path) as f:
            s = json.load(f)
            
        udf_time_sec = 0.0
        if trace_path.exists():
            try:
                with open(trace_path) as f:
                    t = json.load(f)
                    for ev in t.get("traceEvents", []):
                        if ev.get("name") in ("UDF_BATCH_EXECUTION", "UDF_ROW_EXECUTION"):
                            udf_time_sec += ev.get("dur", 0.0) / 1_000_000.0
            except Exception:
                pass
            
        # Extract sample index from run_id (e.g., live_w1_A_s1 -> 1)
        sample_match = re.search(r'_s(\d+)', m["run_id"])
        sample_idx = int(sample_match.group(1)) if sample_match else 1

        data.append({
            "Workload": m["workload"].upper(),
            "Config": m["execution"],
            "Label": get_label(m["execution"]),
            "Depth": m.get("depth"),
            "SampleIdx": sample_idx,
            "WallTime": s["wall_clock_sec"],
            "UDFTime": udf_time_sec,
            "MemoryMB": s["peak_rss_mb"],
            "Rows": s.get("output_rows", 0)
        })

    df = pd.DataFrame(data)
    if df.empty:
        print("No results found.")
        return

    # --- Grouping Logic: Separate Cold and Warm ---
    group_cols = ["Workload", "Config", "Label", "Depth"]
    
    results = []
    for keys, group in df.groupby(group_cols, dropna=False):
        # Cold start is ALWAYS Sample 1
        cold_row = group[group["SampleIdx"] == 1]
        cold_val = cold_row["WallTime"].values[0] if not cold_row.empty else None
        
        # Warm starts are Samples 2, 3, ...
        warm_rows = group[group["SampleIdx"] > 1]
        if not warm_rows.empty:
            warm_mean = warm_rows["WallTime"].mean()
            warm_std = warm_rows["WallTime"].std()
            udf_mean = warm_rows["UDFTime"].mean()
            warm_samples = len(warm_rows)
        else:
            # Fallback: if only 1 sample was taken, it's both cold and warm
            warm_mean = cold_val
            warm_std = 0.0
            udf_mean = cold_row["UDFTime"].values[0] if not cold_row.empty else 0.0
            warm_samples = 1

        results.append({
            "Workload": keys[0],
            "Config": keys[1],
            "Setup": keys[2],
            "Depth": keys[3],
            "Cold (s)": cold_val,
            "Warm_mean": warm_mean,
            "Warm_std": warm_std,
            "UDF Time (s)": udf_mean,
            "Mem (MB)": group["MemoryMB"].mean(),
            "Rows": group["Rows"].mean(),
            "Samples": len(group)
        })

    agg = pd.DataFrame(results)

    # Format strings for the table
    def format_warm(row):
        if pd.isna(row["Warm_std"]) or row["Warm_std"] < 0.001:
            return f"{row['Warm_mean']:.3f}s"
        return f"{row['Warm_mean']:.3f}s ±{row['Warm_std']:.3f}"

    agg["Steady (Warm)"] = agg.apply(format_warm, axis=1)
    agg["Setup (Cold)"] = agg["Cold (s)"].apply(lambda x: f"{x:.2f}s" if x is not None else "-")
    
    # Calculate Overhead
    def get_overhead(row):
        if row["Warm_mean"] > 0:
            overhead_pct = ((row["Warm_mean"] - row["UDF Time (s)"]) / row["Warm_mean"]) * 100
            return f"{max(0, overhead_pct):.1f}%"
        return "0.0%"
    
    agg["Overhead Tax"] = agg.apply(get_overhead, axis=1)
    agg["UDF Time (s)"] = agg["UDF Time (s)"].apply(lambda x: f"{x:.3f}s" if x > 0 else "-")

    # --- Speedup Calculation (Based on Warm Mean) ---
    def get_speedup(row):
        mask = (agg["Workload"] == row["Workload"]) & (agg["Config"] == "A")
        if pd.isna(row["Depth"]):
            mask &= agg["Depth"].isna()
        else:
            mask &= (agg["Depth"] == row["Depth"])
            
        base = agg[mask]
        if not base.empty:
            base_val = base["Warm_mean"].values[0]
            if row["Warm_mean"] > 0:
                return f"{base_val / row['Warm_mean']:.1f}x"
        return "1.0x"

    agg["Speedup"] = agg.apply(get_speedup, axis=1)
    agg = agg.sort_values(["Workload", "Depth", "Config"])

    # --- Write Markdown Report ---
    with open(path / "aggregate.md", "w") as f:
        f.write("# ⛵ Sail vs 🎇 Spark: Professional Benchmark Summary\n\n")

        f.write("## 💡 How to read this report\n")
        f.write("- **Setup (Cold)**: The time for the *first* run (includes model loading).\n")
        f.write("- **Steady (Warm)**: The average time for *subsequent* runs (engine throughput).\n")
        f.write("- **UDF Time (s)**: The actual time spent executing the payload functions.\n")
        f.write("- **Overhead Tax**: The percentage of wall-clock time spent outside the UDF payload (serialization, orchestration).\n")
        f.write("- **Speedup**: Calculated using the **Warm** state to show true engine potential.\n\n")
        
        f.write("## 1. Configuration Legend\n")
        f.write("- **Spark (Row/Pickle)**: Standard row-by-row PySpark. High serialization tax.\n")
        f.write("- **Spark (Pandas/Arrow)**: Vectorized Spark via socket copy.\n")
        f.write("- **Sail (Zero-Copy)**: Shared-memory Rust $\\leftrightarrow$ Python link.\n")
        f.write("- **Sail (SQL-Native)**: Direct UDTF execution inside the Sail engine.\n\n")

        for workload in sorted(agg["Workload"].unique()):
            f.write(f"## Workload: {workload}\n")
            w_df = agg[agg["Workload"] == workload].copy()
            w_df["Depth"] = w_df["Depth"].apply(lambda x: int(x) if pd.notnull(x) else "-")
            
            out_df = w_df[["Setup", "Depth", "Setup (Cold)", "Steady (Warm)", "UDF Time (s)", "Overhead Tax", "Speedup", "Mem (MB)", "Rows", "Samples"]]
            f.write(out_df.to_markdown(index=False))
            f.write("\n\n")

    print(f"✅ Professional 'Cold vs Warm' trace-augmented report generated: {path}/aggregate.md")

    # --- Generate Charts automatically ---
    import subprocess
    import sys
    print("📊 Generating Visualizations...")
    
    scripts = [
        "analysis/plot_overhead_breakdown.py",
        "analysis/plot_speedup.py",
        "analysis/plot_depth_runtime.py"
    ]
    
    for script in scripts:
        script_path = Path(script)
        if script_path.exists():
            try:
                subprocess.run([sys.executable, str(script_path), "--results_dir", str(path)], check=True)
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Failed to run {script}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()
    aggregate(args.results_dir)
