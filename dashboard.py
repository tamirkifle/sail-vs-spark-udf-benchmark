import pandas as pd
import matplotlib.pyplot as plt
import textwrap

# ==========================================
# 1. Data Preparation
# ==========================================
# Wrap text beforehand so the table cells format nicely without overflowing
def wrap(text, width):
    return textwrap.fill(text, width=width)

workloads_data = [
    {"Workload": "W0: Chained Trivial", "Pattern": "input → UDFs → output", "Goal": wrap("Quantifies pure engine orchestration & serialization overhead via simple payload.", 45)},
    {"Workload": "W1: Best-of-N LLM", "Pattern": "prompt → generate N → score → argmax", "Goal": wrap("Archetypal RLHF/Inference. Sail fuses stages to pay IPC costs once.", 45)},
    {"Workload": "W2: Batched Inference", "Pattern": "batch(prompts) → generate → responses", "Goal": wrap("Exercises generator hot path in batch mode; isolates IPC throughput.", 45)},
    {"Workload": "W3: Embedding (RAG)", "Pattern": "text → embed → similarity score", "Goal": wrap("Maximizes relative impact of serialization overhead due to small payloads.", 45)}
]

configs_data = [
    {"Configuration": "Config A: Spark Row", "Engine": "PySpark (@udf)", "Mechanism": "JVM → Pickle → Socket → Python Worker", "Characteristics": wrap("Highest serialization tax. Paid per row/stage.", 40)},
    {"Configuration": "Config B: Spark Pandas", "Engine": "PySpark (@pandas_udf)", "Mechanism": "JVM → Arrow IPC → Socket → Python", "Characteristics": wrap("Removes pickle cost, but retains socket overhead.", 40)},
    {"Configuration": "Config C: Sail Zero-Copy", "Engine": "Sail Rust (mapInArrow)", "Mechanism": "Shared-memory Rust Arrow ↔ PyArrow", "Characteristics": wrap("True zero-copy. Reads directly via C-Data interface.", 40)},
    {"Configuration": "Config D: Sail SQL-Native", "Engine": "Sail Rust (@udtf + LATERAL)", "Mechanism": "Direct execution in partition loop", "Characteristics": wrap("Standard Spark SQL syntax with batch accumulation.", 40)}
]

df_workloads = pd.DataFrame(workloads_data)
df_configs = pd.DataFrame(configs_data)

# ==========================================
# 2. Styling Helper Function
# ==========================================
def render_table(df, title, filename, highlight_sail=False):
    fig, ax = plt.subplots(figsize=(14, 5)) # Wide aspect ratio to fit text naturally
    ax.axis('off')

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)

    # Style cells based on the summary_table aesthetic
    for i, key in enumerate(table.get_celld().keys()):
        cell = table.get_celld()[key]
        row, col = key
        
        # Increase cell height to accommodate wrapped text
        cell.set_height(0.25 if row > 0 else 0.15)
        
        # Header Styling
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#34495E') # Professional dark blue-grey
        else:
            # Alternating Row Colors
            if row % 2 == 0:
                cell.set_facecolor('#F8F9F9')
            else:
                cell.set_facecolor('#FFFFFF')
                
        # Subtle Borders
        cell.set_edgecolor('#BDC3C7')
        cell.set_linewidth(0.5)
        
        # Bold/Highlight Sail configurations for emphasis
        if highlight_sail and row > 0 and col == 0:
            if "Sail" in df.iloc[row-1]["Configuration"]:
                cell.set_text_props(weight='bold', color='#1E8449') # Deep Green

    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=False)
    plt.close()
    print(f"Saved: {filename}")

# ==========================================
# 3. Generate Tables
# ==========================================
render_table(df_workloads, "Workload Matrix Specifications", "tech_spec_workloads.png")
render_table(df_configs, "Execution Configurations", "tech_spec_configs.png", highlight_sail=True)

print("\nTables successfully generated without overflow!")
