import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

custom_palette = ["#9E9E9E", "#FF7043", "#42A5F5", "#26A69A"]
sns.set_palette(custom_palette)

data = [
    {"Workload": "W0", "Setup": "Spark (Row/Pickle)", "Cold Setup (s)": 3.91, "Warm Steady (s)": 3.912, "Speedup": 1.0},
    {"Workload": "W0", "Setup": "Spark (Pandas/Arrow)", "Cold Setup (s)": 4.21, "Warm Steady (s)": 4.209, "Speedup": 0.9},
    {"Workload": "W0", "Setup": "Sail (Zero-Copy)", "Cold Setup (s)": 0.03, "Warm Steady (s)": 0.032, "Speedup": 122.2},
    {"Workload": "W0", "Setup": "Sail (SQL-Native)", "Cold Setup (s)": 0.04, "Warm Steady (s)": 0.035, "Speedup": 111.8},
    
    {"Workload": "W1", "Setup": "Spark (Row/Pickle)", "Cold Setup (s)": 24.05, "Warm Steady (s)": 24.052, "Speedup": 1.0},
    {"Workload": "W1", "Setup": "Spark (Pandas/Arrow)", "Cold Setup (s)": 18.54, "Warm Steady (s)": 18.536, "Speedup": 1.3},
    {"Workload": "W1", "Setup": "Sail (Zero-Copy)", "Cold Setup (s)": 15.80, "Warm Steady (s)": 15.803, "Speedup": 1.5},
    {"Workload": "W1", "Setup": "Sail (SQL-Native)", "Cold Setup (s)": 3.90, "Warm Steady (s)": 3.898, "Speedup": 6.2},
    
    {"Workload": "W2", "Setup": "Spark (Row/Pickle)", "Cold Setup (s)": 6.50, "Warm Steady (s)": 6.504, "Speedup": 1.0},
    {"Workload": "W2", "Setup": "Spark (Pandas/Arrow)", "Cold Setup (s)": 7.08, "Warm Steady (s)": 7.083, "Speedup": 0.9},
    {"Workload": "W2", "Setup": "Sail (Zero-Copy)", "Cold Setup (s)": 3.71, "Warm Steady (s)": 3.711, "Speedup": 1.8},
    {"Workload": "W2", "Setup": "Sail (SQL-Native)", "Cold Setup (s)": 3.73, "Warm Steady (s)": 3.728, "Speedup": 1.7},
    
    {"Workload": "W3", "Setup": "Spark (Row/Pickle)", "Cold Setup (s)": 22.39, "Warm Steady (s)": 22.388, "Speedup": 1.0},
    {"Workload": "W3", "Setup": "Spark (Pandas/Arrow)", "Cold Setup (s)": 14.61, "Warm Steady (s)": 14.610, "Speedup": 1.5},
    {"Workload": "W3", "Setup": "Sail (Zero-Copy)", "Cold Setup (s)": 1.96, "Warm Steady (s)": 1.964, "Speedup": 11.4},
    {"Workload": "W3", "Setup": "Sail (SQL-Native)", "Cold Setup (s)": 0.07, "Warm Steady (s)": 0.069, "Speedup": 324.5},
    
    {"Workload": "W4", "Setup": "Spark (Row/Pickle)", "Cold Setup (s)": 29.77, "Warm Steady (s)": 29.765, "Speedup": 1.0},
    {"Workload": "W4", "Setup": "Spark (Pandas/Arrow)", "Cold Setup (s)": 18.06, "Warm Steady (s)": 18.063, "Speedup": 1.6},
    {"Workload": "W4", "Setup": "Sail (Zero-Copy)", "Cold Setup (s)": 3.90, "Warm Steady (s)": 3.903, "Speedup": 7.6},
    {"Workload": "W4", "Setup": "Sail (SQL-Native)", "Cold Setup (s)": 3.89, "Warm Steady (s)": 3.890, "Speedup": 7.7},
]

df = pd.DataFrame(data)
setup_order = ["Spark (Row/Pickle)", "Spark (Pandas/Arrow)", "Sail (Zero-Copy)", "Sail (SQL-Native)"]
df['Setup'] = pd.Categorical(df['Setup'], categories=setup_order, ordered=True)

plt.figure(figsize=(12, 6))
ax1 = sns.barplot(
    data=df, 
    x="Workload", 
    y="Speedup", 
    hue="Setup",
    edgecolor="white", 
    linewidth=1.5
)

ax1.set_yscale("log")
plt.title("H100 Smoke Test: Relative Performance Multiplier (Log Scale)", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("", fontsize=12)
plt.ylabel("Speedup Multiplier (x) - Log Scale", fontsize=12, fontweight='bold')
sns.despine(left=True, bottom=False)
plt.legend(title="Execution Path", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)

for p in ax1.patches:
    height = p.get_height()
    if height > 0:
        ax1.annotate(f'{height:.1f}x', 
                     (p.get_x() + p.get_width() / 2., height), 
                     ha='center', va='bottom', 
                     fontsize=9, color='black', xytext=(0, 4), 
                     textcoords='offset points')

plt.tight_layout()
plt.savefig("h100_smoke_test_speedup_log.png", bbox_inches='tight', transparent=False)
plt.close()

fig, ax2 = plt.subplots(figsize=(14, 7))
workloads = df['Workload'].unique()
x_indexes = np.arange(len(workloads))
bar_width = 0.2

for i, setup in enumerate(setup_order):
    subset = df[df['Setup'] == setup]
    pos = x_indexes + (i - 1.5) * bar_width
    
    ax2.bar(pos, subset['Warm Steady (s)'], bar_width, 
            label=f"{setup} (Warm)", color=custom_palette[i], edgecolor='white', linewidth=1)
    ax2.bar(pos, subset['Cold Setup (s)'], bar_width, 
            bottom=subset['Warm Steady (s)'], 
            label=f"{setup} (Cold)", color=custom_palette[i], alpha=0.4, edgecolor='white', linewidth=1)

plt.title("H100 Smoke Test: Initialization (Cold) vs. Steady State (Warm)", fontsize=16, fontweight='bold', pad=20)
plt.ylabel("Time (Seconds)", fontsize=12, fontweight='bold')
plt.xticks(x_indexes, workloads, fontsize=11)
sns.despine(left=True)
ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
ax2.xaxis.grid(False)

handles, labels = ax2.get_legend_handles_labels()
plt.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, fontsize=10)

plt.tight_layout()
plt.savefig("h100_smoke_test_execution_time.png", bbox_inches='tight', transparent=False)
plt.close()

fig_tab, ax_tab = plt.subplots(figsize=(14, 6))
ax_tab.axis('off')

df_display = df.copy()
df_display['Cold Setup (s)'] = df_display['Cold Setup (s)'].apply(lambda x: f"{x:.2f}s")
df_display['Warm Steady (s)'] = df_display['Warm Steady (s)'].apply(lambda x: f"{x:.3f}s")
df_display['Speedup'] = df_display['Speedup'].apply(lambda x: f"{x:.1f}x")

table = ax_tab.table(
    cellText=df_display.values,
    colLabels=df_display.columns,
    cellLoc='center',
    loc='center',
    bbox=[0, 0, 1, 1]
)

table.auto_set_font_size(False)
table.set_fontsize(11)

for i, key in enumerate(table.get_celld().keys()):
    cell = table.get_celld()[key]
    row, col = key
    
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#34495E')
    else:
        if row % 2 == 0:
            cell.set_facecolor('#F8F9F9')
        else:
            cell.set_facecolor('#FFFFFF')
            
    cell.set_edgecolor('#BDC3C7')
    cell.set_linewidth(0.5)
    
    if col == 4 and row > 0: 
        if "Sail" in df_display.iloc[row-1]["Setup"]:
            cell.set_text_props(weight='bold', color='#1E8449')

plt.title("H100 Smoke Test Consolidated Results", fontsize=16, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("h100_smoke_test_summary_table.png", bbox_inches='tight', transparent=False)
plt.close()
