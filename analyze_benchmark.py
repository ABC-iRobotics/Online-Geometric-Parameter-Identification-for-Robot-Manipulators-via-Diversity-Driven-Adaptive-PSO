import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np 

# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

CSV_PATH = "Final.csv"

OUT_DIR = Path("benchmark_analysis")
OUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# Method naming
# ------------------------------------------------------------

METHOD_RENAME = {
    "PSO": "PSO",
    "APSO": "APSO",
    "GA": "GA",
    "DE": "DE",
    "PSO_topology": "PSOR",
    "PSO_topology_aging": "PSORA",
    "PSO_final": "Proposed method",
}

METHOD_ORDER = [
    "GA",
    "DE",
    "PSO",
    "APSO",
    "PSOR",
    "PSORA",
    "Proposed method",
]

# ------------------------------------------------------------
# Trajectory naming
# ------------------------------------------------------------

TRAJECTORY_RENAME = {
    "aggressive_sine": "M2F",
    "multi_frequency": "M3F",
    "random_smooth": "RS",
    "slow_sine": "S04F",
    "smooth_sine": "S1F",
}

TRAJECTORY_ORDER = [
    "S1F",
    "S04F",
    "RS",
    "M2F",
    "M3F",
]

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

df = pd.read_csv(CSV_PATH)

# Rename methods
df["method"] = df["method"].replace(METHOD_RENAME)

# Rename trajectories
df["trajectory_id"] = df["trajectory_id"].replace(
    TRAJECTORY_RENAME
)

# Ordered categories
df["method"] = pd.Categorical(
    df["method"],
    categories=METHOD_ORDER,
    ordered=True,
)

df["trajectory_id"] = pd.Categorical(
    df["trajectory_id"],
    categories=TRAJECTORY_ORDER,
    ordered=True,
)

df = df.sort_values(["trajectory_id", "method"])

# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

metrics = [
    "absolute_DH_error_mean",
    "optimized_fitness",
    "improvement_ratio",
    "runtime_per_measurement",
    "theta_error",
    "d_error",
    "a_error",
    "alpha_error",
    "mean_Dx",
    "mean_Dv",
]

# ------------------------------------------------------------
# 1. Overall method summary
# ------------------------------------------------------------

summary = (
    df.groupby("method", observed=True)[metrics]
    .agg(["mean", "std"])
)

summary.to_csv(OUT_DIR / "summary_by_method.csv")

# ------------------------------------------------------------
# 2. Main LaTeX table
# ------------------------------------------------------------

main_metrics = [
    "absolute_DH_error_mean",
    "optimized_fitness",
    "improvement_ratio",
    "runtime_per_measurement",
]

latex_rows = []

for method in METHOD_ORDER:
    group = df[df["method"] == method]

    if group.empty:
        continue

    row = {"Method": method}

    for metric in main_metrics:
        mean = group[metric].mean()
        std = group[metric].std()

        row[metric] = (
            f"{mean:.4e} $\\pm$ {std:.4e}"
        )

    latex_rows.append(row)

latex_df = pd.DataFrame(latex_rows)

latex_df.to_latex(
    OUT_DIR / "main_results_table.tex",
    index=False,
    escape=False,
    caption="Overall comparison of the evaluated optimization methods.",
    label="tab:overall_results",
)

# ------------------------------------------------------------
# 3. Trajectory-wise summary
# ------------------------------------------------------------

traj_summary = (
    df.groupby(
        ["trajectory_id", "method"],
        observed=True,
    )[metrics]
    .agg(["mean", "std"])
)

traj_summary.to_csv(
    OUT_DIR / "summary_by_trajectory.csv"
)

# ------------------------------------------------------------
# 4. Boxplots
# ------------------------------------------------------------

def save_boxplot(metric, ylabel, filename):
    plt.figure(figsize=(11, 5))

    ax = plt.gca()

    df.boxplot(
        column=metric,
        by="method",
        rot=20,
        ax=ax,
        boxprops=dict(linewidth=2.0),
        whiskerprops=dict(linewidth=2.0),
        capprops=dict(linewidth=2.0),
        medianprops=dict(linewidth=3.0),
        flierprops=dict(marker='o', markerfacecolor='black', markersize=5, markeredgecolor='black'),
    )

    # Remove automatic pandas title
    plt.suptitle("")
    ax.set_title("")

    # Axis labels
    ax.set_ylabel(
        ylabel,
        fontsize=16,
        fontweight="bold",
    )

    ax.set_xlabel(
        "",
        fontsize=16,
        fontweight="bold",
    )

    # Tick labels
    ax.tick_params(
        axis="x",
        labelsize=13,
    )

    ax.tick_params(
        axis="y",
        labelsize=13,
    )

    # Make tick labels bold
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")

    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    # Thicker axis lines
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()

    plt.savefig(
        OUT_DIR / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

save_boxplot(
    "absolute_DH_error_mean",
    "Mean absolute DH error",
    "boxplot_dh_error.png",
)

save_boxplot(
    "optimized_fitness",
    "Optimized fitness",
    "boxplot_fitness.png",
)

save_boxplot(
    "runtime_per_measurement",
    "Runtime per iteration [s]",
    "boxplot_runtime.png",
)

save_boxplot(
    "improvement_ratio",
    "Improvement ratio",
    "boxplot_improvement.png",
)

# ------------------------------------------------------------
# 5. Trajectory-wise heatmap
# ------------------------------------------------------------

pivot_dh = df.pivot_table(
    index="trajectory_id",
    columns="method",
    values="optimized_fitness",
    aggfunc="mean",
    observed=True,
)

existing_methods = [
    m for m in METHOD_ORDER
    if m in pivot_dh.columns
]

pivot_dh = pivot_dh[existing_methods]

pivot_dh.to_csv(
    OUT_DIR / "trajectory_method_dh_error_matrix.csv"
)

plt.figure(figsize=(11, 5))

ax = plt.gca()

vmin = pivot_dh.values.min()
vmax = np.percentile(pivot_dh.values, 85)

im = ax.imshow(
    pivot_dh.values,
    aspect="auto",
    vmin=vmin,
    vmax=vmax,
    cmap="viridis",
)

cbar = plt.colorbar(im)
cbar.set_label(
    "Optimized fitness",
    fontsize=15,
    fontweight="bold",
)

# Colorbar tick labels
cbar.ax.tick_params(labelsize=12)

for label in cbar.ax.get_yticklabels():
    label.set_fontweight("bold")

# Axis ticks
ax.set_xticks(range(len(pivot_dh.columns)))
ax.set_xticklabels(
    pivot_dh.columns,
    rotation=20,
    ha="right",
    fontsize=13,
    fontweight="bold",
)

ax.set_yticks(range(len(pivot_dh.index)))
ax.set_yticklabels(
    pivot_dh.index,
    fontsize=13,
    fontweight="bold",
)

# Axis labels
ax.set_xlabel(
    "Optimization Method",
    fontsize=16,
    fontweight="bold",
)

ax.set_ylabel(
    "Trajectory Type",
    fontsize=16,
    fontweight="bold",
)

plt.tight_layout()

plt.savefig(
    OUT_DIR / "heatmap_trajectory_method_dh_error.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

# ------------------------------------------------------------
# 6. Best method ranking
# ------------------------------------------------------------

ranking = (
    df.groupby("method", observed=True)
    .agg(
        mean_dh_error=(
            "absolute_DH_error_mean",
            "mean",
        ),
        std_dh_error=(
            "absolute_DH_error_mean",
            "std",
        ),
        mean_fitness=(
            "optimized_fitness",
            "mean",
        ),
        mean_runtime=(
            "runtime_per_measurement",
            "mean",
        ),
        mean_improvement=(
            "improvement_ratio",
            "mean",
        ),
    )
    .sort_values(
        "mean_dh_error",
        ascending=True,
    )
)

ranking.to_csv(
    OUT_DIR / "method_ranking.csv"
)

# ------------------------------------------------------------
# 7. DH component error table
# ------------------------------------------------------------

dh_component_metrics = [
    "theta_error",
    "d_error",
    "a_error",
    "alpha_error",
]

dh_rows = []

for method in METHOD_ORDER:
    group = df[df["method"] == method]

    if group.empty:
        continue

    row = {"Method": method}

    for metric in dh_component_metrics:
        mean = group[metric].mean()
        std = group[metric].std()

        row[metric] = (
            f"{mean:.4e} $\\pm$ {std:.4e}"
        )

    dh_rows.append(row)

dh_latex_df = pd.DataFrame(dh_rows)

dh_latex_df.to_latex(
    OUT_DIR / "dh_component_error_table.tex",
    index=False,
    escape=False,
    caption="DH parameter component-wise error comparison.",
    label="tab:dh_component_errors",
)

print("Analysis finished.")
print(f"Results saved to: {OUT_DIR.resolve()}")