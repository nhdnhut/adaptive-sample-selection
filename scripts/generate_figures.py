"""
scripts/generate_figures.py

Generate all publication figures from processed results.

Usage:
    python scripts/generate_figures.py [--results-dir results] [--figures-dir figures]

Generates:
    figures/learning_curves.png         -- val accuracy vs epochs (by method)
    figures/accuracy_vs_budget.png      -- accuracy vs data budget
    figures/accuracy_vs_samples.png     -- accuracy vs samples seen
    figures/accuracy_vs_time.png        -- accuracy vs wall-clock time
    figures/f1_vs_budget.png            -- F1 vs data budget
    figures/pareto_frontier.png         -- accuracy vs training time Pareto
    figures/ablation.png                -- ablation component analysis
    figures/failure_analysis.png        -- performance under noise/imbalance

All figures are saved as PNG (300 dpi) and SVG for LaTeX inclusion.
No manual data editing — all values read from results/processed/*.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------
# Style constants
# -----------------------------------------------------------------------

METHOD_COLORS = {
    "random":      "#4e79a7",
    "difficulty":  "#f28e2b",
    "uncertainty": "#e15759",
    "diversity":   "#76b7b2",
    "adaptive":    "#59a14f",
}

METHOD_MARKERS = {
    "random":      "o",
    "difficulty":  "s",
    "uncertainty": "^",
    "diversity":   "D",
    "adaptive":    "*",
}

METHOD_LABELS = {
    "random":      "Random",
    "difficulty":  "Difficulty",
    "uncertainty": "Uncertainty",
    "diversity":   "Diversity",
    "adaptive":    "Adaptive (Ours)",
}

FONT_SIZE = 11
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE + 1,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "legend.fontsize": FONT_SIZE - 1,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def save_fig(fig: plt.Figure, name: str, figures_dir: Path) -> None:
    for ext in ["png", "svg"]:
        path = figures_dir / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight")
    print(f"  Saved: {name}.png / .svg")
    plt.close(fig)


# -----------------------------------------------------------------------
# Figure 1: Learning curves (val accuracy vs epoch per method at one budget)
# -----------------------------------------------------------------------

def plot_learning_curves(all_df: pd.DataFrame, figures_dir: Path, budget_pct: int = 25) -> None:
    df = all_df[all_df["budget_pct"] == budget_pct].copy()
    if df.empty:
        print(f"  [skip] No data for budget={budget_pct}%")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for method in METHOD_COLORS:
        mdf = df[df["method"] == method]
        if mdf.empty:
            continue
        # Average across seeds
        epoch_mean = mdf.groupby("epoch")["val_accuracy"].mean()
        epoch_std = mdf.groupby("epoch")["val_accuracy"].std().fillna(0)
        x = epoch_mean.index
        y = epoch_mean.values
        e = epoch_std.values

        ax.plot(x, y, color=METHOD_COLORS[method], label=METHOD_LABELS[method],
                marker=METHOD_MARKERS[method], markevery=5, linewidth=1.8)
        ax.fill_between(x, y - e, y + e, alpha=0.15, color=METHOD_COLORS[method])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title(f"Learning Curves — Budget {budget_pct}%")
    ax.legend(loc="lower right")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    save_fig(fig, "learning_curves", figures_dir)


# -----------------------------------------------------------------------
# Figure 2: Accuracy vs data budget
# -----------------------------------------------------------------------

def plot_accuracy_vs_budget(mean_std: pd.DataFrame, figures_dir: Path) -> None:
    if mean_std.empty:
        print("  [skip] No mean_std data")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for method in METHOD_COLORS:
        mdf = mean_std[mean_std["method"] == method].sort_values("budget_pct")
        if mdf.empty:
            continue
        x = mdf["budget_pct"].values
        y = mdf["val_accuracy_mean"].values
        e = mdf["val_accuracy_std"].fillna(0).values

        ax.errorbar(
            x, y, yerr=e, label=METHOD_LABELS[method],
            color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
            linewidth=1.8, capsize=4, capthick=1.2,
        )

    ax.set_xlabel("Training Data Budget (%)")
    ax.set_ylabel("Validation Accuracy (mean ± std)")
    ax.set_title("Accuracy vs Data Budget")
    ax.set_xticks([10, 25, 50, 75, 100])
    ax.legend(loc="lower right")
    save_fig(fig, "accuracy_vs_budget", figures_dir)


# -----------------------------------------------------------------------
# Figure 3: Accuracy vs samples seen
# -----------------------------------------------------------------------

def plot_accuracy_vs_samples(all_df: pd.DataFrame, figures_dir: Path) -> None:
    if all_df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for method in METHOD_COLORS:
        mdf = all_df[all_df["method"] == method]
        if mdf.empty:
            continue
        # Average across seeds, bin by samples_seen
        agg = mdf.groupby("samples_seen")["val_accuracy"].mean().reset_index()
        ax.plot(
            agg["samples_seen"], agg["val_accuracy"],
            color=METHOD_COLORS[method], label=METHOD_LABELS[method],
            linewidth=1.8,
        )

    ax.set_xlabel("Cumulative Training Samples Seen")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Learning Efficiency: Accuracy vs Samples Seen")
    ax.legend(loc="lower right")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    save_fig(fig, "accuracy_vs_samples", figures_dir)


# -----------------------------------------------------------------------
# Figure 4: Accuracy vs wall-clock time
# -----------------------------------------------------------------------

def plot_accuracy_vs_time(all_df: pd.DataFrame, figures_dir: Path) -> None:
    if all_df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for method in METHOD_COLORS:
        mdf = all_df[all_df["method"] == method]
        if mdf.empty:
            continue
        agg = mdf.groupby("wall_clock_seconds")["val_accuracy"].mean().reset_index()
        ax.plot(
            agg["wall_clock_seconds"], agg["val_accuracy"],
            color=METHOD_COLORS[method], label=METHOD_LABELS[method],
            linewidth=1.8,
        )

    ax.set_xlabel("Wall-Clock Training Time (seconds)")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Accuracy vs Training Time")
    ax.legend(loc="lower right")
    save_fig(fig, "accuracy_vs_time", figures_dir)


# -----------------------------------------------------------------------
# Figure 5: F1 vs budget
# -----------------------------------------------------------------------

def plot_f1_vs_budget(mean_std: pd.DataFrame, figures_dir: Path) -> None:
    if mean_std.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for method in METHOD_COLORS:
        mdf = mean_std[mean_std["method"] == method].sort_values("budget_pct")
        if mdf.empty:
            continue
        ax.errorbar(
            mdf["budget_pct"], mdf["val_f1_mean"],
            yerr=mdf["val_f1_std"].fillna(0),
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
            linewidth=1.8, capsize=4, capthick=1.2,
        )

    ax.set_xlabel("Training Data Budget (%)")
    ax.set_ylabel("Macro F1 (mean ± std)")
    ax.set_title("Macro F1 vs Data Budget")
    ax.set_xticks([10, 25, 50, 75, 100])
    ax.legend(loc="lower right")
    save_fig(fig, "f1_vs_budget", figures_dir)


# -----------------------------------------------------------------------
# Figure 6: Pareto frontier
# -----------------------------------------------------------------------

def plot_pareto_frontier(best: pd.DataFrame, figures_dir: Path) -> None:
    if best.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for method in METHOD_COLORS:
        mdf = best[best["method"] == method]
        if mdf.empty:
            continue
        ax.scatter(
            mdf["wall_clock_seconds"],
            mdf["val_accuracy"],
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            s=60, alpha=0.7,
        )

    ax.set_xlabel("Total Training Time (seconds)")
    ax.set_ylabel("Best Validation Accuracy")
    ax.set_title("Pareto: Accuracy vs Training Time")
    ax.legend(loc="lower right")
    save_fig(fig, "pareto_frontier", figures_dir)


# -----------------------------------------------------------------------
# Figure 7: Ablation placeholder
# -----------------------------------------------------------------------

def plot_ablation_placeholder(figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.text(0.5, 0.5, "[TBD: E006 ablation results]\nRun E006 ablation experiments first.",
            ha="center", va="center", fontsize=12, transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="#f5f5f5", alpha=0.8))
    ax.set_title("Ablation Study")
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Validation Accuracy")
    save_fig(fig, "ablation", figures_dir)


# -----------------------------------------------------------------------
# Figure 8: Failure analysis placeholder
# -----------------------------------------------------------------------

def plot_failure_placeholder(figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.text(0.5, 0.5, "[TBD: E007-E009 failure analysis results]\nRun failure analysis experiments first.",
            ha="center", va="center", fontsize=12, transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="#f5f5f5", alpha=0.8))
    ax.set_title("Failure Analysis")
    ax.set_xlabel("Noise Level / Condition")
    ax.set_ylabel("Validation Accuracy")
    save_fig(fig, "failure_analysis", figures_dir)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate all publication figures.")
    p.add_argument("--results-dir", default="results", help="Results directory.")
    p.add_argument("--figures-dir", default="figures", help="Output figures directory.")
    p.add_argument("--budget", type=int, default=25, help="Budget % for learning curves.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("[figures] Loading processed results...")
    all_df = pd.DataFrame()
    mean_std = pd.DataFrame()
    best = pd.DataFrame()

    all_path = results_dir / "processed" / "all_epochs.csv"
    mean_path = results_dir / "processed" / "mean_std.csv"
    best_path = results_dir / "processed" / "best_per_run.csv"

    if all_path.exists():
        all_df = pd.read_csv(all_path)
        if "budget_fraction" in all_df.columns:
            all_df["budget_pct"] = (all_df["budget_fraction"] * 100).round(0).astype(int)

    if mean_path.exists():
        mean_std = pd.read_csv(mean_path)
        if "budget_fraction" in mean_std.columns:
            mean_std["budget_pct"] = (mean_std["budget_fraction"] * 100).round(0).astype(int)

    if best_path.exists():
        best = pd.read_csv(best_path)
        if "budget_fraction" in best.columns:
            best["budget_pct"] = (best["budget_fraction"] * 100).round(0).astype(int)

    if all_df.empty and mean_std.empty:
        print("[figures] No processed data found. Run experiments and aggregate first.")
        print("  Generating placeholder figures...")
    else:
        print(f"[figures] Loaded {len(all_df)} epoch rows, {len(mean_std)} mean/std rows")

    print("\n[figures] Generating figures...")
    plot_learning_curves(all_df, figures_dir, budget_pct=args.budget)
    plot_accuracy_vs_budget(mean_std, figures_dir)
    plot_accuracy_vs_samples(all_df, figures_dir)
    plot_accuracy_vs_time(all_df, figures_dir)
    plot_f1_vs_budget(mean_std, figures_dir)
    plot_pareto_frontier(best, figures_dir)
    plot_ablation_placeholder(figures_dir)
    plot_failure_placeholder(figures_dir)

    print(f"\n[figures] All figures saved to {figures_dir}/")


if __name__ == "__main__":
    main()
