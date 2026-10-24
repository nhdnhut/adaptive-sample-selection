"""
scripts/aggregate_results.py

Aggregate raw CSV results into summary tables.

Usage:
    python scripts/aggregate_results.py [--results-dir results]

Outputs:
    results/processed/all_epochs.csv    -- all raw epoch rows merged
    results/processed/best_per_run.csv  -- best val_accuracy per (method, budget, seed)
    results/processed/mean_std.csv      -- mean ± std per (method, budget) across seeds
    results/summaries/experiment_table.md -- publication-ready markdown table
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate experiment results.")
    p.add_argument("--results-dir", default="results", help="Results directory.")
    return p.parse_args()


def load_all_raw(raw_dir: Path) -> pd.DataFrame:
    """Load and concatenate all raw CSV files."""
    csvs = sorted(raw_dir.glob("*.csv"))
    if not csvs:
        print(f"[warn] No CSV files found in {raw_dir}")
        return pd.DataFrame()

    dfs = []
    for csv_path in csvs:
        try:
            df = pd.read_csv(csv_path)
            dfs.append(df)
        except Exception as e:
            print(f"[warn] Failed to read {csv_path}: {e}")

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined["budget_pct"] = (combined["budget_fraction"] * 100).round(0).astype(int)
    return combined


def compute_best_per_run(df: pd.DataFrame) -> pd.DataFrame:
    """For each (method, budget_fraction, seed), find the epoch with best val_accuracy."""
    if df.empty:
        return df
    idx = df.groupby(["method", "budget_fraction", "seed"])["val_accuracy"].idxmax()
    best = df.loc[idx].copy()
    return best.reset_index(drop=True)


def compute_mean_std(best: pd.DataFrame) -> pd.DataFrame:
    """For each (method, budget_fraction), compute mean ± std across seeds."""
    if best.empty:
        return best
    agg = (
        best.groupby(["method", "budget_fraction"])
        .agg(
            val_accuracy_mean=("val_accuracy", "mean"),
            val_accuracy_std=("val_accuracy", "std"),
            val_f1_mean=("val_f1_macro", "mean"),
            val_f1_std=("val_f1_macro", "std"),
            samples_seen_mean=("samples_seen", "mean"),
            wall_clock_mean=("wall_clock_seconds", "mean"),
            n_seeds=("seed", "count"),
        )
        .reset_index()
    )
    agg["budget_pct"] = (agg["budget_fraction"] * 100).round(0).astype(int)
    return agg


def make_markdown_table(mean_std: pd.DataFrame) -> str:
    """Create a publication-style markdown table."""
    if mean_std.empty:
        return "No results available yet."

    lines = []
    lines.append("| Method | Budget | Val Acc (mean ± std) | Val F1 (mean ± std) | Samples Seen | Time (s) |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    methods = ["random", "difficulty", "uncertainty", "diversity", "adaptive"]
    budgets = sorted(mean_std["budget_pct"].unique())

    for method in methods:
        for bpct in budgets:
            row = mean_std[
                (mean_std["method"] == method) & (mean_std["budget_pct"] == bpct)
            ]
            if row.empty:
                lines.append(f"| {method} | {bpct}% | — | — | — | — |")
                continue
            r = row.iloc[0]
            acc_str = f"{r['val_accuracy_mean']:.4f} ± {r['val_accuracy_std']:.4f}" if pd.notna(r.get('val_accuracy_std', np.nan)) else f"{r['val_accuracy_mean']:.4f}"
            f1_str = f"{r['val_f1_mean']:.4f} ± {r['val_f1_std']:.4f}" if pd.notna(r.get('val_f1_std', np.nan)) else f"{r['val_f1_mean']:.4f}"
            lines.append(
                f"| {method} | {bpct}% | {acc_str} | {f1_str} | "
                f"{r['samples_seen_mean']:.0f} | {r['wall_clock_mean']:.0f} |"
            )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    raw_dir = results_dir / "raw"
    proc_dir = results_dir / "processed"
    summ_dir = results_dir / "summaries"

    proc_dir.mkdir(parents=True, exist_ok=True)
    summ_dir.mkdir(parents=True, exist_ok=True)

    print(f"[aggregate] Loading raw results from {raw_dir}...")
    all_df = load_all_raw(raw_dir)

    if all_df.empty:
        print("[aggregate] No data to aggregate. Run experiments first.")
        return

    print(f"[aggregate] Total rows: {len(all_df)}")
    all_df.to_csv(proc_dir / "all_epochs.csv", index=False)

    best = compute_best_per_run(all_df)
    best.to_csv(proc_dir / "best_per_run.csv", index=False)
    print(f"[aggregate] Best-per-run rows: {len(best)}")

    mean_std = compute_mean_std(best)
    mean_std.to_csv(proc_dir / "mean_std.csv", index=False)
    print(f"[aggregate] Mean/std rows: {len(mean_std)}")

    table_md = make_markdown_table(mean_std)
    table_path = summ_dir / "experiment_table.md"
    table_path.write_text(table_md, encoding="utf-8")
    print(f"[aggregate] Markdown table saved: {table_path}")
    print("\n" + table_md)


if __name__ == "__main__":
    main()
