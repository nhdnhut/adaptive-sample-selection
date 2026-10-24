"""
scripts/dashboard.py

Simple CLI research dashboard.

Usage:
    python scripts/dashboard.py

Answers:
  - What has been tested?
  - Which method performs best at each budget?
  - What is the variance?
  - What experiments are missing?
  - Which hypothesis has evidence?
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from datetime import datetime

import pandas as pd
import numpy as np


METHODS = ["random", "difficulty", "uncertainty", "diversity", "adaptive"]
BUDGETS = [10, 25, 50, 75, 100]
SEEDS = [42, 123, 777]

EXPECTED_RUNS = len(METHODS) * len(BUDGETS) * len(SEEDS)  # 75


def load_summaries(results_dir: Path) -> list[dict]:
    summaries = []
    for p in (results_dir / "summaries").glob("summary_*.json"):
        try:
            with p.open() as f:
                summaries.append(json.load(f))
        except Exception:
            pass
    return summaries


def load_mean_std(results_dir: Path) -> pd.DataFrame:
    p = results_dir / "processed" / "mean_std.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "budget_fraction" in df.columns:
        df["budget_pct"] = (df["budget_fraction"] * 100).round(0).astype(int)
    return df


def print_separator(char: str = "─", width: int = 70) -> None:
    print(char * width)


def dashboard(results_dir: Path = Path("results")) -> None:
    print()
    print_separator("═")
    print("  ADAPTIVE SAMPLE SELECTION — RESEARCH DASHBOARD")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_separator("═")

    summaries = load_summaries(results_dir)
    mean_std = load_mean_std(results_dir)

    # ------------------------------------------------------------------
    # What has been tested?
    # ------------------------------------------------------------------
    print("\n[1] TESTED CONFIGURATIONS")
    print_separator()

    completed_runs = {(s["method"], int(s["budget_fraction"] * 100), s["seed"]) for s in summaries}
    n_done = len(completed_runs)
    print(f"  Completed: {n_done} / {EXPECTED_RUNS} primary runs")

    missing = []
    for method in METHODS:
        for budget in BUDGETS:
            for seed in SEEDS:
                if (method, budget, seed) not in completed_runs:
                    missing.append((method, budget, seed))

    if missing:
        print(f"\n  Missing runs ({len(missing)}):")
        for method, budget, seed in missing[:15]:
            print(f"    {method:12} budget={budget:3}%  seed={seed}")
        if len(missing) > 15:
            print(f"    ... and {len(missing) - 15} more")
    else:
        print("  All primary runs complete!")

    # ------------------------------------------------------------------
    # Best method at each budget
    # ------------------------------------------------------------------
    print("\n[2] BEST METHOD PER BUDGET")
    print_separator()

    if mean_std.empty:
        print("  No processed results yet. Run: python scripts/aggregate_results.py")
    else:
        for budget in BUDGETS:
            bdf = mean_std[mean_std["budget_pct"] == budget]
            if bdf.empty:
                print(f"  {budget:3}%: no data")
                continue
            best_row = bdf.loc[bdf["val_accuracy_mean"].idxmax()]
            std = best_row.get("val_accuracy_std", 0)
            print(f"  {budget:3}%: {best_row['method']:12} acc={best_row['val_accuracy_mean']:.4f} ± {std:.4f}")

    # ------------------------------------------------------------------
    # Hypothesis evidence
    # ------------------------------------------------------------------
    print("\n[3] HYPOTHESIS EVIDENCE")
    print_separator()

    if mean_std.empty:
        print("  No evidence yet — run experiments first.")
    else:
        # H1: Adaptive > Random at small budgets
        for budget in [10, 25]:
            bdf = mean_std[mean_std["budget_pct"] == budget]
            adaptive_row = bdf[bdf["method"] == "adaptive"]
            random_row = bdf[bdf["method"] == "random"]
            if not adaptive_row.empty and not random_row.empty:
                delta = adaptive_row.iloc[0]["val_accuracy_mean"] - random_row.iloc[0]["val_accuracy_mean"]
                supported = "✓ SUPPORTED" if delta > 0.005 else ("~ marginal" if delta > 0 else "✗ NOT supported")
                print(f"  H1 (adaptive>random at {budget}%): Δacc={delta:+.4f} → {supported}")

        # H2: Difficulty < random at 10%
        bdf10 = mean_std[mean_std["budget_pct"] == 10]
        diff_row = bdf10[bdf10["method"] == "difficulty"]
        rand_row = bdf10[bdf10["method"] == "random"]
        if not diff_row.empty and not rand_row.empty:
            delta = diff_row.iloc[0]["val_accuracy_mean"] - rand_row.iloc[0]["val_accuracy_mean"]
            supported = "✓ SUPPORTED" if delta < -0.005 else "✗ not supported"
            print(f"  H2 (difficulty<random at 10%): Δacc={delta:+.4f} → {supported}")

    # ------------------------------------------------------------------
    # Quick stats
    # ------------------------------------------------------------------
    print("\n[4] QUICK STATS (ALL RUNS)")
    print_separator()
    if summaries:
        all_accs = [s["best_val_accuracy"] for s in summaries]
        all_times = [s["total_wall_clock_seconds"] for s in summaries]
        print(f"  Runs recorded: {len(summaries)}")
        print(f"  Accuracy range: {min(all_accs):.4f} – {max(all_accs):.4f}")
        print(f"  Time range:     {min(all_times):.0f}s – {max(all_times):.0f}s per run")
    else:
        print("  No runs recorded yet.")

    print_separator("═")
    print("  Run 'python scripts/run_experiment_suite.py' to execute experiments.")
    print("  Run 'python scripts/aggregate_results.py' to update summaries.")
    print_separator("═")
    print()


if __name__ == "__main__":
    dashboard()
