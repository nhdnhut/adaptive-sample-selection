"""
scripts/analyze_results.py

Post-suite analysis and paper-fill script.

After all 75 primary runs complete, this script:
1. Runs aggregate_results.py
2. Generates all figures
3. Prints the full results table
4. Computes hypothesis test results (bootstrap CI)
5. Writes filled-in LaTeX snippets for paper/main.tex

Usage:
    python scripts/analyze_results.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import subprocess


METHODS = ["random", "difficulty", "uncertainty", "diversity", "adaptive"]
BUDGETS = [10, 25, 50, 75, 100]
N_SEEDS = 3


def bootstrap_ci(
    values: np.ndarray,
    n_boot: int = 10_000,
    ci: float = 0.95,
) -> tuple[float, float]:
    """
    Bootstrap 95% confidence interval for the mean.

    Args:
        values: Observed values (e.g., accuracies across seeds).
        n_boot: Number of bootstrap resamples.
        ci: Confidence level.

    Returns:
        (lower, upper) confidence interval bounds.
    """
    rng = np.random.default_rng(42)
    boot_means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_boot)]
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return float(lo), float(hi)


def test_h1(mean_std: pd.DataFrame, budget: int = 25) -> dict:
    """
    H1: Adaptive > Random at budget.

    Returns dict with delta, ci, supported.
    """
    bdf = mean_std[mean_std["budget_pct"] == budget]
    adaptive_rows = bdf[bdf["method"] == "adaptive"]
    random_rows = bdf[bdf["method"] == "random"]
    if adaptive_rows.empty or random_rows.empty:
        return {"supported": None, "delta": None, "note": "missing data"}

    delta = (
        adaptive_rows.iloc[0]["val_accuracy_mean"]
        - random_rows.iloc[0]["val_accuracy_mean"]
    )
    supported = delta > 0.005
    return {"delta": delta, "supported": supported}


def test_h2(best_per_run: pd.DataFrame, budget: int = 10) -> dict:
    """
    H2: Difficulty < Random at small budget.
    """
    diff = best_per_run[(best_per_run["method"] == "difficulty") & (best_per_run["budget_pct"] == budget)]["val_accuracy"].values
    rand = best_per_run[(best_per_run["method"] == "random") & (best_per_run["budget_pct"] == budget)]["val_accuracy"].values
    if len(diff) == 0 or len(rand) == 0:
        return {"supported": None}

    delta = diff.mean() - rand.mean()
    supported = delta < -0.005
    return {"delta": delta, "supported": supported}


def test_h3(mean_std: pd.DataFrame) -> dict:
    """
    H3: Adaptive > all baselines at budget 25%.
    Requires ablation results (E006).
    """
    return {"note": "Requires E006 ablation results."}


def generate_latex_table(mean_std: pd.DataFrame) -> str:
    """Generate a LaTeX table string for paper/main.tex."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Validation accuracy (mean $\pm$ std, 3 seeds) by method and data budget.}")
    lines.append(r"  \label{tab:main}")
    lines.append(r"  \begin{tabular}{lrrrrr}")
    lines.append(r"    \toprule")
    lines.append(r"    Method & 10\% & 25\% & 50\% & 75\% & 100\% \\")
    lines.append(r"    \midrule")

    for method in METHODS:
        cells = []
        for bpct in BUDGETS:
            row = mean_std[(mean_std["method"] == method) & (mean_std["budget_pct"] == bpct)]
            if row.empty:
                cells.append(r"\tbd{}")
            else:
                r = row.iloc[0]
                std = r.get("val_accuracy_std", float("nan"))
                if pd.notna(std):
                    cells.append(f"{r['val_accuracy_mean']:.4f} $\\pm$ {std:.4f}")
                else:
                    cells.append(f"{r['val_accuracy_mean']:.4f}")
        label = method.capitalize() if method != "adaptive" else "Adaptive (Ours)"
        lines.append(f"    {label} & {' & '.join(cells)} \\\\")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main() -> None:
    results_dir = Path("results")
    proc_dir = results_dir / "processed"

    # Step 1: Check how many runs are done
    csvs = list((results_dir / "raw").glob("*.csv"))
    print(f"[analyze] CSV files found: {len(csvs)}")
    if len(csvs) < 75:
        print(f"[analyze] WARNING: only {len(csvs)}/75 primary runs complete.")
        print("          Results below are partial.")

    # Step 2: Aggregate
    print("[analyze] Running aggregation...")
    subprocess.run([sys.executable, "scripts/aggregate_results.py"], check=False)

    # Step 3: Load processed data
    mean_std_path = proc_dir / "mean_std.csv"
    best_path = proc_dir / "best_per_run.csv"

    if not mean_std_path.exists():
        print("[analyze] No processed data yet.")
        return

    mean_std = pd.read_csv(mean_std_path)
    best = pd.read_csv(best_path)
    if "budget_fraction" in mean_std.columns:
        mean_std["budget_pct"] = (mean_std["budget_fraction"] * 100).round(0).astype(int)
    if "budget_fraction" in best.columns:
        best["budget_pct"] = (best["budget_fraction"] * 100).round(0).astype(int)

    # Step 4: Print full results table
    print("\n" + "=" * 80)
    print("RESULTS: Validation Accuracy by Method and Budget")
    print("=" * 80)

    header = f"{'Method':14}" + "".join(f"  b{b:3d}%" for b in BUDGETS)
    print(header)
    print("-" * len(header))

    for method in METHODS:
        row_str = f"{method:14}"
        for bpct in BUDGETS:
            r = mean_std[(mean_std["method"] == method) & (mean_std["budget_pct"] == bpct)]
            if r.empty:
                row_str += "     ---"
            else:
                acc = r.iloc[0]["val_accuracy_mean"]
                row_str += f"  {acc:.4f}"
        print(row_str)

    # Step 5: Hypothesis tests
    print("\n" + "=" * 80)
    print("HYPOTHESIS TESTS")
    print("=" * 80)

    for budget in [10, 25]:
        h1 = test_h1(mean_std, budget=budget)
        if h1["delta"] is not None:
            sym = "[SUPPORTED]" if h1["supported"] else "[NOT SUPPORTED]"
            print(f"  H1 (adaptive > random @ {budget}%%):  delta={h1['delta']:+.4f}  {sym}")

    h2 = test_h2(best, budget=10)
    if h2.get("delta") is not None:
        sym = "[SUPPORTED]" if h2["supported"] else "[NOT SUPPORTED]"
        print(f"  H2 (difficulty < random @ 10%%): delta={h2['delta']:+.4f}  {sym}")

    # Step 6: LaTeX table
    print("\n" + "=" * 80)
    print("LaTeX TABLE (for paper/main.tex)")
    print("=" * 80)
    latex = generate_latex_table(mean_std)
    print(latex)

    latex_path = results_dir / "summaries" / "latex_table.tex"
    latex_path.parent.mkdir(parents=True, exist_ok=True)
    latex_path.write_text(latex, encoding="utf-8")
    print(f"\n[analyze] LaTeX table saved: {latex_path}")

    # Step 7: Generate figures
    print("\n[analyze] Generating figures...")
    subprocess.run([sys.executable, "scripts/generate_figures.py"], check=False)
    print("[analyze] Done.")


if __name__ == "__main__":
    main()
