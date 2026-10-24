"""
scripts/run_experiment_suite.py

Run the full primary experiment suite (E001-E005):
  - 5 methods × 5 budgets × 3 seeds = 75 runs

Usage:
    python scripts/run_experiment_suite.py [--dry-run] [--methods random difficulty]
    python scripts/run_experiment_suite.py --budgets 10 25 --seeds 42

This script runs experiments sequentially (CPU-only, safe for personal machine).
Each run is independent; failed runs print a warning and continue.

Estimated total runtime: 2-6 hours depending on hardware.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

ALL_METHODS = ["random", "difficulty", "uncertainty", "diversity", "adaptive"]
ALL_BUDGETS = [10, 25, 50, 75, 100]
ALL_SEEDS = [42, 123, 777]

EXPERIMENT_IDS = {
    "random":      "E001",
    "difficulty":  "E002",
    "uncertainty": "E003",
    "diversity":   "E004",
    "adaptive":    "E005",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full experiment suite.")
    p.add_argument("--methods", nargs="+", default=ALL_METHODS, help="Methods to run.")
    p.add_argument("--budgets", nargs="+", type=int, default=ALL_BUDGETS, help="Budgets (%%).")
    p.add_argument("--seeds", nargs="+", type=int, default=ALL_SEEDS, help="Seeds.")
    p.add_argument("--dry-run", action="store_true", help="Print commands without running.")
    p.add_argument("--quiet", action="store_true", help="Suppress per-epoch output.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    methods = [m for m in args.methods if m in ALL_METHODS]
    budgets = sorted(set(args.budgets))
    seeds = sorted(set(args.seeds))

    runs = [
        (method, budget, seed)
        for method in methods
        for budget in budgets
        for seed in seeds
    ]

    print(f"Experiment suite: {len(runs)} runs")
    print(f"  Methods: {methods}")
    print(f"  Budgets: {budgets}%")
    print(f"  Seeds:   {seeds}")
    print("-" * 60)

    results = {"success": 0, "failed": 0, "skipped": 0}
    suite_start = time.time()

    for i, (method, budget, seed) in enumerate(runs, 1):
        config_path = f"configs/{method}_{budget:03d}.yaml"
        exp_id = EXPERIMENT_IDS.get(method, "EXP")

        cmd = [
            sys.executable,
            "scripts/train.py",
            "--config", config_path,
            "--seed", str(seed),
        ]
        if args.quiet:
            cmd.append("--quiet")

        print(f"\n[{i:3d}/{len(runs)}] method={method} budget={budget}% seed={seed}")
        print(f"         cmd: {' '.join(cmd)}")

        if args.dry_run:
            results["skipped"] += 1
            continue

        run_start = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(Path(__file__).parent.parent),
                capture_output=False,
                timeout=1200,   # 20-minute timeout per run
            )
            elapsed = time.time() - run_start
            if proc.returncode == 0:
                results["success"] += 1
                print(f"         [OK] Done in {elapsed:.1f}s")
            else:
                results["failed"] += 1
                print(f"         [FAIL] returncode={proc.returncode} in {elapsed:.1f}s")
        except subprocess.TimeoutExpired:
            results["failed"] += 1
            print(f"         [FAIL] TIMEOUT after 1200s")
        except Exception as e:
            results["failed"] += 1
            print(f"         [FAIL] ERROR: {e}")

    total_elapsed = time.time() - suite_start
    print("\n" + "=" * 60)
    print(f"Suite complete in {total_elapsed:.1f}s")
    print(f"  Success: {results['success']}")
    print(f"  Failed:  {results['failed']}")
    print(f"  Skipped: {results['skipped']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
