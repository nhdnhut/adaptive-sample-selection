"""
scripts/run_ablation.py

E006: Ablation study — component contribution analysis.

Tests all 7 non-trivial component subsets of the adaptive selector
at budget 25%, 3 seeds:
  (D), (U), (V), (D+U), (D+V), (U+V), (D+U+V)

Implemented by fixing the phase schedule to constant weights.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data.datasets import load_fashion_mnist, split_train_val
from src.models import build_model
from src.selectors.adaptive_selector import AdaptiveSelector
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.logging import ExperimentLogger
from src.utils.reproducibility import set_seed


ABLATION_CONFIGS = [
    # name, (alpha, beta, gamma)
    ("D_only",    (1.0, 0.0, 0.0)),
    ("U_only",    (0.0, 1.0, 0.0)),
    ("V_only",    (0.0, 0.0, 1.0)),
    ("D_plus_U",  (0.5, 0.5, 0.0)),
    ("D_plus_V",  (0.5, 0.0, 0.5)),
    ("U_plus_V",  (0.0, 0.5, 0.5)),
    ("D_U_V",     (0.4, 0.4, 0.2)),  # equal mix
]

SEEDS = [42, 123, 777]
BUDGET_FRACTION = 0.25


def run_ablation(
    name: str,
    weights: tuple[float, float, float],
    seed: int,
    results_dir: Path,
    num_epochs: int = 30,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Run one ablation configuration and return results."""
    set_seed(seed)
    alpha, beta, gamma = weights

    # Fixed schedule: constant weights for all epochs
    schedule = [(999, alpha, beta, gamma)]

    train_full, _ = load_fashion_mnist("data", augment_train=True)
    pool, val_ds = split_train_val(train_full, val_size=10_000, seed=seed)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    n_select = max(1, int(len(pool) * BUDGET_FRACTION))
    model = build_model("small_cnn", num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    selector = AdaptiveSelector(seed=seed, schedule=schedule, warmup_epochs=1)

    exp_id = f"E006_{name}_s{seed}"

    t_start = time.time()
    with ExperimentLogger(
        experiment_id=exp_id,
        method=f"ablation_{name}",
        budget_fraction=BUDGET_FRACTION,
        seed=seed,
        output_dir=results_dir / "raw",
        metadata={"alpha": alpha, "beta": beta, "gamma": gamma},
    ) as logger:
        trainer = Trainer(
            model=model, pool=pool, val_loader=val_loader,
            selector=selector, device=device, optimizer=optimizer,
            batch_size=128, n_select=n_select,
        )
        history = trainer.train(num_epochs=num_epochs, logger=logger, verbose=False)

    elapsed = time.time() - t_start
    best_acc = max(r["val_accuracy"] for r in history)
    print(f"  {name:12} alpha={alpha:.1f} beta={beta:.1f} gamma={gamma:.1f} | "
          f"seed={seed} best_acc={best_acc:.4f} ({elapsed:.0f}s)")
    return {"name": name, "alpha": alpha, "beta": beta, "gamma": gamma,
            "seed": seed, "best_val_accuracy": best_acc}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E006 Ablation study.")
    p.add_argument("--results-dir", default="results", help="Results directory.")
    p.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)

    print("=" * 60)
    print("E006: Ablation Study — Component Contribution")
    print(f"Budget: {BUDGET_FRACTION:.0%} | Epochs: {args.epochs} | Seeds: {args.seeds}")
    print("=" * 60)

    all_results = []
    for name, weights in ABLATION_CONFIGS:
        for seed in args.seeds:
            result = run_ablation(name, weights, seed, results_dir, args.epochs)
            all_results.append(result)

    # Summary
    import pandas as pd
    df = pd.DataFrame(all_results)
    summary = df.groupby("name")["best_val_accuracy"].agg(["mean", "std"]).reset_index()
    summary = summary.sort_values("mean", ascending=False)

    print("\n[Ablation Summary]")
    for _, row in summary.iterrows():
        print(f"  {row['name']:12} acc={row['mean']:.4f} ± {row['std']:.4f}")

    out_path = results_dir / "processed" / "ablation_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
