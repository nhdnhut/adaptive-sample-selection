"""
scripts/run_robustness_check.py

E010: Architecture robustness — repeat core E001/E005 comparison using SmallMLP.

Tests random vs adaptive at budget 25%, 3 seeds on SmallMLP.
If adaptive outperforms random on MLP too, result generalises across architectures.

Usage:
    python scripts/run_robustness_check.py
"""

from __future__ import annotations

import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data.datasets import load_fashion_mnist, split_train_val
from src.models import build_model
from src.selectors import build_selector
from src.training.trainer import Trainer
from src.utils.logging import ExperimentLogger
from src.utils.reproducibility import set_seed


BUDGET = 0.25
SEEDS = [42, 123, 777]
NUM_EPOCHS = 30
DEVICE = torch.device("cpu")


def run_one(model_name: str, method: str, seed: int, results_dir: Path) -> float:
    set_seed(seed)

    train_full, _ = load_fashion_mnist("data", augment_train=True)
    pool, val_ds = split_train_val(train_full, val_size=10_000, seed=seed)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    n_select = max(1, int(len(pool) * BUDGET))
    dropout = 0.3 if model_name == "small_mlp" else 0.5
    model = build_model(model_name, num_classes=10, dropout_p=dropout).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    selector = build_selector(method, seed=seed, warmup_epochs=1)

    exp_id = f"E010_{model_name}_{method}_s{seed}"

    t_start = time.time()
    with ExperimentLogger(
        experiment_id=exp_id,
        method=method,
        budget_fraction=BUDGET,
        seed=seed,
        output_dir=results_dir / "raw",
        metadata={"model": model_name},
    ) as logger:
        trainer = Trainer(
            model=model, pool=pool, val_loader=val_loader,
            selector=selector, device=DEVICE, optimizer=optimizer,
            batch_size=128, n_select=n_select,
        )
        history = trainer.train(num_epochs=NUM_EPOCHS, logger=logger, verbose=False)

    elapsed = time.time() - t_start
    best_acc = max(r["val_accuracy"] for r in history)
    print(f"  {model_name} {method:12} seed={seed} best_acc={best_acc:.4f} ({elapsed:.0f}s)")
    return best_acc


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="E010 Architecture robustness check.")
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()
    results_dir = Path(args.results_dir)

    print("=" * 60)
    print("E010: Architecture Robustness (SmallMLP)")
    print("=" * 60)

    results = {}
    for model_name in ["small_mlp"]:
        for method in ["random", "adaptive"]:
            accs = []
            for seed in SEEDS:
                acc = run_one(model_name, method, seed, results_dir)
                accs.append(acc)
            import numpy as np
            results[(model_name, method)] = accs
            print(f"  {model_name} {method:12} mean={np.mean(accs):.4f} ± {np.std(accs):.4f}")

    print("\n[E010] Done.")


if __name__ == "__main__":
    main()
