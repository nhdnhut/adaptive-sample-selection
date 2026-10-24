"""
scripts/run_failure_analysis.py

E007-E009: Failure analysis experiments.

E007 — Label noise: 10% and 20% random label flips
E008 — Class imbalance: 5:1 majority-to-minority ratio
E009 — Redundant data: 30% exact duplicate samples

For each condition, compare random vs difficulty vs adaptive at budget 25%.

Usage:
    python scripts/run_failure_analysis.py [--experiments E007 E008 E009]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset, ConcatDataset

from src.data.datasets import load_fashion_mnist, split_train_val
from src.models import build_model
from src.selectors import build_selector
from src.training.trainer import Trainer
from src.utils.logging import ExperimentLogger
from src.utils.reproducibility import set_seed


BUDGET = 0.25
SEEDS = [42, 123, 777]
METHODS = ["random", "difficulty", "adaptive"]
NUM_EPOCHS = 30
DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Data manipulation helpers
# ---------------------------------------------------------------------------

def inject_label_noise(pool, noise_rate: float, seed: int):
    """
    Randomly flip `noise_rate` fraction of labels in pool.
    Returns a new TensorDataset with corrupted labels.
    """
    rng = np.random.default_rng(seed)
    loader = DataLoader(pool, batch_size=len(pool), shuffle=False, num_workers=0)
    images, labels = next(iter(loader))
    labels_np = labels.numpy().copy()

    n_corrupt = int(len(labels_np) * noise_rate)
    corrupt_indices = rng.choice(len(labels_np), size=n_corrupt, replace=False)
    for idx in corrupt_indices:
        # Flip to a random different class
        original = labels_np[idx]
        new_label = rng.choice([c for c in range(10) if c != original])
        labels_np[idx] = new_label

    return TensorDataset(images, torch.from_numpy(labels_np))


def create_imbalanced_pool(pool, imbalance_ratio: float, majority_class: int, seed: int):
    """
    Create an imbalanced pool where majority_class has imbalance_ratio×
    more samples than other classes.
    Undersample all non-majority classes proportionally.
    """
    rng = np.random.default_rng(seed)
    loader = DataLoader(pool, batch_size=len(pool), shuffle=False, num_workers=0)
    images, labels = next(iter(loader))

    labels_np = labels.numpy()
    per_class_indices = {}
    for c in range(10):
        per_class_indices[c] = np.where(labels_np == c)[0]

    # Target size for minority classes
    minority_size = int(len(per_class_indices[majority_class]) / imbalance_ratio)
    minority_size = max(1, minority_size)

    selected = list(per_class_indices[majority_class])
    for c in range(10):
        if c != majority_class:
            chosen = rng.choice(per_class_indices[c], size=min(minority_size, len(per_class_indices[c])), replace=False)
            selected.extend(chosen.tolist())

    rng.shuffle(selected)
    return Subset(pool, selected)


def inject_duplicates(pool, duplicate_fraction: float, seed: int):
    """
    Add duplicate_fraction × N duplicate samples to the pool.
    Duplicates are exact copies of randomly chosen existing samples.
    """
    rng = np.random.default_rng(seed)
    pool_size = len(pool)
    n_dupes = int(pool_size * duplicate_fraction)
    dupe_indices = rng.choice(pool_size, size=n_dupes, replace=True)
    dupe_subset = Subset(pool, dupe_indices.tolist())
    return ConcatDataset([pool, dupe_subset])


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_experiment(
    pool,
    val_loader,
    method: str,
    seed: int,
    exp_id: str,
    results_dir: Path,
    n_select: int,
    metadata: dict = None,
) -> float:
    set_seed(seed)
    model = build_model("small_cnn", num_classes=10).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    selector = build_selector(method, seed=seed, warmup_epochs=1)

    t_start = time.time()
    with ExperimentLogger(
        experiment_id=f"{exp_id}_{method}_s{seed}",
        method=method,
        budget_fraction=BUDGET,
        seed=seed,
        output_dir=results_dir / "raw",
        metadata=metadata or {},
    ) as logger:
        trainer = Trainer(
            model=model, pool=pool, val_loader=val_loader,
            selector=selector, device=DEVICE, optimizer=optimizer,
            batch_size=128, n_select=n_select,
        )
        history = trainer.train(num_epochs=NUM_EPOCHS, logger=logger, verbose=False)

    elapsed = time.time() - t_start
    best_acc = max(r["val_accuracy"] for r in history)
    print(f"  {method:12} seed={seed} best_acc={best_acc:.4f} ({elapsed:.0f}s)")
    return best_acc


# ---------------------------------------------------------------------------
# E007: Label noise
# ---------------------------------------------------------------------------

def run_e007(results_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("E007: Label Noise Robustness")
    print("=" * 60)

    train_full, _ = load_fashion_mnist("data", augment_train=True)
    _, val_ds = split_train_val(train_full, val_size=10_000, seed=42)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    for noise_rate in [0.10, 0.20]:
        print(f"\n  Noise rate: {noise_rate:.0%}")
        base_pool, _ = split_train_val(train_full, val_size=10_000, seed=42)
        n_select = max(1, int(len(base_pool) * BUDGET))

        for seed in SEEDS:
            noisy_pool = inject_label_noise(base_pool, noise_rate, seed)
            for method in METHODS:
                run_experiment(
                    noisy_pool, val_loader, method, seed,
                    f"E007_noise{int(noise_rate*100):02d}", results_dir,
                    n_select, metadata={"noise_rate": noise_rate},
                )


# ---------------------------------------------------------------------------
# E008: Class imbalance
# ---------------------------------------------------------------------------

def run_e008(results_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("E008: Class Imbalance (5:1 ratio)")
    print("=" * 60)

    train_full, _ = load_fashion_mnist("data", augment_train=True)
    base_pool, val_ds = split_train_val(train_full, val_size=10_000, seed=42)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    for seed in SEEDS:
        imbalanced = create_imbalanced_pool(base_pool, imbalance_ratio=5.0, majority_class=0, seed=seed)
        n_select = max(1, int(len(imbalanced) * BUDGET))
        print(f"\n  Seed={seed}, pool size={len(imbalanced)}")
        for method in METHODS:
            run_experiment(
                imbalanced, val_loader, method, seed,
                "E008_imbalance5to1", results_dir, n_select,
                metadata={"imbalance_ratio": 5.0, "majority_class": 0},
            )


# ---------------------------------------------------------------------------
# E009: Redundant data
# ---------------------------------------------------------------------------

def run_e009(results_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("E009: Redundant Data (30% duplicates)")
    print("=" * 60)

    train_full, _ = load_fashion_mnist("data", augment_train=True)
    base_pool, val_ds = split_train_val(train_full, val_size=10_000, seed=42)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    for seed in SEEDS:
        redundant = inject_duplicates(base_pool, duplicate_fraction=0.30, seed=seed)
        n_select = max(1, int(len(base_pool) * BUDGET))  # budget based on original size
        print(f"\n  Seed={seed}, pool size={len(redundant)} (30% dupes added)")
        for method in METHODS:
            run_experiment(
                redundant, val_loader, method, seed,
                "E009_dupes30", results_dir, n_select,
                metadata={"duplicate_fraction": 0.30},
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Failure analysis experiments E007-E009.")
    p.add_argument("--experiments", nargs="+", default=["E007", "E008", "E009"],
                   help="Which experiments to run.")
    p.add_argument("--results-dir", default="results", help="Results directory.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    to_run = set(e.upper() for e in args.experiments)

    if "E007" in to_run:
        run_e007(results_dir)
    if "E008" in to_run:
        run_e008(results_dir)
    if "E009" in to_run:
        run_e009(results_dir)

    print("\n[failure analysis] All done.")


if __name__ == "__main__":
    main()
