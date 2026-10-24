"""
scripts/train.py

Single-experiment entrypoint.

Usage:
    python scripts/train.py --config configs/random_25.yaml [--seed 42]
    python scripts/train.py --config configs/adaptive_10.yaml --seed 123

The script:
  1. Loads the config YAML.
  2. Overrides seed if provided via CLI.
  3. Sets reproducibility.
  4. Loads dataset and splits.
  5. Builds model and selector.
  6. Constructs optimizer.
  7. Runs training with the Trainer.
  8. Saves results to results/raw/.
  9. Prints a summary.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running from project root without installation
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data.datasets import load_fashion_mnist, split_train_val
from src.models import build_model
from src.selectors import build_selector
from src.training.trainer import Trainer
from src.utils.config import load_config, save_config, config_to_dict
from src.utils.logging import ExperimentLogger
from src.utils.reproducibility import set_seed


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train one experiment run from a YAML config."
    )
    p.add_argument("--config", required=True, help="Path to YAML config file.")
    p.add_argument("--seed", type=int, default=None, help="Override seed from config.")
    p.add_argument("--debug", action="store_true", help="Use debug dataset (5k samples).")
    p.add_argument("--quiet", action="store_true", help="Suppress per-epoch stdout.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # CLI seed override
    if args.seed is not None:
        cfg.experiment.seed = args.seed

    seed = cfg.experiment.seed
    set_seed(seed)

    device = torch.device("cpu")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    print(f"[setup] Loading Fashion-MNIST...")
    train_full, test_dataset = load_fashion_mnist(
        data_root=getattr(cfg.dataset, "data_root", "data"),
        augment_train=getattr(cfg.dataset, "augment", True),
    )

    pool, val_dataset = split_train_val(
        train_full,
        val_size=getattr(cfg.dataset, "val_size", 10_000),
        seed=seed,
    )

    if args.debug:
        from src.data.datasets import get_debug_subset
        pool = get_debug_subset(pool, n=5_000, seed=seed)
        print(f"[debug] Using debug subset: {len(pool)} samples")

    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    # ------------------------------------------------------------------
    # Budget
    # ------------------------------------------------------------------
    budget_fraction = cfg.dataset.train_fraction
    n_select = max(1, int(len(pool) * budget_fraction))
    print(f"[setup] Pool size: {len(pool)} | Budget: {budget_fraction:.0%} | n_select: {n_select}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model(
        name=cfg.model.name,
        num_classes=getattr(cfg.model, "num_classes", 10),
        dropout_p=getattr(cfg.model, "dropout_p", 0.5),
    ).to(device)
    print(f"[setup] Model: {cfg.model.name} | Params: {model.count_parameters():,}")

    # ------------------------------------------------------------------
    # Selector
    # ------------------------------------------------------------------
    selector_kwargs = {}
    if hasattr(cfg.selector, "score_every"):
        selector_kwargs["score_every"] = cfg.selector.score_every
    if hasattr(cfg.selector, "warmup_epochs"):
        selector_kwargs["warmup_epochs"] = cfg.selector.warmup_epochs

    selector = build_selector(
        method=cfg.selector.method,
        seed=seed,
        **selector_kwargs,
    )
    print(f"[setup] Selector: {selector.name}")

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=getattr(cfg.training, "weight_decay", 1e-4),
    )

    # ------------------------------------------------------------------
    # Experiment ID
    # ------------------------------------------------------------------
    method_name = selector.name
    budget_pct = int(budget_fraction * 100)
    exp_id = getattr(cfg.experiment, "id", f"EXP")
    experiment_id = f"{exp_id}_{method_name}_b{budget_pct:03d}"

    # ------------------------------------------------------------------
    # Save config
    # ------------------------------------------------------------------
    results_dir = Path(getattr(cfg.experiment, "results_dir", "results"))
    config_out = results_dir / "raw" / f"config_{experiment_id}_s{seed}.json"
    save_config(cfg, config_out)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        pool=pool,
        val_loader=val_loader,
        selector=selector,
        device=device,
        optimizer=optimizer,
        batch_size=cfg.training.batch_size,
        n_select=n_select,
    )

    print(f"\n[train] Starting: {experiment_id} | seed={seed}")
    print(f"        Epochs: {cfg.training.epochs} | batch_size: {cfg.training.batch_size}")
    print("-" * 80)

    t_start = time.time()
    with ExperimentLogger(
        experiment_id=experiment_id,
        method=method_name,
        budget_fraction=budget_fraction,
        seed=seed,
        output_dir=results_dir / "raw",
        metadata=config_to_dict(cfg),
    ) as logger:
        history = trainer.train(
            num_epochs=cfg.training.epochs,
            logger=logger,
            verbose=not args.quiet,
        )

    elapsed = time.time() - t_start

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    best_val_acc = max(r["val_accuracy"] for r in history)
    final_val_acc = history[-1]["val_accuracy"]
    total_samples = history[-1]["samples_seen"]
    total_updates = history[-1]["gradient_updates"]

    print("\n" + "=" * 80)
    print(f"DONE: {experiment_id} | seed={seed}")
    print(f"  Best val accuracy:  {best_val_acc:.4f}")
    print(f"  Final val accuracy: {final_val_acc:.4f}")
    print(f"  Total samples seen: {total_samples:,}")
    print(f"  Total grad updates: {total_updates:,}")
    print(f"  Wall-clock time:    {elapsed:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
