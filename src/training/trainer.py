"""
src/training/trainer.py

Main training loop with selector integration.

The Trainer:
  1. Calls selector.begin_epoch() at the start of each epoch.
  2. Calls selector.select(pool, n, model, device) to get sample indices.
  3. Creates a DataLoader from those indices.
  4. Trains for one epoch on the selected subset.
  5. Evaluates on the validation set.
  6. Logs metrics via ExperimentLogger.
  7. Tracks cumulative samples_seen and gradient_updates.

Key design decisions:
  - The model is set to eval() before calling selector.select()
    so that all scoring uses consistent eval-mode statistics.
  - After selection, model is set back to train() for the gradient step.
  - Selection overhead is measured separately from training time.
  - The training protocol (optimizer, LR, loss) is identical for all selectors.
"""

from __future__ import annotations

import time
from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

from src.selectors.base_selector import BaseSelector
from src.metrics.metrics import evaluate_model, MetricBundle
from src.utils.logging import ExperimentLogger


class Trainer:
    """
    Unified training loop for all sample-selection experiments.

    Args:
        model: Neural network to train.
        pool: Training pool (Dataset or Subset).
        val_loader: DataLoader for validation set.
        selector: Sample selector instance.
        device: Compute device.
        optimizer: PyTorch optimizer (already constructed and linked to model).
        criterion: Loss function. Default: CrossEntropyLoss.
        batch_size: Training batch size.
        n_select: Number of samples to select per epoch.
            This defines the effective data budget.
        num_workers: DataLoader workers. Default 0 for CPU stability.
    """

    def __init__(
        self,
        model: nn.Module,
        pool: Dataset,
        val_loader: DataLoader,
        selector: BaseSelector,
        device: torch.device,
        optimizer: optim.Optimizer,
        criterion: nn.Module | None = None,
        batch_size: int = 128,
        n_select: int = 5000,
        num_workers: int = 0,
    ) -> None:
        self.model = model
        self.pool = pool
        self.val_loader = val_loader
        self.selector = selector
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.batch_size = batch_size
        self.n_select = n_select
        self.num_workers = num_workers

        # Cumulative counters
        self._total_samples_seen: int = 0
        self._total_gradient_updates: int = 0

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        num_epochs: int,
        logger: ExperimentLogger | None = None,
        verbose: bool = True,
    ) -> list[dict]:
        """
        Train for num_epochs and return a list of per-epoch metric dicts.

        Args:
            num_epochs: Total number of epochs to train.
            logger: Optional ExperimentLogger for persistent metric saving.
            verbose: If True, print progress to stdout.

        Returns:
            List of metric dicts, one per epoch.
        """
        history = []
        self.selector.reset()
        self._total_samples_seen = 0
        self._total_gradient_updates = 0

        for epoch in range(1, num_epochs + 1):
            self.selector.begin_epoch()

            # --- Selection phase ---
            t_sel_start = time.time()
            self.model.eval()  # eval mode for scoring
            selected_indices = self.selector.select(
                self.pool, self.n_select, self.model, self.device
            )
            selection_overhead = time.time() - t_sel_start

            # --- Training phase ---
            self.model.train()
            train_loss, train_acc, n_updates = self._train_epoch(selected_indices)
            self._total_samples_seen += len(selected_indices)
            self._total_gradient_updates += n_updates

            # --- Evaluation phase ---
            val_metrics: MetricBundle = evaluate_model(self.model, self.val_loader, self.device)

            row = {
                "epoch": epoch,
                "samples_seen": self._total_samples_seen,
                "gradient_updates": self._total_gradient_updates,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_metrics.accuracy,
                "val_loss": val_metrics.loss,
                "val_f1_macro": val_metrics.f1_macro,
                "selection_overhead_seconds": selection_overhead,
            }
            history.append(row)

            if logger is not None:
                logger.log_epoch(
                    epoch=epoch,
                    samples_seen=self._total_samples_seen,
                    gradient_updates=self._total_gradient_updates,
                    val_accuracy=val_metrics.accuracy,
                    val_loss=val_metrics.loss,
                    val_f1_macro=val_metrics.f1_macro,
                    train_accuracy=train_acc,
                    train_loss=train_loss,
                    selection_overhead_seconds=selection_overhead,
                )

            if verbose:
                print(
                    f"  Epoch {epoch:3d}/{num_epochs} | "
                    f"sel={len(selected_indices)} | "
                    f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                    f"val_acc={val_metrics.accuracy:.4f} val_f1={val_metrics.f1_macro:.4f} | "
                    f"sel_overhead={selection_overhead:.2f}s"
                )

        return history

    # ------------------------------------------------------------------
    # Epoch-level training
    # ------------------------------------------------------------------

    def _train_epoch(self, selected_indices: list[int]) -> tuple[float, float, int]:
        """
        Train for one epoch on selected_indices.

        Args:
            selected_indices: Indices into self.pool.

        Returns:
            Tuple of (mean_loss, mean_accuracy, n_gradient_updates).
        """
        subset = Subset(self.pool, selected_indices)
        loader = DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
        )

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        n_updates = 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            preds = logits.detach().argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_loss += loss.item() * len(labels)
            total_samples += len(labels)
            n_updates += 1

        mean_loss = total_loss / max(total_samples, 1)
        mean_acc = total_correct / max(total_samples, 1)
        return mean_loss, mean_acc, n_updates
