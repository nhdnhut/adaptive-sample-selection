"""
src/utils/logging.py

Structured experiment logging.
Saves per-epoch metrics to CSV and a summary JSON at the end of each run.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any


class ExperimentLogger:
    """
    Logs per-epoch metrics to a CSV file and writes a summary JSON at the end.

    Columns saved per epoch:
        experiment_id, method, budget_fraction, seed, epoch,
        samples_seen, gradient_updates, wall_clock_seconds,
        val_accuracy, val_loss, val_f1_macro,
        train_accuracy, train_loss, selection_overhead_seconds

    Usage::

        logger = ExperimentLogger(
            experiment_id="E001",
            method="random",
            budget_fraction=0.25,
            seed=42,
            output_dir="results/raw",
        )
        with logger:
            for epoch in range(30):
                logger.log_epoch(...)
    """

    # Fixed CSV column order (used in header)
    COLUMNS = [
        "experiment_id",
        "method",
        "budget_fraction",
        "seed",
        "epoch",
        "samples_seen",
        "gradient_updates",
        "wall_clock_seconds",
        "val_accuracy",
        "val_loss",
        "val_f1_macro",
        "train_accuracy",
        "train_loss",
        "selection_overhead_seconds",
    ]

    def __init__(
        self,
        experiment_id: str,
        method: str,
        budget_fraction: float,
        seed: int,
        output_dir: str | Path = "results/raw",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.experiment_id = experiment_id
        self.method = method
        self.budget_fraction = budget_fraction
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.metadata = metadata or {}

        self._rows: list[dict] = []
        self._start_wall: float = 0.0
        self._csv_path: Path | None = None
        self._csv_file = None
        self._csv_writer = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ExperimentLogger":
        self._start_wall = time.time()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        fname = f"{self.experiment_id}_{self.method}_b{int(self.budget_fraction*100):03d}_s{self.seed}.csv"
        self._csv_path = self.output_dir / fname
        self._csv_file = self._csv_path.open("w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.COLUMNS)
        self._csv_writer.writeheader()
        return self

    def __exit__(self, *args) -> None:
        if self._csv_file:
            self._csv_file.close()
        self._write_summary()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_epoch(
        self,
        epoch: int,
        samples_seen: int,
        gradient_updates: int,
        val_accuracy: float,
        val_loss: float,
        val_f1_macro: float,
        train_accuracy: float,
        train_loss: float,
        selection_overhead_seconds: float = 0.0,
    ) -> None:
        """
        Record metrics for one epoch.

        Args:
            epoch: Current epoch number (1-indexed).
            samples_seen: Cumulative training samples processed so far.
            gradient_updates: Cumulative gradient steps so far.
            val_accuracy: Validation accuracy (0–1).
            val_loss: Validation cross-entropy loss.
            val_f1_macro: Macro-averaged F1 on validation set.
            train_accuracy: Training accuracy on selected subset (0–1).
            train_loss: Training loss on selected subset.
            selection_overhead_seconds: Time spent in selector this epoch.
        """
        elapsed = time.time() - self._start_wall
        row = {
            "experiment_id": self.experiment_id,
            "method": self.method,
            "budget_fraction": self.budget_fraction,
            "seed": self.seed,
            "epoch": epoch,
            "samples_seen": samples_seen,
            "gradient_updates": gradient_updates,
            "wall_clock_seconds": round(elapsed, 2),
            "val_accuracy": round(val_accuracy, 6),
            "val_loss": round(val_loss, 6),
            "val_f1_macro": round(val_f1_macro, 6),
            "train_accuracy": round(train_accuracy, 6),
            "train_loss": round(train_loss, 6),
            "selection_overhead_seconds": round(selection_overhead_seconds, 4),
        }
        self._rows.append(row)
        if self._csv_writer:
            self._csv_writer.writerow(row)
            self._csv_file.flush()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _write_summary(self) -> None:
        """Write a JSON summary of the best results for this run."""
        if not self._rows:
            return

        best_val_acc = max(r["val_accuracy"] for r in self._rows)
        best_val_f1 = max(r["val_f1_macro"] for r in self._rows)
        final_row = self._rows[-1]

        summary = {
            "experiment_id": self.experiment_id,
            "method": self.method,
            "budget_fraction": self.budget_fraction,
            "seed": self.seed,
            "best_val_accuracy": best_val_acc,
            "best_val_f1_macro": best_val_f1,
            "final_val_accuracy": final_row["val_accuracy"],
            "final_val_loss": final_row["val_loss"],
            "total_samples_seen": final_row["samples_seen"],
            "total_gradient_updates": final_row["gradient_updates"],
            "total_wall_clock_seconds": final_row["wall_clock_seconds"],
            "num_epochs": len(self._rows),
            "metadata": self.metadata,
        }

        summary_dir = self.output_dir.parent / "summaries"
        summary_dir.mkdir(parents=True, exist_ok=True)
        fname = f"summary_{self.experiment_id}_{self.method}_b{int(self.budget_fraction*100):03d}_s{self.seed}.json"
        with (summary_dir / fname).open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
