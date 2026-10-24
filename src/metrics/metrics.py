"""
src/metrics/metrics.py

Evaluation metrics for model quality and training efficiency.

Functions:
  evaluate_model(model, loader, device) → MetricBundle
  compute_macro_f1(y_true, y_pred)
  compute_accuracy(y_true, y_pred)
  compute_class_accuracy(y_true, y_pred, num_classes)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score


@dataclass
class MetricBundle:
    """Results from a single evaluation pass."""
    accuracy: float
    loss: float
    f1_macro: float
    class_accuracy: Dict[int, float] = field(default_factory=dict)
    num_samples: int = 0


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> MetricBundle:
    """
    Evaluate a model on a DataLoader.

    Args:
        model: Neural network (set to eval mode internally).
        loader: DataLoader for validation or test set.
        device: Compute device.

    Returns:
        MetricBundle with accuracy, loss, F1, and per-class accuracy.
    """
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    total_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels_dev = labels.to(device)

        logits = model(images)
        loss = F.cross_entropy(logits, labels_dev, reduction="sum")

        preds = logits.argmax(dim=1)

        total_loss += loss.item()
        total_samples += len(labels)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    accuracy = float((y_true == y_pred).mean())
    avg_loss = total_loss / max(total_samples, 1)
    f1 = compute_macro_f1(y_true, y_pred)
    class_acc = compute_class_accuracy(y_true, y_pred)

    return MetricBundle(
        accuracy=accuracy,
        loss=avg_loss,
        f1_macro=f1,
        class_accuracy=class_acc,
        num_samples=total_samples,
    )


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute macro-averaged F1 score.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        Macro F1 score in [0, 1].
    """
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Simple accuracy."""
    return float((y_true == y_pred).mean())


def compute_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[int, float]:
    """
    Per-class accuracy.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        Dict mapping class index → accuracy.
    """
    classes = np.unique(y_true)
    result = {}
    for cls in classes:
        mask = y_true == cls
        result[int(cls)] = float((y_pred[mask] == cls).mean()) if mask.sum() > 0 else 0.0
    return result
