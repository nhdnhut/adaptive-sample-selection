"""
src/selectors/_scoring_utils.py

Shared utility functions for computing per-sample scores from model forward passes.

Used by DifficultySelector, UncertaintySelector, and AdaptiveSelector.
All functions assume the model is already in eval mode.

Key functions:
  compute_per_sample_scores(pool, model, device, batch_size)
      → Tuple[losses, entropies, probs, labels]

These are forward-pass-only operations — no gradient computation.
Computational cost: one forward pass over the entire pool.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@torch.no_grad()
def compute_per_sample_scores(
    pool: Dataset,
    model: nn.Module,
    device: torch.device,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-sample difficulty and uncertainty scores via one forward pass.

    Args:
        pool: Training pool Dataset (or Subset).
        model: Neural network in eval mode.
        device: Compute device.
        batch_size: Batch size for the forward pass.

    Returns:
        Tuple of four numpy arrays, each of length len(pool):
          - losses: Cross-entropy loss per sample (difficulty proxy).
          - entropies: Softmax entropy per sample (uncertainty proxy).
          - probs: Softmax probabilities, shape (N, C).
          - labels: Ground-truth class labels.
    """
    model.eval()
    loader = DataLoader(pool, batch_size=batch_size, shuffle=False, num_workers=0)

    all_losses: list[np.ndarray] = []
    all_entropies: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device)
        labels_dev = labels.to(device)

        logits = model(images)
        probs = F.softmax(logits, dim=1)

        # Per-sample cross-entropy loss
        loss_per_sample = F.cross_entropy(logits, labels_dev, reduction="none")

        # Shannon entropy: H = -sum(p * log(p + eps))
        entropy = -(probs * (probs + 1e-9).log()).sum(dim=1)

        all_losses.append(loss_per_sample.cpu().numpy())
        all_entropies.append(entropy.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

    return (
        np.concatenate(all_losses),
        np.concatenate(all_entropies),
        np.concatenate(all_probs, axis=0),
        np.concatenate(all_labels),
    )


@torch.no_grad()
def compute_features(
    pool: Dataset,
    model: nn.Module,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Extract penultimate-layer features from model for all pool samples.

    Requires model to have a get_features(x) method.

    Args:
        pool: Training pool.
        model: Model with get_features() method.
        device: Compute device.
        batch_size: Inference batch size.

    Returns:
        Feature matrix of shape (N, D).
    """
    model.eval()
    loader = DataLoader(pool, batch_size=batch_size, shuffle=False, num_workers=0)
    all_features: list[np.ndarray] = []

    for images, _ in loader:
        images = images.to(device)
        feats = model.get_features(images)
        all_features.append(feats.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def normalize_scores(scores: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Min-max normalise an array of scores to [0, 1].

    Args:
        scores: 1-D float array.
        eps: Small constant to avoid division by zero.

    Returns:
        Normalised scores in [0, 1].
    """
    lo, hi = scores.min(), scores.max()
    return (scores - lo) / (hi - lo + eps)
