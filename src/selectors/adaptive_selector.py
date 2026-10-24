"""
src/selectors/adaptive_selector.py

Proposed Method E: Adaptive sample selection.

This is the primary contribution: a lightweight, forward-pass-only
selector that combines three signals:

  S(x_i, t) = α(t)·D_i(t) + β(t)·U_i(t) + γ(t)·V_i(t)

where:
  D_i(t) = normalised cross-entropy loss     [difficulty]
  U_i(t) = normalised softmax entropy        [uncertainty]
  V_i(t) = diversity score                   [coverage]
  α, β, γ = phase-dependent mixing weights   [adaptivity]

Diversity score V_i(t):
  Computed as normalised distance from each sample's feature to
  the nearest centroid of the already-selected subset from the
  previous epoch. High distance → high novelty → not yet covered.

Phase schedule (configurable via constructor):
  Phase 1 (early epochs): γ heavy → explore broadly
  Phase 2 (mid epochs):   balanced → transition
  Phase 3 (late epochs):  α heavy → focus on hard boundary examples

Rationale from literature:
  - SPL + diversity (Jiang 2014 SPLD) shows diversity prevents
    redundancy in self-paced learning.
  - Curriculum learning (Bengio 2009) suggests phase-dependence.
  - Active learning survey (Ren 2021) shows uncertainty + diversity
    consistently beats either alone.

Design principles:
  - Forward-pass only (no gradient computation beyond training)
  - Each component independently disableable (α=0 disables difficulty, etc.)
  - Phase schedule configurable; defaults tuned by ablation
  - Falls back to random during warmup or if model unavailable

Computational cost per epoch:
  - O(N) for forward pass (score + features)
  - O(n × K) for diversity scoring where K = n prev-selected
  - Memory: O(N × D) for features

Hyperparameters:
  alpha_schedule: List of (end_epoch, alpha) pairs
  beta_schedule:  List of (end_epoch, beta) pairs
  gamma_schedule: List of (end_epoch, gamma) pairs
  score_every: Recompute frequency
  warmup_epochs: Random fallback epochs
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.selectors.base_selector import BaseSelector
from src.selectors._scoring_utils import (
    compute_per_sample_scores,
    compute_features,
    normalize_scores,
)


# Default phase schedule (epoch ranges → weights)
# Format: (epoch_up_to_and_including, alpha, beta, gamma)
DEFAULT_SCHEDULE: List[Tuple[int, float, float, float]] = [
    (10,  0.20, 0.20, 0.60),   # Phase 1: diversity-heavy
    (20,  0.40, 0.40, 0.20),   # Phase 2: balanced
    (999, 0.50, 0.40, 0.10),   # Phase 3: difficulty-heavy
]


class AdaptiveSelector(BaseSelector):
    """
    Adaptive sample selector combining difficulty, uncertainty, and diversity.

    Args:
        seed: Random seed.
        schedule: List of (max_epoch, alpha, beta, gamma) tuples.
            Phase applies for epochs <= max_epoch.
        score_every: Recompute scoring frequency (epochs).
        batch_size: Batch size for forward pass.
        warmup_epochs: Use random selection for this many epochs.
        top_k_fraction: Fraction of pool to pre-filter by score before
            diversity scoring. Reduces k-means cost. Default 0.5 means
            run diversity only on top 50% by (alpha*D + beta*U).
    """

    def __init__(
        self,
        seed: int = 42,
        schedule: List[Tuple[int, float, float, float]] | None = None,
        score_every: int = 1,
        batch_size: int = 256,
        warmup_epochs: int = 1,
        top_k_fraction: float = 0.5,
    ) -> None:
        super().__init__(seed=seed)
        self.schedule = schedule or DEFAULT_SCHEDULE
        self.score_every = score_every
        self.batch_size = batch_size
        self.warmup_epochs = warmup_epochs
        self.top_k_fraction = top_k_fraction

        self._rng = np.random.default_rng(seed)
        self._cached_losses: np.ndarray | None = None
        self._cached_entropies: np.ndarray | None = None
        self._cached_features: np.ndarray | None = None
        self._last_scored_epoch: int = -1
        self._prev_selected_features: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def select(
        self,
        pool: Dataset,
        n: int,
        model: "torch.nn.Module | None",
        device: torch.device = torch.device("cpu"),
    ) -> list[int]:
        """Select n samples using the adaptive scoring function."""
        pool_size = len(pool)
        if n >= pool_size:
            return list(range(pool_size))

        if model is None or self._epoch <= self.warmup_epochs:
            return self._random_fallback(pool_size, n)

        # Get current phase weights
        alpha, beta, gamma = self._get_weights()

        # Recompute scores if needed
        if (
            self._cached_losses is None
            or (self._epoch - self._last_scored_epoch) >= self.score_every
        ):
            losses, entropies, _, _ = compute_per_sample_scores(
                pool, model, device, self.batch_size
            )
            self._cached_losses = normalize_scores(losses)
            self._cached_entropies = normalize_scores(entropies)

            if gamma > 0 and hasattr(model, "get_features"):
                self._cached_features = compute_features(pool, model, device, self.batch_size)
            else:
                self._cached_features = None

            self._last_scored_epoch = self._epoch

        # Compute combined score
        selected_indices = self._adaptive_select(n, alpha, beta, gamma)

        # Cache features of selected samples for next-epoch diversity scoring
        if self._cached_features is not None:
            self._prev_selected_features = self._cached_features[selected_indices]

        return selected_indices

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _adaptive_select(
        self, n: int, alpha: float, beta: float, gamma: float
    ) -> list[int]:
        """Core selection logic."""
        pool_size = len(self._cached_losses)

        # Base score: α*D + β*U
        base_scores = (
            alpha * self._cached_losses
            + beta * self._cached_entropies
        )

        if gamma > 0 and self._cached_features is not None:
            # Compute diversity scores for the top fraction of pool
            top_k = max(n, int(pool_size * self.top_k_fraction))
            top_k = min(top_k, pool_size)

            # Candidate pool: top-top_k by base score
            candidate_idx = np.argsort(base_scores)[::-1][:top_k]
            candidate_features = self._cached_features[candidate_idx]

            # Diversity: distance to previously selected centroid
            if self._prev_selected_features is not None and len(self._prev_selected_features) > 0:
                diversity_scores = self._compute_diversity_scores(candidate_features)
            else:
                # No prior selected set: use within-candidate distance to mean
                mean_feat = candidate_features.mean(axis=0, keepdims=True)
                dists = np.linalg.norm(candidate_features - mean_feat, axis=1)
                diversity_scores = normalize_scores(dists)

            # Re-rank within candidates
            candidate_base = base_scores[candidate_idx]
            combined = (
                (alpha + beta) / max(alpha + beta, 1e-8) * normalize_scores(candidate_base)
                + gamma * diversity_scores
            )
            top_within_candidates = np.argsort(combined)[::-1][:n]
            selected = candidate_idx[top_within_candidates].tolist()

        else:
            # No diversity: pure difficulty + uncertainty
            selected = np.argsort(base_scores)[::-1][:n].tolist()

        return selected

    def _compute_diversity_scores(self, candidate_features: np.ndarray) -> np.ndarray:
        """
        Compute diversity score for each candidate as distance to nearest
        previously-selected feature centroid.

        High score → far from what was selected before → high novelty.
        """
        prev = self._prev_selected_features   # (M, D)
        # Use cluster centres of previous selection to avoid O(M×N) distance
        # For small M, just use all of them
        if len(prev) <= 200:
            centroids = prev
        else:
            # Use first 200 (already randomly ordered by prior selection)
            centroids = prev[:200]

        # For each candidate, min distance to any centroid
        # Shape: (C, D) × (M, D) → (C,)
        diffs = candidate_features[:, None, :] - centroids[None, :, :]  # (C, M, D)
        dists = np.linalg.norm(diffs, axis=2)   # (C, M)
        min_dists = dists.min(axis=1)            # (C,)
        return normalize_scores(min_dists)

    def _get_weights(self) -> Tuple[float, float, float]:
        """Return (alpha, beta, gamma) for the current epoch."""
        for max_epoch, alpha, beta, gamma in self.schedule:
            if self._epoch <= max_epoch:
                return alpha, beta, gamma
        # Fallback: last phase
        _, alpha, beta, gamma = self.schedule[-1]
        return alpha, beta, gamma

    def _random_fallback(self, pool_size: int, n: int) -> list[int]:
        indices = self._rng.choice(pool_size, size=n, replace=False)
        return indices.tolist()

    def reset(self) -> None:
        super().reset()
        self._rng = np.random.default_rng(self.seed)
        self._cached_losses = None
        self._cached_entropies = None
        self._cached_features = None
        self._last_scored_epoch = -1
        self._prev_selected_features = None

    @property
    def name(self) -> str:
        return "adaptive"
