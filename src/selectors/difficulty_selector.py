"""
src/selectors/difficulty_selector.py

Baseline B: Difficulty-based selection.

Selects samples with the highest per-sample training loss.
Intuition: high-loss samples are hard for the current model and may
provide the most informative gradient signal.

This is analogous to Online Hard Example Mining (OHEM, Shrivastava 2016)
but applied to a full pool rather than within a mini-batch.

Selection rule: top-n samples by loss score.

Computational cost:
  - One forward pass over the entire pool per epoch: O(N)
  - Sorting: O(N log N)
  - Memory: O(N)

Limitation (to verify experimentally, H2):
  At very small budgets, this may select noisy/mislabelled samples,
  harming generalisation.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from src.selectors.base_selector import BaseSelector
from src.selectors._scoring_utils import compute_per_sample_scores, normalize_scores


class DifficultySelector(BaseSelector):
    """
    Select the top-n highest-loss samples from the pool.

    Args:
        seed: Random seed for tie-breaking.
        score_every: How often (in epochs) to recompute scores.
            score_every=1 recomputes every epoch (most expensive, most adaptive).
            score_every=5 recomputes every 5 epochs (cheaper, less adaptive).
        batch_size: Batch size for the scoring forward pass.
        warmup_epochs: Use random selection for this many epochs before
            difficulty-based selection starts. Allows the model to develop
            meaningful loss values before selection is applied.
    """

    def __init__(
        self,
        seed: int = 42,
        score_every: int = 1,
        batch_size: int = 256,
        warmup_epochs: int = 1,
    ) -> None:
        super().__init__(seed=seed)
        self.score_every = score_every
        self.batch_size = batch_size
        self.warmup_epochs = warmup_epochs

        self._rng = np.random.default_rng(seed)
        self._cached_scores: np.ndarray | None = None
        self._last_scored_epoch: int = -1

    def select(
        self,
        pool: Dataset,
        n: int,
        model: "torch.nn.Module | None",
        device: torch.device = torch.device("cpu"),
    ) -> list[int]:
        """
        Select n highest-loss samples.

        Falls back to random selection if model is None or during warmup.
        """
        pool_size = len(pool)
        if n >= pool_size:
            return list(range(pool_size))

        # Warmup: random selection for first few epochs
        if model is None or self._epoch <= self.warmup_epochs:
            return self._random_fallback(pool_size, n)

        # Recompute scores if needed
        if (
            self._cached_scores is None
            or (self._epoch - self._last_scored_epoch) >= self.score_every
        ):
            losses, _, _, _ = compute_per_sample_scores(
                pool, model, device, self.batch_size
            )
            self._cached_scores = normalize_scores(losses)
            self._last_scored_epoch = self._epoch

        # Select top-n by loss (highest = hardest)
        top_indices = np.argsort(self._cached_scores)[::-1][:n]
        return top_indices.tolist()

    def _random_fallback(self, pool_size: int, n: int) -> list[int]:
        indices = self._rng.choice(pool_size, size=n, replace=False)
        return indices.tolist()

    def reset(self) -> None:
        super().reset()
        self._rng = np.random.default_rng(self.seed)
        self._cached_scores = None
        self._last_scored_epoch = -1

    @property
    def name(self) -> str:
        return "difficulty"
