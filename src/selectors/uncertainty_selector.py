"""
src/selectors/uncertainty_selector.py

Baseline C: Uncertainty-based selection.

Selects samples with the highest predictive uncertainty, measured by
Shannon entropy of the softmax probability distribution.

H(x) = -sum_c p(c|x) * log(p(c|x))

High entropy → the model is maximally uncertain about the class.
This is analogous to entropy-based active learning query strategies.

Unlike active learning, labels are available for all pool samples.
We use uncertainty purely as a training-efficiency signal.

Selection rule: top-n samples by softmax entropy.

Computational cost:
  - One forward pass over pool per scoring step: O(N)
  - Sorting: O(N log N)

Limitation:
  Entropy alone does not capture diversity. Highly uncertain samples
  may cluster in a few confusing regions, leaving most of the
  distribution under-represented.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from src.selectors.base_selector import BaseSelector
from src.selectors._scoring_utils import compute_per_sample_scores, normalize_scores


class UncertaintySelector(BaseSelector):
    """
    Select the top-n highest-entropy (most uncertain) samples.

    Args:
        seed: Random seed for tie-breaking.
        score_every: Recompute frequency in epochs.
        batch_size: Batch size for scoring forward pass.
        warmup_epochs: Epochs to use random selection before uncertainty kicks in.
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
        pool_size = len(pool)
        if n >= pool_size:
            return list(range(pool_size))

        if model is None or self._epoch <= self.warmup_epochs:
            return self._random_fallback(pool_size, n)

        if (
            self._cached_scores is None
            or (self._epoch - self._last_scored_epoch) >= self.score_every
        ):
            _, entropies, _, _ = compute_per_sample_scores(
                pool, model, device, self.batch_size
            )
            self._cached_scores = normalize_scores(entropies)
            self._last_scored_epoch = self._epoch

        # Top-n by entropy
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
        return "uncertainty"
