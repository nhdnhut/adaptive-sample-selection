"""
src/selectors/random_selector.py

Baseline A: Uniform random sampling.

Selects n samples uniformly at random from the pool without replacement.
No model feedback required.
This is the primary baseline against which all other methods are compared.

Computational complexity:
  - O(n) selection
  - O(1) memory overhead
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from src.selectors.base_selector import BaseSelector


class RandomSelector(BaseSelector):
    """
    Uniform random sampling baseline.

    Selects n indices uniformly at random from the pool, without replacement.
    Reproducible via seed; seed is advanced each epoch for variation.

    Args:
        seed: Base random seed.
    """

    def __init__(self, seed: int = 42, **kwargs) -> None:
        # kwargs (score_every, warmup_epochs) accepted but unused — for factory compatibility
        super().__init__(seed=seed)
        self._rng = np.random.default_rng(seed)

    def select(
        self,
        pool: Dataset,
        n: int,
        model: "torch.nn.Module | None" = None,
        device: torch.device = torch.device("cpu"),
    ) -> list[int]:
        """
        Select n indices uniformly at random.

        Args:
            pool: Training pool.
            n: Number of samples to select.
            model: Ignored.
            device: Ignored.

        Returns:
            List of n unique integer indices.
        """
        pool_size = len(pool)
        if n >= pool_size:
            return list(range(pool_size))
        indices = self._rng.choice(pool_size, size=n, replace=False)
        return indices.tolist()

    def reset(self) -> None:
        super().reset()
        self._rng = np.random.default_rng(self.seed)

    @property
    def name(self) -> str:
        return "random"
