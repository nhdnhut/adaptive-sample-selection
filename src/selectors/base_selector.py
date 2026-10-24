"""
src/selectors/base_selector.py

Abstract base class for all sample selectors.

A selector receives the training pool and returns indices of the samples
to include in the current epoch's training set.

The interface is intentionally minimal:
  - select(pool, n, model, device) → list[int] (indices into pool)

All selectors must implement this method. The model argument may be None
for selectors that do not require model feedback (e.g., random).

Selectors may maintain internal state across epochs (e.g., running scores).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    import torch.nn as nn


class BaseSelector(ABC):
    """Abstract interface for all sample selectors."""

    def __init__(self, seed: int = 42) -> None:
        """
        Args:
            seed: Random seed for reproducibility.
        """
        self.seed = seed
        self._epoch: int = 0

    def begin_epoch(self) -> None:
        """Called at the start of each epoch before select()."""
        self._epoch += 1

    @abstractmethod
    def select(
        self,
        pool: Dataset,
        n: int,
        model: "nn.Module | None",
        device: torch.device,
    ) -> list[int]:
        """
        Select n sample indices from pool.

        Args:
            pool: The full training pool (Subset or Dataset).
            n: Number of samples to select for this epoch.
            model: Current model (may be None for random selector).
                   The model should be in eval mode when passed here.
            device: Compute device.

        Returns:
            List of integer indices (0-indexed into pool) of length n.
            Indices must be unique and in [0, len(pool)).
        """
        ...

    def reset(self) -> None:
        """Reset internal state. Called between independent runs."""
        self._epoch = 0

    @property
    def name(self) -> str:
        """Human-readable selector name."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"{self.name}(seed={self.seed})"
