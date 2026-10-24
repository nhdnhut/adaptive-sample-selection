"""
src/selectors/diversity_selector.py

Baseline D: Diversity-based selection.

Selects a diverse subset of the pool using k-means clustering in the
penultimate-layer feature space.

Algorithm:
  1. Extract features for all pool samples using model.get_features().
  2. Run k-means clustering with k=n clusters.
  3. For each cluster, select the sample closest to the cluster centroid.

Intuition: ensures selected samples span the feature space,
preventing redundancy that arises from difficulty- or uncertainty-only selection.

This is a lightweight approximation to submodular coreset selection
(CRAIG, Mirzasoleiman 2020) that is feasible on CPU without gradient computation.

Computational cost:
  - Feature extraction: O(N) forward pass
  - K-means: O(n_iter × k × N × D) where D is feature dimension
  - Memory: O(N × D) for feature matrix

Limitation:
  K-means clusters may not align with semantic boundaries.
  At very small k (small budget), diversity selection may miss hard examples.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import Dataset

from src.selectors.base_selector import BaseSelector
from src.selectors._scoring_utils import compute_features


class DiversitySelector(BaseSelector):
    """
    Select n diverse samples via k-means clustering in feature space.

    Args:
        seed: Random seed.
        score_every: Feature recomputation frequency (epochs).
        batch_size: Batch size for feature extraction.
        warmup_epochs: Use random selection before diversity kicks in.
        kmeans_init: K-means initialisation strategy ('k-means++' or 'random').
        kmeans_max_iter: Maximum k-means iterations.
    """

    def __init__(
        self,
        seed: int = 42,
        score_every: int = 3,
        batch_size: int = 256,
        warmup_epochs: int = 1,
        kmeans_init: str = "k-means++",
        kmeans_max_iter: int = 100,
    ) -> None:
        super().__init__(seed=seed)
        self.score_every = score_every
        self.batch_size = batch_size
        self.warmup_epochs = warmup_epochs
        self.kmeans_init = kmeans_init
        self.kmeans_max_iter = kmeans_max_iter

        self._rng = np.random.default_rng(seed)
        self._cached_features: np.ndarray | None = None
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

        if model is None or self._epoch <= self.warmup_epochs or not hasattr(model, "get_features"):
            return self._random_fallback(pool_size, n)

        if (
            self._cached_features is None
            or (self._epoch - self._last_scored_epoch) >= self.score_every
        ):
            self._cached_features = compute_features(pool, model, device, self.batch_size)
            self._last_scored_epoch = self._epoch

        return self._kmeans_select(self._cached_features, n)

    def _kmeans_select(self, features: np.ndarray, n: int) -> list[int]:
        """
        Cluster features into n clusters; return index of closest-to-centroid sample per cluster.
        """
        n_samples = features.shape[0]
        # Ensure n ≤ pool size
        k = min(n, n_samples)

        km = MiniBatchKMeans(
            n_clusters=k,
            init=self.kmeans_init,
            max_iter=self.kmeans_max_iter,
            random_state=self.seed + self._epoch,  # vary per epoch
            n_init=3,
            batch_size=min(1024, n_samples),
        )
        km.fit(features)
        centroids = km.cluster_centers_  # (k, D)
        labels = km.labels_             # (N,)

        selected = []
        for cluster_id in range(k):
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            if len(cluster_indices) == 0:
                continue
            cluster_features = features[cluster_indices]
            centroid = centroids[cluster_id]
            # Closest sample to centroid
            dists = np.linalg.norm(cluster_features - centroid, axis=1)
            closest = cluster_indices[np.argmin(dists)]
            selected.append(int(closest))

        # If we got fewer than n (degenerate clusters), fill with random
        if len(selected) < n:
            remaining = list(set(range(features.shape[0])) - set(selected))
            extra = min(n - len(selected), len(remaining))
            if extra > 0:
                extra_idx = self._rng.choice(remaining, size=extra, replace=False)
                selected.extend(extra_idx.tolist())

        return selected[:n]

    def _random_fallback(self, pool_size: int, n: int) -> list[int]:
        indices = self._rng.choice(pool_size, size=n, replace=False)
        return indices.tolist()

    def reset(self) -> None:
        super().reset()
        self._rng = np.random.default_rng(self.seed)
        self._cached_features = None
        self._last_scored_epoch = -1

    @property
    def name(self) -> str:
        return "diversity"
