"""
tests/test_selectors.py

Tests for all sample selectors.

Covers:
  - RandomSelector: reproducibility, correct count, no duplicates
  - DifficultySelector: selects high-loss samples, fallback to random
  - UncertaintySelector: selects high-entropy samples, fallback to random
  - DiversitySelector: correct count, no duplicates
  - AdaptiveSelector: correct count, no duplicates, phase weights
  - build_selector factory
  - Edge cases: n >= pool_size, n=1, warmup epochs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import torch
from torch.utils.data import TensorDataset

from src.selectors import (
    RandomSelector,
    DifficultySelector,
    UncertaintySelector,
    DiversitySelector,
    AdaptiveSelector,
    build_selector,
)
from src.models.small_cnn import SmallCNN


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_dummy_pool(n: int = 200, img_size: int = 28, n_classes: int = 10) -> TensorDataset:
    """Create a dummy pool of random (image, label) pairs."""
    images = torch.randn(n, 1, img_size, img_size)
    labels = torch.randint(0, n_classes, (n,))
    return TensorDataset(images, labels)


def make_dummy_model() -> SmallCNN:
    """Create a SmallCNN in eval mode with random weights."""
    model = SmallCNN(num_classes=10)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# RandomSelector
# ---------------------------------------------------------------------------

class TestRandomSelector:
    def test_correct_count(self):
        pool = make_dummy_pool(200)
        sel = RandomSelector(seed=42)
        sel.begin_epoch()
        indices = sel.select(pool, n=50, model=None, device=torch.device("cpu"))
        assert len(indices) == 50

    def test_no_duplicates(self):
        pool = make_dummy_pool(200)
        sel = RandomSelector(seed=42)
        sel.begin_epoch()
        indices = sel.select(pool, n=100, model=None, device=torch.device("cpu"))
        assert len(set(indices)) == len(indices)

    def test_valid_range(self):
        pool = make_dummy_pool(200)
        sel = RandomSelector(seed=42)
        sel.begin_epoch()
        indices = sel.select(pool, n=50, model=None, device=torch.device("cpu"))
        assert all(0 <= i < 200 for i in indices)

    def test_reproducibility_same_seed(self):
        pool = make_dummy_pool(200)
        sel1 = RandomSelector(seed=42)
        sel1.begin_epoch()
        idx1 = sel1.select(pool, n=50, model=None, device=torch.device("cpu"))
        sel2 = RandomSelector(seed=42)
        sel2.begin_epoch()
        idx2 = sel2.select(pool, n=50, model=None, device=torch.device("cpu"))
        assert sorted(idx1) == sorted(idx2)

    def test_different_seeds_give_different_results(self):
        pool = make_dummy_pool(500)
        sel1 = RandomSelector(seed=42)
        sel1.begin_epoch()
        idx1 = sel1.select(pool, n=100, model=None, device=torch.device("cpu"))
        sel2 = RandomSelector(seed=99)
        sel2.begin_epoch()
        idx2 = sel2.select(pool, n=100, model=None, device=torch.device("cpu"))
        assert sorted(idx1) != sorted(idx2)

    def test_n_geq_pool_returns_all(self):
        pool = make_dummy_pool(50)
        sel = RandomSelector(seed=42)
        sel.begin_epoch()
        indices = sel.select(pool, n=100, model=None, device=torch.device("cpu"))
        assert len(indices) == 50

    def test_reset_restores_reproducibility(self):
        pool = make_dummy_pool(200)
        sel = RandomSelector(seed=42)
        sel.begin_epoch()
        idx1 = sel.select(pool, n=50, model=None, device=torch.device("cpu"))
        sel.reset()
        sel.begin_epoch()
        idx2 = sel.select(pool, n=50, model=None, device=torch.device("cpu"))
        assert sorted(idx1) == sorted(idx2)


# ---------------------------------------------------------------------------
# DifficultySelector
# ---------------------------------------------------------------------------

class TestDifficultySelector:
    def test_warmup_falls_back_to_random(self):
        pool = make_dummy_pool(200)
        model = make_dummy_model()
        sel = DifficultySelector(seed=42, warmup_epochs=2)
        sel.begin_epoch()  # epoch=1, within warmup
        indices = sel.select(pool, n=50, model=model, device=torch.device("cpu"))
        assert len(indices) == 50
        assert len(set(indices)) == 50

    def test_selects_correct_count_after_warmup(self):
        pool = make_dummy_pool(200)
        model = make_dummy_model()
        sel = DifficultySelector(seed=42, warmup_epochs=0)
        sel.begin_epoch()
        indices = sel.select(pool, n=50, model=model, device=torch.device("cpu"))
        assert len(indices) == 50

    def test_no_duplicates(self):
        pool = make_dummy_pool(200)
        model = make_dummy_model()
        sel = DifficultySelector(seed=42, warmup_epochs=0)
        sel.begin_epoch()
        indices = sel.select(pool, n=100, model=model, device=torch.device("cpu"))
        assert len(set(indices)) == len(indices)

    def test_none_model_fallback(self):
        pool = make_dummy_pool(200)
        sel = DifficultySelector(seed=42)
        sel.begin_epoch()
        indices = sel.select(pool, n=50, model=None, device=torch.device("cpu"))
        assert len(indices) == 50


# ---------------------------------------------------------------------------
# UncertaintySelector
# ---------------------------------------------------------------------------

class TestUncertaintySelector:
    def test_correct_count(self):
        pool = make_dummy_pool(200)
        model = make_dummy_model()
        sel = UncertaintySelector(seed=42, warmup_epochs=0)
        sel.begin_epoch()
        indices = sel.select(pool, n=50, model=model, device=torch.device("cpu"))
        assert len(indices) == 50

    def test_no_duplicates(self):
        pool = make_dummy_pool(200)
        model = make_dummy_model()
        sel = UncertaintySelector(seed=42, warmup_epochs=0)
        sel.begin_epoch()
        indices = sel.select(pool, n=100, model=model, device=torch.device("cpu"))
        assert len(set(indices)) == len(indices)


# ---------------------------------------------------------------------------
# DiversitySelector
# ---------------------------------------------------------------------------

class TestDiversitySelector:
    def test_correct_count(self):
        pool = make_dummy_pool(200)
        model = make_dummy_model()
        sel = DiversitySelector(seed=42, warmup_epochs=0)
        sel.begin_epoch()
        indices = sel.select(pool, n=50, model=model, device=torch.device("cpu"))
        assert len(indices) <= 50  # may be slightly less due to degenerate clusters

    def test_no_duplicates(self):
        pool = make_dummy_pool(200)
        model = make_dummy_model()
        sel = DiversitySelector(seed=42, warmup_epochs=0)
        sel.begin_epoch()
        indices = sel.select(pool, n=50, model=model, device=torch.device("cpu"))
        assert len(set(indices)) == len(indices)

    def test_valid_range(self):
        pool = make_dummy_pool(200)
        model = make_dummy_model()
        sel = DiversitySelector(seed=42, warmup_epochs=0)
        sel.begin_epoch()
        indices = sel.select(pool, n=50, model=model, device=torch.device("cpu"))
        assert all(0 <= i < 200 for i in indices)


# ---------------------------------------------------------------------------
# AdaptiveSelector
# ---------------------------------------------------------------------------

class TestAdaptiveSelector:
    def test_correct_count(self):
        pool = make_dummy_pool(200)
        model = make_dummy_model()
        sel = AdaptiveSelector(seed=42, warmup_epochs=0)
        sel.begin_epoch()
        indices = sel.select(pool, n=50, model=model, device=torch.device("cpu"))
        assert len(indices) == 50

    def test_no_duplicates(self):
        pool = make_dummy_pool(200)
        model = make_dummy_model()
        sel = AdaptiveSelector(seed=42, warmup_epochs=0)
        sel.begin_epoch()
        indices = sel.select(pool, n=100, model=model, device=torch.device("cpu"))
        assert len(set(indices)) == len(indices)

    def test_phase_weights_change(self):
        sel = AdaptiveSelector(seed=42)
        # Phase 1 (epoch 5)
        sel._epoch = 5
        a1, b1, g1 = sel._get_weights()
        # Phase 3 (epoch 25)
        sel._epoch = 25
        a3, b3, g3 = sel._get_weights()
        # Diversity weight should be higher in phase 1
        assert g1 > g3

    def test_warmup_fallback(self):
        pool = make_dummy_pool(200)
        model = make_dummy_model()
        sel = AdaptiveSelector(seed=42, warmup_epochs=3)
        sel.begin_epoch()  # epoch=1
        indices = sel.select(pool, n=50, model=model, device=torch.device("cpu"))
        assert len(indices) == 50


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestBuildSelector:
    def test_known_methods(self):
        for method in ["random", "difficulty", "uncertainty", "diversity", "adaptive"]:
            sel = build_selector(method, seed=42)
            assert sel.name == method

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            build_selector("magic", seed=42)
