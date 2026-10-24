"""
tests/test_reproducibility.py

Tests that seed setting produces reproducible results.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from src.utils.reproducibility import set_seed, get_generator
from src.models.small_cnn import SmallCNN
from src.selectors import RandomSelector


class TestSetSeed:
    def test_torch_reproducible(self):
        set_seed(42)
        t1 = torch.randn(10)
        set_seed(42)
        t2 = torch.randn(10)
        assert torch.allclose(t1, t2)

    def test_numpy_reproducible(self):
        set_seed(99)
        a1 = np.random.randn(10)
        set_seed(99)
        a2 = np.random.randn(10)
        np.testing.assert_array_equal(a1, a2)

    def test_model_init_reproducible(self):
        set_seed(42)
        m1 = SmallCNN()
        params1 = [p.data.clone() for p in m1.parameters()]

        set_seed(42)
        m2 = SmallCNN()
        params2 = [p.data.clone() for p in m2.parameters()]

        for p1, p2 in zip(params1, params2):
            assert torch.allclose(p1, p2)

    def test_get_generator_reproducible(self):
        g1 = get_generator(42)
        g2 = get_generator(42)
        t1 = torch.randint(0, 100, (10,), generator=g1)
        t2 = torch.randint(0, 100, (10,), generator=g2)
        assert torch.equal(t1, t2)
