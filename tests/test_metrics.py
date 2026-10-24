"""
tests/test_metrics.py

Tests for evaluation metrics.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.metrics.metrics import (
    compute_accuracy,
    compute_macro_f1,
    compute_class_accuracy,
)


class TestAccuracy:
    def test_perfect_accuracy(self):
        y = np.array([0, 1, 2, 3])
        assert compute_accuracy(y, y) == pytest.approx(1.0)

    def test_zero_accuracy(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        assert compute_accuracy(y_true, y_pred) == pytest.approx(0.0)

    def test_partial_accuracy(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        assert compute_accuracy(y_true, y_pred) == pytest.approx(0.5)


class TestMacroF1:
    def test_perfect_f1(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        assert compute_macro_f1(y, y) == pytest.approx(1.0)

    def test_f1_in_range(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 0, 2, 1, 1, 1])
        f1 = compute_macro_f1(y_true, y_pred)
        assert 0.0 <= f1 <= 1.0


class TestClassAccuracy:
    def test_per_class(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        cls_acc = compute_class_accuracy(y_true, y_pred)
        assert cls_acc[0] == pytest.approx(0.5)  # class 0: 1/2
        assert cls_acc[1] == pytest.approx(1.0)  # class 1: 2/2
