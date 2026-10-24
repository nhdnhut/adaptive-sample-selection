"""
tests/test_training.py

Smoke tests for the full training pipeline.

Tests:
  - Full forward pass works
  - Trainer runs for 2 epochs without error
  - Metrics are in valid ranges
  - Logger saves files
  - All selectors integrate with Trainer correctly
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.models.small_cnn import SmallCNN
from src.models.small_mlp import SmallMLP
from src.selectors import build_selector
from src.training.trainer import Trainer
from src.utils.logging import ExperimentLogger


def make_tiny_dataset(n: int = 500):
    images = torch.randn(n, 1, 28, 28)
    labels = torch.randint(0, 10, (n,))
    return TensorDataset(images, labels)


def make_tiny_val_loader():
    images = torch.randn(100, 1, 28, 28)
    labels = torch.randint(0, 10, (100,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=32, shuffle=False)


class TestTrainerSmoke:
    """Smoke tests — just check the pipeline runs without exceptions."""

    @pytest.mark.parametrize("method", ["random", "difficulty", "uncertainty"])
    def test_train_2_epochs_no_crash(self, method):
        pool = make_tiny_dataset(300)
        val_loader = make_tiny_val_loader()
        model = SmallCNN(num_classes=10)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        # Random selector has no warmup_epochs param; others do
        kwargs = {} if method == "random" else {"warmup_epochs": 0}
        selector = build_selector(method, seed=42, **kwargs)

        trainer = Trainer(
            model=model,
            pool=pool,
            val_loader=val_loader,
            selector=selector,
            device=torch.device("cpu"),
            optimizer=optimizer,
            batch_size=32,
            n_select=100,
        )

        history = trainer.train(num_epochs=2, verbose=False)

        assert len(history) == 2
        for row in history:
            assert 0.0 <= row["val_accuracy"] <= 1.0
            assert row["val_loss"] >= 0.0
            assert 0.0 <= row["val_f1_macro"] <= 1.0
            assert row["samples_seen"] > 0
            assert row["gradient_updates"] > 0

    def test_diversity_selector_train(self):
        pool = make_tiny_dataset(300)
        val_loader = make_tiny_val_loader()
        model = SmallCNN(num_classes=10)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        selector = build_selector("diversity", seed=42, warmup_epochs=0)

        trainer = Trainer(
            model=model,
            pool=pool,
            val_loader=val_loader,
            selector=selector,
            device=torch.device("cpu"),
            optimizer=optimizer,
            batch_size=32,
            n_select=50,
        )
        history = trainer.train(num_epochs=2, verbose=False)
        assert len(history) == 2

    def test_adaptive_selector_train(self):
        pool = make_tiny_dataset(300)
        val_loader = make_tiny_val_loader()
        model = SmallCNN(num_classes=10)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        selector = build_selector("adaptive", seed=42, warmup_epochs=0)

        trainer = Trainer(
            model=model,
            pool=pool,
            val_loader=val_loader,
            selector=selector,
            device=torch.device("cpu"),
            optimizer=optimizer,
            batch_size=32,
            n_select=50,
        )
        history = trainer.train(num_epochs=3, verbose=False)
        assert len(history) == 3

    def test_mlp_model_trains(self):
        pool = make_tiny_dataset(200)
        val_loader = make_tiny_val_loader()
        model = SmallMLP(num_classes=10)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        selector = build_selector("random", seed=42)

        trainer = Trainer(
            model=model,
            pool=pool,
            val_loader=val_loader,
            selector=selector,
            device=torch.device("cpu"),
            optimizer=optimizer,
            batch_size=32,
            n_select=50,
        )
        history = trainer.train(num_epochs=2, verbose=False)
        assert len(history) == 2

    def test_logger_saves_csv(self, tmp_path):
        pool = make_tiny_dataset(200)
        val_loader = make_tiny_val_loader()
        model = SmallCNN(num_classes=10)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        selector = build_selector("random", seed=42)

        trainer = Trainer(
            model=model,
            pool=pool,
            val_loader=val_loader,
            selector=selector,
            device=torch.device("cpu"),
            optimizer=optimizer,
            batch_size=32,
            n_select=50,
        )

        with ExperimentLogger(
            experiment_id="TEST",
            method="random",
            budget_fraction=0.25,
            seed=42,
            output_dir=tmp_path / "raw",
        ) as logger:
            trainer.train(num_epochs=2, logger=logger, verbose=False)

        csv_files = list((tmp_path / "raw").glob("*.csv"))
        assert len(csv_files) == 1

        import pandas as pd
        df = pd.read_csv(csv_files[0])
        assert len(df) == 2  # 2 epochs
        assert "val_accuracy" in df.columns
        assert "samples_seen" in df.columns

    def test_samples_seen_increases_monotonically(self):
        pool = make_tiny_dataset(300)
        val_loader = make_tiny_val_loader()
        model = SmallCNN(num_classes=10)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        selector = build_selector("random", seed=42)

        trainer = Trainer(
            model=model,
            pool=pool,
            val_loader=val_loader,
            selector=selector,
            device=torch.device("cpu"),
            optimizer=optimizer,
            batch_size=32,
            n_select=80,
        )
        history = trainer.train(num_epochs=5, verbose=False)
        samples = [r["samples_seen"] for r in history]
        assert all(samples[i] < samples[i + 1] for i in range(len(samples) - 1))


class TestModelForwardPass:
    def test_small_cnn_output_shape(self):
        model = SmallCNN(num_classes=10)
        x = torch.randn(8, 1, 28, 28)
        out = model(x)
        assert out.shape == (8, 10)

    def test_small_cnn_feature_shape(self):
        model = SmallCNN(num_classes=10)
        x = torch.randn(8, 1, 28, 28)
        feats = model.get_features(x)
        assert feats.shape == (8, 256)

    def test_small_mlp_output_shape(self):
        model = SmallMLP(num_classes=10)
        x = torch.randn(8, 1, 28, 28)
        out = model(x)
        assert out.shape == (8, 10)

    def test_small_mlp_feature_shape(self):
        model = SmallMLP(num_classes=10)
        x = torch.randn(8, 1, 28, 28)
        feats = model.get_features(x)
        assert feats.shape == (8, 128)
