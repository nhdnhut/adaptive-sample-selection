"""
tests/test_config.py

Tests for config loading and validation.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import yaml

from src.utils.config import load_config, config_to_dict, save_config


VALID_CONFIG = {
    "dataset": {"name": "FashionMNIST", "train_fraction": 0.25, "val_size": 1000, "augment": True, "data_root": "data"},
    "model": {"name": "small_cnn", "num_classes": 10, "dropout_p": 0.5},
    "training": {"epochs": 10, "batch_size": 64, "optimizer": "adam", "learning_rate": 0.001, "weight_decay": 1e-4},
    "selector": {"method": "random", "score_every": 1, "warmup_epochs": 1},
    "experiment": {"id": "TEST", "seed": 42, "results_dir": "results"},
}


def write_config(d: dict, path: Path) -> None:
    with path.open("w") as f:
        yaml.dump(d, f)


class TestLoadConfig:
    def test_valid_config_loads(self, tmp_path):
        cfg_path = tmp_path / "test.yaml"
        write_config(VALID_CONFIG, cfg_path)
        cfg = load_config(cfg_path)
        assert cfg.dataset.name == "FashionMNIST"
        assert cfg.training.epochs == 10
        assert cfg.selector.method == "random"
        assert cfg.experiment.seed == 42

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_missing_required_key_raises(self, tmp_path):
        bad = {k: v for k, v in VALID_CONFIG.items() if k != "dataset"}
        cfg_path = tmp_path / "bad.yaml"
        write_config(bad, cfg_path)
        with pytest.raises(ValueError):
            load_config(cfg_path)

    def test_config_to_dict_round_trip(self, tmp_path):
        cfg_path = tmp_path / "test.yaml"
        write_config(VALID_CONFIG, cfg_path)
        cfg = load_config(cfg_path)
        d = config_to_dict(cfg)
        assert d["dataset"]["name"] == "FashionMNIST"
        assert d["experiment"]["seed"] == 42

    def test_save_config_creates_json(self, tmp_path):
        cfg_path = tmp_path / "test.yaml"
        write_config(VALID_CONFIG, cfg_path)
        cfg = load_config(cfg_path)
        out_path = tmp_path / "saved.json"
        save_config(cfg, out_path)
        assert out_path.exists()
        import json
        with out_path.open() as f:
            data = json.load(f)
        assert data["experiment"]["seed"] == 42
