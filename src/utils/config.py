"""
src/utils/config.py

Configuration loading and validation for experiments.
Supports YAML files with dot-access via SimpleNamespace.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> SimpleNamespace:
    """
    Load a YAML config file and return a nested SimpleNamespace.

    Args:
        path: Path to YAML file.

    Returns:
        Nested SimpleNamespace with config values.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If a required top-level key is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw: dict = yaml.safe_load(f)

    if raw is None:
        raw = {}

    _validate_config(raw)
    return _dict_to_namespace(raw)


def config_to_dict(cfg: SimpleNamespace) -> dict:
    """
    Recursively convert a SimpleNamespace back to a plain dict.

    Args:
        cfg: SimpleNamespace (possibly nested).

    Returns:
        Plain dict.
    """
    result = {}
    for key, val in vars(cfg).items():
        if isinstance(val, SimpleNamespace):
            result[key] = config_to_dict(val)
        else:
            result[key] = val
    return result


def save_config(cfg: SimpleNamespace, path: str | Path) -> None:
    """
    Serialise a config to JSON for experiment logging.

    Args:
        cfg: SimpleNamespace config.
        path: Output JSON file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config_to_dict(cfg), f, indent=2)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = {"dataset", "model", "training", "selector", "experiment"}


def _validate_config(raw: dict) -> None:
    """Check that required top-level sections are present."""
    missing = _REQUIRED_KEYS - set(raw.keys())
    if missing:
        raise ValueError(f"Config missing required sections: {missing}")


def _dict_to_namespace(d: Any) -> Any:
    """Recursively convert dicts to SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_dict_to_namespace(item) for item in d]
    return d
