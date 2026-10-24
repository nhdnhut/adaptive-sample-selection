"""
src/utils/reproducibility.py

Utilities for deterministic, reproducible experiments.
Sets all relevant random seeds and configures PyTorch determinism.
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set all random seeds for reproducibility.

    Affects: Python random, NumPy, PyTorch (CPU and CUDA).
    Also sets environment variables for deterministic CUDA ops.

    Args:
        seed: Integer seed value. Recommended: 42, 123, 777.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # CUDA determinism (no-op on CPU-only builds)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Use deterministic algorithms where available.
    # warn_only=True because some ops have no deterministic equivalent.
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Disable cudnn benchmarking (not relevant for CPU but keeps code portable)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_generator(seed: int) -> torch.Generator:
    """
    Return a seeded torch.Generator for dataset splitting operations.

    Args:
        seed: Integer seed value.

    Returns:
        Seeded torch.Generator instance.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g
