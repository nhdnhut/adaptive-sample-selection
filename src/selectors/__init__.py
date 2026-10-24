"""src/selectors package."""
from src.selectors.base_selector import BaseSelector
from src.selectors.random_selector import RandomSelector
from src.selectors.difficulty_selector import DifficultySelector
from src.selectors.uncertainty_selector import UncertaintySelector
from src.selectors.diversity_selector import DiversitySelector
from src.selectors.adaptive_selector import AdaptiveSelector


def build_selector(method: str, seed: int = 42, **kwargs) -> BaseSelector:
    """
    Factory function to instantiate a selector by method name.

    Args:
        method: One of 'random', 'difficulty', 'uncertainty', 'diversity', 'adaptive'.
        seed: Random seed.
        **kwargs: Additional kwargs passed to the selector constructor.

    Returns:
        Instantiated BaseSelector.

    Raises:
        ValueError: If method is not recognised.
    """
    method = method.lower()
    mapping = {
        "random": RandomSelector,
        "difficulty": DifficultySelector,
        "uncertainty": UncertaintySelector,
        "diversity": DiversitySelector,
        "adaptive": AdaptiveSelector,
    }
    if method not in mapping:
        raise ValueError(f"Unknown selector method: '{method}'. Choose from: {list(mapping)}")
    return mapping[method](seed=seed, **kwargs)


__all__ = [
    "BaseSelector",
    "RandomSelector",
    "DifficultySelector",
    "UncertaintySelector",
    "DiversitySelector",
    "AdaptiveSelector",
    "build_selector",
]
