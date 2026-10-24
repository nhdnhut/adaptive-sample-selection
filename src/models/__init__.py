"""src/models package."""
from src.models.small_cnn import SmallCNN, build_small_cnn
from src.models.small_mlp import SmallMLP, build_small_mlp


def build_model(name: str, num_classes: int = 10, **kwargs):
    """
    Factory function to build a model by name.

    Args:
        name: One of 'small_cnn', 'small_mlp'.
        num_classes: Number of output classes.
        **kwargs: Additional kwargs passed to the model constructor.

    Returns:
        Instantiated model.

    Raises:
        ValueError: If name is not recognised.
    """
    name = name.lower()
    if name == "small_cnn":
        return build_small_cnn(num_classes=num_classes, **kwargs)
    if name == "small_mlp":
        return build_small_mlp(num_classes=num_classes, **kwargs)
    raise ValueError(f"Unknown model name: '{name}'. Choose from: small_cnn, small_mlp")


__all__ = [
    "SmallCNN",
    "SmallMLP",
    "build_small_cnn",
    "build_small_mlp",
    "build_model",
]
