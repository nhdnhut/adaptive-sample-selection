"""
src/models/small_mlp.py

SmallMLP: a compact fully-connected network for Fashion-MNIST.
Used as a secondary architecture for robustness checks (E010).

Architecture:
  FC(784 → 256) → BN → ReLU → Dropout(0.3)
  FC(256 → 128) → BN → ReLU
  FC(128 → num_classes)

Parameter count: ~235k
Input: (B, 1, 28, 28) — flattened internally
Output: (B, num_classes) — raw logits
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SmallMLP(nn.Module):
    """
    Compact MLP for Fashion-MNIST.

    Args:
        num_classes: Number of output classes (default 10).
        dropout_p: Dropout probability (default 0.3).
    """

    def __init__(self, num_classes: int = 10, dropout_p: float = 0.3) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, 1, 28, 28).

        Returns:
            Logits of shape (B, num_classes).
        """
        return self.net(x)

    @torch.no_grad()
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract 128-dim penultimate features.

        Args:
            x: (B, 1, 28, 28)

        Returns:
            (B, 128) feature tensor.
        """
        self.eval()
        # Run through everything except the last Linear
        for layer in list(self.net.children())[:-1]:
            x = layer(x)
        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_small_mlp(num_classes: int = 10, dropout_p: float = 0.3) -> SmallMLP:
    """Convenience constructor."""
    return SmallMLP(num_classes=num_classes, dropout_p=dropout_p)
