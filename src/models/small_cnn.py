"""
src/models/small_cnn.py

SmallCNN: a compact convolutional network for Fashion-MNIST.

Architecture:
  Conv(1→32, 3×3) → BN → ReLU → MaxPool(2)
  Conv(32→64, 3×3) → BN → ReLU → MaxPool(2)
  FC(64×5×5 → 256) → ReLU → Dropout(p)
  FC(256 → num_classes)

Parameter count: ~450k
Input: (B, 1, 28, 28)
Output: (B, num_classes) — raw logits

Feature extraction: call model.get_features(x) to get the 256-dim
penultimate-layer activations (used by diversity selector).

Intentional design choices:
- BatchNorm for stable training across small budgets
- Dropout to prevent overfitting on small subsets
- MaxPool instead of stride-conv for simplicity
- No residual connections (keeps it interpretable)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    """
    Compact CNN for Fashion-MNIST.

    Args:
        num_classes: Number of output classes (default 10).
        dropout_p: Dropout probability in the classifier head (default 0.5).
    """

    def __init__(self, num_classes: int = 10, dropout_p: float = 0.5) -> None:
        super().__init__()

        # Feature extractor (convolutional backbone)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 28×28
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # 14×14

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 14×14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # 7×7
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming normal init for conv layers; xavier for linear."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 1, 28, 28).

        Returns:
            Logits of shape (B, num_classes).
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten → (B, 64*7*7)
        x = self.classifier(x)
        return x

    @torch.no_grad()
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract penultimate-layer features (before final linear).

        Used by the diversity selector to compute embedding-space distances.

        Args:
            x: Input tensor of shape (B, 1, 28, 28).

        Returns:
            Feature tensor of shape (B, 256).
        """
        self.eval()
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # Apply only the first two ops of classifier (Linear + ReLU)
        x = self.classifier[0](x)  # Linear(64*7*7 → 256)
        x = self.classifier[1](x)  # ReLU
        return x

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_small_cnn(num_classes: int = 10, dropout_p: float = 0.5) -> SmallCNN:
    """Convenience constructor matching the config-driven build pattern."""
    return SmallCNN(num_classes=num_classes, dropout_p=dropout_p)
