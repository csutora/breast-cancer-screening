"""
Standalone mammogram patch classifier for benign/malignant classification.

Decoupled from the Mask R-CNN detector so it can be trained and iterated on
independently.
"""

import torch
from torch import nn
from torchvision.models import (
    convnext_tiny, convnext_small, convnext_base,
    ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights, ConvNeXt_Base_Weights,
)

_BACKBONES = {
    "convnext_tiny":  (convnext_tiny,  ConvNeXt_Tiny_Weights.DEFAULT,  768),
    "convnext_small": (convnext_small, ConvNeXt_Small_Weights.DEFAULT, 768),
    "convnext_base":  (convnext_base,  ConvNeXt_Base_Weights.DEFAULT, 1024),
}


class MammoClassifier(nn.Module):
    """
    ConvNeXt-based patch classifier for mammogram mass pathology.

    Args:
        backbone:       one of 'convnext_tiny', 'convnext_small', 'convnext_base'
        freeze_stages:  number of feature stages to freeze (0-8, default 4)
        head_hidden:    hidden dimension in the classifier head
        num_classes:    number of output classes (default 2: benign/malignant)
        dropout:        dropout probability in the head
    """

    def __init__(
        self,
        backbone: str = "convnext_small",
        freeze_stages: int = 4,
        head_hidden: int = 256,
        num_classes: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        if backbone not in _BACKBONES:
            raise ValueError(f"Unknown backbone {backbone!r}, choose from {list(_BACKBONES)}")

        factory, weights, feat_dim = _BACKBONES[backbone]
        self.features = factory(weights=weights).features

        # Freeze early stages
        for i, block in enumerate(self.features):
            if i < freeze_stages:
                for p in block.parameters():
                    p.requires_grad = False

        # ImageNet normalization — pretrained weights expect this
        self.register_buffer('img_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('img_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) float32 image patches in [0, 1]

        Returns:
            logits: (B, num_classes)
        """
        x = (x - self.img_mean) / self.img_std
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)
