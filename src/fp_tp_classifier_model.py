"""
ResNet-based classifier for detector box quality (FP vs TP).

Label convention:
- 0: false positive box
- 1: true positive box
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import (
    resnet18,
    resnet34,
    resnet50,
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
)

_BACKBONES = {
    "resnet18": (resnet18, ResNet18_Weights.DEFAULT, 512),
    "resnet34": (resnet34, ResNet34_Weights.DEFAULT, 512),
    "resnet50": (resnet50, ResNet50_Weights.DEFAULT, 2048),
}


class FPTPBoxClassifier(nn.Module):
    """Pretrained ResNet classifier for FP/TP patch classification."""

    def __init__(
        self,
        backbone: str = "resnet34",
        freeze_stages: int = 2,
        head_hidden: int = 256,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()

        if backbone not in _BACKBONES:
            raise ValueError(f"Unknown backbone {backbone!r}, choose from {sorted(_BACKBONES)}")

        factory, weights, feat_dim = _BACKBONES[backbone]
        model = factory(weights=weights)
        model.fc = nn.Identity()
        self.backbone = model

        self._freeze_early_stages(int(freeze_stages))

        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.head = nn.Sequential(
            nn.Linear(feat_dim, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(head_hidden, num_classes),
        )

    def _freeze_early_stages(self, freeze_stages: int) -> None:
        # Stage order:
        # 0: stem (conv1+bn1), 1: layer1, 2: layer2, 3: layer3, 4: layer4
        if freeze_stages <= 0:
            return

        stages = [
            [self.backbone.conv1, self.backbone.bn1],
            [self.backbone.layer1],
            [self.backbone.layer2],
            [self.backbone.layer3],
            [self.backbone.layer4],
        ]

        for stage_idx, modules in enumerate(stages):
            if stage_idx >= freeze_stages:
                break
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.img_mean) / self.img_std
        feats = self.backbone(x)
        return self.head(feats)
