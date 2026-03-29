import torch
from torch import nn
import torchvision.transforms.functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models import convnext_small, ResNet50_Weights, ConvNeXt_Small_Weights

class MammoModel(nn.Module):
    def __init__(self, num_seg_classes=2, num_classifier_classes=2):
        super(MammoModel, self).__init__()
        self.detector = maskrcnn_resnet50_fpn_v2(
            weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
            weights_backbone=ResNet50_Weights.DEFAULT,
            trainable_backbone_layers=3,
        )
        
        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_seg_classes)

        in_features_mask = self.detector.roi_heads.mask_predictor.conv5_mask.in_channels
        self.detector.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_seg_classes)

        self.classifier_backbone = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT).features
        # Freeze all stages except the last one (features[6] and features[7])
        for i, block in enumerate(self.classifier_backbone):
            if i < 6:
                for p in block.parameters():
                    p.requires_grad = False
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classifier_classes),
        )

    def forward(self, images, targets=None):
        detector_out = self.detector(images, targets)  # losses in train, dets in eval

        if self.training:
            # Crop GT patches and compute classifier loss
            cls_losses = []
            for image, target in zip(images, targets):
                boxes = target["boxes"]
                labels = torch.tensor(
                    [0 if p == "BENIGN" else 1 for p in target["pathology"]],
                    device=image.device
                )
                patches = [F.resize(self._pad_to_square(image[:, y1:y2, x1:x2]), [224, 224]).unsqueeze(0)
                    for x1, y1, x2, y2 in boxes.int().tolist()]
                if patches:
                    feats = self.pool(self.classifier_backbone(torch.cat(patches))).flatten(1)
                    logits = self.classifier(feats)
                    cls_losses.append(nn.functional.cross_entropy(logits, labels))

            if cls_losses:
                detector_out["loss_pathology"] = torch.stack(cls_losses).mean()
            return detector_out  # all losses including pathology

        else:
            for image, det in zip(images, detector_out):
                patches = [F.resize(self._pad_to_square(image[:, y1:y2, x1:x2]), [224, 224]).unsqueeze(0)
                    for x1, y1, x2, y2 in det["boxes"].int().tolist()]
                if patches:
                    feats = self.pool(self.classifier_backbone(torch.cat(patches))).flatten(1)
                    det["pathology_logits"] = self.classifier(feats)
            return detector_out

    def _pad_to_square(self, patch):
        _, h, w = patch.shape
        diff = abs(h - w)
        if h < w:
            patch = F.pad(patch, [0, 0, diff // 2, diff - diff // 2])  # pad height
        else:
            patch = F.pad(patch, [diff // 2, diff - diff // 2, 0, 0])  # pad width
        return patch
