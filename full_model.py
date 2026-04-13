"""
Unified detector + classifier pipeline.

Loads:
- detector checkpoint (e.g. models/<run_id>/detector_resnet101_best.pth)
- classifier checkpoint from classifier_model/classifier_best.pth

The detector predicts lesion boxes/masks, then the classifier predicts
benign/malignant labels from detector-cropped patches.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as nnf
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import box_iou

from classify_model import MammoClassifier


_SUPPORTED_BACKBONES = {"resnet101", "resnet152"}


def _unwrap_state_dict(blob):
    if isinstance(blob, dict) and "model_state_dict" in blob:
        return blob["model_state_dict"]
    return blob


def _pathology_to_label(pathology: str) -> int | None:
    raw = str(pathology).strip().upper()
    if "MALIGNANT" in raw:
        return 1
    if "BENIGN" in raw:
        return 0
    return None


def _build_detector_model(
    num_classes: int = 2,
    trainable_backbone_layers: int = 5,
    backbone_name: str = "resnet101",
) -> MaskRCNN:
    if backbone_name not in _SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unsupported backbone '{backbone_name}'. Choose one of: {sorted(_SUPPORTED_BACKBONES)}"
        )

    # We intentionally avoid pretrained backbone downloads here because
    # checkpoints are loaded immediately after model creation.
    backbone = resnet_fpn_backbone(
        backbone_name=backbone_name,
        weights=None,
        trainable_layers=trainable_backbone_layers,
    )
    return MaskRCNN(backbone=backbone, num_classes=num_classes, min_size=1024, max_size=1024)


class FullMammoPipeline(nn.Module):
    """Unified inference/training utility around detector + classifier."""

    def __init__(
        self,
        detector_checkpoint: str,
        classifier_dir: str = "./classifier_model",
        detector_backbone: str = "resnet101",
        detector_trainable_backbone_layers: int = 5,
        patch_size: int | None = None,
        inference_score_threshold: float = 0.05,
    ):
        super().__init__()

        self.detector = _build_detector_model(
            num_classes=2,
            trainable_backbone_layers=detector_trainable_backbone_layers,
            backbone_name=detector_backbone,
        )
        self._load_detector_weights(detector_checkpoint)

        self.classifier, self.classifier_hparams = self._load_classifier(classifier_dir)
        if patch_size is None:
            patch_size = int(self.classifier_hparams.get("patch_size", 224))
        self.patch_size = int(patch_size)
        self.inference_score_threshold = float(inference_score_threshold)

    def _load_detector_weights(self, detector_checkpoint: str) -> None:
        ckpt_path = Path(detector_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Detector checkpoint not found: {ckpt_path}")

        blob = torch.load(ckpt_path, map_location="cpu")
        state_dict = _unwrap_state_dict(blob)
        self.detector.load_state_dict(state_dict, strict=True)

    @staticmethod
    def _load_classifier(classifier_dir: str) -> tuple[MammoClassifier, dict]:
        cls_dir = Path(classifier_dir)
        if not cls_dir.exists():
            raise FileNotFoundError(f"Classifier directory not found: {cls_dir}")

        hparams_path = cls_dir / "hyperparams.json"
        ckpt_path = cls_dir / "classifier_best.pth"

        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Classifier checkpoint not found: {ckpt_path}. "
                "Expected classifier_model/classifier_best.pth"
            )

        hparams = {}
        if hparams_path.exists():
            with hparams_path.open("r", encoding="utf-8") as f:
                hparams = json.load(f)

        # Fine-tuned checkpoints store original classifier settings under
        # base_classifier_hparams; prefer those when top-level keys are absent.
        base_hparams = hparams.get("base_classifier_hparams", {})
        cls_hparams = dict(base_hparams) if isinstance(base_hparams, dict) else {}
        for k in ("backbone", "freeze_stages", "head_hidden", "dropout"):
            if k in hparams:
                cls_hparams[k] = hparams[k]

        blob = torch.load(ckpt_path, map_location="cpu")
        state_dict = _unwrap_state_dict(blob)

        # Infer dimensions from checkpoint as a safe fallback.
        head_in = state_dict.get("head.1.weight").shape[1] if "head.1.weight" in state_dict else None
        head_hidden = state_dict.get("head.1.weight").shape[0] if "head.1.weight" in state_dict else None
        num_classes = state_dict.get("head.4.weight").shape[0] if "head.4.weight" in state_dict else 2

        backbone = cls_hparams.get("backbone")
        if backbone is None:
            if head_in == 1024:
                backbone = "convnext_base"
            else:
                backbone = "convnext_small"

        model = MammoClassifier(
            backbone=str(backbone),
            freeze_stages=int(cls_hparams.get("freeze_stages", 4)),
            head_hidden=int(cls_hparams.get("head_hidden", head_hidden if head_hidden is not None else 256)),
            num_classes=int(num_classes),
            dropout=float(cls_hparams.get("dropout", 0.2)),
        )

        model.load_state_dict(state_dict, strict=True)
        return model, cls_hparams

    def freeze_detector(self) -> None:
        for p in self.detector.parameters():
            p.requires_grad = False
        self.detector.eval()

    def unfreeze_detector(self) -> None:
        for p in self.detector.parameters():
            p.requires_grad = True

    @staticmethod
    def _pad_to_square(patch: torch.Tensor) -> torch.Tensor:
        _, h, w = patch.shape
        if h == w:
            return patch

        diff = abs(h - w)
        if h < w:
            top = diff // 2
            bottom = diff - top
            return nnf.pad(patch, (0, 0, top, bottom))

        left = diff // 2
        right = diff - left
        return nnf.pad(patch, (left, right, 0, 0))

    def _crop_patch(self, image: torch.Tensor, box_xyxy: torch.Tensor) -> torch.Tensor | None:
        _, h, w = image.shape
        x1, y1, x2, y2 = [int(v) for v in box_xyxy.tolist()]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return None

        patch = image[:, y1:y2, x1:x2]
        if patch.numel() == 0:
            return None

        patch = self._pad_to_square(patch)
        patch = nnf.interpolate(
            patch.unsqueeze(0),
            size=(self.patch_size, self.patch_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return patch

    def _classify_boxes(self, image: torch.Tensor, boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num_classes = int(getattr(self.classifier.head[-1], "out_features", 2))
        if boxes.numel() == 0:
            empty_logits = torch.empty((0, num_classes), device=image.device, dtype=image.dtype)
            empty_valid = torch.empty((0,), device=image.device, dtype=torch.bool)
            return empty_logits, empty_valid

        patches = []
        valid_indices: list[int] = []
        for i, box in enumerate(boxes):
            patch = self._crop_patch(image, box)
            if patch is not None:
                patches.append(patch)
                valid_indices.append(i)

        valid_mask = torch.zeros((boxes.shape[0],), device=image.device, dtype=torch.bool)
        if valid_indices:
            valid_idx_t = torch.tensor(valid_indices, dtype=torch.long, device=image.device)
            valid_mask[valid_idx_t] = True

        if not patches:
            # Keep output aligned with detector boxes even when all crops are invalid.
            logits = torch.zeros((boxes.shape[0], num_classes), device=image.device, dtype=image.dtype)
            return logits, valid_mask

        patch_batch = torch.stack(patches, dim=0)
        logits_valid = self.classifier(patch_batch)

        if len(valid_indices) == int(boxes.shape[0]):
            return logits_valid, valid_mask

        logits_full = torch.zeros(
            (boxes.shape[0], logits_valid.shape[1]),
            device=logits_valid.device,
            dtype=logits_valid.dtype,
        )
        valid_idx_t = torch.tensor(valid_indices, dtype=torch.long, device=logits_valid.device)
        logits_full[valid_idx_t] = logits_valid
        return logits_full, valid_mask

    @torch.no_grad()
    def forward(self, images: list[torch.Tensor]) -> list[dict]:
        """Inference: detector outputs plus classifier logits/probabilities."""
        detector_out = self.detector(images)
        outputs = []

        for image, det in zip(images, detector_out):
            scores = det.get("scores", torch.empty((0,), device=image.device))
            keep = scores >= self.inference_score_threshold if scores.numel() else scores
            boxes = det["boxes"][keep] if scores.numel() else det["boxes"]

            logits, valid_mask = self._classify_boxes(image, boxes)
            probs = (
                torch.softmax(logits, dim=-1)
                if logits.numel() > 0
                else torch.empty((0, 2), device=image.device, dtype=image.dtype)
            )

            out = {k: v for k, v in det.items()}
            if scores.numel():
                out["boxes"] = boxes
                out["scores"] = scores[keep]
                if "labels" in out:
                    out["labels"] = out["labels"][keep]
                if "masks" in out:
                    out["masks"] = out["masks"][keep]
            out["pathology_logits"] = logits
            out["pathology_probs"] = probs
            out["pathology_valid"] = valid_mask
            outputs.append(out)

        return outputs

    def build_classifier_training_batch(
        self,
        images: list[torch.Tensor],
        targets: list[dict],
        iou_match_thr: float = 0.3,
        score_threshold: float | None = None,
        max_patches_per_image: int = 32,
        detections: list[dict] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Create classifier training patches from detector predictions matched to GT.

        Returns:
            patches: (N, 3, patch_size, patch_size)
            labels:  (N,) with 0=benign, 1=malignant
            stats:   bookkeeping counts
        """
        if score_threshold is None:
            score_threshold = self.inference_score_threshold

        if detections is None:
            with torch.no_grad():
                detections = self.detector(images)

        patches = []
        labels = []

        stats = {
            "num_images": len(images),
            "num_pred_boxes": 0,
            "num_matched_boxes": 0,
            "num_training_patches": 0,
            "num_skipped_unknown_label": 0,
            "num_skipped_empty_crop": 0,
        }

        for image, det, tgt in zip(images, detections, targets):
            gt_boxes = tgt.get("boxes")
            gt_pathology = tgt.get("pathology", [])
            if gt_boxes is None or gt_boxes.numel() == 0 or len(gt_pathology) == 0:
                continue

            pred_scores = det.get("scores", torch.empty((0,), device=image.device))
            pred_boxes = det.get("boxes", torch.empty((0, 4), device=image.device))

            if pred_boxes.numel() == 0:
                continue

            if pred_scores.numel() > 0:
                keep = pred_scores >= float(score_threshold)
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]

            if pred_boxes.numel() == 0:
                continue

            stats["num_pred_boxes"] += int(pred_boxes.shape[0])

            iou = box_iou(pred_boxes, gt_boxes)
            matched_iou, matched_idx = iou.max(dim=1)

            keep_idx = torch.nonzero(matched_iou >= float(iou_match_thr)).flatten()
            if keep_idx.numel() == 0:
                continue

            if max_patches_per_image > 0 and keep_idx.numel() > max_patches_per_image:
                keep_idx = keep_idx[:max_patches_per_image]

            stats["num_matched_boxes"] += int(keep_idx.numel())

            for pred_i in keep_idx.tolist():
                gt_i = int(matched_idx[pred_i].item())
                if gt_i >= len(gt_pathology):
                    continue

                cls_label = _pathology_to_label(gt_pathology[gt_i])
                if cls_label is None:
                    stats["num_skipped_unknown_label"] += 1
                    continue

                patch = self._crop_patch(image, pred_boxes[pred_i])
                if patch is None:
                    stats["num_skipped_empty_crop"] += 1
                    continue

                patches.append(patch)
                labels.append(cls_label)

        if patches:
            patch_batch = torch.stack(patches, dim=0)
            label_tensor = torch.tensor(labels, dtype=torch.long, device=patch_batch.device)
        else:
            device = images[0].device if images else torch.device("cpu")
            patch_batch = torch.empty((0, 3, self.patch_size, self.patch_size), device=device)
            label_tensor = torch.empty((0,), dtype=torch.long, device=device)

        stats["num_training_patches"] = int(label_tensor.numel())
        return patch_batch, label_tensor, stats

    def build_gt_classifier_training_batch(
        self,
        images: list[torch.Tensor],
        targets: list[dict],
        max_patches_per_image: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Create classifier patches directly from GT boxes.

        Useful as a fallback when detector-matched patches are too sparse.
        """
        patches = []
        labels = []

        stats = {
            "num_images": len(images),
            "num_gt_boxes": 0,
            "num_training_patches": 0,
            "num_skipped_unknown_label": 0,
            "num_skipped_empty_crop": 0,
        }

        for image, tgt in zip(images, targets):
            gt_boxes = tgt.get("boxes")
            gt_pathology = tgt.get("pathology", [])
            if gt_boxes is None or gt_boxes.numel() == 0 or len(gt_pathology) == 0:
                continue

            n = min(int(gt_boxes.shape[0]), len(gt_pathology))
            if n <= 0:
                continue

            stats["num_gt_boxes"] += n
            added_for_image = 0

            for gt_i in range(n):
                if max_patches_per_image > 0 and added_for_image >= max_patches_per_image:
                    break

                cls_label = _pathology_to_label(gt_pathology[gt_i])
                if cls_label is None:
                    stats["num_skipped_unknown_label"] += 1
                    continue

                patch = self._crop_patch(image, gt_boxes[gt_i])
                if patch is None:
                    stats["num_skipped_empty_crop"] += 1
                    continue

                patches.append(patch)
                labels.append(cls_label)
                added_for_image += 1

        if patches:
            patch_batch = torch.stack(patches, dim=0)
            label_tensor = torch.tensor(labels, dtype=torch.long, device=patch_batch.device)
        else:
            device = images[0].device if images else torch.device("cpu")
            patch_batch = torch.empty((0, 3, self.patch_size, self.patch_size), device=device)
            label_tensor = torch.empty((0,), dtype=torch.long, device=device)

        stats["num_training_patches"] = int(label_tensor.numel())
        return patch_batch, label_tensor, stats


def load_full_model(
    detector_checkpoint: str,
    classifier_dir: str = "./classifier_model",
    detector_backbone: str = "resnet101",
    detector_trainable_backbone_layers: int = 5,
    patch_size: int | None = None,
    inference_score_threshold: float = 0.05,
    device: str | torch.device | None = None,
) -> FullMammoPipeline:
    """Convenience helper to instantiate and move unified model."""
    model = FullMammoPipeline(
        detector_checkpoint=detector_checkpoint,
        classifier_dir=classifier_dir,
        detector_backbone=detector_backbone,
        detector_trainable_backbone_layers=detector_trainable_backbone_layers,
        patch_size=patch_size,
        inference_score_threshold=inference_score_threshold,
    )

    if device is not None:
        model.to(device)
    return model
