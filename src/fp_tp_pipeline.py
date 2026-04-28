"""
Inference pipeline: detector + FP/TP classifier filter.

Expected FP/TP classifier directory structure:
    <fp_tp_classifier_dir>/
      - fp_tp_classifier_best.pth
      - hyperparams.json

Label convention:
- class 0: FP
- class 1: TP
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as nnf
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import nms

from fp_tp_classifier_model import FPTPBoxClassifier


_SUPPORTED_DETECTOR_BACKBONES = {"resnet101", "resnet152"}


def _unwrap_state_dict(blob):
    if isinstance(blob, dict) and "model_state_dict" in blob:
        return blob["model_state_dict"]
    return blob


def _build_detector(backbone_name: str, trainable_layers: int = 5) -> MaskRCNN:
    if backbone_name not in _SUPPORTED_DETECTOR_BACKBONES:
        raise ValueError(
            f"Unsupported detector backbone '{backbone_name}'. "
            f"Choose one of: {sorted(_SUPPORTED_DETECTOR_BACKBONES)}"
        )

    backbone = resnet_fpn_backbone(
        backbone_name=backbone_name,
        weights=None,
        trainable_layers=trainable_layers,
    )
    return MaskRCNN(backbone=backbone, num_classes=2, min_size=1024, max_size=1024)


def _load_detector_with_fallback(checkpoint_path: str, backbone_name: str) -> tuple[MaskRCNN, str]:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {ckpt_path}")

    blob = torch.load(ckpt_path, map_location="cpu")
    state_dict = _unwrap_state_dict(blob)

    model = _build_detector(backbone_name=backbone_name, trainable_layers=5)
    try:
        model.load_state_dict(state_dict, strict=True)
        return model, backbone_name
    except RuntimeError as exc:
        alt_backbones = [b for b in _SUPPORTED_DETECTOR_BACKBONES if b != backbone_name]
        for alt in alt_backbones:
            alt_model = _build_detector(backbone_name=alt, trainable_layers=5)
            try:
                alt_model.load_state_dict(state_dict, strict=True)
                return alt_model, alt
            except RuntimeError:
                continue
        raise RuntimeError(
            "Failed to load detector checkpoint. Try detector_backbone='resnet101' or 'resnet152'."
        ) from exc


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


class DetectorFPTPFilterPipeline(nn.Module):
    """Two-stage detector with FP suppression by box-quality classifier."""

    def __init__(
        self,
        detector_checkpoint: str,
        fp_tp_classifier_dir: str,
        detector_backbone: str = "resnet101",
        detector_score_threshold: float = 0.05,
        tp_prob_threshold: float = 0.50,
        post_nms_iou_threshold: float = 0.30,
        patch_size: int | None = None,
        context_factor: float | None = None,
    ):
        super().__init__()

        detector, detected_backbone = _load_detector_with_fallback(detector_checkpoint, detector_backbone)
        self.detector = detector
        self.detector_backbone = detected_backbone

        self.fp_tp_classifier, self.fp_tp_hparams = self._load_fp_tp_classifier(fp_tp_classifier_dir)

        if patch_size is None:
            patch_size = int(self.fp_tp_hparams.get("patch_size", 224))
        if context_factor is None:
            context_factor = float(self.fp_tp_hparams.get("context_factor", 1.2))

        self.patch_size = int(patch_size)
        self.context_factor = float(context_factor)
        self.detector_score_threshold = float(detector_score_threshold)
        self.tp_prob_threshold = float(tp_prob_threshold)
        self.post_nms_iou_threshold = float(post_nms_iou_threshold)

    @staticmethod
    def _load_fp_tp_classifier(fp_tp_classifier_dir: str) -> tuple[FPTPBoxClassifier, dict]:
        model_dir = Path(fp_tp_classifier_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"FP/TP classifier directory not found: {model_dir}")

        hparams_path = model_dir / "hyperparams.json"
        ckpt_path = model_dir / "fp_tp_classifier_best.pth"

        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"FP/TP checkpoint not found: {ckpt_path}. "
                "Expected file name: fp_tp_classifier_best.pth"
            )

        hparams = {}
        if hparams_path.exists():
            with hparams_path.open("r", encoding="utf-8") as f:
                hparams = json.load(f)

        blob = torch.load(ckpt_path, map_location="cpu")
        state_dict = _unwrap_state_dict(blob)

        model = FPTPBoxClassifier(
            backbone=str(hparams.get("backbone", "resnet34")),
            freeze_stages=int(hparams.get("freeze_stages", 2)),
            head_hidden=int(hparams.get("head_hidden", 256)),
            dropout=float(hparams.get("dropout", 0.3)),
            num_classes=2,
        )
        model.load_state_dict(state_dict, strict=True)

        return model, hparams

    @staticmethod
    def _expand_box_xyxy(box_xyxy: torch.Tensor, h: int, w: int, context_factor: float) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = [float(v) for v in box_xyxy.tolist()]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        bw *= float(context_factor)
        bh *= float(context_factor)

        nx1 = int(max(0.0, cx - 0.5 * bw))
        ny1 = int(max(0.0, cy - 0.5 * bh))
        nx2 = int(min(float(w), cx + 0.5 * bw))
        ny2 = int(min(float(h), cy + 0.5 * bh))

        if nx2 <= nx1:
            nx2 = min(w, nx1 + 1)
        if ny2 <= ny1:
            ny2 = min(h, ny1 + 1)

        return nx1, ny1, nx2, ny2

    def _crop_patch(self, image: torch.Tensor, box_xyxy: torch.Tensor) -> torch.Tensor | None:
        _, h, w = image.shape
        x1, y1, x2, y2 = self._expand_box_xyxy(box_xyxy, h=h, w=w, context_factor=self.context_factor)

        if x2 <= x1 or y2 <= y1:
            return None

        patch = image[:, y1:y2, x1:x2]
        if patch.numel() == 0:
            return None

        patch = _pad_to_square(patch)
        patch = nnf.interpolate(
            patch.unsqueeze(0),
            size=(self.patch_size, self.patch_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return patch

    def _classify_boxes(self, image: torch.Tensor, boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if boxes.numel() == 0:
            empty_probs = torch.empty((0,), device=image.device, dtype=image.dtype)
            empty_valid = torch.empty((0,), device=image.device, dtype=torch.bool)
            return empty_probs, empty_valid

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
            probs = torch.zeros((boxes.shape[0],), device=image.device, dtype=image.dtype)
            return probs, valid_mask

        patch_batch = torch.stack(patches, dim=0)
        logits = self.fp_tp_classifier(patch_batch)
        probs_valid = torch.softmax(logits, dim=-1)[:, 1]

        probs_full = torch.zeros((boxes.shape[0],), device=probs_valid.device, dtype=probs_valid.dtype)
        valid_idx_t = torch.tensor(valid_indices, dtype=torch.long, device=probs_valid.device)
        probs_full[valid_idx_t] = probs_valid
        return probs_full, valid_mask

    @torch.no_grad()
    def forward(self, images: list[torch.Tensor]) -> list[dict]:
        detector_out = self.detector(images)
        outputs = []

        for image, det in zip(images, detector_out):
            scores = det.get("scores", torch.empty((0,), device=image.device))
            keep_det = scores >= self.detector_score_threshold if scores.numel() else scores

            out = {k: v for k, v in det.items()}
            if scores.numel() > 0:
                boxes = det["boxes"][keep_det]
                det_scores = scores[keep_det]
                labels = det["labels"][keep_det] if "labels" in det else None
                masks = det["masks"][keep_det] if "masks" in det else None
            else:
                boxes = det.get("boxes", torch.empty((0, 4), device=image.device))
                det_scores = torch.empty((boxes.shape[0],), device=image.device, dtype=image.dtype)
                labels = None
                masks = None

            tp_probs, valid_mask = self._classify_boxes(image, boxes)
            combined_scores = tp_probs
            if det_scores.numel() == tp_probs.numel():
                combined_scores = det_scores * tp_probs

            keep_cls = valid_mask & (tp_probs >= self.tp_prob_threshold)
            selected_idx = torch.nonzero(keep_cls).flatten()

            # Dedupe overlapping positive boxes by fused score (detector * TP prob).
            if selected_idx.numel() > 0 and self.post_nms_iou_threshold > 0:
                selected_local = nms(
                    boxes[selected_idx],
                    combined_scores[selected_idx],
                    self.post_nms_iou_threshold,
                )
                selected_idx = selected_idx[selected_local]

            if boxes.numel() > 0:
                out["boxes"] = boxes[selected_idx]
                out["scores"] = combined_scores[selected_idx]
                out["detector_scores"] = det_scores[selected_idx]
                if labels is not None:
                    out["labels"] = labels[selected_idx]
                if masks is not None:
                    out["masks"] = masks[selected_idx]
            else:
                out["boxes"] = boxes
                out["scores"] = torch.empty((0,), device=image.device, dtype=tp_probs.dtype)
                out["detector_scores"] = torch.empty((0,), device=image.device, dtype=tp_probs.dtype)

            # Keep pre-filter candidates for diagnostic evaluation.
            out["fp_tp_candidate_boxes"] = boxes
            out["fp_tp_candidate_detector_scores"] = det_scores
            if masks is not None:
                out["fp_tp_candidate_masks"] = masks
            else:
                out["fp_tp_candidate_masks"] = torch.empty((0, 1, 1, 1), device=image.device, dtype=image.dtype)

            out["fp_tp_probs"] = tp_probs
            out["combined_scores"] = combined_scores
            out["fp_tp_valid"] = valid_mask
            out["fp_tp_keep"] = keep_cls
            out["fp_tp_selected_idx"] = selected_idx
            outputs.append(out)

        return outputs


def load_detector_fp_tp_pipeline(
    detector_checkpoint: str,
    fp_tp_classifier_dir: str,
    detector_backbone: str = "resnet101",
    detector_score_threshold: float = 0.05,
    tp_prob_threshold: float = 0.5,
    post_nms_iou_threshold: float = 0.30,
    patch_size: int | None = None,
    context_factor: float | None = None,
    device: str | torch.device | None = None,
) -> DetectorFPTPFilterPipeline:
    model = DetectorFPTPFilterPipeline(
        detector_checkpoint=detector_checkpoint,
        fp_tp_classifier_dir=fp_tp_classifier_dir,
        detector_backbone=detector_backbone,
        detector_score_threshold=detector_score_threshold,
        tp_prob_threshold=tp_prob_threshold,
        post_nms_iou_threshold=post_nms_iou_threshold,
        patch_size=patch_size,
        context_factor=context_factor,
    )
    if device is not None:
        model.to(device)
    return model
