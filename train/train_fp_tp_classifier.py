"""
Train a second-stage FP/TP bounding-box classifier on detector proposals.

The detector is frozen and generates candidate boxes. Each box is labeled via
IoU against GT boxes. By default we use one-to-one greedy matching so each GT
supports at most one TP detection (duplicate high-IoU detections become FP):
- TP (1): matched IoU >= tp_iou_thr
- FP (0): matched IoU <= fp_iou_max, or duplicate around already-matched GT
- ignored: fp_iou_max < IoU < tp_iou_thr

Class imbalance is handled by two mechanisms:
- Per-image negative mining via max_neg_pos_ratio
- Optional inverse-frequency class weights in CrossEntropyLoss (class_weight=auto)

Example:
    python train_fp_tp_classifier.py \
      --detector_checkpoint ./models/<run_id>/detector_resnet101_best.pth

Sweep:
    python train_fp_tp_classifier.py \
      --detector_checkpoint ./models/<run_id>/detector_resnet101_best.pth \
      --sweep
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as nnf
from torch.utils.data import DataLoader
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import box_iou

try:
    import wandb
except ImportError:
    wandb = None

from config import DatasetConfig, PreprocessConfig, Representation
from dataset import CBISDDSMDataset, _collate_fn, build_sample_index
from fp_tp_classifier_model import FPTPBoxClassifier


_SUPPORTED_DETECTOR_BACKBONES = {"resnet101", "resnet152"}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _unwrap_state_dict(blob):
    if isinstance(blob, dict) and "model_state_dict" in blob:
        return blob["model_state_dict"]
    return blob


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def _roc_auc(y_true: list[int], y_score: list[float]) -> float:
    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_score_np = np.asarray(y_score, dtype=np.float64)
    pos = int((y_true_np == 1).sum())
    neg = int((y_true_np == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(-y_score_np)
    y_sorted = y_true_np[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    tpr = np.concatenate(([0.0], tp / float(pos), [1.0]))
    fpr = np.concatenate(([0.0], fp / float(neg), [1.0]))
    return float(np.trapezoid(tpr, fpr))


def _compute_metrics(y_true: list[int], y_prob: list[float], threshold: float = 0.5) -> dict:
    if not y_true:
        return {
            "n": 0,
            "loss": float("nan"),
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "specificity": float("nan"),
            "f1": float("nan"),
            "balanced_accuracy": float("nan"),
            "auc_roc": float("nan"),
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
        }

    y_pred = [1 if p >= threshold else 0 for p in y_prob]
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)

    return {
        "n": len(y_true),
        "accuracy": _safe_div(tp + tn, len(y_true)),
        "precision": _safe_div(tp, tp + fp),
        "recall": recall,
        "specificity": specificity,
        "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": (recall + specificity) / 2 if not (math.isnan(recall) or math.isnan(specificity)) else float("nan"),
        "auc_roc": _roc_auc(y_true, y_prob),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _build_detector_model(backbone_name: str, trainable_layers: int = 5) -> MaskRCNN:
    if backbone_name not in _SUPPORTED_DETECTOR_BACKBONES:
        raise ValueError(
            f"Unsupported detector backbone {backbone_name!r}. "
            f"Choose one of {sorted(_SUPPORTED_DETECTOR_BACKBONES)}"
        )

    backbone = resnet_fpn_backbone(
        backbone_name=backbone_name,
        weights=None,
        trainable_layers=trainable_layers,
    )
    return MaskRCNN(backbone=backbone, num_classes=2, min_size=1024, max_size=1024)


def _load_detector(checkpoint_path: str, backbone_name: str, device: torch.device) -> MaskRCNN:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Detector checkpoint not found: {ckpt_path}")

    blob = torch.load(ckpt_path, map_location="cpu")
    state_dict = _unwrap_state_dict(blob)

    model = _build_detector_model(backbone_name=backbone_name, trainable_layers=5)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        alt_backbones = [b for b in _SUPPORTED_DETECTOR_BACKBONES if b != backbone_name]
        loaded = False
        for alt in alt_backbones:
            alt_model = _build_detector_model(backbone_name=alt, trainable_layers=5)
            try:
                alt_model.load_state_dict(state_dict, strict=True)
                model = alt_model
                loaded = True
                print(f"[fp_tp] Detector checkpoint matched backbone '{alt}' (overriding '{backbone_name}').")
                break
            except RuntimeError:
                continue
        if not loaded:
            raise RuntimeError(
                "Failed to load detector checkpoint. Try --detector_backbone resnet101/resnet152."
            ) from exc

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def _to_device_targets(targets: list[dict], device: torch.device) -> list[dict]:
    moved = []
    for tgt in targets:
        moved.append({k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in tgt.items()})
    return moved


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


def _crop_patch(
    image: torch.Tensor,
    box_xyxy: torch.Tensor,
    patch_size: int,
    context_factor: float,
) -> torch.Tensor | None:
    _, h, w = image.shape
    x1, y1, x2, y2 = _expand_box_xyxy(box_xyxy, h=h, w=w, context_factor=context_factor)

    if x2 <= x1 or y2 <= y1:
        return None

    patch = image[:, y1:y2, x1:x2]
    if patch.numel() == 0:
        return None

    patch = _pad_to_square(patch)
    patch = nnf.interpolate(
        patch.unsqueeze(0),
        size=(patch_size, patch_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return patch


def _limit_negatives(labels: torch.Tensor, max_neg_pos_ratio: float) -> torch.Tensor:
    # max_neg_pos_ratio <= 0 means keep all negatives (no downsampling).
    if labels.numel() == 0 or max_neg_pos_ratio <= 0:
        return torch.arange(labels.numel(), device=labels.device)

    pos_idx = torch.nonzero(labels == 1).flatten()
    neg_idx = torch.nonzero(labels == 0).flatten()

    if pos_idx.numel() == 0:
        max_neg = min(int(max_neg_pos_ratio), int(neg_idx.numel()))
    else:
        max_neg = min(int(math.ceil(max_neg_pos_ratio * pos_idx.numel())), int(neg_idx.numel()))

    if max_neg <= 0:
        return pos_idx
    if neg_idx.numel() > max_neg:
        perm = torch.randperm(neg_idx.numel(), device=labels.device)
        neg_idx = neg_idx[perm[:max_neg]]

    return torch.cat([pos_idx, neg_idx], dim=0)


def _assign_fp_tp_labels(
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    tp_iou_thr: float,
    fp_iou_max: float,
    label_assignment: str,
) -> tuple[torch.Tensor, dict]:
    """
    Assign per-proposal labels for box-quality classification.

    Returns:
        labels: Long tensor with values in {-1, 0, 1}
            -1: ignore, 0: FP, 1: TP
        meta: assignment stats
    """
    n_pred = int(pred_boxes.shape[0])
    labels = torch.full((n_pred,), -1, dtype=torch.long, device=pred_boxes.device)
    meta = {
        "num_labeled_tp": 0,
        "num_labeled_fp": 0,
        "num_labeled_fp_duplicate": 0,
        "num_ignored_ambiguous": 0,
    }

    if n_pred == 0:
        return labels, meta

    if gt_boxes.numel() == 0:
        labels[:] = 0
        meta["num_labeled_fp"] = n_pred
        return labels, meta

    ious = box_iou(pred_boxes, gt_boxes)
    max_iou, best_gt = ious.max(dim=1)

    if label_assignment == "max_iou":
        labels[max_iou >= float(tp_iou_thr)] = 1
        labels[max_iou <= float(fp_iou_max)] = 0
        meta["num_labeled_tp"] = int((labels == 1).sum().item())
        meta["num_labeled_fp"] = int((labels == 0).sum().item())
        meta["num_ignored_ambiguous"] = int((labels == -1).sum().item())
        return labels, meta

    # Greedy 1:1 assignment: each GT can explain at most one TP proposal.
    # Additional high-IoU duplicates are treated as FP.
    if label_assignment != "greedy":
        raise ValueError(f"Unsupported label_assignment={label_assignment!r}")

    used_gt = torch.zeros((gt_boxes.shape[0],), dtype=torch.bool, device=pred_boxes.device)
    order = torch.argsort(max_iou, descending=True)

    for pred_idx_t in order:
        pred_idx = int(pred_idx_t.item())
        iou_val = float(max_iou[pred_idx].item())
        gt_idx = int(best_gt[pred_idx].item())

        if iou_val >= float(tp_iou_thr):
            if not bool(used_gt[gt_idx].item()):
                labels[pred_idx] = 1
                used_gt[gt_idx] = True
                meta["num_labeled_tp"] += 1
            else:
                labels[pred_idx] = 0
                meta["num_labeled_fp"] += 1
                meta["num_labeled_fp_duplicate"] += 1
        elif iou_val <= float(fp_iou_max):
            labels[pred_idx] = 0
            meta["num_labeled_fp"] += 1
        else:
            meta["num_ignored_ambiguous"] += 1

    return labels, meta


def build_fp_tp_training_batch(
    images: list[torch.Tensor],
    targets: list[dict],
    detections: list[dict],
    patch_size: int,
    det_score_threshold: float,
    tp_iou_thr: float,
    fp_iou_max: float,
    label_assignment: str,
    context_factor: float,
    max_patches_per_image: int,
    max_neg_pos_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    patches = []
    labels = []

    stats = {
        "num_images": len(images),
        "num_pred_boxes": 0,
        "num_after_score_threshold": 0,
        "num_labeled_tp": 0,
        "num_labeled_fp": 0,
        "num_labeled_fp_duplicate": 0,
        "num_ignored_ambiguous": 0,
        "num_skipped_invalid_crop": 0,
        "num_training_patches": 0,
    }

    for image, target, det in zip(images, targets, detections):
        pred_scores = det.get("scores", torch.empty((0,), device=image.device))
        pred_boxes = det.get("boxes", torch.empty((0, 4), device=image.device))
        gt_boxes = target.get("boxes", torch.empty((0, 4), device=image.device))

        if pred_boxes.numel() == 0:
            continue

        stats["num_pred_boxes"] += int(pred_boxes.shape[0])

        if pred_scores.numel() > 0:
            keep = pred_scores >= float(det_score_threshold)
            pred_boxes = pred_boxes[keep]
        if pred_boxes.numel() == 0:
            continue

        stats["num_after_score_threshold"] += int(pred_boxes.shape[0])

        if max_patches_per_image > 0 and pred_boxes.shape[0] > max_patches_per_image:
            pred_boxes = pred_boxes[:max_patches_per_image]

        assigned_labels, assign_meta = _assign_fp_tp_labels(
            pred_boxes=pred_boxes,
            gt_boxes=gt_boxes,
            tp_iou_thr=tp_iou_thr,
            fp_iou_max=fp_iou_max,
            label_assignment=label_assignment,
        )
        stats["num_labeled_tp"] += int(assign_meta["num_labeled_tp"])
        stats["num_labeled_fp"] += int(assign_meta["num_labeled_fp"])
        stats["num_labeled_fp_duplicate"] += int(assign_meta["num_labeled_fp_duplicate"])
        stats["num_ignored_ambiguous"] += int(assign_meta["num_ignored_ambiguous"])

        patch_list = []
        label_list = []

        for box, label_t in zip(pred_boxes, assigned_labels):
            label = int(label_t.item())
            if label < 0:
                continue

            patch = _crop_patch(
                image=image,
                box_xyxy=box,
                patch_size=patch_size,
                context_factor=context_factor,
            )
            if patch is None:
                stats["num_skipped_invalid_crop"] += 1
                continue

            patch_list.append(patch)
            label_list.append(label)

        if patch_list:
            labels_t = torch.tensor(label_list, dtype=torch.long, device=image.device)
            keep_idx = _limit_negatives(labels_t, max_neg_pos_ratio=max_neg_pos_ratio)
            for idx in keep_idx.tolist():
                patches.append(patch_list[idx])
                labels.append(label_list[idx])

    if patches:
        patch_batch = torch.stack(patches, dim=0)
        label_batch = torch.tensor(labels, dtype=torch.long, device=patch_batch.device)
    else:
        device = images[0].device if images else torch.device("cpu")
        patch_batch = torch.empty((0, 3, patch_size, patch_size), device=device)
        label_batch = torch.empty((0,), dtype=torch.long, device=device)

    stats["num_training_patches"] = int(label_batch.numel())
    return patch_batch, label_batch, stats


@torch.no_grad()
def compute_class_weights(
    detector,
    loader,
    device,
    patch_size: int,
    det_score_threshold: float,
    tp_iou_thr: float,
    fp_iou_max: float,
    label_assignment: str,
    context_factor: float,
    max_patches_per_image: int,
    max_neg_pos_ratio: float,
) -> torch.Tensor | None:
    # Inverse-frequency weighting on the mined training patches:
    # w_c = N / (2 * n_c), where n_c is class count for class c.
    counts = [0, 0]

    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = _to_device_targets(targets, device)
        detections = detector(images)

        _, labels, _ = build_fp_tp_training_batch(
            images=images,
            targets=targets,
            detections=detections,
            patch_size=patch_size,
            det_score_threshold=det_score_threshold,
            tp_iou_thr=tp_iou_thr,
            fp_iou_max=fp_iou_max,
            label_assignment=label_assignment,
            context_factor=context_factor,
            max_patches_per_image=max_patches_per_image,
            max_neg_pos_ratio=max_neg_pos_ratio,
        )

        for value in labels.tolist():
            counts[value] += 1

    total = sum(counts)
    if total == 0:
        print("[fp_tp] No labeled patches found for class-weight estimation.")
        return None

    weights = [total / (2 * max(c, 1)) for c in counts]
    print(f"[fp_tp] Class counts: fp={counts[0]}, tp={counts[1]}")
    print(f"[fp_tp] Class weights: {weights}")
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _run_epoch(
    mode: str,
    detector,
    classifier,
    loader,
    criterion,
    optimizer,
    device,
    args,
) -> dict:
    is_train = mode == "train"
    if is_train:
        classifier.train()
    else:
        classifier.eval()
    detector.eval()

    total_loss = 0.0
    total_samples = 0
    y_true: list[int] = []
    y_prob: list[float] = []

    sum_stats = {
        "num_pred_boxes": 0,
        "num_after_score_threshold": 0,
        "num_labeled_tp": 0,
        "num_labeled_fp": 0,
        "num_labeled_fp_duplicate": 0,
        "num_ignored_ambiguous": 0,
        "num_skipped_invalid_crop": 0,
        "num_training_patches": 0,
    }

    for batch_idx, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = _to_device_targets(targets, device)

        with torch.no_grad():
            detections = detector(images)

        patches, labels, stats = build_fp_tp_training_batch(
            images=images,
            targets=targets,
            detections=detections,
            patch_size=args.patch_size,
            det_score_threshold=args.det_score_threshold,
            tp_iou_thr=args.tp_iou_thr,
            fp_iou_max=args.fp_iou_max,
            label_assignment=args.label_assignment,
            context_factor=args.context_factor,
            max_patches_per_image=args.max_patches_per_image,
            max_neg_pos_ratio=args.max_neg_pos_ratio,
        )

        for key in sum_stats:
            sum_stats[key] += int(stats.get(key, 0))

        if labels.numel() == 0:
            continue

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = classifier(patches)
        loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            optimizer.step()

        probs = torch.softmax(logits, dim=1)[:, 1]

        batch_n = int(labels.numel())
        total_loss += float(loss.item()) * batch_n
        total_samples += batch_n
        y_true.extend(labels.tolist())
        y_prob.extend(probs.detach().cpu().tolist())

        if is_train and (batch_idx + 1) % 10 == 0:
            running = _compute_metrics(y_true, y_prob, threshold=args.cls_threshold)
            print(
                f"  [{mode}] batch {batch_idx + 1}/{len(loader)} "
                f"loss={loss.item():.4f} acc={running['accuracy']:.4f} "
                f"f1={running['f1']:.4f} patches={total_samples}"
            )

    metrics = _compute_metrics(y_true, y_prob, threshold=args.cls_threshold)
    metrics["loss"] = _safe_div(total_loss, total_samples)
    metrics["samples"] = total_samples
    metrics.update({f"stats/{k}": v for k, v in sum_stats.items()})
    return metrics


def _coerce_sweep_value(key: str, value):
    float_keys = {
        "lr",
        "weight_decay",
        "dropout",
        "label_smoothing",
        "det_score_threshold",
        "tp_iou_thr",
        "fp_iou_max",
        "context_factor",
        "max_neg_pos_ratio",
        "cls_threshold",
        "val_split",
    }
    int_keys = {
        "epochs",
        "warmup_epochs",
        "early_stopping_patience",
        "batch_size",
        "num_workers",
        "patch_size",
        "head_hidden",
        "freeze_stages",
        "max_patches_per_image",
        "seed",
    }
    if key in float_keys:
        return float(value)
    if key in int_keys:
        return int(value)
    return value


def _apply_run_overrides(args, run_config: dict | None) -> None:
    if not run_config:
        return
    for key, value in run_config.items():
        if hasattr(args, key):
            setattr(args, key, _coerce_sweep_value(key, value))


def _build_sweep_config(args) -> dict:
    return {
        "method": "bayes",
        "metric": {"name": "val/balanced_accuracy", "goal": "maximize"},
        "parameters": {
            "lr": {"values": [1e-5, 3e-5, 1e-4, 3e-4]},
            "weight_decay": {"values": [1e-5, 1e-4, 1e-3]},
            "dropout": {"values": [0.1, 0.2, 0.3]},
            "freeze_stages": {"values": [1, 2]},
            "head_hidden": {"values": [256, 512]},
            "det_score_threshold": {"values": [0.05, 0.1, 0.2]},
            "tp_iou_thr": {"values": [0.4, 0.5, 0.6]},
            "fp_iou_max": {"values": [0.1, 0.2, 0.3]},
            "label_assignment": {"value": args.label_assignment},
            "context_factor": {"values": [1.0, 1.1, 1.2, 1.3]},
            "max_neg_pos_ratio": {"values": [2.0, 3.0, 4.0]},
            "cls_threshold": {"values": [0.4, 0.5, 0.6]},
            "batch_size": {"value": args.batch_size},
            "epochs": {"value": args.epochs},
            "patch_size": {"value": args.patch_size},
            "backbone": {"value": args.backbone},
            "label_smoothing": {"value": args.label_smoothing},
            "class_weight": {"value": args.class_weight},
        },
    }


def run_training(base_args, run_config: dict | None = None, log_to_wandb: bool = False) -> dict:
    args = copy.deepcopy(base_args)
    _apply_run_overrides(args, run_config)

    if args.label_assignment == "max_iou":
        print(
            "[fp_tp][warning] label_assignment='max_iou' allows duplicate detections around the "
            "same GT to be labeled TP. For FP suppression, 'greedy' is usually preferred."
        )

    _seed_everything(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    detector = _load_detector(
        checkpoint_path=args.detector_checkpoint,
        backbone_name=args.detector_backbone,
        device=device,
    )

    classifier = FPTPBoxClassifier(
        backbone=args.backbone,
        freeze_stages=args.freeze_stages,
        head_hidden=args.head_hidden,
        dropout=args.dropout,
        num_classes=2,
    ).to(device)

    trainable = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    total = sum(p.numel() for p in classifier.parameters())
    print(f"[fp_tp] Parameters: {trainable:,} trainable / {total:,} total")

    cfg = DatasetConfig(
        data_root=args.data_root,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        preprocess=PreprocessConfig(
            target_size=(1024, 1024),
            representation=Representation.MMS_PSEUDO_COLOR,
            remove_pectoral=False,
            orient_breast=True,
            tight_crop=True,
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        augment=args.augment,
    )

    all_train_samples = build_sample_index(os.path.join(cfg.data_root, "train"), cfg.train_csv)
    patients = sorted({s["patient_id"] for s in all_train_samples})
    random.Random(args.seed).shuffle(patients)
    val_count = int(len(patients) * args.val_split)
    val_patients = set(patients[:val_count])

    train_samples = [s for s in all_train_samples if s["patient_id"] not in val_patients]
    val_samples = [s for s in all_train_samples if s["patient_id"] in val_patients]

    print(
        f"[dataset] Train: {len(train_samples)} images | "
        f"Val: {len(val_samples)} images | "
        f"Val patients: {len(val_patients)}"
    )
    if args.max_patches_per_image <= 0:
        print("[fp_tp] Proposal cap: disabled (using all detector boxes above score threshold).")
    else:
        print(f"[fp_tp] Proposal cap: up to {int(args.max_patches_per_image)} boxes/image.")

    if args.max_neg_pos_ratio <= 0:
        print("[fp_tp] Negative sampling: disabled (keeping all FP-labeled patches).")
    else:
        print(f"[fp_tp] Negative sampling: max_neg_pos_ratio={float(args.max_neg_pos_ratio):.2f}.")

    print(f"[fp_tp] Label assignment mode: {args.label_assignment}")

    train_loader = DataLoader(
        CBISDDSMDataset(train_samples, cfg, train=True),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        CBISDDSMDataset(val_samples, cfg, train=False),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=_collate_fn,
    )

    if args.class_weight == "auto":
        class_weights = compute_class_weights(
            detector=detector,
            loader=train_loader,
            device=device,
            patch_size=args.patch_size,
            det_score_threshold=args.det_score_threshold,
            tp_iou_thr=args.tp_iou_thr,
            fp_iou_max=args.fp_iou_max,
            label_assignment=args.label_assignment,
            context_factor=args.context_factor,
            max_patches_per_image=args.max_patches_per_image,
            max_neg_pos_ratio=args.max_neg_pos_ratio,
        )
    else:
        class_weights = None

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)

    optimizer = torch.optim.AdamW(
        [p for p in classifier.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    warmup_epochs = max(0, int(args.warmup_epochs))

    def lr_lambda(epoch_idx):
        if warmup_epochs > 0 and epoch_idx < warmup_epochs:
            return (epoch_idx + 1) / warmup_epochs
        progress = (epoch_idx - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if log_to_wandb and wandb is not None and wandb.run is not None:
        run_dir = Path(args.output_dir) / wandb.run.id
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path(args.output_dir) / f"fp_tp_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    save_hparams = {
        "detector_checkpoint": args.detector_checkpoint,
        "detector_backbone": args.detector_backbone,
        "data_root": args.data_root,
        "train_csv": args.train_csv,
        "test_csv": args.test_csv,
        "val_split": args.val_split,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "augment": bool(args.augment),
        "backbone": args.backbone,
        "freeze_stages": args.freeze_stages,
        "head_hidden": args.head_hidden,
        "dropout": args.dropout,
        "patch_size": args.patch_size,
        "label_smoothing": args.label_smoothing,
        "class_weight": args.class_weight,
        "det_score_threshold": args.det_score_threshold,
        "tp_iou_thr": args.tp_iou_thr,
        "fp_iou_max": args.fp_iou_max,
        "label_assignment": args.label_assignment,
        "context_factor": args.context_factor,
        "max_patches_per_image": args.max_patches_per_image,
        "max_neg_pos_ratio": args.max_neg_pos_ratio,
        "cls_threshold": args.cls_threshold,
        "seed": args.seed,
        "sweep_enabled": bool(base_args.sweep),
        "sweep_count": int(base_args.sweep_count),
        "sweep_run_overrides": run_config or {},
        "label_semantics": {"0": "fp", "1": "tp"},
    }
    with (run_dir / "hyperparams.json").open("w", encoding="utf-8") as f:
        json.dump(save_hparams, f, indent=2)

    best_bal_acc = float("-inf")
    best_f1 = float("-inf")
    best_auc = float("-inf")
    best_epoch = -1
    epochs_without_improvement = 0

    patience = max(1, int(args.early_stopping_patience))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = _run_epoch(
            mode="train",
            detector=detector,
            classifier=classifier,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            args=args,
        )

        with torch.no_grad():
            val_metrics = _run_epoch(
                mode="val",
                detector=detector,
                classifier=classifier,
                loader=val_loader,
                criterion=criterion,
                optimizer=None,
                device=device,
                args=args,
            )

        scheduler.step()
        elapsed = time.time() - t0

        epoch_ckpt = run_dir / f"fp_tp_classifier_epoch{epoch:03d}.pth"
        torch.save(classifier.state_dict(), epoch_ckpt)

        if val_metrics["samples"] > 0 and val_metrics["balanced_accuracy"] > best_bal_acc:
            best_bal_acc = val_metrics["balanced_accuracy"]
            best_f1 = max(best_f1, val_metrics["f1"])
            best_auc = max(best_auc, val_metrics["auc_roc"])
            best_epoch = epoch
            torch.save(classifier.state_dict(), run_dir / "fp_tp_classifier_best.pth")
            print(
                "  -> New best classifier saved "
                f"(bal_acc={best_bal_acc:.4f}, f1={val_metrics['f1']:.4f}, auc={val_metrics['auc_roc']:.4f})"
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
            f"train_f1={train_metrics['f1']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"val_auc={val_metrics['auc_roc']:.4f} "
            f"train_patches={train_metrics['samples']} val_patches={val_metrics['samples']} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} time={elapsed:.1f}s"
        )

        if log_to_wandb and wandb is not None and wandb.run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_metrics["loss"],
                    "train/accuracy": train_metrics["accuracy"],
                    "train/precision": train_metrics["precision"],
                    "train/recall": train_metrics["recall"],
                    "train/specificity": train_metrics["specificity"],
                    "train/f1": train_metrics["f1"],
                    "train/balanced_accuracy": train_metrics["balanced_accuracy"],
                    "train/auc_roc": train_metrics["auc_roc"],
                    "val/loss": val_metrics["loss"],
                    "val/accuracy": val_metrics["accuracy"],
                    "val/precision": val_metrics["precision"],
                    "val/recall": val_metrics["recall"],
                    "val/specificity": val_metrics["specificity"],
                    "val/f1": val_metrics["f1"],
                    "val/balanced_accuracy": val_metrics["balanced_accuracy"],
                    "val/auc_roc": val_metrics["auc_roc"],
                    "train/patches": train_metrics["samples"],
                    "val/patches": val_metrics["samples"],
                    "train/num_pred_boxes": train_metrics["stats/num_pred_boxes"],
                    "train/num_labeled_tp": train_metrics["stats/num_labeled_tp"],
                    "train/num_labeled_fp": train_metrics["stats/num_labeled_fp"],
                    "train/num_labeled_fp_duplicate": train_metrics["stats/num_labeled_fp_duplicate"],
                    "train/num_ignored_ambiguous": train_metrics["stats/num_ignored_ambiguous"],
                    "val/num_pred_boxes": val_metrics["stats/num_pred_boxes"],
                    "val/num_labeled_tp": val_metrics["stats/num_labeled_tp"],
                    "val/num_labeled_fp": val_metrics["stats/num_labeled_fp"],
                    "val/num_labeled_fp_duplicate": val_metrics["stats/num_labeled_fp_duplicate"],
                    "val/num_ignored_ambiguous": val_metrics["stats/num_ignored_ambiguous"],
                    "val/best_balanced_accuracy": best_bal_acc,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

        if epochs_without_improvement >= patience:
            print(
                f"  -> Early stopping at epoch {epoch} "
                f"(best_bal_acc={best_bal_acc:.4f}, patience={patience})"
            )
            break

    print("\nFP/TP classifier training complete.")
    if best_epoch > 0:
        print(
            f"Best epoch: {best_epoch} | best val_bal_acc={best_bal_acc:.4f} "
            f"| best val_f1={best_f1:.4f} | best val_auc={best_auc:.4f}"
        )
        print(f"Best checkpoint: {run_dir / 'fp_tp_classifier_best.pth'}")
    else:
        print("No labeled validation patches found for model selection.")

    summary = {
        "best_epoch": int(best_epoch),
        "best_val_balanced_accuracy": float(best_bal_acc) if math.isfinite(best_bal_acc) else None,
        "best_val_f1": float(best_f1) if math.isfinite(best_f1) else None,
        "best_val_auc": float(best_auc) if math.isfinite(best_auc) else None,
        "run_dir": str(run_dir),
    }

    if log_to_wandb and wandb is not None and wandb.run is not None:
        wandb.run.summary["best_epoch"] = summary["best_epoch"]
        wandb.run.summary["val/best_balanced_accuracy"] = summary["best_val_balanced_accuracy"]
        wandb.run.summary["val/best_f1"] = summary["best_val_f1"]
        wandb.run.summary["val/best_auc_roc"] = summary["best_val_auc"]

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train detector-box FP/TP classifier (ResNet)")
    parser.add_argument("--detector_checkpoint", required=True)
    parser.add_argument("--detector_backbone", choices=["resnet101", "resnet152"], default="resnet101")

    parser.add_argument("--data_root", default="./data/cbis-ddsm")
    parser.add_argument("--train_csv", default="./meta/cbis-ddsm/mass_case_description_train_set.csv")
    parser.add_argument("--test_csv", default="./meta/cbis-ddsm/mass_case_description_test_set.csv")
    parser.add_argument("--val_split", type=float, default=0.15)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--early_stopping_patience", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--augment", action="store_true")

    parser.add_argument("--backbone", choices=["resnet18", "resnet34", "resnet50"], default="resnet34")
    parser.add_argument("--freeze_stages", type=int, default=2)
    parser.add_argument("--head_hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--class_weight", choices=["auto", "none"], default="auto")

    parser.add_argument("--det_score_threshold", type=float, default=0.10)
    parser.add_argument("--tp_iou_thr", type=float, default=0.50)
    parser.add_argument("--fp_iou_max", type=float, default=0.20)
    parser.add_argument(
        "--label_assignment",
        choices=["greedy", "max_iou"],
        default="greedy",
        help="How to map detector boxes to TP/FP labels. 'greedy' enforces one-to-one GT matching.",
    )
    parser.add_argument("--context_factor", type=float, default=1.2)
    parser.add_argument(
        "--max_patches_per_image",
        type=int,
        default=36,
        help="Cap detector proposals per image before FP/TP labeling; <=0 keeps all boxes above det_score_threshold.",
    )
    parser.add_argument(
        "--max_neg_pos_ratio",
        type=float,
        default=3.0,
        help="Max FP:TP ratio after labeling; <=0 disables negative downsampling and keeps all FP patches.",
    )
    parser.add_argument("--cls_threshold", type=float, default=0.50,
                        help="TP probability threshold for metric computation")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="./models_fp_tp")

    parser.add_argument("--sweep", action="store_true",
                        help="Run W&B sweep")
    parser.add_argument("--sweep_count", type=int, default=24)
    parser.add_argument("--wandb_project", default="hadamlab-fp-tp-classifier")
    parser.add_argument("--wandb_entity", default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.fp_iou_max >= args.tp_iou_thr:
        raise ValueError(
            f"Expected fp_iou_max < tp_iou_thr, got {args.fp_iou_max} >= {args.tp_iou_thr}."
        )

    if args.sweep:
        if wandb is None:
            raise RuntimeError("Sweep requested but wandb is not installed. Install it with: pip install wandb")
        if args.sweep_count < 1:
            raise ValueError(f"sweep_count must be >= 1, got {args.sweep_count}")

        wandb_target = {"project": args.wandb_project}
        wandb_entity = args.wandb_entity or os.getenv("WANDB_ENTITY")
        if wandb_entity:
            wandb_target["entity"] = wandb_entity

        sweep_config = _build_sweep_config(args)
        sweep_id = wandb.sweep(sweep_config, **wandb_target)

        base_config = {
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "backbone": args.backbone,
            "dropout": args.dropout,
            "freeze_stages": args.freeze_stages,
            "head_hidden": args.head_hidden,
            "det_score_threshold": args.det_score_threshold,
            "tp_iou_thr": args.tp_iou_thr,
            "fp_iou_max": args.fp_iou_max,
            "context_factor": args.context_factor,
            "max_neg_pos_ratio": args.max_neg_pos_ratio,
            "cls_threshold": args.cls_threshold,
            "patch_size": args.patch_size,
            "label_smoothing": args.label_smoothing,
            "class_weight": args.class_weight,
            "val_split": args.val_split,
        }

        def sweep_run():
            wandb.init(**wandb_target, config=base_config)
            run_training(base_args=args, run_config=dict(wandb.config), log_to_wandb=True)
            wandb.finish()

        wandb.agent(sweep_id, function=sweep_run, count=args.sweep_count)
    else:
        run_training(base_args=args, run_config=None, log_to_wandb=False)


if __name__ == "__main__":
    main()
