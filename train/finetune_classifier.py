"""
Fine-tune the standalone classifier while freezing a pretrained detector.

Workflow:
1) Load detector checkpoint (e.g. models/nb7u0jeg/detector_resnet101_best.pth)
2) Load pretrained classifier from classifier_model/
3) Freeze detector parameters
4) Train classifier on detector-predicted crops matched to GT pathology labels
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

import torch
import numpy as np
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:
    wandb = None

from config import DatasetConfig, PreprocessConfig, Representation
from dataset import CBISDDSMDataset, _collate_fn, build_sample_index
from full_model import load_full_model


def _to_device_targets(targets: list[dict], device: torch.device) -> list[dict]:
    moved = []
    for tgt in targets:
        moved.append({k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in tgt.items()})
    return moved


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def _finite_or_nan(value: float) -> float:
    return float(value) if math.isfinite(value) else float("nan")


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


def _threshold_schedule(epoch: int, args) -> tuple[float, float]:
    """Curriculum from strict thresholds to final thresholds."""
    final_score = float(args.score_threshold)
    final_iou = float(args.iou_match_thr)
    curriculum_epochs = max(0, int(args.curriculum_epochs))

    if curriculum_epochs <= 0 or epoch > curriculum_epochs:
        return final_score, final_iou

    strict_score = float(args.strict_score_threshold)
    strict_iou = float(args.strict_iou_match_thr)

    if curriculum_epochs == 1:
        score = strict_score
        iou = strict_iou
    else:
        alpha = (epoch - 1) / float(curriculum_epochs - 1)
        score = strict_score + alpha * (final_score - strict_score)
        iou = strict_iou + alpha * (final_iou - strict_iou)

    score = max(0.0, score)
    iou = min(1.0, max(0.0, iou))
    return score, iou


def _coerce_sweep_value(key: str, value):
    float_keys = {
        "lr",
        "weight_decay",
        "label_smoothing",
        "val_split",
        "score_threshold",
        "iou_match_thr",
        "strict_score_threshold",
        "strict_iou_match_thr",
    }
    int_keys = {
        "epochs",
        "warmup_epochs",
        "early_stopping_patience",
        "batch_size",
        "num_workers",
        "max_patches_per_image",
        "curriculum_epochs",
        "min_patches_per_batch",
        "max_gt_fallback_patches_per_batch",
        "max_gt_fallback_patches_per_image",
        "patch_size",
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


def _build_bayesian_grid_sweep_config(args) -> dict:
    return {
        "method": "bayes",
        "metric": {"name": "val/best_balanced_accuracy", "goal": "maximize"},
        "parameters": {
            "lr": {"values": [2e-5, 5e-5, 1e-4, 5e-4]},
            "weight_decay": {"values": [1e-5, 1e-4, 1e-3]},
            "label_smoothing": {"values": [0.0, 0.05, 0.1]},
            "score_threshold": {"values": [0.05, 0.10, 0.15]},
            "iou_match_thr": {"values": [0.30, 0.40, 0.50]},
            "strict_score_threshold": {"values": [0.20, 0.25, 0.30]},
            "strict_iou_match_thr": {"values": [0.50, 0.60]},
            "min_patches_per_batch": {"values": [6, 8, 10, 12]},
            "batch_size": {"value": args.batch_size},
            "epochs": {"value": args.epochs},
            "warmup_epochs": {"value": args.warmup_epochs},
            "curriculum_epochs": {"value": args.curriculum_epochs},
            "max_patches_per_image": {"value": args.max_patches_per_image},
            "max_gt_fallback_patches_per_batch": {"value": args.max_gt_fallback_patches_per_batch},
            "max_gt_fallback_patches_per_image": {"value": args.max_gt_fallback_patches_per_image},
            "class_weight": {"value": args.class_weight},
            "augment": {"value": args.augment},
        },
    }


def _build_patches_with_optional_gt_fallback(
    model,
    images,
    targets,
    args,
    score_threshold: float,
    iou_match_thr: float,
    for_train: bool,
):
    patches, labels, stats = model.build_classifier_training_batch(
        images=images,
        targets=targets,
        iou_match_thr=iou_match_thr,
        score_threshold=score_threshold,
        max_patches_per_image=args.max_patches_per_image,
    )

    gt_added = 0
    use_fallback = (not args.disable_gt_fallback) and (for_train or args.gt_fallback_in_val)
    desired = max(0, int(args.min_patches_per_batch))

    if use_fallback and labels.numel() < desired:
        needed = desired - int(labels.numel())
        gt_patches, gt_labels, _gt_stats = model.build_gt_classifier_training_batch(
            images=images,
            targets=targets,
            max_patches_per_image=args.max_gt_fallback_patches_per_image,
        )
        if gt_labels.numel() > 0 and needed > 0:
            max_add = max(0, int(args.max_gt_fallback_patches_per_batch))
            if max_add == 0:
                take = min(needed, int(gt_labels.numel()))
            else:
                take = min(needed, max_add, int(gt_labels.numel()))

            if take > 0:
                gt_patches = gt_patches[:take]
                gt_labels = gt_labels[:take]
                if patches.numel() == 0:
                    patches = gt_patches
                    labels = gt_labels
                else:
                    patches = torch.cat([patches, gt_patches], dim=0)
                    labels = torch.cat([labels, gt_labels], dim=0)
                gt_added = int(take)

    out_stats = dict(stats)
    out_stats["detector_training_patches"] = int(stats.get("num_training_patches", 0))
    out_stats["gt_fallback_patches"] = int(gt_added)
    out_stats["num_training_patches"] = int(labels.numel())
    return patches, labels, out_stats


@torch.no_grad()
def compute_class_weights(model, loader, device, args, score_threshold: float, iou_match_thr: float) -> torch.Tensor | None:
    counts = [0, 0]
    model.detector.eval()
    model.classifier.eval()

    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = _to_device_targets(targets, device)
        _, labels, _stats = _build_patches_with_optional_gt_fallback(
            model=model,
            images=images,
            targets=targets,
            args=args,
            iou_match_thr=iou_match_thr,
            score_threshold=score_threshold,
            for_train=True,
        )
        for l in labels.tolist():
            counts[l] += 1

    total = sum(counts)
    if total == 0:
        print("[finetune] No classifier patches found for class-weight estimation; using unweighted loss.")
        return None

    weights = [total / (2 * max(c, 1)) for c in counts]
    print(f"[finetune] Class counts: benign={counts[0]}, malignant={counts[1]}")
    print(f"[finetune] Class weights: {weights}")
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    args,
    epoch: int,
    score_threshold: float,
    iou_match_thr: float,
) -> dict:
    model.classifier.train()
    model.detector.eval()

    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    tp = tn = fp = fn = 0
    all_true = []
    all_prob = []
    matched_boxes = 0
    training_patches = 0
    gt_fallback_patches = 0

    for batch_idx, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = _to_device_targets(targets, device)

        patches, labels, stats = _build_patches_with_optional_gt_fallback(
            model=model,
            images=images,
            targets=targets,
            args=args,
            iou_match_thr=iou_match_thr,
            score_threshold=score_threshold,
            for_train=True,
        )

        matched_boxes += int(stats["num_matched_boxes"])
        training_patches += int(stats["num_training_patches"])
        gt_fallback_patches += int(stats.get("gt_fallback_patches", 0))

        if labels.numel() == 0:
            continue

        logits = model.classifier(patches)
        loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.classifier.parameters(), max_norm=1.0)
        optimizer.step()

        preds = logits.argmax(dim=1)
        probs = torch.softmax(logits, dim=1)[:, 1]

        labels_list = labels.tolist()
        preds_list = preds.tolist()
        all_true.extend(labels_list)
        all_prob.extend(probs.detach().cpu().tolist())

        tp += sum(1 for t, p in zip(labels_list, preds_list) if t == 1 and p == 1)
        tn += sum(1 for t, p in zip(labels_list, preds_list) if t == 0 and p == 0)
        fp += sum(1 for t, p in zip(labels_list, preds_list) if t == 0 and p == 1)
        fn += sum(1 for t, p in zip(labels_list, preds_list) if t == 1 and p == 0)

        total_correct += int((preds == labels).sum().item())
        batch_n = int(labels.numel())
        total_samples += batch_n
        total_loss += float(loss.item()) * batch_n

        if (batch_idx + 1) % 10 == 0:
            acc = _safe_div(total_correct, total_samples)
            print(
                f"  [epoch {epoch}] batch {batch_idx + 1}/{len(loader)} "
                f"loss={loss.item():.4f} acc={acc:.4f} patches={total_samples}"
            )

    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)

    return {
        "loss": _safe_div(total_loss, total_samples),
        "accuracy": _safe_div(total_correct, total_samples),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "auc_roc": _roc_auc(all_true, all_prob),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "samples": total_samples,
        "matched_boxes": matched_boxes,
        "training_patches": training_patches,
        "gt_fallback_patches": gt_fallback_patches,
    }


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    args,
    score_threshold: float,
    iou_match_thr: float,
) -> dict:
    model.classifier.eval()
    model.detector.eval()

    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    tp = tn = fp = fn = 0
    all_true = []
    all_prob = []
    matched_boxes = 0
    training_patches = 0
    gt_fallback_patches = 0

    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = _to_device_targets(targets, device)

        patches, labels, stats = _build_patches_with_optional_gt_fallback(
            model=model,
            images=images,
            targets=targets,
            args=args,
            iou_match_thr=iou_match_thr,
            score_threshold=score_threshold,
            for_train=False,
        )

        matched_boxes += int(stats["num_matched_boxes"])
        training_patches += int(stats["num_training_patches"])
        gt_fallback_patches += int(stats.get("gt_fallback_patches", 0))

        if labels.numel() == 0:
            continue

        logits = model.classifier(patches)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)
        probs = torch.softmax(logits, dim=1)[:, 1]

        labels_list = labels.tolist()
        preds_list = preds.tolist()
        all_true.extend(labels_list)
        all_prob.extend(probs.detach().cpu().tolist())

        tp += sum(1 for t, p in zip(labels_list, preds_list) if t == 1 and p == 1)
        tn += sum(1 for t, p in zip(labels_list, preds_list) if t == 0 and p == 0)
        fp += sum(1 for t, p in zip(labels_list, preds_list) if t == 0 and p == 1)
        fn += sum(1 for t, p in zip(labels_list, preds_list) if t == 1 and p == 0)

        total_correct += int((preds == labels).sum().item())
        batch_n = int(labels.numel())
        total_samples += batch_n
        total_loss += float(loss.item()) * batch_n

    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)

    return {
        "loss": _safe_div(total_loss, total_samples),
        "accuracy": _safe_div(total_correct, total_samples),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "auc_roc": _roc_auc(all_true, all_prob),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "samples": total_samples,
        "matched_boxes": matched_boxes,
        "training_patches": training_patches,
        "gt_fallback_patches": gt_fallback_patches,
    }


def run_finetuning(base_args, run_config: dict | None = None, log_to_wandb: bool = False) -> dict:
    args = copy.deepcopy(base_args)
    _apply_run_overrides(args, run_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    patch_size = None if args.patch_size <= 0 else args.patch_size
    model = load_full_model(
        detector_checkpoint=args.detector_checkpoint,
        classifier_dir=args.classifier_dir,
        detector_backbone=args.detector_backbone,
        patch_size=patch_size,
        inference_score_threshold=args.score_threshold,
        device=device,
    )
    model.freeze_detector()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,}")

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
    random.Random(42).shuffle(patients)
    val_count = int(len(patients) * args.val_split)
    val_patients = set(patients[:val_count])

    train_samples = [s for s in all_train_samples if s["patient_id"] not in val_patients]
    val_samples = [s for s in all_train_samples if s["patient_id"] in val_patients]

    print(
        f"[dataset] Train: {len(train_samples)} images | "
        f"Val: {len(val_samples)} images | "
        f"Val patients: {len(val_patients)}"
    )

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

    optimizer = torch.optim.AdamW(
        [p for p in model.classifier.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # classify_train-style warmup + cosine schedule
    warmup_epochs = max(0, int(args.warmup_epochs))

    def lr_lambda(epoch_idx):
        if warmup_epochs > 0 and epoch_idx < warmup_epochs:
            return (epoch_idx + 1) / warmup_epochs
        progress = (epoch_idx - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    init_score_threshold, init_iou_match_thr = _threshold_schedule(epoch=1, args=args)

    if args.class_weight == "auto":
        class_weights = compute_class_weights(
            model,
            train_loader,
            device,
            args,
            score_threshold=init_score_threshold,
            iou_match_thr=init_iou_match_thr,
        )
    else:
        class_weights = None

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )

    if log_to_wandb and wandb is not None and wandb.run is not None:
        run_dir = Path(args.output_dir) / wandb.run.id
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path(args.output_dir) / f"finetune_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    save_hparams = {
        "detector_checkpoint": args.detector_checkpoint,
        "classifier_dir": args.classifier_dir,
        "detector_backbone": args.detector_backbone,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "class_weight": args.class_weight,
        "warmup_epochs": args.warmup_epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "batch_size": args.batch_size,
        "score_threshold": args.score_threshold,
        "iou_match_thr": args.iou_match_thr,
        "curriculum_epochs": args.curriculum_epochs,
        "strict_score_threshold": args.strict_score_threshold,
        "strict_iou_match_thr": args.strict_iou_match_thr,
        "max_patches_per_image": args.max_patches_per_image,
        "min_patches_per_batch": args.min_patches_per_batch,
        "max_gt_fallback_patches_per_batch": args.max_gt_fallback_patches_per_batch,
        "max_gt_fallback_patches_per_image": args.max_gt_fallback_patches_per_image,
        "disable_gt_fallback": args.disable_gt_fallback,
        "gt_fallback_in_val": args.gt_fallback_in_val,
        "patch_size": model.patch_size,
        "base_classifier_hparams": model.classifier_hparams,
        "sweep_enabled": bool(base_args.sweep),
        "sweep_count": int(base_args.sweep_count),
        "sweep_run_overrides": run_config or {},
    }
    with (run_dir / "hyperparams.json").open("w", encoding="utf-8") as f:
        json.dump(save_hparams, f, indent=2)

    best_val_acc = float("-inf")
    best_val_auc = float("-inf")
    best_val_bal_acc = float("-inf")
    best_epoch = -1
    epochs_without_improvement = 0
    patience = max(1, int(args.early_stopping_patience))

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_score_thr, train_iou_thr = _threshold_schedule(epoch=epoch, args=args)
        val_score_thr = float(args.score_threshold)
        val_iou_thr = float(args.iou_match_thr)

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            args,
            epoch,
            score_threshold=train_score_thr,
            iou_match_thr=train_iou_thr,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            args,
            score_threshold=val_score_thr,
            iou_match_thr=val_iou_thr,
        )
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_auc={val_metrics['auc_roc']:.4f} "
            f"val_sens={val_metrics['sensitivity']:.4f} val_spec={val_metrics['specificity']:.4f} "
            f"train_patches={train_metrics['samples']} val_patches={val_metrics['samples']} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} time={elapsed:.1f}s"
        )
        print(
            f"  thresholds train(score/iou)={train_score_thr:.3f}/{train_iou_thr:.3f} "
            f"val(score/iou)={val_score_thr:.3f}/{val_iou_thr:.3f}"
        )
        print(
            f"  matched_boxes train/val={train_metrics['matched_boxes']}/{val_metrics['matched_boxes']} "
            f"| gt_fallback_patches train/val={train_metrics['gt_fallback_patches']}/{val_metrics['gt_fallback_patches']}"
        )

        epoch_ckpt = run_dir / f"classifier_epoch{epoch:03d}.pth"
        torch.save(model.classifier.state_dict(), epoch_ckpt)

        if val_metrics["samples"] > 0 and val_metrics["auc_roc"] > best_val_auc:
            best_val_auc = val_metrics["auc_roc"]

        bal_acc_improved = False
        if val_metrics["samples"] > 0 and val_metrics["balanced_accuracy"] > best_val_bal_acc:
            best_val_bal_acc = val_metrics["balanced_accuracy"]
            bal_acc_improved = True

        if val_metrics["samples"] > 0 and val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch
            torch.save(model.classifier.state_dict(), run_dir / "classifier_best.pth")
            print(
                "  -> New best classifier saved "
                f"(val_acc={best_val_acc:.4f}, bal_acc={val_metrics['balanced_accuracy']:.4f})"
            )

        if bal_acc_improved:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(
                f"  -> Early stopping check: no balanced-accuracy improvement "
                f"for {epochs_without_improvement}/{patience} epoch(s)"
            )
            if epochs_without_improvement >= patience:
                print(
                    f"  -> Early stopping triggered at epoch {epoch} "
                    f"(best_bal_acc={best_val_bal_acc:.4f}, patience={patience})"
                )
                break

        if log_to_wandb and wandb is not None and wandb.run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_metrics["loss"],
                    "train/accuracy": train_metrics["accuracy"],
                    "train/balanced_accuracy": train_metrics["balanced_accuracy"],
                    "val/loss": val_metrics["loss"],
                    "val/accuracy": val_metrics["accuracy"],
                    "val/balanced_accuracy": val_metrics["balanced_accuracy"],
                    "val/auc_roc": val_metrics["auc_roc"],
                    "val/sensitivity": val_metrics["sensitivity"],
                    "val/specificity": val_metrics["specificity"],
                    "val/best_accuracy": _finite_or_nan(best_val_acc),
                    "val/best_balanced_accuracy": _finite_or_nan(best_val_bal_acc),
                    "val/best_auc_roc": _finite_or_nan(best_val_auc),
                    "train/patches": train_metrics["samples"],
                    "val/patches": val_metrics["samples"],
                    "train/matched_boxes": train_metrics["matched_boxes"],
                    "val/matched_boxes": val_metrics["matched_boxes"],
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

    print("\nFine-tuning complete.")
    if best_epoch > 0:
        print(
            f"Best epoch: {best_epoch} | "
            f"best val_acc={best_val_acc:.4f} | "
            f"best val_bal_acc={best_val_bal_acc:.4f} | "
            f"best val_auc={best_val_auc:.4f}"
        )
        print(f"Best checkpoint: {run_dir / 'classifier_best.pth'}")
    else:
        print("No validation patches were available for model selection.")
        print(f"Last checkpoint: {run_dir / f'classifier_epoch{args.epochs:03d}.pth'}")

    summary = {
        "best_epoch": int(best_epoch),
        "best_val_acc": _finite_or_nan(best_val_acc),
        "best_val_bal_acc": _finite_or_nan(best_val_bal_acc),
        "best_val_auc": _finite_or_nan(best_val_auc),
        "run_dir": str(run_dir),
    }

    if log_to_wandb and wandb is not None and wandb.run is not None:
        wandb.run.summary["best_epoch"] = summary["best_epoch"]
        wandb.run.summary["val/best_accuracy"] = summary["best_val_acc"]
        wandb.run.summary["val/best_balanced_accuracy"] = summary["best_val_bal_acc"]
        wandb.run.summary["val/best_auc_roc"] = summary["best_val_auc"]

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune classifier with frozen detector (detector-predicted lesions)."
    )
    parser.add_argument("--detector_checkpoint", required=True)
    parser.add_argument("--classifier_dir", default="./classifier_model")
    parser.add_argument("--detector_backbone", choices=["resnet101", "resnet152"], default="resnet101")

    parser.add_argument("--data_root", default="./data/cbis-ddsm")
    parser.add_argument("--train_csv", default="./meta/cbis-ddsm/mass_case_description_train_set.csv")
    parser.add_argument("--test_csv", default="./meta/cbis-ddsm/mass_case_description_test_set.csv")
    parser.add_argument("--val_split", type=float, default=0.15)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--class_weight", choices=["auto", "none"], default="auto")
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--early_stopping_patience", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--augment", action="store_true")

    parser.add_argument("--score_threshold", type=float, default=0.05)
    parser.add_argument("--iou_match_thr", type=float, default=0.30)
    parser.add_argument("--max_patches_per_image", type=int, default=32)
    parser.add_argument("--curriculum_epochs", type=int, default=2)
    parser.add_argument("--strict_score_threshold", type=float, default=0.20)
    parser.add_argument("--strict_iou_match_thr", type=float, default=0.50)
    parser.add_argument("--min_patches_per_batch", type=int, default=8)
    parser.add_argument("--max_gt_fallback_patches_per_batch", type=int, default=16)
    parser.add_argument("--max_gt_fallback_patches_per_image", type=int, default=8)
    parser.add_argument("--disable_gt_fallback", action="store_true")
    parser.add_argument("--gt_fallback_in_val", action="store_true")

    parser.add_argument("--patch_size", type=int, default=0,
                        help="Override classifier patch size; 0 means read from classifier_model/hyperparams.json")
    parser.add_argument("--output_dir", default="./models_cls_finetune")
    parser.add_argument("--sweep", action="store_true",
                        help="Run W&B Bayesian sweep over a discrete hyperparameter grid")
    parser.add_argument("--sweep_count", type=int, default=24,
                        help="Maximum number of Bayesian sweep runs")
    parser.add_argument("--sweep_trial", type=int, default=-1,
                        help="Deprecated (legacy preset sweeps); ignored for Bayesian sweeps")
    parser.add_argument("--wandb_project", default="hadamlab-classifier-finetune")
    parser.add_argument("--wandb_entity", default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.sweep:
        if wandb is None:
            raise RuntimeError(
                "Sweep requested but wandb is not installed. Install it with: pip install wandb"
            )
        if args.sweep_count < 1:
            raise ValueError(f"sweep_count must be >= 1, got {args.sweep_count}")

        wandb_target = {"project": args.wandb_project}
        wandb_entity = args.wandb_entity or os.getenv("WANDB_ENTITY")
        if wandb_entity:
            wandb_target["entity"] = wandb_entity

        sweep_config = _build_bayesian_grid_sweep_config(args)
        sweep_id = wandb.sweep(sweep_config, **wandb_target)

        base_config = {
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "class_weight": args.class_weight,
            "warmup_epochs": args.warmup_epochs,
            "early_stopping_patience": args.early_stopping_patience,
            "batch_size": args.batch_size,
            "val_split": args.val_split,
            "score_threshold": args.score_threshold,
            "iou_match_thr": args.iou_match_thr,
            "max_patches_per_image": args.max_patches_per_image,
            "curriculum_epochs": args.curriculum_epochs,
            "strict_score_threshold": args.strict_score_threshold,
            "strict_iou_match_thr": args.strict_iou_match_thr,
            "min_patches_per_batch": args.min_patches_per_batch,
            "max_gt_fallback_patches_per_batch": args.max_gt_fallback_patches_per_batch,
            "max_gt_fallback_patches_per_image": args.max_gt_fallback_patches_per_image,
            "augment": args.augment,
        }

        def sweep_run():
            wandb.init(**wandb_target, config=base_config)
            run_finetuning(base_args=args, run_config=dict(wandb.config), log_to_wandb=True)
            wandb.finish()

        wandb.agent(sweep_id, function=sweep_run, count=args.sweep_count)
    else:
        run_finetuning(base_args=args, run_config=None, log_to_wandb=False)


if __name__ == "__main__":
    main()
