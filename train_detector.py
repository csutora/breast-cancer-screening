"""
Standalone detector training script for CBIS-DDSM mammograms.

This script trains only the detection/segmentation model (Mask R-CNN) with a
larger ResNet-101 FPN backbone.

Example:
    python train_detector.py --data_root ./data/cbis-ddsm --epochs 20
"""

import argparse
import json
import os
import random
import time

import torch
import wandb
from torch.utils.data import DataLoader
from torchvision.models import ResNet101_Weights
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import box_iou
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

from config import DatasetConfig, PreprocessConfig, Representation
from dataset import CBISDDSMDataset, _collate_fn, build_sample_index


def build_detector_model(num_classes: int = 2, trainable_backbone_layers: int = 5) -> MaskRCNN:
    """
    Build a Mask R-CNN detector with a ResNet-101 FPN backbone.

    num_classes includes background (2 = background + mass).
    """
    backbone = resnet_fpn_backbone(
        backbone_name="resnet101",
        weights=ResNet101_Weights.DEFAULT,
        trainable_layers=trainable_backbone_layers,
    )
    model = MaskRCNN(backbone=backbone, num_classes=num_classes, min_size=1024, max_size=1024)
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch, warmup_scheduler=None):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [
            {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
            for t in targets
        ]

        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # Step the warmup scheduler per batch if it exists
        if warmup_scheduler is not None:
            warmup_scheduler.step()

        running_loss += losses.item()

        if (batch_idx + 1) % 10 == 0:
            print(
                f"  [epoch {epoch}] batch {batch_idx + 1}/{len(data_loader)} "
                f"loss={losses.item():.4f} "
                f"({', '.join(f'{k}={v.item():.4f}' for k, v in loss_dict.items())})"
            )
            wandb.log(
                {
                    "batch/total_loss": losses.item(),
                    **{f"batch/{k}": v.item() for k, v in loss_dict.items()},
                    "batch": (epoch - 1) * len(data_loader) + batch_idx,
                }
            )

    return running_loss / max(len(data_loader), 1)


@torch.no_grad()
def compute_validation_loss(model, data_loader, device):
    """Compute validation loss while freezing BatchNorm running stats."""
    was_training = model.training
    model.train()

    bn_modules = []
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            bn_modules.append((module, module.training))
            module.eval()

    total_loss = 0.0
    n_batches = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [
            {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
            for t in targets
        ]

        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())
        total_loss += losses.item()
        n_batches += 1

    for module, was_module_training in bn_modules:
        module.train(was_module_training)

    model.train(was_training)
    return total_loss, total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_detector(model, data_loader, device, score_threshold: float = 0.5):
    """
    Evaluate detector quality on the validation split.

    Returns:
        mean_box_iou, mean_mask_iou
    """
    model.eval()
    box_ious = []
    mask_ious = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [
            {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
            for t in targets
        ]

        detections = model(images)

        for det, tgt in zip(detections, targets):
            gt_boxes = tgt["boxes"]
            gt_masks = tgt["masks"]

            if len(gt_boxes) == 0:
                continue

            keep = det["scores"] >= score_threshold
            pred_boxes = det["boxes"][keep]
            pred_masks = det["masks"][keep]

            if len(pred_boxes) == 0:
                continue

            iou_matrix = box_iou(pred_boxes, gt_boxes)
            matched_iou, matched_idx = iou_matrix.max(dim=1)
            box_ious.extend(matched_iou.cpu().tolist())

            for n, m in enumerate(matched_idx):
                pred_mask = (pred_masks[n, 0] > 0.5).float()
                gt_mask = gt_masks[m].float()
                intersection = (pred_mask * gt_mask).sum()
                union = (pred_mask + gt_mask).clamp(0, 1).sum()
                if union > 0:
                    mask_ious.append((intersection / union).item())

    mean_box_iou = sum(box_ious) / max(len(box_ious), 1)
    mean_mask_iou = sum(mask_ious) / max(len(mask_ious), 1)

    print(f"  box_iou={mean_box_iou:.4f}  mask_iou={mean_mask_iou:.4f}")
    return mean_box_iou, mean_mask_iou


def train(config, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("Running on CPU — consider using a GPU for faster training")

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
        batch_size=config["batch_size"],
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        augment=args.augment,
    )

    all_train_samples = build_sample_index(os.path.join(cfg.data_root, "train"), cfg.train_csv)
    patients = sorted({s["patient_id"] for s in all_train_samples})
    random.Random(42).shuffle(patients)
    val_patients = set(patients[: int(len(patients) * args.val_split)])

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

    model = build_detector_model(
        num_classes=2,
        trainable_backbone_layers=int(config.get("trainable_backbone_layers", args.trainable_backbone_layers)),
    )

    if args.pretrain_weights:
        ckpt = torch.load(args.pretrain_weights, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pre-trained weights from {args.pretrain_weights}")
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=config["lr"], weight_decay=config["weight_decay"])
    
    # Cosine annealing over epochs; keep a small floor learning rate.
    lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["epochs"],
        eta_min=float(config.get("eta_min", args.eta_min)),
    )

    run_dir = os.path.join(args.output_dir, wandb.run.id)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "hyperparams.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    best_box_iou = 0.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    early_stopping_patience = max(1, int(args.early_stopping_patience))

    for epoch in range(1, config["epochs"] + 1):
        t0 = time.time()
        
        # Warmup only during the first epoch
        if epoch == 1:
            warmup_scheduler = LinearLR(optimizer, start_factor=0.001, total_iters=len(train_loader))
        else:
            warmup_scheduler = None

        # Pass the warmup_scheduler to the modified train_one_epoch
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, warmup_scheduler)
        val_total_loss, val_loss = compute_validation_loss(model, val_loader, device)
        
        # Kept original IoU evaluation
        box_iou, mask_iou = evaluate_detector(
            model,
            val_loader,
            device,
            score_threshold=args.eval_score_threshold,
        )
        
        # Step the epoch-level cosine scheduler
        lr_scheduler.step()
        
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}/{config['epochs']} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"box_iou={box_iou:.4f} mask_iou={mask_iou:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} time={elapsed:.1f}s"
        )

        wandb.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/loss_total": val_total_loss,
                "val/box_iou": box_iou,
                "val/mask_iou": mask_iou,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        ckpt_path = os.path.join(run_dir, f"detector_resnet101_epoch{epoch:03d}.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "box_iou": box_iou,
                "mask_iou": mask_iou,
            },
            ckpt_path,
        )

        if box_iou > best_box_iou:
            best_box_iou = box_iou
            best_path = os.path.join(run_dir, "detector_resnet101_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  -> New best model saved to {best_path} (box_iou={box_iou:.4f})")
            wandb.run.summary["best_box_iou"] = best_box_iou

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            wandb.run.summary["best_val_loss"] = best_val_loss
        else:
            epochs_without_improvement += 1
            print(
                f"  -> Early stopping check: no val_loss improvement "
                f"for {epochs_without_improvement}/{early_stopping_patience} epoch(s)"
            )
            if epochs_without_improvement >= early_stopping_patience:
                print(
                    f"  -> Early stopping triggered at epoch {epoch} "
                    f"(best_val_loss={best_val_loss:.4f}, patience={early_stopping_patience})"
                )
                wandb.run.summary["early_stopped"] = True
                wandb.run.summary["early_stop_epoch"] = epoch
                break

    print(f"\nTraining complete. Best box IoU: {best_box_iou:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train detector-only Mask R-CNN (ResNet-101 backbone)")
    parser.add_argument("--data_root", default="./data/cbis-ddsm")
    parser.add_argument(
        "--train_csv",
        default="./meta/cbis-ddsm/mass_case_description_train_set.csv",
    )
    parser.add_argument(
        "--test_csv",
        default="./meta/cbis-ddsm/mass_case_description_test_set.csv",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_dir", default="./models")
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--eval_score_threshold", type=float, default=0.5)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--trainable_backbone_layers", type=int, default=5)
    parser.add_argument("--pretrain_weights", default=None,
                        help="Path to pre-trained checkpoint (e.g. from pretrain_balloon.py)")
    parser.add_argument("--sweep", action="store_true", help="Run a small wandb sweep")
    parser.add_argument("--sweep_count", type=int, default=8, help="Number of sweep runs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    wandb_project = "hadamlab-detector-only"
    wandb_entity = os.getenv("WANDB_ENTITY")

    wandb_target = {"project": wandb_project}
    if wandb_entity:
        wandb_target["entity"] = wandb_entity

    config = {
        "epochs": args.epochs,
        "lr": args.lr,
        "eta_min": args.eta_min,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "val_split": args.val_split,
        "eval_score_threshold": args.eval_score_threshold,
        "trainable_backbone_layers": args.trainable_backbone_layers,
        "early_stopping_patience": args.early_stopping_patience,
        "augment": args.augment,
    }

    if args.sweep:
        sweep_config = {
            "method": "bayes",
            "metric": {"name": "val/box_iou", "goal": "maximize"},
            "parameters": {
                "lr": {"min": 1e-5, "max": 1e-3},
                "eta_min": {"values": [1e-7, 1e-6, 1e-5]},
                "weight_decay": {"values": [1e-5, 1e-4, 1e-3]},
                "batch_size": {"values": [16, 32]},
                "trainable_backbone_layers": {"values": [4, 5]},
                "epochs": {"value": args.epochs},
            },
        }

        sweep_id = wandb.sweep(sweep_config, **wandb_target)

        def sweep_run():
            wandb.init(**wandb_target)
            train(dict(wandb.config), args)
            wandb.finish()

        wandb.agent(sweep_id, function=sweep_run, count=args.sweep_count)
    else:
        wandb.init(**wandb_target, config=config)
        train(config, args)
        wandb.finish()


if __name__ == "__main__":
    main()
