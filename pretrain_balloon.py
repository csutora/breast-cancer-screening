"""
Pre-training script on the Balloon dataset.

Trains the same Mask R-CNN ResNet-101 architecture used for mammograms
on balloon instance segmentation, so the detector heads learn general
object detection before fine-tuning on CBIS-DDSM.

Example:
    python pretrain_balloon.py --data_root ./data/balloon --meta_root ./meta/balloon
"""

import argparse
import os
import time

import torch
import wandb
from torch.utils.data import DataLoader
from torchvision.models import ResNet101_Weights
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torchvision.ops import box_iou

from balloon_dataset import BalloonDataset, _collate_fn


def build_model(num_classes: int = 2, trainable_backbone_layers: int = 5) -> MaskRCNN:
    backbone = resnet_fpn_backbone(
        backbone_name="resnet101",
        weights=ResNet101_Weights.DEFAULT,
        trainable_layers=trainable_backbone_layers,
    )
    return MaskRCNN(backbone=backbone, num_classes=num_classes, min_size=1024, max_size=1024)


def train_one_epoch(model, optimizer, data_loader, device, epoch, warmup_scheduler=None):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if warmup_scheduler is not None:
            warmup_scheduler.step()

        running_loss += losses.item()

        if (batch_idx + 1) % 5 == 0:
            print(
                f"  [epoch {epoch}] batch {batch_idx + 1}/{len(data_loader)} "
                f"loss={losses.item():.4f} "
                f"({', '.join(f'{k}={v.item():.4f}' for k, v in loss_dict.items())})"
            )
            wandb.log({
                "batch/total_loss": losses.item(),
                **{f"batch/{k}": v.item() for k, v in loss_dict.items()},
                "batch": (epoch - 1) * len(data_loader) + batch_idx,
            })

    return running_loss / max(len(data_loader), 1)


@torch.no_grad()
def compute_validation_loss(model, data_loader, device):
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()

    total_loss, n = 0.0, 0
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()} for t in targets]
        losses = sum(model(images, targets).values())
        total_loss += losses.item()
        n += 1

    model.train()
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, data_loader, device, score_threshold=0.5):
    model.eval()
    box_ious, mask_ious = [], []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()} for t in targets]

        for det, tgt in zip(model(images), targets):
            gt_boxes, gt_masks = tgt["boxes"], tgt["masks"]
            if len(gt_boxes) == 0:
                continue
            keep = det["scores"] >= score_threshold
            pred_boxes, pred_masks = det["boxes"][keep], det["masks"][keep]
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


def train(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    train_dataset = BalloonDataset(
        image_dir=os.path.join(args.data_root, "train"),
        annotation_json=os.path.join(args.meta_root, "train.json"),
        target_size=(args.image_size, args.image_size),
    )
    val_dataset = BalloonDataset(
        image_dir=os.path.join(args.data_root, "test"),
        annotation_json=os.path.join(args.meta_root, "test.json"),
        target_size=(args.image_size, args.image_size),
    )

    print(f"[dataset] Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=_collate_fn,
    )

    model = build_model(num_classes=2, trainable_backbone_layers=args.trainable_backbone_layers)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min)

    os.makedirs(args.output_dir, exist_ok=True)
    best_box_iou = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        warmup_scheduler = LinearLR(optimizer, start_factor=0.001, total_iters=len(train_loader)) \
            if epoch == 1 else None

        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, warmup_scheduler)
        val_loss = compute_validation_loss(model, val_loader, device)
        box_iou_val, mask_iou_val = evaluate(model, val_loader, device, args.eval_score_threshold)
        lr_scheduler.step()

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"box_iou={box_iou_val:.4f} mask_iou={mask_iou_val:.4f} "
            f"time={time.time() - t0:.1f}s"
        )
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/box_iou": box_iou_val,
            "val/mask_iou": mask_iou_val,
            "lr": optimizer.param_groups[0]["lr"],
        })

        ckpt_path = os.path.join(args.output_dir, f"balloon_epoch{epoch:03d}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "box_iou": box_iou_val,
            "mask_iou": mask_iou_val,
        }, ckpt_path)

        if box_iou_val > best_box_iou:
            best_box_iou = box_iou_val
            best_path = os.path.join(args.output_dir, "balloon_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  -> Best model saved (box_iou={box_iou_val:.4f}): {best_path}")
            wandb.run.summary["best_box_iou"] = best_box_iou

    print(f"\nPre-training complete. Best box IoU: {best_box_iou:.4f}")
    print(f"Use --pretrain_weights {best_path} in train_detector.py to fine-tune.")


def main():
    parser = argparse.ArgumentParser(description="Pre-train Mask R-CNN on Balloon dataset")
    parser.add_argument("--data_root", default="./data/balloon")
    parser.add_argument("--meta_root", default="./meta/balloon")
    parser.add_argument("--output_dir", default="./models/pretrain_balloon")
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--trainable_backbone_layers", type=int, default=5)
    parser.add_argument("--eval_score_threshold", type=float, default=0.5)
    args = parser.parse_args()

    wandb.init(project="hadamlab-pretrain-balloon", config=vars(args))
    train(args)
    wandb.finish()


if __name__ == "__main__":
    main()
