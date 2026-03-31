"""
Training script for CBIS-DDSM mammogram mass detection with Mask R-CNN.

Normal run:
    python train.py --data_root ./data/cbis-ddsm --epochs 10

Wandb sweep:
    wandb sweep sweep.yaml
    wandb agent <sweep_id>
"""

import argparse
import json
import os
import random
import time

import torch
import wandb
from torch.utils.data import DataLoader

from config import DatasetConfig, PreprocessConfig, Representation
from dataset import build_sample_index, CBISDDSMDataset, _collate_fn
from model import MammoModel


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [
            {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in t.items()}
            for t in targets
        ]

        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        n_batches += 1

        if (batch_idx + 1) % 10 == 0:
            print(
                f"  [epoch {epoch}] batch {batch_idx + 1}/{len(data_loader)}  "
                f"loss={losses.item():.4f}  "
                f"({', '.join(f'{k}={v.item():.4f}' for k, v in loss_dict.items())})"
            )
            wandb.log({
                "batch/total_loss": losses.item(),
                **{f"batch/{k}": v.item() for k, v in loss_dict.items()},
                "batch": (epoch - 1) * len(data_loader) + batch_idx,
            })

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss


@torch.no_grad()
def compute_validation_loss(model, data_loader, device):
    """Compute validation loss with targets (requires train mode for torchvision detectors)."""
    was_training = model.training
    model.train()

    # Keep detector in train mode to obtain losses, but freeze BatchNorm stats on val data.
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
            {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in t.items()}
            for t in targets
        ]

        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())
        total_loss += losses.item()
        n_batches += 1

    for module, was_module_training in bn_modules:
        module.train(was_module_training)

    model.train(was_training)
    avg_loss = total_loss / max(n_batches, 1)
    return total_loss, avg_loss


@torch.no_grad()
def evaluate(model, data_loader, device):
    """
    Simple dev-split metrics:
      - Mean box IoU  (predicted vs GT, matched by highest IoU)
      - Mean mask IoU (predicted vs GT, same matching)
      - Classifier accuracy (pathology: benign/malignant)
    """
    from torchvision.ops import box_iou

    model.eval()

    box_ious, mask_ious = [], []
    cls_correct, cls_total = 0, 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()} for t in targets]

        detections = model(images)

        for det, tgt in zip(detections, targets):
            gt_boxes = tgt["boxes"]
            pred_boxes = det["boxes"]
            pred_masks = det["masks"]

            if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                continue

            iou_matrix = box_iou(pred_boxes, gt_boxes)
            matched_iou, matched_idx = iou_matrix.max(dim=1)

            box_ious.extend(matched_iou.cpu().tolist())

            gt_masks = tgt["masks"]
            for n, m in enumerate(matched_idx):
                pred_mask = (pred_masks[n, 0] > 0.5).float()
                gt_mask = gt_masks[m].float()
                intersection = (pred_mask * gt_mask).sum()
                union = (pred_mask + gt_mask).clamp(0, 1).sum()
                if union > 0:
                    mask_ious.append((intersection / union).item())

            if "pathology_logits" in det:
                gt_labels = torch.tensor(
                    [0 if p == "BENIGN" else 1 for p in tgt["pathology"]],
                    device=device
                )
                for n, (iou_val, m) in enumerate(zip(matched_iou, matched_idx)):
                    if iou_val >= 0.5:
                        pred_cls = det["pathology_logits"][n].argmax()
                        cls_correct += int(pred_cls == gt_labels[m])
                        cls_total += 1

    mean_box_iou  = sum(box_ious)  / max(len(box_ious), 1)
    mean_mask_iou = sum(mask_ious) / max(len(mask_ious), 1)
    cls_accuracy  = cls_correct / max(cls_total, 1)

    print(f"  box_iou={mean_box_iou:.4f}  mask_iou={mean_mask_iou:.4f}  "
          f"cls_acc={cls_accuracy:.4f} ({cls_correct}/{cls_total})")

    return mean_box_iou, mean_mask_iou, cls_accuracy


# ---------------------------------------------------------------------------
# Core training logic (called by main or sweep agent)
# ---------------------------------------------------------------------------

def train(config, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("Running on CPU — consider using a GPU for faster training")

    # --- Dataset ---
    cfg = DatasetConfig(
        data_root=args.data_root,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        preprocess=PreprocessConfig(
            target_size=(1024, 1024),
            representation=Representation.MMS_PSEUDO_COLOR,
            remove_pectoral=False,
        ),
        batch_size=config["batch_size"],
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Patient-level train/val split from the official train set
    all_train_samples = build_sample_index(
        os.path.join(cfg.data_root, "train"), cfg.train_csv
    )
    patients = sorted({s["patient_id"] for s in all_train_samples})
    random.Random(42).shuffle(patients)
    val_patients = set(patients[:int(len(patients) * 0.15)])

    train_samples = [s for s in all_train_samples if s["patient_id"] not in val_patients]
    val_samples   = [s for s in all_train_samples if s["patient_id"] in val_patients]

    print(f"[dataset] Train: {len(train_samples)} images  |  "
          f"Val: {len(val_samples)} images  |  "
          f"Val patients: {len(val_patients)}")

    train_loader = DataLoader(
        CBISDDSMDataset(train_samples, cfg, train=True),
        batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        CBISDDSMDataset(val_samples, cfg, train=False),
        batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        collate_fn=_collate_fn,
    )

    # --- Model ---
    model = MammoModel(num_seg_classes=2, num_classifier_classes=2)
    model.to(device)

    # --- Optimizer + scheduler ---
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=config["lr"], weight_decay=config["weight_decay"]
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config["lr_step_size"], gamma=config["lr_gamma"]
    )

    # --- Per-run output folder: models/<run_id>/ ---
    run_dir = os.path.join(args.output_dir, wandb.run.id)
    os.makedirs(run_dir, exist_ok=True)

    # --- Save hyperparams ---
    with open(os.path.join(run_dir, "hyperparams.json"), "w") as f:
        json.dump(config, f, indent=2)

    # --- Training loop ---
    best_box_iou = 0.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    early_stopping_patience = max(1, int(args.early_stopping_patience))

    for epoch in range(1, config["epochs"] + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        val_total_loss, val_loss = compute_validation_loss(model, val_loader, device)
        box_iou, mask_iou, cls_acc = evaluate(model, val_loader, device)
        lr_scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}/{config['epochs']}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"box_iou={box_iou:.4f}  mask_iou={mask_iou:.4f}  cls_acc={cls_acc:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  time={elapsed:.1f}s"
        )

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/loss_total": val_total_loss,
            "val/box_iou": box_iou,
            "val/mask_iou": mask_iou,
            "val/cls_acc": cls_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })

        # Save checkpoint locally — not uploaded to wandb
        ckpt_path = os.path.join(run_dir, f"maskrcnn_epoch{epoch:03d}.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "box_iou": box_iou,
                "mask_iou": mask_iou,
                "cls_acc": cls_acc,
            },
            ckpt_path,
        )

        if box_iou > best_box_iou:
            best_box_iou = box_iou
            best_path = os.path.join(run_dir, "maskrcnn_best.pth")
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Mask R-CNN on CBIS-DDSM")
    parser.add_argument("--data_root", default="./data/cbis-ddsm")
    parser.add_argument("--train_csv",
                        default="./meta/cbis-ddsm/mass_case_description_train_set.csv")
    parser.add_argument("--test_csv",
                        default="./meta/cbis-ddsm/mass_case_description_test_set.csv")
    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--lr",           type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size",   type=int,   default=2)
    parser.add_argument("--lr_step_size", type=int,   default=3)
    parser.add_argument("--lr_gamma",     type=float, default=0.5)
    parser.add_argument("--num_workers",  type=int,   default=8)
    parser.add_argument("--output_dir",   default="./models")
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    parser.add_argument("--sweep",        action="store_true",
                        help="Run as a wandb sweep agent")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    wandb_project = "hadamlab"
    wandb_entity = os.getenv("WANDB_ENTITY")

    sweep_target = {"project": wandb_project}
    if wandb_entity:
        sweep_target["entity"] = wandb_entity

    if args.sweep:
        # Sweep agent: wandb injects config via wandb.config
        def sweep_run():
            wandb.init()
            train(dict(wandb.config), args)

        sweep_id = wandb.sweep(SWEEP_CONFIG, **sweep_target)
        wandb.agent(sweep_id, function=sweep_run)
    else:
        # Normal single run
        config = {
            "epochs":       args.epochs,
            "lr":           args.lr,
            "weight_decay": args.weight_decay,
            "batch_size":   args.batch_size,
            "lr_step_size": args.lr_step_size,
            "lr_gamma":     args.lr_gamma,
            "early_stopping_patience": args.early_stopping_patience,
        }
        wandb.init(**sweep_target, config=config)
        train(config, args)
        wandb.finish()


# ---------------------------------------------------------------------------
# Sweep config
# ---------------------------------------------------------------------------

SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val/box_iou", "goal": "maximize"},
    "parameters": {
        "lr":           {"min": 1e-5, "max": 1e-3},
        "weight_decay": {"values": [1e-5, 1e-4, 1e-3]},
        "batch_size":   {"values": [2, 4]},
        "lr_step_size": {"values": [2, 3, 5]},
        "lr_gamma":     {"values": [0.1, 0.5]},
        "epochs":       {"value": 10},
    },
}


if __name__ == "__main__":
    main()
