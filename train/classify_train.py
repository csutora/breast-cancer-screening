"""
Standalone classifier training for mammogram mass pathology.

Normal run:
    python classify_train.py --data_root ./data/cbis-ddsm --epochs 30

Wandb sweep:
    python classify_train.py --sweep
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import wandb

from config import Representation
from classify_dataset import build_patch_dataloaders, preprocess_samples
from classify_model import MammoClassifier
from dataset import build_sample_index


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def compute_class_weights(loader: torch.utils.data.DataLoader) -> torch.Tensor:
    """Compute inverse-frequency class weights from the training set."""
    counts = [0, 0]
    for _, labels, _ in loader:
        for l in labels.tolist():
            counts[l] += 1
    total = sum(counts)
    weights = [total / (2 * max(c, 1)) for c in counts]
    print(f"[classify] Class counts: benign={counts[0]}, malignant={counts[1]}")
    print(f"[classify] Class weights: {weights}")
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, optimizer, loader, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels, _metas) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        if (batch_idx + 1) % 20 == 0:
            print(
                f"  [epoch {epoch}] batch {batch_idx + 1}/{len(loader)}  "
                f"loss={loss.item():.4f}  acc={correct / total:.4f}"
            )

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_true: list[int] = []
    all_prob: list[float] = []
    all_pred: list[int] = []

    for images, labels, _metas in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)

        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)

        all_true.extend(labels.cpu().tolist())
        all_prob.extend(probs.cpu().tolist())
        all_pred.extend(preds.cpu().tolist())

    n = len(all_true)
    avg_loss = total_loss / max(n, 1)

    tp = sum(1 for t, p in zip(all_true, all_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(all_true, all_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(all_true, all_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(all_true, all_pred) if t == 1 and p == 0)

    accuracy = (tp + tn) / max(n, 1)
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    auc = _roc_auc(all_true, all_prob)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auc_roc": auc,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "n": n,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config: dict, args, preprocessed: dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, _test_loader = build_patch_dataloaders(
        preprocessed_train=preprocessed["train"],
        preprocessed_val=preprocessed["val"],
        preprocessed_test=preprocessed["test"],
        patch_size=config["patch_size"],
        context_factor=config["context_factor"],
        label_source=config["label_source"],
        batch_size=config["batch_size"],
    )

    # Class-weighted loss
    if config.get("class_weight", "auto") == "auto":
        weights = compute_class_weights(train_loader).to(device)
    else:
        weights = None
    label_smoothing = config.get("label_smoothing", 0.1)
    criterion = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)

    model = MammoClassifier(
        backbone=config["backbone"],
        freeze_stages=config["freeze_stages"],
        head_hidden=config["head_hidden"],
        dropout=config["dropout"],
    )
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[classify] Parameters: {trainable:,} trainable / {total:,} total")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    # Warmup + cosine decay schedule
    warmup_epochs = config.get("warmup_epochs", 3)
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, config["epochs"] - warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    run_dir = os.path.join(args.output_dir, wandb.run.id)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "hyperparams.json"), "w") as f:
        json.dump(config, f, indent=2)

    best_auc = 0.0
    best_bal_acc = 0.0
    patience_counter = 0
    patience = config.get("early_stopping_patience", 5)

    for epoch in range(1, config["epochs"] + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, optimizer, train_loader, criterion, device, epoch,
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}/{config['epochs']}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  val_acc={val_metrics['accuracy']:.4f}  "
            f"val_auc={val_metrics['auc_roc']:.4f}  "
            f"val_sens={val_metrics['sensitivity']:.4f}  "
            f"val_spec={val_metrics['specificity']:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  time={elapsed:.1f}s"
        )

        bal_acc = (val_metrics["sensitivity"] + val_metrics["specificity"]) / 2
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_metrics["loss"],
            "val/accuracy": val_metrics["accuracy"],
            "val/auc_roc": val_metrics["auc_roc"],
            "val/sensitivity": val_metrics["sensitivity"],
            "val/specificity": val_metrics["specificity"],
            "val/balanced_accuracy": bal_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })

        if val_metrics["auc_roc"] > best_auc:
            best_auc = val_metrics["auc_roc"]
            wandb.run.summary["best_auc_roc"] = best_auc

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_path = os.path.join(run_dir, "classifier_best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  -> New best model (bal_acc={best_bal_acc:.4f}, AUC={val_metrics['auc_roc']:.4f}) saved to {best_path}")
            wandb.run.summary["best_balanced_accuracy"] = best_bal_acc

        # Early stopping on balanced accuracy
        if bal_acc >= best_bal_acc:
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  -> Early stopping at epoch {epoch} (patience={patience})")
                wandb.run.summary["early_stopped"] = True
                wandb.run.summary["early_stop_epoch"] = epoch
                break

    # Stochastic Weight Averaging
    print(f"\nTraining complete. Best AUC-ROC: {best_auc:.4f}")


# ---------------------------------------------------------------------------
# Sweep config
# ---------------------------------------------------------------------------

SWEEP_CONFIG = {
    "method": "bayes",
    "metric": {"name": "val/balanced_accuracy", "goal": "maximize"},
    "parameters": {
        "lr":              {"min": 1e-5, "max": 1e-3},
        "weight_decay":    {"values": [1e-4, 1e-3, 1e-2]},
        "dropout":         {"values": [0.3, 0.4, 0.5, 0.6]},
        "freeze_stages":   {"values": [4, 6, 7]},
        "head_hidden":     {"values": [256, 512, 1024]},
        "context_factor":  {"values": [1.5, 2.0, 2.5]},
        "label_source":    {"value": "pathology"},
        "backbone":        {"value": "convnext_small"},
        "batch_size":      {"values": [16, 32, 64, 128]},
        "patch_size":      {"value": 224},
        "epochs":          {"value": 30},
        "label_smoothing":  {"values": [0.0, 0.05, 0.1, 0.15]},
        "warmup_epochs":    {"values": [2, 3, 5]},
        "class_weight":    {"value": "auto"},
        "early_stopping_patience": {"value": 12},
    },
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train standalone mammogram classifier")
    parser.add_argument("--data_root", default="./data/cbis-ddsm")
    parser.add_argument("--train_csv", default="./meta/cbis-ddsm/mass_case_description_train_set.csv")
    parser.add_argument("--test_csv", default="./meta/cbis-ddsm/mass_case_description_test_set.csv")
    parser.add_argument("--output_dir", default="./models_cls")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers (0 = auto-detect from CPU count)")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--context_factor", type=float, default=1.5)
    parser.add_argument("--label_source", choices=["pathology", "assessment"], default="pathology")
    parser.add_argument("--backbone", default="convnext_small")
    parser.add_argument("--freeze_stages", type=int, default=4)
    parser.add_argument("--head_hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--class_weight", choices=["auto", "none"], default="auto")
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--early_stopping_patience", type=int, default=12)

    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep_count", type=int, default=30,
                        help="Number of sweep runs to execute")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    wandb_project = "hadamlab-classify"
    wandb_entity = os.getenv("WANDB_ENTITY")

    sweep_target = {"project": wandb_project}
    if wandb_entity:
        sweep_target["entity"] = wandb_entity

    # --- Preprocess once, reuse across all runs ---
    import random as _random
    num_workers = args.num_workers
    if num_workers == 0:
        num_workers = min(os.cpu_count() or 4, 32)

    print(f"Preprocessing with {num_workers} workers...")

    train_samples = build_sample_index(
        os.path.join(args.data_root, "train"), args.train_csv,
    )
    test_samples = build_sample_index(
        os.path.join(args.data_root, "test"), args.test_csv,
    )

    # Patient-level train/val split
    patients = sorted({s["patient_id"] for s in train_samples})
    _random.Random(42).shuffle(patients)
    val_patients = set(patients[:int(len(patients) * 0.15)])
    train_split = [s for s in train_samples if s["patient_id"] not in val_patients]
    val_split = [s for s in train_samples if s["patient_id"] in val_patients]

    preprocessed = {
        "train": preprocess_samples(train_split, num_workers=num_workers),
        "val": preprocess_samples(val_split, num_workers=num_workers),
        "test": preprocess_samples(test_samples, num_workers=num_workers),
    }
    print(f"Preprocessing complete. Train: {len(preprocessed['train'])}, "
          f"Val: {len(preprocessed['val'])}, Test: {len(preprocessed['test'])}")

    if args.sweep:
        def sweep_run():
            wandb.init()
            train(dict(wandb.config), args, preprocessed)

        sweep_id = wandb.sweep(SWEEP_CONFIG, **sweep_target)
        wandb.agent(sweep_id, function=sweep_run, count=args.sweep_count)
    else:
        config = {
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "patch_size": args.patch_size,
            "context_factor": args.context_factor,
            "label_source": args.label_source,
            "backbone": args.backbone,
            "freeze_stages": args.freeze_stages,
            "head_hidden": args.head_hidden,
            "dropout": args.dropout,
            "class_weight": args.class_weight,
            "label_smoothing": args.label_smoothing,
            "warmup_epochs": args.warmup_epochs,
            "early_stopping_patience": args.early_stopping_patience,
        }
        wandb.init(**sweep_target, config=config)
        train(config, args, preprocessed)
        wandb.finish()


if __name__ == "__main__":
    main()
