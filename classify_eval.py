"""
Evaluate a trained standalone mammogram classifier.

Produces per-density and per-size breakdowns alongside standard metrics.

Usage:
    python classify_eval.py --checkpoint ./models_cls/<run_id>/classifier_best.pth
    python classify_eval.py --checkpoint ./models_cls/<run_id>/classifier_best.pth --split train
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import Representation
from dataset import build_sample_index
from classify_dataset import PatchDataset, preprocess_samples
from classify_model import MammoClassifier


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


def _classification_metrics(y_true: list[int], y_pred: list[int], y_prob: list[float]) -> dict:
    n = len(y_true)
    if n == 0:
        return {"n": 0, "accuracy": float("nan"), "sensitivity": float("nan"),
                "specificity": float("nan"), "f1": float("nan"),
                "auc_roc": float("nan"), "balanced_accuracy": float("nan")}

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    sens = _safe_div(tp, tp + fn)
    spec = _safe_div(tn, tn + fp)

    return {
        "n": n,
        "accuracy": _safe_div(tp + tn, n),
        "sensitivity": sens,
        "specificity": spec,
        "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
        "balanced_accuracy": (sens + spec) / 2 if not (math.isnan(sens) or math.isnan(spec)) else float("nan"),
        "auc_roc": _roc_auc(y_true, y_prob),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def _tta_predict(model, images, device):
    """Test-time augmentation: average predictions over 4 flip variants."""
    variants = [
        images,
        torch.flip(images, dims=[-1]),       # hflip
        torch.flip(images, dims=[-2]),       # vflip
        torch.flip(images, dims=[-2, -1]),   # both
    ]
    batch = torch.cat(variants, dim=0).to(device)
    logits = model(batch)
    probs = torch.softmax(logits, dim=1)[:, 1]
    # Reshape to (4, B) and average across augmentations
    probs = probs.view(4, images.size(0)).mean(dim=0)
    return probs


def _find_threshold_for_sensitivity(
    y_true: list[int], y_prob: list[float], target_sens: float = 0.9,
) -> tuple[float, dict]:
    """Find the highest threshold that achieves at least target_sens sensitivity."""
    y_true_np = np.asarray(y_true)
    y_prob_np = np.asarray(y_prob)

    best_threshold = 0.5
    best_spec = 0.0

    for threshold in np.arange(0.01, 0.99, 0.01):
        preds = (y_prob_np >= threshold).astype(int)
        tp = int(((preds == 1) & (y_true_np == 1)).sum())
        tn = int(((preds == 0) & (y_true_np == 0)).sum())
        fp = int(((preds == 1) & (y_true_np == 0)).sum())
        fn = int(((preds == 0) & (y_true_np == 1)).sum())
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        if sens >= target_sens and spec > best_spec:
            best_threshold = threshold
            best_spec = spec

    # Compute final metrics at chosen threshold
    preds = (y_prob_np >= best_threshold).astype(int).tolist()
    metrics = _classification_metrics(y_true, preds, y_prob)
    metrics["threshold"] = float(best_threshold)
    return best_threshold, metrics


@torch.no_grad()
def run_evaluation(
    model: MammoClassifier,
    loader: DataLoader,
    device: torch.device,
    tta: bool = False,
    threshold: float = 0.5,
) -> dict:
    model.eval()

    all_true: list[int] = []
    all_pred: list[int] = []
    all_prob: list[float] = []
    all_density: list[int] = []
    all_area: list[int] = []

    for images, labels, metas in loader:
        if tta:
            probs = _tta_predict(model, images, device)
        else:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]

        preds = (probs >= threshold).long()

        all_true.extend(labels.tolist())
        all_pred.extend(preds.cpu().tolist())
        all_prob.extend(probs.cpu().tolist())
        all_density.extend([m["density"] for m in metas])
        all_area.extend([m["lesion_area_px"] for m in metas])

    # Overall metrics
    overall = _classification_metrics(all_true, all_pred, all_prob)

    # Per-density breakdown
    density_metrics = {}
    for d in sorted(set(all_density)):
        mask = [i for i, dd in enumerate(all_density) if dd == d]
        y_t = [all_true[i] for i in mask]
        y_p = [all_pred[i] for i in mask]
        y_s = [all_prob[i] for i in mask]
        density_metrics[str(d)] = _classification_metrics(y_t, y_p, y_s)

    # Per-size breakdown (terciles)
    areas = np.asarray(all_area, dtype=np.float64)
    if len(areas) > 0:
        p33, p66 = np.percentile(areas, [33, 66])
        size_buckets = {
            "small": [i for i, a in enumerate(all_area) if a <= p33],
            "medium": [i for i, a in enumerate(all_area) if p33 < a <= p66],
            "large": [i for i, a in enumerate(all_area) if a > p66],
        }
        size_metrics = {}
        for name, idxs in size_buckets.items():
            y_t = [all_true[i] for i in idxs]
            y_p = [all_pred[i] for i in idxs]
            y_s = [all_prob[i] for i in idxs]
            size_metrics[name] = _classification_metrics(y_t, y_p, y_s)
        size_metrics["thresholds_px"] = {"p33": float(p33), "p66": float(p66)}
    else:
        size_metrics = {}

    return {
        "overall": overall,
        "per_density": density_metrics,
        "per_size": size_metrics,
    }


def _to_json_safe(obj):
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def main():
    parser = argparse.ArgumentParser(description="Evaluate standalone mammogram classifier")
    parser.add_argument("--checkpoint", required=True, help="Path to classifier_best.pth")
    parser.add_argument("--data_root", default="./data/cbis-ddsm")
    parser.add_argument("--train_csv", default="./meta/cbis-ddsm/mass_case_description_train_set.csv")
    parser.add_argument("--test_csv", default="./meta/cbis-ddsm/mass_case_description_test_set.csv")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation (flip variants)")
    parser.add_argument("--threshold", type=float, default=0.42,
                        help="Classification threshold (default 0.42, tuned for ~0.85 sens)")
    parser.add_argument("--target_sens", type=float, default=None,
                        help="Auto-tune threshold to achieve this sensitivity (e.g. 0.9)")
    parser.add_argument("--output_json", default="./outputs/classify_eval.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config from checkpoint
    ckpt_path = Path(args.checkpoint)
    ckpt_dir = ckpt_path.parent
    config_path = ckpt_dir / "hyperparams.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}

    # Load model
    model = MammoClassifier(
        backbone=config.get("backbone", "convnext_small"),
        freeze_stages=config.get("freeze_stages", 4),
        head_hidden=config.get("head_hidden", 256),
        dropout=config.get("dropout", 0.2),
    )
    state_dict = torch.load(ckpt_path, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    # Build dataset
    split_csv = args.train_csv if args.split == "train" else args.test_csv
    samples = build_sample_index(
        Path(args.data_root) / args.split, split_csv,
    )

    preprocessed = preprocess_samples(samples, num_workers=args.num_workers)

    dataset = PatchDataset(
        preprocessed,
        patch_size=config.get("patch_size", 224),
        context_factor=config.get("context_factor", 1.5),
        label_source=config.get("label_source", "pathology"),
        augment=False,
    )

    def collate(batch):
        images, labels, metas = zip(*batch)
        return torch.stack(images), torch.tensor(labels, dtype=torch.long), list(metas)

    loader = DataLoader(
        dataset, batch_size=32, shuffle=False,
        num_workers=min(os.cpu_count() or 4, 8), pin_memory=True,
        collate_fn=collate, persistent_workers=True, prefetch_factor=4,
    )

    # If target_sens is set, first collect all probabilities to find threshold
    if args.target_sens is not None:
        # Run once to collect probabilities
        raw_metrics = run_evaluation(model, loader, device, tta=args.tta, threshold=0.5)
        # Collect all probs and labels from the raw run
        all_true_for_thresh = []
        all_prob_for_thresh = []
        for images, labels, metas in loader:
            if args.tta:
                probs = _tta_predict(model, images, device)
            else:
                images_dev = images.to(device)
                logits = model(images_dev)
                probs = torch.softmax(logits, dim=1)[:, 1]
            all_true_for_thresh.extend(labels.tolist())
            all_prob_for_thresh.extend(probs.cpu().tolist())

        threshold, thresh_metrics = _find_threshold_for_sensitivity(
            all_true_for_thresh, all_prob_for_thresh, target_sens=args.target_sens,
        )
        print(f"[classify_eval] Auto-tuned threshold={threshold:.2f} for target sensitivity>={args.target_sens}")
        print(f"[classify_eval]   -> sens={thresh_metrics['sensitivity']:.4f} spec={thresh_metrics['specificity']:.4f}")
    else:
        threshold = args.threshold

    metrics = run_evaluation(model, loader, device, tta=args.tta, threshold=threshold)

    results = {
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "num_patches": len(dataset),
        "config": config,
        "metrics": metrics,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(_to_json_safe(results), f, indent=2)

    # Print summary
    ov = metrics["overall"]
    print(f"\n[classify_eval] checkpoint: {ckpt_path}")
    print(f"[classify_eval] split: {args.split} | patches: {len(dataset)} | threshold: {threshold:.2f}")
    print(f"[classify_eval] AUC-ROC:      {ov['auc_roc']:.4f}")
    print(f"[classify_eval] Accuracy:     {ov['accuracy']:.4f}")
    print(f"[classify_eval] Sensitivity:  {ov['sensitivity']:.4f}")
    print(f"[classify_eval] Specificity:  {ov['specificity']:.4f}")
    print(f"[classify_eval] F1:           {ov['f1']:.4f}")
    print(f"[classify_eval] Balanced acc: {ov['balanced_accuracy']:.4f}")

    print(f"\n  Per density:")
    for d, dm in sorted(metrics["per_density"].items()):
        print(f"    Density {d}: acc={dm['accuracy']:.3f} sens={dm['sensitivity']:.3f} spec={dm['specificity']:.3f} n={dm['n']}")

    if metrics["per_size"]:
        print(f"\n  Per size:")
        for s in ["small", "medium", "large"]:
            sm = metrics["per_size"].get(s, {})
            if sm:
                print(f"    {s:>6}: acc={sm['accuracy']:.3f} sens={sm['sensitivity']:.3f} spec={sm['specificity']:.3f} n={sm['n']}")

    print(f"\n[classify_eval] JSON saved to: {out_path}")


if __name__ == "__main__":
    main()
