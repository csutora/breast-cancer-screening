"""
Evaluate trained mammogram models on CBIS-DDSM.

Metrics:
- Detection: mAP@50, mAP@50:95, FROC curve (+ mean sensitivity at common FP/image points)
- Segmentation: Dice, IoU
- Classification (pathology): AUC-ROC, sensitivity, specificity, F1
- Density-aware breakdown: detection sensitivity and classification accuracy per density

Usage:
    python eval.py --model_dir ./models
    python eval.py --checkpoint ./models/<run_id>/maskrcnn_best.pth
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from config import DatasetConfig, PreprocessConfig, Representation
from dataset import build_sample_index, CBISDDSMDataset, _collate_fn
from model import MammoModel


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def _to_float(value: float) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return float(value)


def _pathology_to_label(pathology: str) -> int | None:
    raw = str(pathology).strip().upper()
    if "MALIGNANT" in raw:
        return 1
    if "BENIGN" in raw:
        return 0
    return None


def _resolve_checkpoint(model_dir: Path, checkpoint: str | None) -> Path:
    def _latest_from_dir(base: Path) -> Path | None:
        best = sorted(base.glob("**/maskrcnn_best.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
        if best:
            return best[0]
        epochs = sorted(base.glob("**/maskrcnn_epoch*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
        if epochs:
            return epochs[0]
        return None

    if checkpoint:
        cp = Path(checkpoint)
        if not cp.is_absolute():
            cp = (Path.cwd() / cp).resolve()
        if cp.is_dir():
            found = _latest_from_dir(cp)
            if found is None:
                raise FileNotFoundError(f"No checkpoint files found in directory: {cp}")
            return found
        if not cp.exists():
            raise FileNotFoundError(f"Checkpoint not found: {cp}")
        return cp

    found = _latest_from_dir(model_dir)
    if found is None:
        raise FileNotFoundError(
            f"No checkpoints found under {model_dir}. Expected maskrcnn_best.pth or maskrcnn_epoch*.pth"
        )
    return found


def _load_model(checkpoint_path: Path, device: torch.device) -> MammoModel:
    model = MammoModel(num_seg_classes=2, num_classifier_classes=2)
    blob = torch.load(checkpoint_path, map_location=device)
    if isinstance(blob, dict) and "model_state_dict" in blob:
        state_dict = blob["model_state_dict"]
    else:
        state_dict = blob
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def _greedy_match(pred_boxes: torch.Tensor, pred_scores: torch.Tensor, gt_boxes: torch.Tensor, iou_thr: float) -> list[tuple[int, int, float]]:
    """
    Greedy one-to-one matching by descending prediction confidence.
    Returns list of (pred_idx, gt_idx, iou).
    """
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return []

    order = torch.argsort(pred_scores, descending=True)
    used_gt: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    iou_mat = box_iou(pred_boxes, gt_boxes)
    for pred_idx_t in order:
        pred_idx = int(pred_idx_t.item())
        ious = iou_mat[pred_idx]
        best_iou = -1.0
        best_gt = -1
        for gt_idx in range(ious.shape[0]):
            if gt_idx in used_gt:
                continue
            iou_val = float(ious[gt_idx].item())
            if iou_val > best_iou:
                best_iou = iou_val
                best_gt = gt_idx
        if best_gt >= 0 and best_iou >= iou_thr:
            used_gt.add(best_gt)
            matches.append((pred_idx, best_gt, best_iou))

    return matches


def _average_precision(recalls: np.ndarray, precisions: np.ndarray) -> float:
    if recalls.size == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _compute_ap(records: list[dict], iou_thr: float) -> float:
    total_gt = int(sum(r["gt_boxes"].shape[0] for r in records))
    if total_gt == 0:
        return float("nan")

    preds: list[tuple[float, int, int]] = []
    for image_idx, rec in enumerate(records):
        scores = rec["pred_scores"]
        for pred_idx in range(scores.shape[0]):
            preds.append((float(scores[pred_idx].item()), image_idx, pred_idx))

    if not preds:
        return 0.0

    preds.sort(key=lambda x: x[0], reverse=True)
    used_gt: dict[int, set[int]] = {i: set() for i in range(len(records))}

    tps: list[int] = []
    fps: list[int] = []

    for _, image_idx, pred_idx in preds:
        rec = records[image_idx]
        gt_boxes = rec["gt_boxes"]
        if gt_boxes.numel() == 0:
            tps.append(0)
            fps.append(1)
            continue

        pred_box = rec["pred_boxes"][pred_idx].unsqueeze(0)
        ious = box_iou(pred_box, gt_boxes).squeeze(0)

        best_iou = -1.0
        best_gt = -1
        for gt_idx in range(ious.shape[0]):
            if gt_idx in used_gt[image_idx]:
                continue
            iou_val = float(ious[gt_idx].item())
            if iou_val > best_iou:
                best_iou = iou_val
                best_gt = gt_idx

        if best_gt >= 0 and best_iou >= iou_thr:
            used_gt[image_idx].add(best_gt)
            tps.append(1)
            fps.append(0)
        else:
            tps.append(0)
            fps.append(1)

    tps_cum = np.cumsum(np.asarray(tps, dtype=np.float64))
    fps_cum = np.cumsum(np.asarray(fps, dtype=np.float64))

    recalls = tps_cum / float(total_gt)
    precisions = tps_cum / np.maximum(tps_cum + fps_cum, 1e-12)
    return _average_precision(recalls, precisions)


def _compute_froc(records: list[dict], iou_thr: float = 0.5) -> tuple[list[dict], dict]:
    num_images = max(len(records), 1)
    total_gt = int(sum(r["gt_boxes"].shape[0] for r in records))

    all_scores: list[float] = []
    for rec in records:
        all_scores.extend([float(s.item()) for s in rec["pred_scores"]])

    if not all_scores:
        points = [{"score_threshold": 1.0, "sensitivity": 0.0, "fp_per_image": 0.0}]
    else:
        thresholds = [float("inf")] + sorted(set(all_scores), reverse=True)
        points = []
        for thr in thresholds:
            tp_total = 0
            fp_total = 0
            for rec in records:
                keep = rec["pred_scores"] >= thr
                pred_boxes = rec["pred_boxes"][keep]
                pred_scores = rec["pred_scores"][keep]
                matches = _greedy_match(pred_boxes, pred_scores, rec["gt_boxes"], iou_thr=iou_thr)
                tp = len(matches)
                fp = int(pred_boxes.shape[0]) - tp
                tp_total += tp
                fp_total += fp

            sensitivity = _safe_div(tp_total, total_gt) if total_gt > 0 else float("nan")
            fp_per_image = float(fp_total) / float(num_images)
            points.append(
                {
                    "score_threshold": float(thr if math.isfinite(thr) else 1.1),
                    "sensitivity": sensitivity,
                    "fp_per_image": fp_per_image,
                }
            )

    xs = np.asarray([p["fp_per_image"] for p in points], dtype=np.float64)
    ys = np.asarray([0.0 if p["sensitivity"] is None else p["sensitivity"] for p in points], dtype=np.float64)

    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    ys = np.maximum.accumulate(ys)

    targets = np.asarray([0.25, 0.5, 1.0, 2.0, 4.0, 8.0], dtype=np.float64)
    sens_at_targets = np.interp(targets, xs, ys, left=0.0, right=float(ys[-1]) if ys.size else 0.0)

    summary = {
        "iou_threshold": iou_thr,
        "sensitivity_at_fp_per_image": {
            f"{t:g}": float(v) for t, v in zip(targets.tolist(), sens_at_targets.tolist())
        },
        "froc_score_mean": float(np.mean(sens_at_targets)),
    }
    return points, summary


def _roc_auc(y_true: list[int], y_score: list[float]) -> float:
    if not y_true:
        return float("nan")
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

    tpr = tp / float(pos)
    fpr = fp / float(neg)

    tpr = np.concatenate(([0.0], tpr, [1.0]))
    fpr = np.concatenate(([0.0], fpr, [1.0]))

    return float(np.trapezoid(tpr, fpr))


def _classification_stats(y_true: list[int], y_prob: list[float], threshold: float = 0.5) -> dict:
    if not y_true:
        return {
            "n": 0,
            "auc_roc": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "f1": float("nan"),
            "accuracy": float("nan"),
        }

    y_pred = [1 if p >= threshold else 0 for p in y_prob]

    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * tp, (2 * tp) + fp + fn)
    accuracy = _safe_div(tp + tn, len(y_true))

    return {
        "n": len(y_true),
        "auc_roc": _roc_auc(y_true, y_prob),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
    }


def _evaluate_records(records: list[dict], iou_match_thr: float = 0.5) -> dict:
    iou_thresholds = np.arange(0.5, 0.96, 0.05)
    ap_by_thr = {f"{thr:.2f}": _compute_ap(records, float(thr)) for thr in iou_thresholds}
    map50 = ap_by_thr["0.50"]
    map5095 = float(np.nanmean(list(ap_by_thr.values()))) if ap_by_thr else float("nan")

    froc_points, froc_summary = _compute_froc(records, iou_thr=iou_match_thr)

    dice_scores: list[float] = []
    iou_scores: list[float] = []
    cls_true: list[int] = []
    cls_prob: list[float] = []

    density_images: dict[int, int] = {}
    density_det: dict[int, dict[str, int]] = {}
    density_cls: dict[int, dict[str, int]] = {}

    for rec in records:
        density = int(rec["density"])
        density_images[density] = density_images.get(density, 0) + 1

        gt_boxes = rec["gt_boxes"]
        pred_boxes = rec["pred_boxes"]
        pred_scores = rec["pred_scores"]
        gt_masks = rec["gt_masks"]
        pred_masks = rec["pred_masks"]
        gt_pathology = rec["gt_pathology"]
        pred_logits = rec["pred_logits"]

        matches = _greedy_match(pred_boxes, pred_scores, gt_boxes, iou_thr=iou_match_thr)
        gt_to_pred = {gt_idx: pred_idx for pred_idx, gt_idx, _ in matches}

        det_item = density_det.setdefault(density, {"tp": 0, "gt": 0})
        det_item["tp"] += len(matches)
        det_item["gt"] += int(gt_boxes.shape[0])

        for gt_idx in range(gt_boxes.shape[0]):
            pred_idx = gt_to_pred.get(gt_idx)
            if pred_idx is None:
                dice_scores.append(0.0)
                iou_scores.append(0.0)
                continue

            gt_m = (gt_masks[gt_idx] > 0).to(torch.uint8)
            pred_m = (pred_masks[pred_idx, 0] > 0.5).to(torch.uint8)

            inter = int((gt_m & pred_m).sum().item())
            union = int((gt_m | pred_m).sum().item())
            gt_area = int(gt_m.sum().item())
            pred_area = int(pred_m.sum().item())

            dice = _safe_div(2 * inter, gt_area + pred_area)
            iou_val = _safe_div(inter, union)
            dice_scores.append(0.0 if math.isnan(dice) else dice)
            iou_scores.append(0.0 if math.isnan(iou_val) else iou_val)

            gt_label = _pathology_to_label(gt_pathology[gt_idx])
            if gt_label is None:
                continue
            if pred_logits.numel() == 0 or pred_idx >= pred_logits.shape[0]:
                continue

            probs = torch.softmax(pred_logits[pred_idx], dim=-1)
            malignant_prob = float(probs[1].item())
            pred_label = 1 if malignant_prob >= 0.5 else 0

            cls_true.append(gt_label)
            cls_prob.append(malignant_prob)

            cls_item = density_cls.setdefault(density, {"correct": 0, "n": 0})
            cls_item["correct"] += int(pred_label == gt_label)
            cls_item["n"] += 1

    all_densities = sorted(set(range(1, 5)) | set(density_images) | set(density_det) | set(density_cls))
    density_metrics = {}
    for d in all_densities:
        det = density_det.get(d, {"tp": 0, "gt": 0})
        cls = density_cls.get(d, {"correct": 0, "n": 0})
        density_metrics[str(d)] = {
            "n_images": int(density_images.get(d, 0)),
            "n_gt_lesions": int(det["gt"]),
            "detection_sensitivity_iou50": _safe_div(det["tp"], det["gt"]),
            "classification_accuracy": _safe_div(cls["correct"], cls["n"]),
            "classification_n": int(cls["n"]),
        }

    class_stats = _classification_stats(cls_true, cls_prob)

    results = {
        "detection": {
            "mAP@50": map50,
            "mAP@50:95": map5095,
            "ap_by_iou": ap_by_thr,
            "froc": froc_summary,
        },
        "segmentation": {
            "dice": float(np.mean(dice_scores)) if dice_scores else float("nan"),
            "iou": float(np.mean(iou_scores)) if iou_scores else float("nan"),
            "n_gt_lesions": len(dice_scores),
        },
        "classification": class_stats,
        "density_metrics": density_metrics,
        "froc_curve": froc_points,
    }
    return results


def _json_ready(obj):
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, float):
        return _to_float(obj)
    return obj


def _write_froc_csv(points: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("score_threshold,fp_per_image,sensitivity\n")
        for p in points:
            score = p.get("score_threshold")
            fp_img = p.get("fp_per_image")
            sens = p.get("sensitivity")
            f.write(f"{score},{fp_img},{sens}\n")


def _write_froc_png(points: list[dict], out_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for FROC plot output. Install it or omit --froc_png."
        ) from exc

    out_png.parent.mkdir(parents=True, exist_ok=True)

    xs = np.asarray([float(p.get("fp_per_image", 0.0)) for p in points], dtype=np.float64)
    ys = np.asarray([float(p.get("sensitivity", 0.0) or 0.0) for p in points], dtype=np.float64)

    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    ys = np.maximum.accumulate(ys)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.5)
    ax.set_xlabel("False positives per image")
    ax.set_ylabel("Sensitivity")
    ax.set_title("FROC curve")
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained mammogram model checkpoints.")
    parser.add_argument("--model_dir", default="./models", help="Directory containing run subfolders/checkpoints.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path (or directory) to evaluate.")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--data_root", default="./data/cbis-ddsm")
    parser.add_argument("--train_csv", default="./meta/cbis-ddsm/mass_case_description_train_set.csv")
    parser.add_argument("--test_csv", default="./meta/cbis-ddsm/mass_case_description_test_set.csv")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output_json", default="./outputs/eval_metrics.json")
    parser.add_argument("--froc_csv", default="./outputs/froc_curve.csv")
    parser.add_argument("--froc_png", default="./outputs/froc_curve.png")
    parser.add_argument("--iou_match_thr", type=float, default=0.5, help="IoU threshold for lesion-level matching.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = Path(args.model_dir).resolve()
    checkpoint_path = _resolve_checkpoint(model_dir=model_dir, checkpoint=args.checkpoint)

    split_csv = args.train_csv if args.split == "train" else args.test_csv
    samples = build_sample_index(Path(args.data_root) / args.split, split_csv)

    cfg = DatasetConfig(
        data_root=args.data_root,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        preprocess=PreprocessConfig(
            target_size=(1024, 800),
            representation=Representation.PSEUDO_COLOR,
        ),
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        augment=False,
    )

    dataset = CBISDDSMDataset(samples, cfg, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=_collate_fn,
    )

    model = _load_model(checkpoint_path, device)

    records: list[dict] = []
    with torch.no_grad():
        for sample_idx, (images, targets) in enumerate(loader):
            image = images[0].to(device)
            target = targets[0]

            det = model([image])[0]

            rec = {
                "density": int(samples[sample_idx]["density"]),
                "gt_boxes": target["boxes"].cpu(),
                "gt_masks": target["masks"].cpu(),
                "gt_pathology": list(target["pathology"]),
                "pred_boxes": det["boxes"].detach().cpu(),
                "pred_scores": det["scores"].detach().cpu(),
                "pred_masks": det["masks"].detach().cpu(),
                "pred_logits": det.get("pathology_logits", torch.empty((0, 2))).detach().cpu(),
            }
            records.append(rec)

    metrics = _evaluate_records(records, iou_match_thr=args.iou_match_thr)

    results = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "num_images": len(records),
        "device": str(device),
        "metrics": metrics,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(_json_ready(results), f, indent=2)

    _write_froc_csv(metrics["froc_curve"], Path(args.froc_csv))
    if args.froc_png:
        _write_froc_png(metrics["froc_curve"], Path(args.froc_png))

    print(f"[eval] checkpoint: {checkpoint_path}")
    print(f"[eval] split: {args.split} | images: {len(records)}")
    print(
        "[eval] detection: "
        f"mAP@50={metrics['detection']['mAP@50']:.4f}, "
        f"mAP@50:95={metrics['detection']['mAP@50:95']:.4f}, "
        f"FROC={metrics['detection']['froc']['froc_score_mean']:.4f}"
    )
    print(
        "[eval] segmentation: "
        f"Dice={metrics['segmentation']['dice']:.4f}, "
        f"IoU={metrics['segmentation']['iou']:.4f}"
    )
    print(
        "[eval] classification: "
        f"AUC={metrics['classification']['auc_roc']:.4f}, "
        f"Sens={metrics['classification']['sensitivity']:.4f}, "
        f"Spec={metrics['classification']['specificity']:.4f}, "
        f"F1={metrics['classification']['f1']:.4f}"
    )
    print(f"[eval] metrics json: {out_json}")
    print(f"[eval] froc csv: {args.froc_csv}")
    if args.froc_png:
        print(f"[eval] froc png: {args.froc_png}")


if __name__ == "__main__":
    main()
