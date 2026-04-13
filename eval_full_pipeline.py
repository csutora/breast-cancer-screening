"""
Evaluate the unified detector + classifier pipeline.

Combines detector-style metrics (mAP, FROC, segmentation, thresholded PR/F1)
with classifier-style metrics (AUC, sensitivity/specificity, per-density and
per-size breakdowns).

Example:
    python eval_full_pipeline.py \
      --detector_checkpoint ./models/nb7u0jeg/detector_resnet101_best.pth \
      --classifier_dir ./classifier_model \
      --split test
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from config import DatasetConfig, PreprocessConfig, Representation
from dataset import CBISDDSMDataset, _collate_fn, build_sample_index
from full_model import load_full_model


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def _to_float(value):
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return float(value)


def _json_ready(obj):
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, float):
        return _to_float(obj)
    return obj


def _pathology_to_label(pathology: str) -> int | None:
    raw = str(pathology).strip().upper()
    if "MALIGNANT" in raw:
        return 1
    if "BENIGN" in raw:
        return 0
    return None


def _greedy_match(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_thr: float,
) -> list[tuple[int, int, float]]:
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return []

    order = torch.argsort(pred_scores, descending=True)
    iou_mat = box_iou(pred_boxes, gt_boxes)

    used_gt: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for pred_idx_t in order:
        pred_idx = int(pred_idx_t.item())
        row = iou_mat[pred_idx]
        best_iou = -1.0
        best_gt = -1
        for gt_idx in range(row.shape[0]):
            if gt_idx in used_gt:
                continue
            iou_val = float(row[gt_idx].item())
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
        return {
            "n": 0,
            "accuracy": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "f1": float("nan"),
            "auc_roc": float("nan"),
            "balanced_accuracy": float("nan"),
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
        }

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
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _find_threshold_for_sensitivity(
    y_true: list[int], y_prob: list[float], target_sens: float,
) -> tuple[float, dict]:
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
        sens = _safe_div(tp, tp + fn)
        spec = _safe_div(tn, tn + fp)
        if sens >= target_sens and spec > best_spec:
            best_threshold = float(threshold)
            best_spec = float(spec)

    preds = (y_prob_np >= best_threshold).astype(int).tolist()
    metrics = _classification_metrics(y_true, preds, y_prob)
    metrics["threshold"] = float(best_threshold)
    return best_threshold, metrics


def _write_froc_csv(points: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("score_threshold,fp_per_image,sensitivity\n")
        for p in points:
            f.write(f"{p.get('score_threshold')},{p.get('fp_per_image')},{p.get('sensitivity')}\n")


def _write_froc_png(points: list[dict], out_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for FROC plot output.") from exc

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


def _parse_float_list(raw: str | None) -> list[float]:
    if raw is None:
        return []
    text = str(raw).strip()
    if not text:
        return []

    values: list[float] = []
    for token in text.split(","):
        t = token.strip()
        if not t:
            continue
        values.append(float(t))
    return values


def _metric_or_neg_inf(value: float) -> float:
    return float(value) if isinstance(value, float) and math.isfinite(value) else float("-inf")


def _classification_breakdowns(
    y_true: list[int],
    y_pred: list[int],
    y_prob: list[float],
    y_density: list[int],
    y_area: list[int],
) -> tuple[dict, dict]:
    per_density = {}
    for d in sorted(set(y_density)):
        idxs = [i for i, dd in enumerate(y_density) if dd == d]
        y_t = [y_true[i] for i in idxs]
        y_p = [y_pred[i] for i in idxs]
        y_s = [y_prob[i] for i in idxs]
        per_density[str(d)] = _classification_metrics(y_t, y_p, y_s)

    per_size = {}
    if y_area:
        areas = np.asarray(y_area, dtype=np.float64)
        p33, p66 = np.percentile(areas, [33, 66])
        buckets = {
            "small": [i for i, a in enumerate(y_area) if a <= p33],
            "medium": [i for i, a in enumerate(y_area) if p33 < a <= p66],
            "large": [i for i, a in enumerate(y_area) if a > p66],
        }
        for name, idxs in buckets.items():
            y_t = [y_true[i] for i in idxs]
            y_p = [y_pred[i] for i in idxs]
            y_s = [y_prob[i] for i in idxs]
            per_size[name] = _classification_metrics(y_t, y_p, y_s)
        per_size["thresholds_px"] = {"p33": float(p33), "p66": float(p66)}

    return per_density, per_size


def _evaluate_records(
    records: list[dict],
    score_threshold: float,
    iou_match_thr: float,
    cls_threshold: float,
) -> tuple[dict, dict]:
    iou_thresholds = np.arange(0.5, 0.96, 0.05)
    ap_by_thr = {f"{thr:.2f}": _compute_ap(records, float(thr)) for thr in iou_thresholds}
    map50 = ap_by_thr.get("0.50", float("nan"))
    map5095 = float(np.nanmean(list(ap_by_thr.values()))) if ap_by_thr else float("nan")

    froc_points, froc_summary = _compute_froc(records, iou_thr=iou_match_thr)

    dice_scores: list[float] = []
    iou_scores: list[float] = []
    cls_true: list[int] = []
    cls_prob: list[float] = []
    cls_pred: list[int] = []
    cls_density: list[int] = []
    cls_area: list[int] = []

    cls_true_valid: list[int] = []
    cls_prob_valid: list[float] = []
    cls_pred_valid: list[int] = []
    cls_density_valid: list[int] = []
    cls_area_valid: list[int] = []
    cls_invalid_matched = 0
    cls_matched_total = 0

    density_det: dict[int, dict[str, int]] = {}

    tp_total = 0
    fp_total = 0
    fn_total = 0
    matched_ious: list[float] = []

    for rec in records:
        density = int(rec["density"])
        gt_boxes = rec["gt_boxes"]
        gt_masks = rec["gt_masks"]
        gt_pathology = rec["gt_pathology"]
        pred_boxes = rec["pred_boxes"]
        pred_scores = rec["pred_scores"]
        pred_masks_compact = rec["pred_masks_compact"]
        pred_mask_lookup = rec["pred_mask_lookup"]
        pred_probs = rec["pred_probs"]
        pred_valid = rec["pred_valid"]

        # Keep classifier probabilities aligned with detector outputs.
        if pred_scores.shape[0] != pred_probs.shape[0]:
            if pred_probs.numel() == 0:
                pred_probs = torch.full((pred_scores.shape[0], 2), 0.5, dtype=torch.float32)
            elif pred_probs.shape[0] < pred_scores.shape[0]:
                pad_n = int(pred_scores.shape[0] - pred_probs.shape[0])
                cls_dim = int(pred_probs.shape[1]) if pred_probs.ndim == 2 and pred_probs.shape[1] > 0 else 2
                pad = torch.full((pad_n, cls_dim), 0.5, dtype=pred_probs.dtype)
                pred_probs = torch.cat([pred_probs, pad], dim=0)
            else:
                pred_probs = pred_probs[: pred_scores.shape[0]]

        if pred_scores.shape[0] != pred_valid.shape[0]:
            if pred_valid.numel() == 0:
                pred_valid = torch.zeros((pred_scores.shape[0],), dtype=torch.bool)
            elif pred_valid.shape[0] < pred_scores.shape[0]:
                pad_n = int(pred_scores.shape[0] - pred_valid.shape[0])
                pad = torch.zeros((pad_n,), dtype=torch.bool)
                pred_valid = torch.cat([pred_valid, pad], dim=0)
            else:
                pred_valid = pred_valid[: pred_scores.shape[0]]

        keep = pred_scores >= float(score_threshold)
        keep_idx = torch.nonzero(keep).flatten()
        pred_boxes_thr = pred_boxes[keep]
        pred_scores_thr = pred_scores[keep]
        pred_probs_thr = pred_probs[keep] if pred_probs.numel() else pred_probs
        pred_valid_thr = pred_valid[keep] if pred_valid.numel() else pred_valid

        matches = _greedy_match(pred_boxes_thr, pred_scores_thr, gt_boxes, iou_thr=iou_match_thr)
        gt_to_pred = {gt_idx: pred_idx for pred_idx, gt_idx, _ in matches}

        tp = len(matches)
        fp = int(pred_boxes_thr.shape[0]) - tp
        fn = int(gt_boxes.shape[0]) - tp
        tp_total += tp
        fp_total += fp
        fn_total += fn
        matched_ious.extend([float(i) for _, _, i in matches])

        det_item = density_det.setdefault(density, {"tp": 0, "gt": 0})
        det_item["tp"] += tp
        det_item["gt"] += int(gt_boxes.shape[0])

        for gt_idx in range(gt_boxes.shape[0]):
            pred_idx = gt_to_pred.get(gt_idx)
            if pred_idx is None:
                dice_scores.append(0.0)
                iou_scores.append(0.0)
                continue

            orig_pred_idx = int(keep_idx[pred_idx].item()) if pred_idx < keep_idx.shape[0] else -1
            gt_m = (gt_masks[gt_idx] > 0).to(torch.uint8)
            gt_area = int(gt_m.sum().item())

            compact_idx = -1
            if 0 <= orig_pred_idx < pred_mask_lookup.shape[0]:
                compact_idx = int(pred_mask_lookup[orig_pred_idx].item())

            if compact_idx < 0 or compact_idx >= pred_masks_compact.shape[0]:
                dice_scores.append(0.0)
                iou_scores.append(0.0)
            else:
                pred_m = (pred_masks_compact[compact_idx, 0] > 0.5).to(torch.uint8)

                inter = int((gt_m & pred_m).sum().item())
                union = int((gt_m | pred_m).sum().item())
                pred_area = int(pred_m.sum().item())

                dice = _safe_div(2 * inter, gt_area + pred_area)
                iou_val = _safe_div(inter, union)
                dice_scores.append(0.0 if math.isnan(dice) else dice)
                iou_scores.append(0.0 if math.isnan(iou_val) else iou_val)

            if pred_probs_thr.numel() == 0 or pred_idx >= pred_probs_thr.shape[0]:
                continue

            gt_label = _pathology_to_label(gt_pathology[gt_idx] if gt_idx < len(gt_pathology) else "UNKNOWN")
            if gt_label is None:
                continue

            cls_matched_total += 1

            malignant_prob = float(pred_probs_thr[pred_idx, 1].item())
            pred_label = 1 if malignant_prob >= float(cls_threshold) else 0
            pred_is_valid = bool(pred_valid_thr[pred_idx].item()) if pred_idx < pred_valid_thr.shape[0] else False

            cls_true.append(gt_label)
            cls_prob.append(malignant_prob)
            cls_pred.append(pred_label)
            cls_density.append(density)
            cls_area.append(gt_area)

            if pred_is_valid:
                cls_true_valid.append(gt_label)
                cls_prob_valid.append(malignant_prob)
                cls_pred_valid.append(pred_label)
                cls_density_valid.append(density)
                cls_area_valid.append(gt_area)
            else:
                cls_invalid_matched += 1

    det_precision = _safe_div(tp_total, tp_total + fp_total)
    det_recall = _safe_div(tp_total, tp_total + fn_total)
    det_f1 = _safe_div(2 * tp_total, (2 * tp_total) + fp_total + fn_total)

    cls_overall = _classification_metrics(cls_true, cls_pred, cls_prob)
    cls_per_density, cls_per_size = _classification_breakdowns(
        cls_true, cls_pred, cls_prob, cls_density, cls_area
    )

    cls_overall_valid = _classification_metrics(cls_true_valid, cls_pred_valid, cls_prob_valid)
    cls_per_density_valid, cls_per_size_valid = _classification_breakdowns(
        cls_true_valid, cls_pred_valid, cls_prob_valid, cls_density_valid, cls_area_valid
    )

    density_detection = {}
    for d in sorted(density_det):
        item = density_det[d]
        density_detection[str(d)] = {
            "n_gt_lesions": int(item["gt"]),
            "detection_sensitivity_iou50": _safe_div(item["tp"], item["gt"]),
        }

    metrics = {
        "detection_full": {
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
        "detection_thresholded": {
            "score_threshold": float(score_threshold),
            "precision": det_precision,
            "recall": det_recall,
            "f1": det_f1,
            "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else float("nan"),
            "tp": int(tp_total),
            "fp": int(fp_total),
            "fn": int(fn_total),
        },
        "classification": {
            "threshold": float(cls_threshold),
            "overall": cls_overall,
            "per_density": cls_per_density,
            "per_size": cls_per_size,
            "valid_only": {
                "overall": cls_overall_valid,
                "per_density": cls_per_density_valid,
                "per_size": cls_per_size_valid,
            },
            "invalid_crop_analysis": {
                "num_matched_for_classification": int(cls_matched_total),
                "num_invalid_crops_matched": int(cls_invalid_matched),
                "num_valid_crops_matched": int(cls_matched_total - cls_invalid_matched),
                "invalid_rate_among_matched": _safe_div(cls_invalid_matched, cls_matched_total),
            },
        },
        "density_detection": density_detection,
        "froc_curve": froc_points,
    }

    cls_arrays = {
        "y_true": cls_true,
        "y_prob": cls_prob,
        "y_true_valid": cls_true_valid,
        "y_prob_valid": cls_prob_valid,
    }
    return metrics, cls_arrays


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate unified detector+classifier pipeline")
    parser.add_argument("--detector_checkpoint", required=True)
    parser.add_argument("--classifier_dir", default="./classifier_model")
    parser.add_argument("--detector_backbone", choices=["resnet101", "resnet152"], default="resnet101")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--data_root", default="./data/cbis-ddsm")
    parser.add_argument("--train_csv", default="./meta/cbis-ddsm/mass_case_description_train_set.csv")
    parser.add_argument("--test_csv", default="./meta/cbis-ddsm/mass_case_description_test_set.csv")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--score_threshold", type=float, default=0.6)
    parser.add_argument("--iou_match_thr", type=float, default=0.5)
    parser.add_argument("--cls_threshold", type=float, default=0.42)
    parser.add_argument(
        "--score_thresholds",
        type=str,
        default="",
        help="Optional comma-separated detector score thresholds for threshold sweep (e.g. 0.3,0.5,0.7)",
    )
    parser.add_argument(
        "--iou_match_thrs",
        type=str,
        default="",
        help="Optional comma-separated IoU match thresholds for threshold sweep (e.g. 0.3,0.5)",
    )
    parser.add_argument(
        "--cls_thresholds",
        type=str,
        default="",
        help="Optional comma-separated classifier thresholds for threshold sweep (e.g. 0.3,0.5,0.7)",
    )
    parser.add_argument("--target_sens", type=float, default=None,
                        help="Auto-tune classifier threshold to meet target sensitivity")
    parser.add_argument("--output_json", default="./outputs/eval_full_pipeline_metrics.json")
    parser.add_argument("--froc_csv", default="./outputs/eval_full_pipeline_froc.csv")
    parser.add_argument("--froc_png", default="./outputs/eval_full_pipeline_froc.png")
    args = parser.parse_args()

    score_thresholds = _parse_float_list(args.score_thresholds) or [float(args.score_threshold)]
    iou_match_thrs = _parse_float_list(args.iou_match_thrs) or [float(args.iou_match_thr)]
    cls_thresholds_explicit = bool(str(args.cls_thresholds).strip())
    cls_thresholds = _parse_float_list(args.cls_thresholds) or [float(args.cls_threshold)]

    if args.target_sens is not None and cls_thresholds_explicit:
        raise ValueError("Use either --target_sens or --cls_thresholds, not both.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_csv = args.train_csv if args.split == "train" else args.test_csv
    samples = build_sample_index(Path(args.data_root) / args.split, split_csv)

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

    model = load_full_model(
        detector_checkpoint=args.detector_checkpoint,
        classifier_dir=args.classifier_dir,
        detector_backbone=args.detector_backbone,
        inference_score_threshold=0.0,
        device=device,
    )
    model.eval()

    min_iou_for_masks = float(min(iou_match_thrs)) if iou_match_thrs else float(args.iou_match_thr)
    print(f"[eval_full] storing compact prediction masks with min_iou_for_masks={min_iou_for_masks:.3f}")

    records: list[dict] = []
    with torch.no_grad():
        for sample_idx, (images, targets) in enumerate(loader):
            image = images[0].to(device)
            det = model([image])[0]
            target = targets[0]

            gt_boxes = target["boxes"].cpu()
            gt_masks = target["masks"].cpu()
            pred_boxes = det.get("boxes", torch.empty((0, 4))).detach().cpu()
            pred_scores = det.get("scores", torch.empty((0,))).detach().cpu()
            pred_masks = det.get("masks", torch.empty((0, 1, 1, 1))).detach().cpu()
            pred_probs = det.get("pathology_probs", torch.empty((0, 2))).detach().cpu()
            pred_valid = det.get(
                "pathology_valid",
                torch.ones((pred_boxes.shape[0],), dtype=torch.bool, device=pred_boxes.device),
            ).detach().cpu()

            if pred_masks.numel() > 0 and pred_boxes.numel() > 0 and gt_boxes.numel() > 0:
                iou_mat = box_iou(pred_boxes, gt_boxes)
                best_iou, _ = iou_mat.max(dim=1)
                keep_mask_idx = torch.nonzero(best_iou >= min_iou_for_masks).flatten()
            else:
                keep_mask_idx = torch.empty((0,), dtype=torch.long)

            if keep_mask_idx.numel() > 0:
                pred_masks_compact = pred_masks[keep_mask_idx].to(torch.float16)
            else:
                pred_masks_compact = torch.empty((0, 1, 1, 1), dtype=torch.float16)

            pred_mask_lookup = torch.full((pred_boxes.shape[0],), -1, dtype=torch.long)
            if keep_mask_idx.numel() > 0:
                pred_mask_lookup[keep_mask_idx] = torch.arange(keep_mask_idx.shape[0], dtype=torch.long)

            records.append(
                {
                    "density": int(samples[sample_idx]["density"]),
                    "gt_boxes": gt_boxes,
                    "gt_masks": gt_masks,
                    "gt_pathology": list(target["pathology"]),
                    "pred_boxes": pred_boxes,
                    "pred_scores": pred_scores,
                    "pred_masks_compact": pred_masks_compact,
                    "pred_mask_lookup": pred_mask_lookup,
                    "pred_probs": pred_probs,
                    "pred_valid": pred_valid,
                }
            )

    sweep_results: list[dict] = []
    for score_thr in score_thresholds:
        for iou_thr in iou_match_thrs:
            if args.target_sens is not None:
                probe_metrics, cls_arrays = _evaluate_records(
                    records,
                    score_threshold=score_thr,
                    iou_match_thr=iou_thr,
                    cls_threshold=float(args.cls_threshold),
                )
                y_true = cls_arrays["y_true"]
                y_prob = cls_arrays["y_prob"]

                cls_thr = float(args.cls_threshold)
                tuned_metrics = None
                if y_true:
                    cls_thr, tuned_metrics = _find_threshold_for_sensitivity(y_true, y_prob, float(args.target_sens))
                    print(
                        f"[eval_full] Auto-tuned cls_threshold={cls_thr:.2f} "
                        f"for target_sens={args.target_sens} at score/iou={score_thr:.3f}/{iou_thr:.3f} "
                        f"(sens={tuned_metrics['sensitivity']:.4f}, spec={tuned_metrics['specificity']:.4f})"
                    )

                metrics, _ = _evaluate_records(
                    records,
                    score_threshold=score_thr,
                    iou_match_thr=iou_thr,
                    cls_threshold=cls_thr,
                )
                sweep_results.append(
                    {
                        "score_threshold": float(score_thr),
                        "iou_match_thr": float(iou_thr),
                        "cls_threshold": float(cls_thr),
                        "target_sens": None if args.target_sens is None else float(args.target_sens),
                        "target_sens_metrics": tuned_metrics,
                        "metrics": metrics,
                    }
                )
            else:
                for cls_thr in cls_thresholds:
                    metrics, _ = _evaluate_records(
                        records,
                        score_threshold=score_thr,
                        iou_match_thr=iou_thr,
                        cls_threshold=cls_thr,
                    )
                    sweep_results.append(
                        {
                            "score_threshold": float(score_thr),
                            "iou_match_thr": float(iou_thr),
                            "cls_threshold": float(cls_thr),
                            "target_sens": None,
                            "target_sens_metrics": None,
                            "metrics": metrics,
                        }
                    )

    primary = sweep_results[0]
    metrics = primary["metrics"]

    results = {
        "detector_checkpoint": str(Path(args.detector_checkpoint).resolve()),
        "classifier_dir": str(Path(args.classifier_dir).resolve()),
        "detector_backbone": args.detector_backbone,
        "split": args.split,
        "num_images": len(records),
        "device": str(device),
        "threshold_grid": {
            "score_thresholds": [float(v) for v in score_thresholds],
            "iou_match_thrs": [float(v) for v in iou_match_thrs],
            "cls_thresholds": [float(v) for v in cls_thresholds],
            "target_sens": None if args.target_sens is None else float(args.target_sens),
        },
        "primary_thresholds": {
            "score_threshold": primary["score_threshold"],
            "iou_match_thr": primary["iou_match_thr"],
            "cls_threshold": primary["cls_threshold"],
        },
        "metrics": metrics,
        "threshold_sweep": sweep_results,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(_json_ready(results), f, indent=2)

    _write_froc_csv(metrics["froc_curve"], Path(args.froc_csv))
    if args.froc_png:
        _write_froc_png(metrics["froc_curve"], Path(args.froc_png))

    det_full = metrics["detection_full"]
    seg = metrics["segmentation"]
    det_thr = metrics["detection_thresholded"]
    cls = metrics["classification"]["overall"]
    cls_valid = metrics["classification"]["valid_only"]["overall"]
    cls_invalid = metrics["classification"]["invalid_crop_analysis"]

    print(f"[eval_full] detector_checkpoint: {args.detector_checkpoint}")
    print(f"[eval_full] classifier_dir: {args.classifier_dir}")
    print(f"[eval_full] split: {args.split} | images: {len(records)}")
    print(
        "[eval_full] primary thresholds: "
        f"score={primary['score_threshold']:.3f}, "
        f"iou={primary['iou_match_thr']:.3f}, "
        f"cls={primary['cls_threshold']:.3f}"
    )
    print(
        "[eval_full] detection (full): "
        f"mAP@50={det_full['mAP@50']:.4f}, "
        f"mAP@50:95={det_full['mAP@50:95']:.4f}, "
        f"FROC={det_full['froc']['froc_score_mean']:.4f}"
    )
    print(f"[eval_full] segmentation: Dice={seg['dice']:.4f}, IoU={seg['iou']:.4f}")
    print(
        "[eval_full] detection (score-thresholded): "
        f"precision={det_thr['precision']:.4f}, recall={det_thr['recall']:.4f}, "
        f"f1={det_thr['f1']:.4f}, mean_matched_iou={det_thr['mean_matched_iou']:.4f}"
    )
    print(
        "[eval_full] classification: "
        f"AUC={cls['auc_roc']:.4f}, acc={cls['accuracy']:.4f}, "
        f"sens={cls['sensitivity']:.4f}, spec={cls['specificity']:.4f}, f1={cls['f1']:.4f}"
    )
    print(
        "[eval_full] classification (valid_only): "
        f"AUC={cls_valid['auc_roc']:.4f}, acc={cls_valid['accuracy']:.4f}, "
        f"sens={cls_valid['sensitivity']:.4f}, spec={cls_valid['specificity']:.4f}, "
        f"f1={cls_valid['f1']:.4f}"
    )
    print(
        "[eval_full] invalid crop analysis: "
        f"matched={cls_invalid['num_matched_for_classification']}, "
        f"invalid={cls_invalid['num_invalid_crops_matched']}, "
        f"invalid_rate={cls_invalid['invalid_rate_among_matched']:.4f}"
    )
    print(f"[eval_full] metrics json: {out_json}")
    print(f"[eval_full] froc csv: {args.froc_csv}")
    if args.froc_png:
        print(f"[eval_full] froc png: {args.froc_png}")

    if len(sweep_results) > 1:
        best_det = max(
            sweep_results,
            key=lambda x: _metric_or_neg_inf(x["metrics"]["detection_thresholded"]["f1"]),
        )
        best_cls = max(
            sweep_results,
            key=lambda x: _metric_or_neg_inf(x["metrics"]["classification"]["overall"]["balanced_accuracy"]),
        )

        print(
            "[eval_full] best detection_thresholded F1: "
            f"{best_det['metrics']['detection_thresholded']['f1']:.4f} "
            f"at score/iou/cls={best_det['score_threshold']:.3f}/"
            f"{best_det['iou_match_thr']:.3f}/{best_det['cls_threshold']:.3f}"
        )
        print(
            "[eval_full] best classification balanced_accuracy: "
            f"{best_cls['metrics']['classification']['overall']['balanced_accuracy']:.4f} "
            f"at score/iou/cls={best_cls['score_threshold']:.3f}/"
            f"{best_cls['iou_match_thr']:.3f}/{best_cls['cls_threshold']:.3f}"
        )


if __name__ == "__main__":
    main()
