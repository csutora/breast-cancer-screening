"""
Evaluate the unified detector + classifier pipeline.

Combines detector-style metrics (mAP, FROC, segmentation, thresholded PR/F1)
with classifier-style metrics (AUC, sensitivity/specificity, per-density and
per-size breakdowns).

Optional FP/TP filtering:
- If --fp_tp_classifier_dir is provided, detector outputs are first filtered by
    the FP/TP box-quality classifier.
- FROC and detection metrics are then computed on these filtered boxes.
- Pathology (benign/malignant) classification is run only on filtered boxes.

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
from fp_tp_pipeline import load_detector_fp_tp_pipeline


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


def _collect_device_info(device: torch.device) -> dict:
    cuda_available = bool(torch.cuda.is_available())
    gpu_count = int(torch.cuda.device_count()) if cuda_available else 0
    selected_index = None
    selected_name = None

    if device.type == "cuda" and cuda_available and gpu_count > 0:
        try:
            selected_index = int(device.index) if device.index is not None else int(torch.cuda.current_device())
            selected_name = str(torch.cuda.get_device_name(selected_index))
        except Exception:
            selected_index = None
            selected_name = None

    return {
        "selected_device": str(device),
        "selected_device_type": str(device.type),
        "cuda_available": cuda_available,
        "gpu_count": gpu_count,
        "selected_gpu_index": selected_index,
        "selected_gpu_name": selected_name,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda) if torch.version.cuda is not None else None,
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
    }


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


def _write_froc_png(
    points: list[dict],
    out_png: Path,
    points_per_size: dict[str, list[dict]] | None = None,
) -> None:
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
    ax.plot(xs, ys, marker="o", markersize=3, linewidth=1.5, label="overall")

    if points_per_size:
        for name, series in points_per_size.items():
            if not series:
                continue
            sx = np.asarray([float(p.get("fp_per_image", 0.0)) for p in series], dtype=np.float64)
            sy = np.asarray([float(p.get("sensitivity", 0.0) or 0.0) for p in series], dtype=np.float64)
            if sx.size == 0 or sy.size == 0:
                continue
            order_s = np.argsort(sx)
            sx = sx[order_s]
            sy = sy[order_s]
            sy = np.maximum.accumulate(sy)
            ax.plot(sx, sy, linewidth=1.5, label=f"{name}")

    ax.set_xlabel("False positives per image")
    ax.set_ylabel("Sensitivity")
    ax.set_title("FROC curve")
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    if points_per_size:
        ax.legend(loc="lower right")
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


def _with_stem_suffix(path: Path, suffix: str) -> Path:
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def _drop_keys_recursive(obj, drop_keys: set[str]) -> None:
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            if key in drop_keys:
                obj.pop(key, None)
                continue
            _drop_keys_recursive(obj[key], drop_keys)
        return

    if isinstance(obj, list):
        for item in obj:
            _drop_keys_recursive(item, drop_keys)


def _select_heavy_metrics_index(sweep_results: list[dict], mode: str) -> int:
    if not sweep_results:
        return 0

    policy = str(mode).strip().lower()
    if policy == "primary":
        return 0

    if policy == "best_det_f1":
        best = max(
            enumerate(sweep_results),
            key=lambda kv: _metric_or_neg_inf(kv[1]["metrics"]["detection_thresholded"]["f1"]),
        )
        return int(best[0])

    if policy == "best_det_recall":
        best = max(
            enumerate(sweep_results),
            key=lambda kv: _metric_or_neg_inf(kv[1]["metrics"]["detection_thresholded"]["recall"]),
        )
        return int(best[0])

    if policy == "best_cls_balanced_accuracy":
        best = max(
            enumerate(sweep_results),
            key=lambda kv: _metric_or_neg_inf(kv[1]["metrics"]["classification"]["overall"]["balanced_accuracy"]),
        )
        return int(best[0])

    return 0


def _build_size_buckets(areas: list[int]) -> tuple[dict[str, list[int]], dict[str, float]]:
    if not areas:
        return {}, {}

    arr = np.asarray(areas, dtype=np.float64)
    p33, p66 = np.percentile(arr, [33, 66])
    buckets = {
        "small": [i for i, a in enumerate(areas) if a <= p33],
        "medium": [i for i, a in enumerate(areas) if p33 < a <= p66],
        "large": [i for i, a in enumerate(areas) if a > p66],
    }
    stats = {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p33": float(p33),
        "p66": float(p66),
    }
    return buckets, stats


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
        buckets, size_stats = _build_size_buckets(y_area)
        for name, idxs in buckets.items():
            y_t = [y_true[i] for i in idxs]
            y_p = [y_pred[i] for i in idxs]
            y_s = [y_prob[i] for i in idxs]
            per_size[name] = _classification_metrics(y_t, y_p, y_s)
        per_size["thresholds_px"] = {"p33": size_stats["p33"], "p66": size_stats["p66"]}
        per_size["area_stats_px"] = size_stats

    return per_density, per_size


def _compute_froc_per_size(records: list[dict], iou_thr: float) -> tuple[dict, dict]:
    lesion_entries: list[tuple[int, int, int]] = []
    all_areas: list[int] = []

    for rec_idx, rec in enumerate(records):
        gt_masks = rec["gt_masks"]
        for gt_idx in range(gt_masks.shape[0]):
            area = int((gt_masks[gt_idx] > 0).sum().item())
            lesion_entries.append((rec_idx, gt_idx, area))
            all_areas.append(area)

    if not all_areas:
        return {}, {}

    _, size_stats = _build_size_buckets(all_areas)
    p33 = float(size_stats["p33"])
    p66 = float(size_stats["p66"])

    bucket_names = ["small", "medium", "large"]
    idx_by_bucket: dict[str, dict[int, list[int]]] = {name: {} for name in bucket_names}

    for rec_idx, gt_idx, area in lesion_entries:
        if area <= p33:
            bucket = "small"
        elif area <= p66:
            bucket = "medium"
        else:
            bucket = "large"

        img_map = idx_by_bucket[bucket]
        img_map.setdefault(rec_idx, []).append(gt_idx)

    curves = {}
    summaries = {}
    for bucket in bucket_names:
        bucket_records: list[dict] = []
        gt_total = 0
        for rec_idx, rec in enumerate(records):
            idxs = idx_by_bucket[bucket].get(rec_idx, [])
            if idxs:
                idx_t = torch.tensor(idxs, dtype=torch.long)
                gt_boxes = rec["gt_boxes"][idx_t]
            else:
                gt_boxes = torch.empty((0, 4), dtype=rec["gt_boxes"].dtype)

            gt_total += int(gt_boxes.shape[0])
            bucket_records.append(
                {
                    "gt_boxes": gt_boxes,
                    "pred_boxes": rec["pred_boxes"],
                    "pred_scores": rec["pred_scores"],
                }
            )

        points, summary = _compute_froc(bucket_records, iou_thr=iou_thr)
        summary["n_gt_lesions"] = int(gt_total)
        summary["thresholds_px"] = {"p33": p33, "p66": p66}
        curves[bucket] = points
        summaries[bucket] = summary

    summaries["thresholds_px"] = {"p33": p33, "p66": p66}
    summaries["area_stats_px"] = size_stats
    return curves, summaries


def _evaluate_records(
    records: list[dict],
    score_threshold: float,
    iou_match_thr: float,
    cls_threshold: float,
    compute_heavy: bool = True,
) -> tuple[dict, dict]:
    if compute_heavy:
        iou_thresholds = np.arange(0.5, 0.96, 0.05)
        ap_by_thr = {f"{thr:.2f}": _compute_ap(records, float(thr)) for thr in iou_thresholds}
        map50 = ap_by_thr.get("0.50", float("nan"))
        map5095 = float(np.nanmean(list(ap_by_thr.values()))) if ap_by_thr else float("nan")

        froc_points, froc_summary = _compute_froc(records, iou_thr=iou_match_thr)
        froc_points_per_size, froc_summary_per_size = _compute_froc_per_size(records, iou_thr=iou_match_thr)
    else:
        ap_by_thr = {}
        map50 = float("nan")
        map5095 = float("nan")
        froc_points = []
        froc_points_per_size = {}
        froc_summary = {
            "iou_threshold": float(iou_match_thr),
            "sensitivity_at_fp_per_image": {},
            "froc_score_mean": float("nan"),
            "skipped": True,
        }
        froc_summary_per_size = {"skipped": True}

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
    lesion_areas_px: list[int] = []
    lesion_matched: list[bool] = []
    lesion_matched_iou: list[float] = []

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
        gt_to_iou = {gt_idx: iou for _, gt_idx, iou in matches}

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
            gt_m = (gt_masks[gt_idx] > 0).to(torch.uint8)
            gt_area = int(gt_m.sum().item())
            lesion_areas_px.append(gt_area)

            pred_idx = gt_to_pred.get(gt_idx)
            if pred_idx is None:
                lesion_matched.append(False)
                lesion_matched_iou.append(float("nan"))
                dice_scores.append(0.0)
                iou_scores.append(0.0)
                continue

            lesion_matched.append(True)
            lesion_matched_iou.append(float(gt_to_iou.get(gt_idx, float("nan"))))

            orig_pred_idx = int(keep_idx[pred_idx].item()) if pred_idx < keep_idx.shape[0] else -1

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

    segmentation_per_size = {}
    detection_per_size = {}
    lesion_size_px = {}
    if lesion_areas_px:
        size_buckets, size_stats = _build_size_buckets(lesion_areas_px)
        lesion_size_px = size_stats

        for name, idxs in size_buckets.items():
            seg_dice = [dice_scores[i] for i in idxs]
            seg_iou = [iou_scores[i] for i in idxs]
            segmentation_per_size[name] = {
                "n_gt_lesions": len(idxs),
                "dice": float(np.mean(seg_dice)) if seg_dice else float("nan"),
                "iou": float(np.mean(seg_iou)) if seg_iou else float("nan"),
            }

            gt_n = len(idxs)
            tp_n = sum(1 for i in idxs if lesion_matched[i])
            fn_n = gt_n - tp_n
            ious = [lesion_matched_iou[i] for i in idxs if lesion_matched[i] and math.isfinite(lesion_matched_iou[i])]
            detection_per_size[name] = {
                "n_gt_lesions": int(gt_n),
                "tp": int(tp_n),
                "fn": int(fn_n),
                "recall": _safe_div(tp_n, gt_n),
                "mean_matched_iou": float(np.mean(ious)) if ious else float("nan"),
            }

        thresholds = {"p33": size_stats["p33"], "p66": size_stats["p66"]}
        segmentation_per_size["thresholds_px"] = thresholds
        detection_per_size["thresholds_px"] = thresholds

    metrics = {
        "detection_full": {
            "mAP@50": map50,
            "mAP@50:95": map5095,
            "ap_by_iou": ap_by_thr,
            "froc": froc_summary,
            "froc_per_size": froc_summary_per_size,
        },
        "segmentation": {
            "dice": float(np.mean(dice_scores)) if dice_scores else float("nan"),
            "iou": float(np.mean(iou_scores)) if iou_scores else float("nan"),
            "n_gt_lesions": len(dice_scores),
            "per_size": segmentation_per_size,
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
            "per_size": detection_per_size,
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
        "lesion_size_px": lesion_size_px,
        "density_detection": density_detection,
        "froc_curve": froc_points,
        "froc_curve_per_size": froc_points_per_size,
    }

    cls_arrays = {
        "y_true": cls_true,
        "y_prob": cls_prob,
        "y_true_valid": cls_true_valid,
        "y_prob_valid": cls_prob_valid,
    }
    return metrics, cls_arrays


def _compute_fp_tp_classifier_metrics(
    records: list[dict],
    iou_match_thr: float,
    score_threshold: float,
    fp_tp_threshold: float | None,
) -> dict:
    if fp_tp_threshold is None:
        return {
            "enabled": False,
            "reason": "fp_tp_filter_disabled",
        }

    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []
    y_density: list[int] = []
    y_area: list[int] = []

    y_true_valid: list[int] = []
    y_pred_valid: list[int] = []
    y_prob_valid: list[float] = []
    y_density_valid: list[int] = []
    y_area_valid: list[int] = []

    num_candidates_total = 0
    num_candidates_after_score = 0
    num_valid_candidates_after_score = 0

    for rec in records:
        cand_boxes = rec.get("fp_tp_candidate_boxes", torch.empty((0, 4)))
        if cand_boxes.numel() == 0:
            continue

        n_boxes = int(cand_boxes.shape[0])
        cand_scores = rec.get("fp_tp_candidate_detector_scores", torch.empty((0,)))
        cand_probs = rec.get("fp_tp_candidate_probs", torch.empty((0,)))
        cand_valid = rec.get("fp_tp_candidate_valid", torch.empty((0,), dtype=torch.bool))

        if cand_scores.shape[0] != n_boxes:
            if cand_scores.numel() == 0:
                cand_scores = torch.zeros((n_boxes,), dtype=torch.float32)
            elif cand_scores.shape[0] < n_boxes:
                pad_n = n_boxes - int(cand_scores.shape[0])
                cand_scores = torch.cat([cand_scores, torch.zeros((pad_n,), dtype=cand_scores.dtype)], dim=0)
            else:
                cand_scores = cand_scores[:n_boxes]

        if cand_probs.shape[0] != n_boxes:
            if cand_probs.numel() == 0:
                cand_probs = torch.zeros((n_boxes,), dtype=torch.float32)
            elif cand_probs.shape[0] < n_boxes:
                pad_n = n_boxes - int(cand_probs.shape[0])
                cand_probs = torch.cat([cand_probs, torch.zeros((pad_n,), dtype=cand_probs.dtype)], dim=0)
            else:
                cand_probs = cand_probs[:n_boxes]

        if cand_valid.shape[0] != n_boxes:
            if cand_valid.numel() == 0:
                cand_valid = torch.zeros((n_boxes,), dtype=torch.bool)
            elif cand_valid.shape[0] < n_boxes:
                pad_n = n_boxes - int(cand_valid.shape[0])
                cand_valid = torch.cat([cand_valid, torch.zeros((pad_n,), dtype=torch.bool)], dim=0)
            else:
                cand_valid = cand_valid[:n_boxes]

        num_candidates_total += n_boxes

        keep = cand_scores >= float(score_threshold)
        boxes = cand_boxes[keep]
        scores = cand_scores[keep]
        probs = cand_probs[keep]
        valid = cand_valid[keep]

        if boxes.numel() == 0:
            continue

        n_kept = int(boxes.shape[0])
        num_candidates_after_score += n_kept
        num_valid_candidates_after_score += int(valid.sum().item())

        gt_boxes = rec["gt_boxes"]
        matches = _greedy_match(boxes, scores, gt_boxes, iou_thr=iou_match_thr)
        true_labels = torch.zeros((n_kept,), dtype=torch.int64)
        for pred_idx, _, _ in matches:
            if 0 <= pred_idx < n_kept:
                true_labels[pred_idx] = 1

        pred_probs = probs.float()
        pred_labels = (valid & (pred_probs >= float(fp_tp_threshold))).to(torch.int64)

        density = int(rec.get("density", 0))
        areas = ((boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)).tolist()

        for i in range(n_kept):
            t = int(true_labels[i].item())
            p = int(pred_labels[i].item())
            s = float(pred_probs[i].item())
            a = int(max(0.0, float(areas[i])))
            v = bool(valid[i].item())

            y_true.append(t)
            y_pred.append(p)
            y_prob.append(s)
            y_density.append(density)
            y_area.append(a)

            if v:
                y_true_valid.append(t)
                y_pred_valid.append(p)
                y_prob_valid.append(s)
                y_density_valid.append(density)
                y_area_valid.append(a)

    overall = _classification_metrics(y_true, y_pred, y_prob)
    per_density, per_size = _classification_breakdowns(y_true, y_pred, y_prob, y_density, y_area)

    overall_valid = _classification_metrics(y_true_valid, y_pred_valid, y_prob_valid)
    per_density_valid, per_size_valid = _classification_breakdowns(
        y_true_valid, y_pred_valid, y_prob_valid, y_density_valid, y_area_valid
    )

    return {
        "enabled": True,
        "iou_match_thr": float(iou_match_thr),
        "score_threshold": float(score_threshold),
        "fp_tp_threshold": float(fp_tp_threshold),
        "overall": overall,
        "per_density": per_density,
        "per_size": per_size,
        "valid_only": {
            "overall": overall_valid,
            "per_density": per_density_valid,
            "per_size": per_size_valid,
        },
        "candidate_analysis": {
            "num_candidates_total": int(num_candidates_total),
            "num_candidates_after_score": int(num_candidates_after_score),
            "num_valid_candidates_after_score": int(num_valid_candidates_after_score),
            "valid_rate_after_score": _safe_div(num_valid_candidates_after_score, num_candidates_after_score),
        },
    }


def _build_eval_records(
    loader: DataLoader,
    samples: list[dict],
    device: torch.device,
    pathology_model,
    use_fp_tp_filter: bool,
    fp_tp_model,
    min_iou_for_masks: float,
) -> list[dict]:
    records: list[dict] = []
    with torch.no_grad():
        for sample_idx, (images, targets) in enumerate(loader):
            image = images[0].to(device)
            if use_fp_tp_filter:
                det = fp_tp_model([image])[0]
                pred_boxes_dev = det.get("boxes", torch.empty((0, 4), device=image.device))
                logits, valid_mask = pathology_model._classify_boxes(image, pred_boxes_dev)
                pred_probs_dev = (
                    torch.softmax(logits, dim=-1)
                    if logits.numel() > 0
                    else torch.empty((0, 2), device=image.device, dtype=image.dtype)
                )
                fp_tp_candidate_boxes_dev = det.get("fp_tp_candidate_boxes", torch.empty((0, 4), device=image.device))
                fp_tp_candidate_scores_dev = det.get(
                    "fp_tp_candidate_detector_scores",
                    torch.empty((fp_tp_candidate_boxes_dev.shape[0],), device=image.device, dtype=image.dtype),
                )
                fp_tp_candidate_probs_dev = det.get(
                    "fp_tp_probs",
                    torch.empty((fp_tp_candidate_boxes_dev.shape[0],), device=image.device, dtype=image.dtype),
                )
                fp_tp_candidate_valid_dev = det.get(
                    "fp_tp_valid",
                    torch.zeros((fp_tp_candidate_boxes_dev.shape[0],), device=image.device, dtype=torch.bool),
                )
            else:
                det = pathology_model([image])[0]
                pred_boxes_dev = det.get("boxes", torch.empty((0, 4), device=image.device))
                pred_probs_dev = det.get("pathology_probs", torch.empty((0, 2), device=image.device))
                valid_mask = det.get(
                    "pathology_valid",
                    torch.ones((pred_boxes_dev.shape[0],), dtype=torch.bool, device=image.device),
                )
                fp_tp_candidate_boxes_dev = torch.empty((0, 4), device=image.device)
                fp_tp_candidate_scores_dev = torch.empty((0,), device=image.device, dtype=image.dtype)
                fp_tp_candidate_probs_dev = torch.empty((0,), device=image.device, dtype=image.dtype)
                fp_tp_candidate_valid_dev = torch.empty((0,), device=image.device, dtype=torch.bool)

            target = targets[0]

            gt_boxes = target["boxes"].cpu()
            gt_masks = target["masks"].cpu()
            pred_boxes = pred_boxes_dev.detach().cpu()
            pred_scores = det.get("scores", torch.empty((0,), device=image.device)).detach().cpu()
            pred_masks = det.get("masks", torch.empty((0, 1, 1, 1), device=image.device)).detach().cpu()
            pred_probs = pred_probs_dev.detach().cpu()
            pred_valid = valid_mask.detach().cpu()
            fp_tp_candidate_boxes = fp_tp_candidate_boxes_dev.detach().cpu()
            fp_tp_candidate_scores = fp_tp_candidate_scores_dev.detach().cpu()
            fp_tp_candidate_probs = fp_tp_candidate_probs_dev.detach().cpu()
            fp_tp_candidate_valid = fp_tp_candidate_valid_dev.detach().cpu()

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
                    "fp_tp_candidate_boxes": fp_tp_candidate_boxes,
                    "fp_tp_candidate_detector_scores": fp_tp_candidate_scores,
                    "fp_tp_candidate_probs": fp_tp_candidate_probs,
                    "fp_tp_candidate_valid": fp_tp_candidate_valid,
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate unified detector+classifier pipeline")
    parser.add_argument("--detector_checkpoint", required=True)
    parser.add_argument("--classifier_dir", default="./classifier_model")
    parser.add_argument(
        "--fp_tp_classifier_dir",
        default="",
        help="Optional FP/TP classifier dir. If set, detection/FROC run after FP/TP filtering.",
    )
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
        "--fp_tp_threshold",
        type=float,
        default=0.5,
        help="TP probability threshold for FP/TP filtering stage.",
    )
    parser.add_argument(
        "--fp_tp_post_nms_iou",
        type=float,
        default=0.30,
        help="NMS IoU used after FP/TP filtering.",
    )
    parser.add_argument(
        "--fp_tp_thresholds",
        type=str,
        default="",
        help="Optional comma-separated FP/TP TP-prob thresholds (e.g. 0.2,0.3,0.5).",
    )
    parser.add_argument(
        "--fp_tp_post_nms_ious",
        type=str,
        default="",
        help="Optional comma-separated post-NMS IoUs for FP/TP stage (e.g. 0.0,0.3,0.5).",
    )
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
    parser.add_argument(
        "--heavy_metrics_mode",
        choices=["all", "primary", "best_det_f1", "best_det_recall", "best_cls_balanced_accuracy"],
        default="all",
        help=(
            "How to compute heavy metrics (mAP/FROC). "
            "'all' computes for every sweep point; other modes compute heavy metrics for one selected sweep point only."
        ),
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
    heavy_metrics_mode = str(args.heavy_metrics_mode).strip().lower()
    compute_heavy_for_grid = heavy_metrics_mode == "all"

    if args.target_sens is not None and cls_thresholds_explicit:
        raise ValueError("Use either --target_sens or --cls_thresholds, not both.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_info = _collect_device_info(device)

    print(
        "[eval_full] runtime device: "
        f"selected={device_info['selected_device']} "
        f"(type={device_info['selected_device_type']}), "
        f"cuda_available={device_info['cuda_available']}, "
        f"gpu_count={device_info['gpu_count']}"
    )
    if device_info["selected_device_type"] == "cuda":
        print(
            "[eval_full] runtime gpu: "
            f"index={device_info['selected_gpu_index']}, "
            f"name={device_info['selected_gpu_name']}"
        )
    else:
        print("[eval_full] runtime gpu: not in use (running on CPU)")

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

    pathology_model = load_full_model(
        detector_checkpoint=args.detector_checkpoint,
        classifier_dir=args.classifier_dir,
        detector_backbone=args.detector_backbone,
        inference_score_threshold=0.0,
        device=device,
    )
    pathology_model.eval()

    use_fp_tp_filter = bool(str(args.fp_tp_classifier_dir).strip())
    fp_tp_thresholds: list[float] = []
    fp_tp_post_nms_ious: list[float] = []
    if use_fp_tp_filter:
        fp_tp_thresholds = _parse_float_list(args.fp_tp_thresholds) or [float(args.fp_tp_threshold)]
        fp_tp_post_nms_ious = _parse_float_list(args.fp_tp_post_nms_ious) or [float(args.fp_tp_post_nms_iou)]
    elif str(args.fp_tp_thresholds).strip() or str(args.fp_tp_post_nms_ious).strip():
        print(
            "[eval_full] warning: fp_tp_threshold grid args were provided but fp_tp filtering is disabled; ignoring."
        )

    fp_tp_model = None
    if use_fp_tp_filter:
        fp_tp_model = load_detector_fp_tp_pipeline(
            detector_checkpoint=args.detector_checkpoint,
            fp_tp_classifier_dir=args.fp_tp_classifier_dir,
            detector_backbone=args.detector_backbone,
            detector_score_threshold=0.0,
            tp_prob_threshold=float(fp_tp_thresholds[0]),
            post_nms_iou_threshold=float(fp_tp_post_nms_ious[0]),
            device=device,
        )
        fp_tp_model.eval()

    min_iou_for_masks = float(min(iou_match_thrs)) if iou_match_thrs else float(args.iou_match_thr)
    print(f"[eval_full] storing compact prediction masks with min_iou_for_masks={min_iou_for_masks:.3f}")

    records_detector_unfiltered: list[dict] = []
    if use_fp_tp_filter:
        print("[eval_full] building detector-only records (no FP/TP filtering) for side-by-side comparison metrics")
        records_detector_unfiltered = _build_eval_records(
            loader=loader,
            samples=samples,
            device=device,
            pathology_model=pathology_model,
            use_fp_tp_filter=False,
            fp_tp_model=None,
            min_iou_for_masks=min_iou_for_masks,
        )

    fp_tp_grid: list[tuple[float | None, float | None]]
    if use_fp_tp_filter:
        fp_tp_grid = [(float(t), float(n)) for t in fp_tp_thresholds for n in fp_tp_post_nms_ious]
    else:
        fp_tp_grid = [(None, None)]

    sweep_results: list[dict] = []
    records: list[dict] = []
    baseline_metrics_cache: dict[tuple[float, float, float, bool], dict] = {}
    for fp_tp_thr, fp_tp_nms in fp_tp_grid:
        if use_fp_tp_filter and fp_tp_model is not None:
            fp_tp_model.tp_prob_threshold = float(fp_tp_thr)
            fp_tp_model.post_nms_iou_threshold = float(fp_tp_nms)
            print(
                "[eval_full] running FP/TP grid point: "
                f"tp_prob_threshold={fp_tp_thr:.3f}, post_nms_iou={fp_tp_nms:.3f}"
            )

        records = _build_eval_records(
            loader=loader,
            samples=samples,
            device=device,
            pathology_model=pathology_model,
            use_fp_tp_filter=use_fp_tp_filter,
            fp_tp_model=fp_tp_model,
            min_iou_for_masks=min_iou_for_masks,
        )

        for score_thr in score_thresholds:
            for iou_thr in iou_match_thrs:
                if args.target_sens is not None:
                    _, cls_arrays = _evaluate_records(
                        records,
                        score_threshold=score_thr,
                        iou_match_thr=iou_thr,
                        cls_threshold=float(args.cls_threshold),
                        compute_heavy=compute_heavy_for_grid,
                    )
                    y_true = cls_arrays["y_true"]
                    y_prob = cls_arrays["y_prob"]

                    cls_thr = float(args.cls_threshold)
                    tuned_metrics = None
                    if y_true:
                        cls_thr, tuned_metrics = _find_threshold_for_sensitivity(y_true, y_prob, float(args.target_sens))
                        fp_tp_context = (
                            f" fp_tp={fp_tp_thr:.3f}/{fp_tp_nms:.3f}" if use_fp_tp_filter else ""
                        )
                        print(
                            f"[eval_full] Auto-tuned cls_threshold={cls_thr:.2f} "
                            f"for target_sens={args.target_sens} at score/iou={score_thr:.3f}/{iou_thr:.3f}"
                            f"{fp_tp_context} "
                            f"(sens={tuned_metrics['sensitivity']:.4f}, spec={tuned_metrics['specificity']:.4f})"
                        )

                    metrics, _ = _evaluate_records(
                        records,
                        score_threshold=score_thr,
                        iou_match_thr=iou_thr,
                        cls_threshold=cls_thr,
                        compute_heavy=compute_heavy_for_grid,
                    )

                    baseline_records = records_detector_unfiltered if use_fp_tp_filter else records
                    baseline_key = (
                        float(score_thr),
                        float(iou_thr),
                        float(cls_thr),
                        bool(compute_heavy_for_grid),
                    )
                    if baseline_key in baseline_metrics_cache:
                        metrics_detector_unfiltered = baseline_metrics_cache[baseline_key]
                    else:
                        metrics_detector_unfiltered, _ = _evaluate_records(
                            baseline_records,
                            score_threshold=score_thr,
                            iou_match_thr=iou_thr,
                            cls_threshold=cls_thr,
                            compute_heavy=compute_heavy_for_grid,
                        )
                        baseline_metrics_cache[baseline_key] = metrics_detector_unfiltered

                    fp_tp_classifier_metrics = _compute_fp_tp_classifier_metrics(
                        records=records,
                        iou_match_thr=iou_thr,
                        score_threshold=score_thr,
                        fp_tp_threshold=fp_tp_thr,
                    )

                    sweep_results.append(
                        {
                            "fp_tp_threshold": fp_tp_thr,
                            "fp_tp_post_nms_iou": fp_tp_nms,
                            "score_threshold": float(score_thr),
                            "iou_match_thr": float(iou_thr),
                            "cls_threshold": float(cls_thr),
                            "target_sens": None if args.target_sens is None else float(args.target_sens),
                            "target_sens_metrics": tuned_metrics,
                            "metrics": metrics,
                            "metrics_detector_unfiltered": metrics_detector_unfiltered,
                            "fp_tp_classifier_metrics": fp_tp_classifier_metrics,
                        }
                    )
                else:
                    for cls_thr in cls_thresholds:
                        metrics, _ = _evaluate_records(
                            records,
                            score_threshold=score_thr,
                            iou_match_thr=iou_thr,
                            cls_threshold=cls_thr,
                            compute_heavy=compute_heavy_for_grid,
                        )

                        baseline_records = records_detector_unfiltered if use_fp_tp_filter else records
                        baseline_key = (
                            float(score_thr),
                            float(iou_thr),
                            float(cls_thr),
                            bool(compute_heavy_for_grid),
                        )
                        if baseline_key in baseline_metrics_cache:
                            metrics_detector_unfiltered = baseline_metrics_cache[baseline_key]
                        else:
                            metrics_detector_unfiltered, _ = _evaluate_records(
                                baseline_records,
                                score_threshold=score_thr,
                                iou_match_thr=iou_thr,
                                cls_threshold=cls_thr,
                                compute_heavy=compute_heavy_for_grid,
                            )
                            baseline_metrics_cache[baseline_key] = metrics_detector_unfiltered

                        fp_tp_classifier_metrics = _compute_fp_tp_classifier_metrics(
                            records=records,
                            iou_match_thr=iou_thr,
                            score_threshold=score_thr,
                            fp_tp_threshold=fp_tp_thr,
                        )

                        sweep_results.append(
                            {
                                "fp_tp_threshold": fp_tp_thr,
                                "fp_tp_post_nms_iou": fp_tp_nms,
                                "score_threshold": float(score_thr),
                                "iou_match_thr": float(iou_thr),
                                "cls_threshold": float(cls_thr),
                                "target_sens": None,
                                "target_sens_metrics": None,
                                "metrics": metrics,
                                "metrics_detector_unfiltered": metrics_detector_unfiltered,
                                "fp_tp_classifier_metrics": fp_tp_classifier_metrics,
                            }
                        )

    heavy_selected_index = 0
    if heavy_metrics_mode != "all" and sweep_results:
        heavy_selected_index = _select_heavy_metrics_index(sweep_results, heavy_metrics_mode)
        selected = sweep_results[heavy_selected_index]

        selected_fp_tp_thr = selected.get("fp_tp_threshold")
        selected_fp_tp_nms = selected.get("fp_tp_post_nms_iou")
        if use_fp_tp_filter and fp_tp_model is not None and selected_fp_tp_thr is not None and selected_fp_tp_nms is not None:
            fp_tp_model.tp_prob_threshold = float(selected_fp_tp_thr)
            fp_tp_model.post_nms_iou_threshold = float(selected_fp_tp_nms)

        selected_records = _build_eval_records(
            loader=loader,
            samples=samples,
            device=device,
            pathology_model=pathology_model,
            use_fp_tp_filter=use_fp_tp_filter,
            fp_tp_model=fp_tp_model,
            min_iou_for_masks=min_iou_for_masks,
        )

        heavy_metrics, _ = _evaluate_records(
            selected_records,
            score_threshold=float(selected["score_threshold"]),
            iou_match_thr=float(selected["iou_match_thr"]),
            cls_threshold=float(selected["cls_threshold"]),
            compute_heavy=True,
        )
        selected["metrics"] = heavy_metrics

        baseline_records = records_detector_unfiltered if use_fp_tp_filter else selected_records
        heavy_metrics_detector_unfiltered, _ = _evaluate_records(
            baseline_records,
            score_threshold=float(selected["score_threshold"]),
            iou_match_thr=float(selected["iou_match_thr"]),
            cls_threshold=float(selected["cls_threshold"]),
            compute_heavy=True,
        )
        selected["metrics_detector_unfiltered"] = heavy_metrics_detector_unfiltered

        selected["fp_tp_classifier_metrics"] = _compute_fp_tp_classifier_metrics(
            records=selected_records,
            iou_match_thr=float(selected["iou_match_thr"]),
            score_threshold=float(selected["score_threshold"]),
            fp_tp_threshold=selected.get("fp_tp_threshold"),
        )

        print(
            "[eval_full] heavy metrics mode: "
            f"{heavy_metrics_mode} -> selected sweep index {heavy_selected_index}"
        )

    first_entry = sweep_results[0]
    primary = sweep_results[heavy_selected_index]
    metrics = primary["metrics"]
    metrics_detector_unfiltered = primary.get("metrics_detector_unfiltered")
    fp_tp_classifier_metrics = primary.get("fp_tp_classifier_metrics")

    results = {
        "detector_checkpoint": str(Path(args.detector_checkpoint).resolve()),
        "classifier_dir": str(Path(args.classifier_dir).resolve()),
        "runtime": {
            "device_info": device_info,
        },
        "fp_tp_filter": {
            "enabled": bool(use_fp_tp_filter),
            "fp_tp_classifier_dir": (
                str(Path(args.fp_tp_classifier_dir).resolve()) if use_fp_tp_filter else None
            ),
            "fp_tp_threshold": primary["fp_tp_threshold"] if use_fp_tp_filter else None,
            "fp_tp_post_nms_iou": primary["fp_tp_post_nms_iou"] if use_fp_tp_filter else None,
            "fp_tp_thresholds": [float(v) for v in fp_tp_thresholds] if use_fp_tp_filter else [],
            "fp_tp_post_nms_ious": [float(v) for v in fp_tp_post_nms_ious] if use_fp_tp_filter else [],
        },
        "detector_backbone": args.detector_backbone,
        "split": args.split,
        "num_images": len(samples),
        "device": str(device),
        "threshold_grid": {
            "heavy_metrics_mode": heavy_metrics_mode,
            "fp_tp_thresholds": [float(v) for v in fp_tp_thresholds] if use_fp_tp_filter else [],
            "fp_tp_post_nms_ious": [float(v) for v in fp_tp_post_nms_ious] if use_fp_tp_filter else [],
            "score_thresholds": [float(v) for v in score_thresholds],
            "iou_match_thrs": [float(v) for v in iou_match_thrs],
            "cls_thresholds": [float(v) for v in cls_thresholds],
            "target_sens": None if args.target_sens is None else float(args.target_sens),
        },
        "heavy_metrics_selection": {
            "mode": heavy_metrics_mode,
            "selected_sweep_index": int(heavy_selected_index),
            "grid_first_thresholds": {
                "fp_tp_threshold": first_entry["fp_tp_threshold"],
                "fp_tp_post_nms_iou": first_entry["fp_tp_post_nms_iou"],
                "score_threshold": first_entry["score_threshold"],
                "iou_match_thr": first_entry["iou_match_thr"],
                "cls_threshold": first_entry["cls_threshold"],
            },
        },
        "primary_thresholds": {
            "fp_tp_threshold": primary["fp_tp_threshold"],
            "fp_tp_post_nms_iou": primary["fp_tp_post_nms_iou"],
            "score_threshold": primary["score_threshold"],
            "iou_match_thr": primary["iou_match_thr"],
            "cls_threshold": primary["cls_threshold"],
        },
        "metrics": metrics,
        "metrics_detector_unfiltered": metrics_detector_unfiltered,
        "fp_tp_classifier_metrics": fp_tp_classifier_metrics,
        "threshold_sweep": sweep_results,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    froc_csv_filtered = Path(args.froc_csv)
    froc_png_filtered = Path(args.froc_png) if args.froc_png else None
    _write_froc_csv(metrics["froc_curve"], froc_csv_filtered)
    if args.froc_png:
        _write_froc_png(
            metrics["froc_curve"],
            froc_png_filtered,
            metrics.get("froc_curve_per_size", {}),
        )

    froc_csv_detector_only = None
    froc_png_detector_only = None
    if use_fp_tp_filter and isinstance(metrics_detector_unfiltered, dict):
        det_froc_curve = metrics_detector_unfiltered.get("froc_curve", [])
        det_froc_curve_per_size = metrics_detector_unfiltered.get("froc_curve_per_size", {})

        froc_csv_detector_only = _with_stem_suffix(froc_csv_filtered, "_detector_only")
        _write_froc_csv(det_froc_curve, froc_csv_detector_only)

        if froc_png_filtered is not None:
            froc_png_detector_only = _with_stem_suffix(froc_png_filtered, "_detector_only")
            _write_froc_png(det_froc_curve, froc_png_detector_only, det_froc_curve_per_size)

    results["artifacts"] = {
        "froc_csv": str(froc_csv_filtered),
        "froc_png": str(froc_png_filtered) if froc_png_filtered is not None else None,
        "froc_csv_detector_only": (
            str(froc_csv_detector_only) if froc_csv_detector_only is not None else None
        ),
        "froc_png_detector_only": (
            str(froc_png_detector_only) if froc_png_detector_only is not None else None
        ),
    }

    # Keep detailed FROC points in dedicated CSV/PNG artifacts only.
    _drop_keys_recursive(results, {"froc_curve", "froc_curve_per_size"})

    # Rewrite JSON once artifact paths are known.
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(_json_ready(results), f, indent=2)

    det_full = metrics["detection_full"]
    seg = metrics["segmentation"]
    det_thr = metrics["detection_thresholded"]
    cls = metrics["classification"]["overall"]
    cls_valid = metrics["classification"]["valid_only"]["overall"]
    cls_invalid = metrics["classification"]["invalid_crop_analysis"]
    det_thr_unfiltered = None
    cls_unfiltered = None
    if isinstance(metrics_detector_unfiltered, dict):
        det_thr_unfiltered = metrics_detector_unfiltered.get("detection_thresholded")
        cls_unfiltered = metrics_detector_unfiltered.get("classification", {}).get("overall")

    print(f"[eval_full] detector_checkpoint: {args.detector_checkpoint}")
    print(f"[eval_full] classifier_dir: {args.classifier_dir}")
    if use_fp_tp_filter:
        print(f"[eval_full] fp_tp_classifier_dir: {args.fp_tp_classifier_dir}")
        print(
            "[eval_full] fp_tp filtering: "
            f"tp_prob_threshold={primary['fp_tp_threshold']:.3f}, "
            f"post_nms_iou={primary['fp_tp_post_nms_iou']:.3f}"
        )
        if len(fp_tp_thresholds) > 1 or len(fp_tp_post_nms_ious) > 1:
            print(
                "[eval_full] fp_tp threshold grid: "
                f"tp_prob=[{','.join(f'{v:g}' for v in fp_tp_thresholds)}] "
                f"post_nms_iou=[{','.join(f'{v:g}' for v in fp_tp_post_nms_ious)}]"
            )
        print("[eval_full] detection/FROC computed after FP/TP filtering; pathology classification runs on filtered boxes.")
    else:
        print("[eval_full] fp_tp filtering: disabled")
    print(f"[eval_full] split: {args.split} | images: {len(samples)}")
    print(
        "[eval_full] heavy metrics mode: "
        f"{heavy_metrics_mode} | selected_sweep_index={heavy_selected_index}"
    )
    print(
        "[eval_full] primary thresholds: "
        f"fp_tp={primary['fp_tp_threshold'] if primary['fp_tp_threshold'] is not None else float('nan'):.3f}, "
        f"post_nms={primary['fp_tp_post_nms_iou'] if primary['fp_tp_post_nms_iou'] is not None else float('nan'):.3f}, "
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
    if isinstance(det_thr_unfiltered, dict):
        print(
            "[eval_full] detection (detector-only, no FP/TP filter): "
            f"precision={det_thr_unfiltered.get('precision', float('nan')):.4f}, "
            f"recall={det_thr_unfiltered.get('recall', float('nan')):.4f}, "
            f"f1={det_thr_unfiltered.get('f1', float('nan')):.4f}, "
            f"mean_matched_iou={det_thr_unfiltered.get('mean_matched_iou', float('nan')):.4f}"
        )
    size_stats = metrics.get("lesion_size_px", {})
    if size_stats:
        print(
            "[eval_full] lesion area (px): "
            f"min={size_stats.get('min', float('nan')):.1f}, "
            f"p33={size_stats.get('p33', float('nan')):.1f}, "
            f"p66={size_stats.get('p66', float('nan')):.1f}, "
            f"max={size_stats.get('max', float('nan')):.1f}"
        )

    det_per_size = det_thr.get("per_size", {})
    if all(k in det_per_size for k in ("small", "medium", "large")):
        print(
            "[eval_full] detection recall by lesion size: "
            f"small={det_per_size['small'].get('recall', float('nan')):.4f}, "
            f"medium={det_per_size['medium'].get('recall', float('nan')):.4f}, "
            f"large={det_per_size['large'].get('recall', float('nan')):.4f}"
        )

    seg_per_size = seg.get("per_size", {})
    if all(k in seg_per_size for k in ("small", "medium", "large")):
        print(
            "[eval_full] segmentation Dice by lesion size: "
            f"small={seg_per_size['small'].get('dice', float('nan')):.4f}, "
            f"medium={seg_per_size['medium'].get('dice', float('nan')):.4f}, "
            f"large={seg_per_size['large'].get('dice', float('nan')):.4f}"
        )
    print(
        "[eval_full] classification: "
        f"AUC={cls['auc_roc']:.4f}, acc={cls['accuracy']:.4f}, "
        f"sens={cls['sensitivity']:.4f}, spec={cls['specificity']:.4f}, f1={cls['f1']:.4f}"
    )
    if isinstance(cls_unfiltered, dict):
        print(
            "[eval_full] classification (detector-only, no FP/TP filter): "
            f"AUC={cls_unfiltered.get('auc_roc', float('nan')):.4f}, "
            f"acc={cls_unfiltered.get('accuracy', float('nan')):.4f}, "
            f"sens={cls_unfiltered.get('sensitivity', float('nan')):.4f}, "
            f"spec={cls_unfiltered.get('specificity', float('nan')):.4f}, "
            f"f1={cls_unfiltered.get('f1', float('nan')):.4f}"
        )
    if isinstance(fp_tp_classifier_metrics, dict) and fp_tp_classifier_metrics.get("enabled"):
        fp_tp_overall = fp_tp_classifier_metrics.get("overall", {})
        fp_tp_cand = fp_tp_classifier_metrics.get("candidate_analysis", {})
        print(
            "[eval_full] fp/tp classifier: "
            f"AUC={fp_tp_overall.get('auc_roc', float('nan')):.4f}, "
            f"acc={fp_tp_overall.get('accuracy', float('nan')):.4f}, "
            f"sens={fp_tp_overall.get('sensitivity', float('nan')):.4f}, "
            f"spec={fp_tp_overall.get('specificity', float('nan')):.4f}, "
            f"f1={fp_tp_overall.get('f1', float('nan')):.4f}, "
            f"n_after_score={fp_tp_cand.get('num_candidates_after_score', 0)}"
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
    print(f"[eval_full] froc csv: {froc_csv_filtered}")
    if args.froc_png:
        print(f"[eval_full] froc png: {froc_png_filtered}")
    if froc_csv_detector_only is not None:
        print(f"[eval_full] froc csv (detector-only): {froc_csv_detector_only}")
    if froc_png_detector_only is not None:
        print(f"[eval_full] froc png (detector-only): {froc_png_detector_only}")

    if len(sweep_results) > 1:
        best_det = max(
            sweep_results,
            key=lambda x: _metric_or_neg_inf(x["metrics"]["detection_thresholded"]["f1"]),
        )
        best_cls = max(
            sweep_results,
            key=lambda x: _metric_or_neg_inf(x["metrics"]["classification"]["overall"]["balanced_accuracy"]),
        )

        if use_fp_tp_filter:
            print(
                "[eval_full] best detection_thresholded F1: "
                f"{best_det['metrics']['detection_thresholded']['f1']:.4f} "
                f"at fp_tp/post_nms/score/iou/cls="
                f"{best_det['fp_tp_threshold']:.3f}/{best_det['fp_tp_post_nms_iou']:.3f}/"
                f"{best_det['score_threshold']:.3f}/{best_det['iou_match_thr']:.3f}/{best_det['cls_threshold']:.3f}"
            )
            print(
                "[eval_full] best classification balanced_accuracy: "
                f"{best_cls['metrics']['classification']['overall']['balanced_accuracy']:.4f} "
                f"at fp_tp/post_nms/score/iou/cls="
                f"{best_cls['fp_tp_threshold']:.3f}/{best_cls['fp_tp_post_nms_iou']:.3f}/"
                f"{best_cls['score_threshold']:.3f}/{best_cls['iou_match_thr']:.3f}/{best_cls['cls_threshold']:.3f}"
            )
        else:
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
