"""
Evaluate detector-only checkpoints and export qualitative detection examples.

This script is intended for debugging detector behavior on specific models by
saving side-by-side GT vs prediction overlays for difficult cases.

Usage:
    python eval_detector.py --checkpoint ./models/<run_id>/detector_resnet152_best.pth
    python eval_detector.py --model_dir ./models --split test --num_examples 24 --num_good_examples 24
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models import ResNet101_Weights, ResNet152_Weights
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import box_iou

from config import DatasetConfig, PreprocessConfig, Representation
from dataset import CBISDDSMDataset, _collate_fn, build_sample_index


_BACKBONE_WEIGHTS = {
    "resnet101": ResNet101_Weights.DEFAULT,
    "resnet152": ResNet152_Weights.DEFAULT,
}


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def _to_float(value: float) -> float | None:
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


def _build_detector_model(
    num_classes: int = 2,
    trainable_backbone_layers: int = 5,
    backbone_name: str = "resnet152",
) -> MaskRCNN:
    if backbone_name not in _BACKBONE_WEIGHTS:
        raise ValueError(f"Unsupported backbone '{backbone_name}'. Choose one of: {sorted(_BACKBONE_WEIGHTS)}")

    backbone = resnet_fpn_backbone(
        backbone_name=backbone_name,
        weights=_BACKBONE_WEIGHTS[backbone_name],
        trainable_layers=trainable_backbone_layers,
    )
    return MaskRCNN(backbone=backbone, num_classes=num_classes, min_size=1024, max_size=1024)


def _resolve_checkpoint(model_dir: Path, checkpoint: str | None) -> Path:
    def _latest_from_dir(base: Path) -> Path | None:
        patterns = [
            "**/detector_resnet152_best.pth",
            "**/detector_resnet152_epoch*.pth",
            "**/detector_resnet101_best.pth",
            "**/detector_resnet101_epoch*.pth",
            "**/detector_*_best.pth",
            "**/detector_*_epoch*.pth",
            "**/maskrcnn_best.pth",
            "**/maskrcnn_epoch*.pth",
        ]
        for pattern in patterns:
            candidates = sorted(base.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates:
                return candidates[0]
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
            f"No checkpoints found under {model_dir}. Expected detector_resnet152_best.pth or detector_resnet152_epoch*.pth"
        )
    return found


def _load_model(checkpoint_path: Path, device: torch.device, backbone_name: str) -> MaskRCNN:
    model = _build_detector_model(num_classes=2, trainable_backbone_layers=5, backbone_name=backbone_name)
    blob = torch.load(checkpoint_path, map_location=device)
    if isinstance(blob, dict) and "model_state_dict" in blob:
        state_dict = blob["model_state_dict"]
    else:
        state_dict = blob

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        alt_backbones = [b for b in _BACKBONE_WEIGHTS if b != backbone_name]
        loaded = False
        for alt in alt_backbones:
            alt_model = _build_detector_model(num_classes=2, trainable_backbone_layers=5, backbone_name=alt)
            try:
                alt_model.load_state_dict(state_dict, strict=True)
                model = alt_model
                loaded = True
                print(f"[eval_detector] checkpoint matched backbone '{alt}' (overriding --backbone={backbone_name})")
                break
            except RuntimeError:
                continue
        if not loaded:
            raise RuntimeError(
                f"Failed to load checkpoint with backbone '{backbone_name}'. "
                "Try the other backbone via --backbone (resnet101/resnet152)."
            ) from exc

    model.to(device)
    model.eval()
    return model


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


def _tensor_to_bgr_u8(image: torch.Tensor) -> np.ndarray:
    arr = image.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    arr = np.clip(arr, 0.0, 1.0)
    arr_u8 = (arr * 255.0).astype(np.uint8)
    return cv2.cvtColor(arr_u8, cv2.COLOR_RGB2BGR)


def _draw_overlay(
    image_bgr: np.ndarray,
    boxes: np.ndarray,
    labels: list[str],
    masks: np.ndarray | None,
    colors: list[tuple[int, int, int]],
) -> np.ndarray:
    out = image_bgr.copy()

    if masks is not None and masks.size > 0:
        for idx in range(masks.shape[0]):
            mask = masks[idx] > 0
            if not np.any(mask):
                continue
            color = np.array(colors[idx], dtype=np.float32)
            blended = (0.65 * out[mask].astype(np.float32)) + (0.35 * color)
            out[mask] = np.clip(blended, 0, 255).astype(np.uint8)

    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        color = colors[idx]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            labels[idx],
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return out


def _build_example_canvas(
    image_bgr: np.ndarray,
    gt_boxes: np.ndarray,
    gt_masks: np.ndarray,
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_masks: np.ndarray,
    matches: list[tuple[int, int, float]],
    meta_line: str,
    stats_line: str,
) -> np.ndarray:
    matched_pred_iou = {pred_idx: iou for pred_idx, _, iou in matches}

    gt_labels = [f"GT {i + 1}" for i in range(gt_boxes.shape[0])]
    gt_colors = [(60, 220, 60) for _ in range(gt_boxes.shape[0])]
    gt_view = _draw_overlay(image_bgr, gt_boxes, gt_labels, gt_masks, gt_colors)

    pred_labels: list[str] = []
    pred_colors: list[tuple[int, int, int]] = []
    for i in range(pred_boxes.shape[0]):
        score = float(pred_scores[i])
        if i in matched_pred_iou:
            iou = matched_pred_iou[i]
            pred_labels.append(f"TP {score:.2f} IoU {iou:.2f}")
            pred_colors.append((60, 220, 60))
        else:
            pred_labels.append(f"FP {score:.2f}")
            pred_colors.append((40, 40, 220))

    pred_view = _draw_overlay(image_bgr, pred_boxes, pred_labels, pred_masks, pred_colors)

    panel = np.hstack([gt_view, pred_view])
    panel = cv2.copyMakeBorder(panel, 70, 0, 0, 0, cv2.BORDER_CONSTANT, value=(10, 10, 10))

    cv2.putText(panel, "Ground Truth", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(
        panel,
        "Predictions",
        (image_bgr.shape[1] + 20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(panel, meta_line, (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(panel, stats_line, (20, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    return panel


def _sanitize_name(text: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)
    return out.strip("_") or "sample"


def _push_topk(heap: list[tuple[float, int, dict]], payload: tuple[float, int, dict], k: int) -> None:
    if k <= 0:
        return
    if len(heap) < k:
        heapq.heappush(heap, payload)
    elif payload[0] > heap[0][0]:
        heapq.heapreplace(heap, payload)


def _save_ranked_examples(
    ranked: list[tuple[float, int, dict]],
    out_dir: Path,
    key_name: str,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[dict] = []

    for rank, (_, _, ex) in enumerate(ranked, start=1):
        stats = ex["stats"]
        meta_line = (
            f"idx={stats['sample_idx']} patient={stats['patient_id']} {stats['side']} {stats['view']} "
            f"density={stats['density']}"
        )
        stats_line = (
            f"GT={stats['gt']} Pred={stats['pred']} TP={stats['tp']} FP={stats['fp']} FN={stats['fn']} "
            f"meanIoU={stats['mean_match_iou']:.3f}"
        )

        gt_masks_np = ex["gt_masks"].astype(np.uint8)
        if ex["pred_masks"].size:
            pred_masks_np = ex["pred_masks"][:, 0].astype(np.float32) > 0.5
        else:
            pred_masks_np = np.zeros((0, 1, 1), dtype=bool)

        canvas = _build_example_canvas(
            image_bgr=ex["image_bgr"],
            gt_boxes=ex["gt_boxes"],
            gt_masks=gt_masks_np,
            pred_boxes=ex["pred_boxes"],
            pred_scores=ex["pred_scores"],
            pred_masks=pred_masks_np,
            matches=ex["matches"],
            meta_line=meta_line,
            stats_line=stats_line,
        )

        slug = _sanitize_name(f"{stats['patient_id']}_{stats['side']}_{stats['view']}")
        out_name = f"{rank:03d}_idx{stats['sample_idx']:05d}_{slug}.png"
        out_path = out_dir / out_name
        cv2.imwrite(str(out_path), canvas)

        saved.append(
            {
                "rank": rank,
                "file": out_name,
                key_name: ex[key_name],
                **stats,
            }
        )

    return saved


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


def _evaluate_records_detector(
    records: list[dict],
    iou_match_thr: float,
    seg_dice_sum: float,
    seg_iou_sum: float,
    seg_count: int,
    density_images: dict[int, int],
    density_det: dict[int, dict[str, int]],
) -> dict:
    iou_thresholds = np.arange(0.5, 0.96, 0.05)
    ap_by_thr = {f"{thr:.2f}": _compute_ap(records, float(thr)) for thr in iou_thresholds}
    map50 = ap_by_thr["0.50"]
    map5095 = float(np.nanmean(list(ap_by_thr.values()))) if ap_by_thr else float("nan")

    froc_points, froc_summary = _compute_froc(records, iou_thr=iou_match_thr)

    all_densities = sorted(set(range(1, 5)) | set(density_images) | set(density_det))
    density_metrics = {}
    for d in all_densities:
        det = density_det.get(d, {"tp": 0, "gt": 0})
        density_metrics[str(d)] = {
            "n_images": int(density_images.get(d, 0)),
            "n_gt_lesions": int(det["gt"]),
            "detection_sensitivity_iou50": _safe_div(det["tp"], det["gt"]),
            "classification_accuracy": float("nan"),
            "classification_n": 0,
        }

    return {
        "detection": {
            "mAP@50": map50,
            "mAP@50:95": map5095,
            "ap_by_iou": ap_by_thr,
            "froc": froc_summary,
        },
        "segmentation": {
            "dice": _safe_div(seg_dice_sum, seg_count),
            "iou": _safe_div(seg_iou_sum, seg_count),
            "n_gt_lesions": int(seg_count),
        },
        "classification": {
            "enabled": False,
            "reason": "Detector-only evaluation does not include pathology classification.",
            "n": 0,
            "auc_roc": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "f1": float("nan"),
            "accuracy": float("nan"),
        },
        "density_metrics": density_metrics,
        "froc_curve": froc_points,
    }


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
    parser = argparse.ArgumentParser(description="Evaluate detector-only checkpoints with qualitative outputs.")
    parser.add_argument("--model_dir", default="./models", help="Directory containing run subfolders/checkpoints.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path (or directory) to evaluate.")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--data_root", default="./data/cbis-ddsm")
    parser.add_argument("--train_csv", default="./meta/cbis-ddsm/mass_case_description_train_set.csv")
    parser.add_argument("--test_csv", default="./meta/cbis-ddsm/mass_case_description_test_set.csv")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--backbone", choices=["resnet101", "resnet152"], default="resnet152")
    parser.add_argument("--score_threshold", type=float, default=0.3)
    parser.add_argument("--iou_match_thr", type=float, default=0.5)
    parser.add_argument("--num_examples", type=int, default=20, help="Number of difficult (bad) examples to save.")
    parser.add_argument("--num_good_examples", type=int, default=20, help="Number of best (good) examples to save.")
    parser.add_argument(
        "--include_empty_images_in_good",
        action="store_true",
        help="If set, allow images with no GT and no predictions in good examples.",
    )
    parser.add_argument("--max_images", type=int, default=0, help="If >0, evaluate only first N images.")
    parser.add_argument("--output_dir", default="./outputs/detector_examples")
    parser.add_argument("--output_json", default="./outputs/eval_detector_metrics.json")
    parser.add_argument("--froc_csv", default=None, help="Optional FROC CSV output path.")
    parser.add_argument("--froc_png", default=None, help="Optional FROC PNG output path.")
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

    model = _load_model(checkpoint_path, device, backbone_name=args.backbone)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    matched_ious: list[float] = []
    image_stats: list[dict] = []
    records: list[dict] = []
    density_images: dict[int, int] = {}
    density_det: dict[int, dict[str, int]] = {}
    seg_dice_sum = 0.0
    seg_iou_sum = 0.0
    seg_count = 0

    top_bad_examples: list[tuple[float, int, dict]] = []
    top_good_examples: list[tuple[float, int, dict]] = []
    counter = 0

    max_images = int(args.max_images) if args.max_images and args.max_images > 0 else None
    num_bad_examples = max(0, int(args.num_examples))
    num_good_examples = max(0, int(args.num_good_examples))

    with torch.no_grad():
        for sample_idx, (images, targets) in enumerate(loader):
            if max_images is not None and sample_idx >= max_images:
                break

            image = images[0].to(device)
            target = targets[0]
            det = model([image])[0]

            gt_boxes = target["boxes"].detach().cpu()
            gt_masks = target["masks"].detach().cpu()

            pred_boxes_all = det["boxes"].detach().cpu()
            pred_scores_all = det["scores"].detach().cpu()
            pred_masks_all = det["masks"].detach().cpu() if "masks" in det else torch.zeros((0, 1, 1, 1))

            if pred_masks_all.shape[0] != pred_boxes_all.shape[0]:
                pred_masks_all = torch.zeros((0, 1, 1, 1))

            sample_meta = samples[sample_idx]
            density = int(sample_meta.get("density", 0))

            density_images[density] = density_images.get(density, 0) + 1

            matches_all = _greedy_match(pred_boxes_all, pred_scores_all, gt_boxes, iou_thr=float(args.iou_match_thr))
            det_item = density_det.setdefault(density, {"tp": 0, "gt": 0})
            det_item["tp"] += len(matches_all)
            det_item["gt"] += int(gt_boxes.shape[0])

            gt_to_pred_all = {gt_idx: pred_idx for pred_idx, gt_idx, _ in matches_all}
            for gt_idx in range(gt_boxes.shape[0]):
                pred_idx = gt_to_pred_all.get(gt_idx)
                seg_count += 1

                if pred_idx is None or pred_masks_all.numel() == 0 or pred_idx >= pred_masks_all.shape[0]:
                    continue

                gt_m = (gt_masks[gt_idx] > 0).to(torch.uint8)
                pred_m = (pred_masks_all[pred_idx, 0] > 0.5).to(torch.uint8)

                inter = int((gt_m & pred_m).sum().item())
                union = int((gt_m | pred_m).sum().item())
                gt_area = int(gt_m.sum().item())
                pred_area = int(pred_m.sum().item())

                dice = _safe_div(2 * inter, gt_area + pred_area)
                iou_val = _safe_div(inter, union)
                seg_dice_sum += 0.0 if math.isnan(dice) else float(dice)
                seg_iou_sum += 0.0 if math.isnan(iou_val) else float(iou_val)

            records.append(
                {
                    "density": density,
                    "gt_boxes": gt_boxes,
                    "pred_boxes": pred_boxes_all,
                    "pred_scores": pred_scores_all,
                }
            )

            keep = pred_scores_all >= float(args.score_threshold)
            pred_boxes = pred_boxes_all[keep]
            pred_scores = pred_scores_all[keep]
            pred_masks = pred_masks_all[keep] if pred_masks_all.shape[0] == pred_boxes_all.shape[0] else torch.zeros((0, 1, 1, 1))

            matches = _greedy_match(pred_boxes, pred_scores, gt_boxes, iou_thr=float(args.iou_match_thr))
            tp = len(matches)
            fp = int(pred_boxes.shape[0]) - tp
            fn = int(gt_boxes.shape[0]) - tp

            total_tp += tp
            total_fp += fp
            total_fn += fn

            sample_ious = [m[2] for m in matches]
            matched_ious.extend(sample_ious)
            mean_iou = float(np.mean(sample_ious)) if sample_ious else 0.0

            image_stat = {
                "sample_idx": sample_idx,
                "patient_id": sample_meta["patient_id"],
                "side": sample_meta["side"],
                "view": sample_meta["view"],
                "density": int(sample_meta.get("density", 0)),
                "gt": int(gt_boxes.shape[0]),
                "pred": int(pred_boxes.shape[0]),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "mean_match_iou": mean_iou,
            }
            image_stats.append(image_stat)

            if num_bad_examples > 0 or num_good_examples > 0:
                bad_score = float(fp + fn + (1.0 - mean_iou if gt_boxes.shape[0] > 0 else 0.0))

                # Prefer examples with high true positives, low errors, and strong IoU.
                good_score = float((2.0 * tp) - fp - fn + mean_iou)

                image_bgr = _tensor_to_bgr_u8(images[0])

                example_payload = {
                    "stats": image_stat,
                    "image_bgr": image_bgr,
                    "gt_boxes": gt_boxes.numpy(),
                    "gt_masks": gt_masks.numpy(),
                    "pred_boxes": pred_boxes.numpy(),
                    "pred_scores": pred_scores.numpy(),
                    "pred_masks": pred_masks.numpy(),
                    "matches": matches,
                    "bad_score": bad_score,
                    "good_score": good_score,
                }

                bad_entry = (bad_score, counter, example_payload)
                counter += 1
                _push_topk(top_bad_examples, bad_entry, num_bad_examples)

                allow_for_good = args.include_empty_images_in_good or (gt_boxes.shape[0] > 0 or pred_boxes.shape[0] > 0)
                if allow_for_good:
                    good_entry = (good_score, counter, example_payload)
                    counter += 1
                    _push_topk(top_good_examples, good_entry, num_good_examples)

    precision = _safe_div(total_tp, total_tp + total_fp)
    recall = _safe_div(total_tp, total_tp + total_fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    mean_iou = float(np.mean(matched_ious)) if matched_ious else float("nan")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hardest = sorted(top_bad_examples, key=lambda x: x[0], reverse=True)
    best = sorted(top_good_examples, key=lambda x: x[0], reverse=True)

    bad_examples_dir = out_dir / "bad_cases"
    good_examples_dir = out_dir / "good_cases"

    saved_bad_examples = _save_ranked_examples(hardest, bad_examples_dir, key_name="bad_score")
    saved_good_examples = _save_ranked_examples(best, good_examples_dir, key_name="good_score")

    metrics = _evaluate_records_detector(
        records,
        iou_match_thr=float(args.iou_match_thr),
        seg_dice_sum=seg_dice_sum,
        seg_iou_sum=seg_iou_sum,
        seg_count=seg_count,
        density_images=density_images,
        density_det=density_det,
    )

    out_json = Path(args.output_json)
    default_froc_csv = out_json.with_name(f"{out_json.stem}_froc.csv")
    default_froc_png = out_json.with_name(f"{out_json.stem}_froc.png")
    out_froc_csv = Path(args.froc_csv) if args.froc_csv else default_froc_csv
    out_froc_png = Path(args.froc_png) if args.froc_png else default_froc_png

    results = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "device": str(device),
        "num_images": len(records),
        "score_threshold": float(args.score_threshold),
        "iou_match_thr": float(args.iou_match_thr),
        "metrics": metrics,
        "detection_summary": {
            "tp": int(total_tp),
            "fp": int(total_fp),
            "fn": int(total_fn),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_matched_iou": mean_iou,
        },
        "saved_examples_dir": str(out_dir.resolve()),
        "saved_bad_examples_dir": str(bad_examples_dir.resolve()),
        "saved_good_examples_dir": str(good_examples_dir.resolve()),
        "saved_bad_examples": saved_bad_examples,
        "saved_good_examples": saved_good_examples,
        "image_stats": image_stats,
        "froc_csv": str(out_froc_csv.resolve()),
        "froc_png": str(out_froc_png.resolve()),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(_json_ready(results), f, indent=2)

    _write_froc_csv(metrics["froc_curve"], out_froc_csv)

    froc_png_written = True
    try:
        _write_froc_png(metrics["froc_curve"], out_froc_png)
    except RuntimeError as exc:
        froc_png_written = False
        print(f"[eval_detector] warning: {exc}")

    print(f"[eval_detector] checkpoint: {checkpoint_path}")
    print(f"[eval_detector] split: {args.split} | images: {len(records)}")
    print(
        "[eval_detector] detection (full): "
        f"mAP@50={metrics['detection']['mAP@50']:.4f}, "
        f"mAP@50:95={metrics['detection']['mAP@50:95']:.4f}, "
        f"FROC={metrics['detection']['froc']['froc_score_mean']:.4f}"
    )
    print(
        "[eval_detector] segmentation: "
        f"Dice={metrics['segmentation']['dice']:.4f}, "
        f"IoU={metrics['segmentation']['iou']:.4f}"
    )
    print(
        "[eval_detector] detection (score-thresholded): "
        f"precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}, mean_matched_iou={mean_iou:.4f}"
    )
    print(
        f"[eval_detector] bad examples: {bad_examples_dir.resolve()} "
        f"({len(saved_bad_examples)} saved)"
    )
    print(
        f"[eval_detector] good examples: {good_examples_dir.resolve()} "
        f"({len(saved_good_examples)} saved)"
    )
    print(f"[eval_detector] metrics json: {out_json.resolve()}")
    print(f"[eval_detector] froc csv: {out_froc_csv.resolve()}")
    if froc_png_written:
        print(f"[eval_detector] froc png: {out_froc_png.resolve()}")


if __name__ == "__main__":
    main()
