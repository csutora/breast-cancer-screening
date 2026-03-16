"""Lesion-level feature extraction and plotting utilities for CBIS-DDSM masks.

This module is designed to be imported from notebooks or called via a small
command-line runner script. It computes interpretable lesion descriptors:
- size (area, equivalent diameter, bbox size)
- position (centroid in pixels and normalized coordinates)
- boundary clarity proxies (edge strength at lesion boundary)
- boundary shape descriptors (compactness and solidity)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

from dataset import build_sample_index


@dataclass(frozen=True)
class AnalysisInputs:
    """Input paths for lesion analysis."""

    data_root: str = "./data/cbis-ddsm"
    train_csv: str = "./meta/cbis-ddsm/mass_case_description_train_set.csv"
    test_csv: str = "./meta/cbis-ddsm/mass_case_description_test_set.csv"


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep the largest connected foreground component in a binary mask."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    return (labels == largest).astype(np.uint8)


def _boundary_edge_strength(gray_image: np.ndarray, lesion_mask: np.ndarray) -> float:
    """Estimate boundary clarity by averaging gradient magnitude along lesion edges."""
    gx = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)

    boundary = cv2.morphologyEx(lesion_mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    boundary_pixels = grad_mag[boundary > 0]
    if boundary_pixels.size == 0:
        return 0.0

    return float(boundary_pixels.mean())


def _extract_single_lesion_features(
    image: np.ndarray,
    mask: np.ndarray,
    split: str,
    image_key: str,
    patient_id: str,
    side: str,
    view: str,
    pathology: str,
    assessment: int,
    mask_path: str,
) -> dict | None:
    """Extract numeric lesion descriptors for one lesion mask."""
    binary = (mask > 127).astype(np.uint8)
    if binary.sum() == 0:
        return None

    binary = _largest_component(binary)
    h, w = binary.shape

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    area_px = float(cv2.contourArea(contour))
    if area_px <= 0:
        return None

    perimeter_px = float(cv2.arcLength(contour, closed=True))
    x, y, bw, bh = cv2.boundingRect(contour)

    moments = cv2.moments(contour)
    if abs(moments["m00"]) < 1e-8:
        cx, cy = float(x + bw / 2.0), float(y + bh / 2.0)
    else:
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull)) if hull is not None else 0.0

    area_ratio = area_px / float(h * w)
    equiv_diameter_px = float(np.sqrt(4.0 * area_px / np.pi))
    aspect_ratio = float(bw) / float(bh) if bh > 0 else 0.0

    compactness = (
        (perimeter_px * perimeter_px) / (4.0 * np.pi * area_px)
        if area_px > 0 and perimeter_px > 0
        else 0.0
    )
    solidity = (area_px / hull_area) if hull_area > 0 else 0.0
    boundary_clarity = _boundary_edge_strength(image, binary)

    return {
        "split": split,
        "image_key": image_key,
        "patient_id": patient_id,
        "side": side,
        "view": view,
        "pathology": pathology,
        "assessment": int(assessment),
        "mask_path": mask_path,
        "image_height": int(h),
        "image_width": int(w),
        "area_px": area_px,
        "area_ratio": area_ratio,
        "equiv_diameter_px": equiv_diameter_px,
        "bbox_x": int(x),
        "bbox_y": int(y),
        "bbox_w": int(bw),
        "bbox_h": int(bh),
        "bbox_area_px": int(bw * bh),
        "bbox_aspect_ratio": aspect_ratio,
        "centroid_x": cx,
        "centroid_y": cy,
        "centroid_x_norm": cx / float(w),
        "centroid_y_norm": cy / float(h),
        "perimeter_px": perimeter_px,
        "compactness": compactness,
        "solidity": solidity,
        "boundary_clarity": boundary_clarity,
    }


def _iter_samples(inputs: AnalysisInputs) -> Iterable[tuple[str, dict]]:
    """Yield (split, sample) pairs from train and test splits."""
    train_samples = build_sample_index(
        os.path.join(inputs.data_root, "train"),
        inputs.train_csv,
    )
    test_samples = build_sample_index(
        os.path.join(inputs.data_root, "test"),
        inputs.test_csv,
    )

    for sample in train_samples:
        yield "train", sample
    for sample in test_samples:
        yield "test", sample


def extract_lesion_features(inputs: AnalysisInputs) -> pd.DataFrame:
    """Build a lesion-level table with size, position, and boundary descriptors."""
    records: list[dict] = []

    for split, sample in _iter_samples(inputs):
        image = cv2.imread(sample["image_path"], cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        for mask_info in sample["masks"]:
            mask = cv2.imread(mask_info["path"], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            rec = _extract_single_lesion_features(
                image=image,
                mask=mask,
                split=split,
                image_key=sample["image_key"],
                patient_id=sample["patient_id"],
                side=sample["side"],
                view=sample["view"],
                pathology=mask_info.get("pathology", "UNKNOWN"),
                assessment=mask_info.get("assessment", 0),
                mask_path=mask_info["path"],
            )
            if rec is not None:
                records.append(rec)

    return pd.DataFrame.from_records(records)


def save_features_csv(df: pd.DataFrame, output_csv: str) -> None:
    """Save extracted lesion features to CSV."""
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


def plot_feature_distributions(df: pd.DataFrame, output_dir: str) -> list[str]:
    """Create histograms and a position map for lesion descriptors."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "matplotlib is required for plotting. Install it with: pip install matplotlib"
        ) from exc

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    plot_specs = [
        ("area_ratio", "Lesion area ratio", "Area ratio (lesion pixels / image pixels)", 40),
        ("equiv_diameter_px", "Equivalent lesion diameter", "Equivalent diameter (pixels)", 40),
        ("boundary_clarity", "Boundary clarity", "Boundary clarity (mean gradient at edge)", 40),
        ("compactness", "Boundary compactness", "Compactness (P^2 / 4piA)", 40),
        ("solidity", "Boundary solidity", "Solidity (A / convex hull area)", 40),
    ]

    pathology_values = [p for p in sorted(df["pathology"].dropna().unique())]

    for column, title, xlabel, bins in plot_specs:
        if column not in df.columns or df[column].dropna().empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))

        if pathology_values:
            for pathology in pathology_values:
                vals = df.loc[df["pathology"] == pathology, column].dropna().to_numpy()
                if vals.size == 0:
                    continue
                ax.hist(vals, bins=bins, alpha=0.45, label=pathology)
            ax.legend()
        else:
            vals = df[column].dropna().to_numpy()
            ax.hist(vals, bins=bins, alpha=0.8)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.grid(alpha=0.25)

        out_path = out_dir / f"hist_{column}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        saved.append(str(out_path))

    if {"centroid_x_norm", "centroid_y_norm"}.issubset(df.columns):
        pos_df = df[["centroid_x_norm", "centroid_y_norm"]].dropna()
        if not pos_df.empty:
            fig, ax = plt.subplots(figsize=(6, 6))
            hb = ax.hexbin(
                pos_df["centroid_x_norm"],
                pos_df["centroid_y_norm"],
                gridsize=25,
                cmap="viridis",
                mincnt=1,
            )
            fig.colorbar(hb, ax=ax, label="Lesion count")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.invert_yaxis()
            ax.set_title("Normalized lesion centroid map")
            ax.set_xlabel("x / width")
            ax.set_ylabel("y / height")
            ax.grid(alpha=0.25)

            out_path = out_dir / "lesion_position_hexbin.png"
            fig.tight_layout()
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            saved.append(str(out_path))

    return saved


def run_full_analysis(inputs: AnalysisInputs, output_csv: str, output_dir: str) -> tuple[pd.DataFrame, list[str]]:
    """Convenience wrapper: extract features, save CSV, and generate plots."""
    df = extract_lesion_features(inputs)
    save_features_csv(df, output_csv)
    plots = plot_feature_distributions(df, output_dir)
    return df, plots
