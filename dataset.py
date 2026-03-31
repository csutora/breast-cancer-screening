"""
PyTorch Dataset + DataLoader for CBIS-DDSM mammograms.

Outputs torchvision Mask R-CNN compatible target dicts:
    {
        "boxes":      FloatTensor[N, 4]  (x1, y1, x2, y2),
        "labels":     Int64Tensor[N]     (1 = mass; 0 = background is implicit),
        "masks":      UInt8Tensor[N, H, W],
        "image_id":   Int64Tensor[1],
        "area":       FloatTensor[N],
        "iscrowd":    Int64Tensor[N],
        "pathology":  list[str]          (extra: per-instance pathology),
        "assessment": Int64Tensor[N]     (extra: per-instance BI-RADS),
    }
"""

from __future__ import annotations

import os
import re
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import Dataset as _Dataset, DataLoader
except ImportError:
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment,misc]

    class _Dataset:  # type: ignore[no-redef]
        """Stub so the class definition doesn't crash without torch."""
        pass

from config import DatasetConfig, PreprocessConfig, ClassificationTarget
from preprocess import MammogramPreprocessor, SpatialTransform


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

# Image filename pattern:  P_00001_LEFT_CC_density_3.png
_IMG_RE = re.compile(
    r"^(P_\d+)_(LEFT|RIGHT)_(CC|MLO)_density_(\d)\.png$", re.IGNORECASE
)

# Mask filename pattern:   P_00001_LEFT_CC_density_3_mass_0.png
_MASK_RE = re.compile(
    r"^(P_\d+)_(LEFT|RIGHT)_(CC|MLO)_density_(\d)_(mass|calc)_(\d+)\.png$", re.IGNORECASE
)


def _image_key(patient_id: str, side: str, view: str) -> str:
    """Canonical key for an image (patient + laterality + view)."""
    return f"{patient_id}_{side}_{view}".upper()


def _parse_image_filename(fname: str) -> Optional[dict]:
    m = _IMG_RE.match(fname)
    if not m:
        return None
    return {
        "patient_id": m.group(1).upper(),
        "side": m.group(2).upper(),
        "view": m.group(3).upper(),
        "density": int(m.group(4)),
        "filename": fname,
    }


def _parse_mask_filename(fname: str) -> Optional[dict]:
    m = _MASK_RE.match(fname)
    if not m:
        return None
    return {
        "patient_id": m.group(1).upper(),
        "side": m.group(2).upper(),
        "view": m.group(3).upper(),
        "density": int(m.group(4)),
        "abnormality_type": m.group(5).lower(),
        "mask_index": int(m.group(6)),
        "filename": fname,
    }


# ---------------------------------------------------------------------------
# Metadata integration
# ---------------------------------------------------------------------------

def build_sample_index(
    split_dir: str,
    csv_path: str,
) -> list[dict]:
    """
    Scan `split_dir/images/` and `split_dir/masks/`, cross-reference with
    the CSV metadata, and produce a list of per-image sample dicts.

    `split_dir` is e.g. `cbis_ddsm/train` or `cbis_ddsm/test`.

    Each sample dict:
        {
            "image_path": str,
            "image_key":  str,
            "patient_id": str,
            "side":       str,
            "view":       str,
            "density":    int,
            "masks": [
                {
                    "path":             str,
                    "mask_index":       int,
                    "abnormality_type": str,
                    "pathology":        str,
                    "assessment":       int,
                },
                ...
            ],
        }
    """
    images_dir = os.path.join(split_dir, "images")
    masks_dir = os.path.join(split_dir, "masks")

    # --- Parse image filenames ---
    img_lookup: dict[str, dict] = {}
    for fname in sorted(os.listdir(images_dir)):
        parsed = _parse_image_filename(fname)
        if parsed is None:
            continue
        key = _image_key(parsed["patient_id"], parsed["side"], parsed["view"])
        img_lookup[key] = {
            **parsed,
            "image_path": os.path.join(images_dir, fname),
        }

    # --- Parse mask filenames ---
    mask_lookup: dict[str, list[dict]] = {}
    for fname in sorted(os.listdir(masks_dir)):
        parsed = _parse_mask_filename(fname)
        if parsed is None:
            continue
        # Only keep mass masks (skip calc)
        if parsed["abnormality_type"] != "mass":
            continue
        key = _image_key(parsed["patient_id"], parsed["side"], parsed["view"])
        entry = {
            "path": os.path.join(masks_dir, fname),
            "mask_index": parsed["mask_index"],
            "abnormality_type": parsed["abnormality_type"],
        }
        mask_lookup.setdefault(key, []).append(entry)

    # --- Load CSV metadata ---
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Build a lookup: (patient_id, side, view, abnormality_id) -> row
    meta_lookup: dict[tuple, pd.Series] = {}
    for _, row in df.iterrows():
        pid = row["patient_id"].strip().upper()
        side = row["left or right breast"].strip().upper()
        view = row["image view"].strip().upper()
        abn_id = int(row["abnormality id"])
        meta_lookup[(pid, side, view, abn_id)] = row

    # --- Merge everything ---
    samples = []
    for key, img_info in img_lookup.items():
        masks_for_image = mask_lookup.get(key, [])
        if not masks_for_image:
            continue  # skip images with no mass masks

        # Enrich each mask with metadata from CSV
        enriched_masks = []
        for m in sorted(masks_for_image, key=lambda x: x["mask_index"]):
            # The mask_index in the filename corresponds to abnormality_id - 1
            # in the CSV (0-indexed vs 1-indexed). Try both mappings.
            abn_id_candidates = [m["mask_index"] + 1, m["mask_index"]]
            meta_row = None
            for abn_id in abn_id_candidates:
                meta_row = meta_lookup.get(
                    (img_info["patient_id"], img_info["side"], img_info["view"], abn_id)
                )
                if meta_row is not None:
                    break

            pathology = "UNKNOWN"
            assessment = 0
            if meta_row is not None:
                assessment = int(meta_row.get("assessment", 0))
                # Index-time label rule from BI-RADS assessment:
                # 0-3 -> BENIGN, 4-5 -> MALIGNANT.
                if 0 <= assessment <= 3:
                    pathology = "BENIGN"
                elif assessment in (4, 5):
                    pathology = "MALIGNANT"

            enriched_masks.append({
                **m,
                "pathology": pathology,
                "assessment": assessment,
            })

        samples.append({
            "image_path": img_info["image_path"],
            "image_key": key,
            "patient_id": img_info["patient_id"],
            "side": img_info["side"],
            "view": img_info["view"],
            "density": img_info["density"],
            "masks": enriched_masks,
        })

    return samples


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def _augment(
    image: np.ndarray,
    masks: list[np.ndarray],
    cfg: DatasetConfig,
    rng: random.Random,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    On-the-fly augmentations applied AFTER preprocessing.

    Works on (H, W, 3) float32 images and list of (H, W) uint8 masks.
    """
    h, w = image.shape[:2]

    # Horizontal flip
    if cfg.aug_hflip and rng.random() < 0.5:
        image = np.fliplr(image).copy()
        masks = [np.fliplr(m).copy() for m in masks]

    # Vertical flip
    if cfg.aug_vflip and rng.random() < 0.5:
        image = np.flipud(image).copy()
        masks = [np.flipud(m).copy() for m in masks]

    # Random rotation
    if cfg.aug_rotation_degrees > 0:
        angle = rng.uniform(-cfg.aug_rotation_degrees, cfg.aug_rotation_degrees)
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(
            image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        masks = [
            cv2.warpAffine(
                m, M, (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            for m in masks
        ]

    # Contrast jitter (image only)
    if cfg.aug_contrast_jitter > 0:
        factor = rng.uniform(
            1 - cfg.aug_contrast_jitter, 1 + cfg.aug_contrast_jitter
        )
        mean = image.mean()
        image = np.clip((image - mean) * factor + mean, 0, 1).astype(np.float32)

    # Gaussian noise (image only)
    if cfg.aug_gaussian_noise_std > 0:
        noise = np.random.normal(0, cfg.aug_gaussian_noise_std, image.shape).astype(
            np.float32
        )
        image = np.clip(image + noise, 0, 1)

    return image, masks


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CBISDDSMDataset(_Dataset):
    """
    PyTorch Dataset for CBIS-DDSM mammograms.

    Each __getitem__ returns:
        image:  FloatTensor (3, H, W) in [0, 1]
        target: dict compatible with torchvision Mask R-CNN
    """

    def __init__(
        self,
        samples: list[dict],
        cfg: DatasetConfig,
        train: bool = True,
    ):
        self.samples = samples
        self.cfg = cfg
        self.train = train
        self.preprocessor = MammogramPreprocessor(cfg.preprocess)
        self.rng = random.Random(42)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        sample = self.samples[idx]

        # --- Load image ---
        image = cv2.imread(sample["image_path"], cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {sample['image_path']}")

        # --- Preprocess image ---
        image_3ch, tx = self.preprocessor(image, view=sample["view"])

        # --- Load and transform masks ---
        processed_masks = []
        pathologies = []
        assessments = []

        for mask_info in sample["masks"]:
            raw_mask = cv2.imread(mask_info["path"], cv2.IMREAD_GRAYSCALE)
            if raw_mask is None:
                continue
            # Binarize (some masks might have anti-aliasing artefacts)
            raw_mask = (raw_mask > 127).astype(np.uint8) * 255
            # Apply same spatial transforms as the image
            transformed = self.preprocessor.transform_mask(raw_mask, tx)
            # Skip empty masks (can happen after cropping/resizing)
            if transformed.sum() == 0:
                continue
            processed_masks.append(transformed)
            pathologies.append(mask_info["pathology"])
            assessments.append(mask_info["assessment"])

        # --- Augmentation (training only) ---
        if self.train and self.cfg.augment and processed_masks:
            image_3ch, processed_masks = _augment(
                image_3ch, processed_masks, self.cfg, self.rng
            )

        # --- Build torchvision target dict ---
        boxes = []
        binary_masks = []
        areas = []

        for m in processed_masks:
            # Binarize again after augmentation (rotation can blur edges)
            bm = (m > 127).astype(np.uint8)
            coords = cv2.findNonZero(bm)
            if coords is None:
                continue
            x, y, bw, bh = cv2.boundingRect(coords)
            # Skip degenerate boxes
            if bw < 2 or bh < 2:
                continue
            boxes.append([x, y, x + bw, y + bh])
            binary_masks.append(bm)
            areas.append(float(bm.sum()))

        n = len(boxes)
        if n == 0:
            # Fallback: empty annotations (torchvision handles this)
            h_out, w_out = image_3ch.shape[:2]
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, h_out, w_out), dtype=torch.uint8),
                "image_id": torch.tensor([idx], dtype=torch.int64),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
                "pathology": [],
                "assessment": torch.zeros((0,), dtype=torch.int64),
            }
        else:
            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.ones((n,), dtype=torch.int64),  # 1 = mass
                "masks": torch.as_tensor(
                    np.stack(binary_masks), dtype=torch.uint8
                ),
                "image_id": torch.tensor([idx], dtype=torch.int64),
                "area": torch.as_tensor(areas, dtype=torch.float32),
                "iscrowd": torch.zeros((n,), dtype=torch.int64),
                "pathology": pathologies[:n],
                "assessment": torch.as_tensor(
                    assessments[:n], dtype=torch.int64
                ),
            }

        # Image: (H, W, 3) float32 -> (3, H, W) tensor
        image_tensor = torch.from_numpy(
            image_3ch.transpose(2, 0, 1).copy()
        ).float()

        return image_tensor, target

    def get_patient_ids(self) -> list[str]:
        """Unique patient IDs in this dataset."""
        return sorted(set(s["patient_id"] for s in self.samples))


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def _collate_fn(batch):
    """Custom collate: images and targets have variable-length annotations."""
    images, targets = zip(*batch)
    return list(images), list(targets)


def build_dataloaders(
    cfg: DatasetConfig,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and test DataLoaders from the CBIS-DDSM dataset.

    Uses the official train/test split from the CSV files (no patient leakage
    by construction since the CSVs have disjoint patient sets).
    """
    train_samples = build_sample_index(os.path.join(cfg.data_root, "train"), cfg.train_csv)
    test_samples = build_sample_index(os.path.join(cfg.data_root, "test"), cfg.test_csv)

    print(f"[dataset] Train: {len(train_samples)} images, "
          f"{sum(len(s['masks']) for s in train_samples)} mass annotations")
    print(f"[dataset] Test:  {len(test_samples)} images, "
          f"{sum(len(s['masks']) for s in test_samples)} mass annotations")

    train_ds = CBISDDSMDataset(train_samples, cfg, train=True)
    test_ds = CBISDDSMDataset(test_samples, cfg, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=_collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=_collate_fn,
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Quick test / usage example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = DatasetConfig(
        data_root="./cbis_ddsm",
        train_csv="./mass_case_description_train_set.csv",
        test_csv="./mass_case_description_test_set.csv",
        preprocess=PreprocessConfig(target_size=(1024, 800)),
    )

    # If data isn't downloaded yet, just test the index building:
    if os.path.isdir(os.path.join(cfg.data_root, "train", "images")):
        train_loader, test_loader = build_dataloaders(cfg)

        # Grab one batch
        images, targets = next(iter(train_loader))
        print(f"\nBatch: {len(images)} images")
        for i, (img, tgt) in enumerate(zip(images, targets)):
            print(
                f"  [{i}] image {img.shape}, "
                f"{tgt['boxes'].shape[0]} masses, "
                f"pathology={tgt['pathology']}"
            )
    else:
        print(f"Data root '{cfg.data_root}' not found — skipping dataloader test.")
        print("Once the data is downloaded, run:")
        print(f"  python dataset.py")
