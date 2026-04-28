"""
Patch-level dataset for standalone mammogram classifier training.

Extracts lesion patches from GT bounding boxes with:
- Integer-friendly downscaling (snap crop to multiples of patch_size)
- Configurable context margin around the GT box
- Classifier-specific augmentation (offset, flip, rotate, jitter)
- Both pathology and BI-RADS labels available

Preprocessing is done once in parallel and held in RAM. Patch extraction
and augmentation happen on-the-fly per __getitem__ call.
"""

from __future__ import annotations

import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import PreprocessConfig, Representation
from dataset import build_sample_index
from preprocess import MammogramPreprocessor


# ---------------------------------------------------------------------------
# Integer-friendly patch extraction
# ---------------------------------------------------------------------------

def _snap_to_multiple(size: int, base: int) -> int:
    """Round up to the nearest multiple of base that is >= size."""
    return max(base, math.ceil(size / base) * base)


def extract_patch(
    image_3ch: np.ndarray,
    mask_or_box,
    patch_size: int = 224,
    context_factor: float = 1.5,
    offset_jitter: float = 0.0,
    rotation_degrees: float = 0.0,
    rng: Optional[random.Random] = None,
) -> tuple[np.ndarray, int]:
    """
    Extract a square patch around the lesion with integer-friendly downscaling.

    Rotation is applied pre-crop: a larger region is extracted, rotated, then
    center-cropped to crop_size so the rotated area is filled with real tissue
    instead of black padding.

    Args:
        image_3ch:        (H, W, 3) float32 preprocessed image
        mask_or_box:      (H, W) uint8 binary mask, or (x, y, w, h) box tuple
        patch_size:       target output size (pixels, square)
        context_factor:   how much to expand the GT box (1.0 = tight, 1.5 = 50% margin)
        offset_jitter:    max random offset as fraction of box size (0.0 = centered)
        rotation_degrees: max random rotation in degrees (0.0 = no rotation)
        rng:              random.Random instance for reproducibility

    Returns:
        patch:          (patch_size, patch_size, 3) float32
        crop_size:      the snapped crop size before resize (for metadata)
    """
    h, w = image_3ch.shape[:2]

    if isinstance(mask_or_box, tuple):
        # Detector box: (x, y, w, h)
        bx, by, bw, bh = mask_or_box
        cx = bx + bw // 2
        cy = by + bh // 2
    else:
        # GT mask
        binary = (mask_or_box > 127).astype(np.uint8)
        coords = cv2.findNonZero(binary)
        if coords is None:
            cx, cy = w // 2, h // 2
            bw = bh = min(h, w) // 2
        else:
            bx, by, bw, bh = cv2.boundingRect(coords)
            cx = bx + bw // 2
            cy = by + bh // 2
        box_size = max(bw, bh)

    # Expand by context factor
    expanded = int(box_size * context_factor)

    # Snap to integer multiple of patch_size
    crop_size = _snap_to_multiple(expanded, patch_size)

    # Random offset (training augmentation)
    if offset_jitter > 0 and rng is not None:
        max_offset = int(box_size * offset_jitter)
        cx += rng.randint(-max_offset, max_offset)
        cy += rng.randint(-max_offset, max_offset)

    # Compute padded crop size for rotation margin
    if rotation_degrees > 0 and rng is not None:
        angle = rng.uniform(-rotation_degrees, rotation_degrees)
        theta_max = math.radians(rotation_degrees)
        padded_size = math.ceil(crop_size * (math.cos(theta_max) + math.sin(theta_max)))
    else:
        angle = 0.0
        padded_size = crop_size

    # Compute crop bounds, clamped to image
    half = padded_size // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, x1 + padded_size)
    y2 = min(h, y1 + padded_size)
    x1 = max(0, x2 - padded_size)
    y1 = max(0, y2 - padded_size)

    patch = image_3ch[y1:y2, x1:x2]

    # If the crop is smaller than padded_size (image too small), pad
    ph, pw = patch.shape[:2]
    if ph < padded_size or pw < padded_size:
        canvas = np.zeros((padded_size, padded_size, 3), dtype=patch.dtype)
        pad_y = (padded_size - ph) // 2
        pad_x = (padded_size - pw) // 2
        canvas[pad_y:pad_y + ph, pad_x:pad_x + pw] = patch
        patch = canvas

    # Apply rotation pre-crop, then center-crop to crop_size
    if angle != 0.0:
        ph, pw = patch.shape[:2]
        center = (pw / 2, ph / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        patch = cv2.warpAffine(
            patch, M, (pw, ph),
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        dy = (ph - crop_size) // 2
        dx = (pw - crop_size) // 2
        patch = patch[dy:dy + crop_size, dx:dx + crop_size]
    elif padded_size > crop_size:
        ph, pw = patch.shape[:2]
        dy = (ph - crop_size) // 2
        dx = (pw - crop_size) // 2
        patch = patch[dy:dy + crop_size, dx:dx + crop_size]

    # Integer downscale
    if crop_size != patch_size:
        patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_AREA)

    return patch, crop_size


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def _augment_patch(
    patch: np.ndarray,
    rng: random.Random,
    hflip: bool = True,
    vflip: bool = True,
    contrast_jitter: float = 0.15,
    gamma_range: tuple[float, float] = (0.8, 1.2),
    gaussian_noise_std: float = 0.02,
    cutout_fraction: float = 0.15,
) -> np.ndarray:
    """Apply post-crop augmentations to a (H, W, 3) float32 patch.

    Rotation is handled pre-crop in extract_patch() to avoid black-corner
    artifacts.  Only artifact-free augmentations belong here.
    """
    if hflip and rng.random() < 0.5:
        patch = np.fliplr(patch).copy()

    if vflip and rng.random() < 0.5:
        patch = np.flipud(patch).copy()

    if contrast_jitter > 0:
        factor = rng.uniform(1 - contrast_jitter, 1 + contrast_jitter)
        mean = patch.mean()
        patch = np.clip((patch - mean) * factor + mean, 0, 1).astype(np.float32)

    # Gamma/brightness jitter
    if gamma_range and rng.random() < 0.5:
        gamma = rng.uniform(gamma_range[0], gamma_range[1])
        patch = np.clip(patch ** gamma, 0, 1).astype(np.float32)

    if gaussian_noise_std > 0:
        noise = np.random.default_rng().normal(0, gaussian_noise_std, patch.shape).astype(np.float32)
        patch = np.clip(patch + noise, 0, 1).astype(np.float32)

    # Random cutout
    if cutout_fraction > 0 and rng.random() < 0.5:
        h, w = patch.shape[:2]
        ch = int(h * cutout_fraction)
        cw = int(w * cutout_fraction)
        y0 = rng.randint(0, h - ch)
        x0 = rng.randint(0, w - cw)
        patch = patch.copy()
        patch[y0:y0 + ch, x0:x0 + cw] = 0.0

    return patch


# ---------------------------------------------------------------------------
# Parallel preprocessing
# ---------------------------------------------------------------------------

def _max_source_size(
    box_size: int,
    max_context_factor: float = 3.0,
    patch_size: int = 224,
    rotation_degrees: float = 15.0,
    offset_jitter: float = 0.2,
) -> int:
    """Compute a source crop large enough for any sweep config."""
    expanded = int(box_size * max_context_factor)
    crop_size = _snap_to_multiple(expanded, patch_size)
    theta = math.radians(rotation_degrees)
    with_rotation = math.ceil(crop_size * (math.cos(theta) + math.sin(theta)))
    offset_px = int(box_size * offset_jitter)
    return with_rotation + 2 * offset_px


def _preprocess_one_sample(args: tuple) -> list[dict]:
    """Preprocess a single sample (image + masks). Runs in a worker process.

    Stores a per-lesion source crop (not the full image) to keep RAM usage
    reasonable (~5MB per lesion instead of ~200MB per image).
    """
    sample, representation, remove_pectoral = args

    preprocessor = MammogramPreprocessor(PreprocessConfig(
        target_size=None,
        representation=representation,
        remove_pectoral=remove_pectoral,
    ))

    image = cv2.imread(sample["image_path"], cv2.IMREAD_GRAYSCALE)
    if image is None:
        return []

    image_3ch, tx = preprocessor(image, view=sample["view"])
    h, w = image_3ch.shape[:2]

    lesions = []
    for mask_info in sample["masks"]:
        raw_mask = cv2.imread(mask_info["path"], cv2.IMREAD_GRAYSCALE)
        if raw_mask is None:
            continue
        raw_mask = (raw_mask > 127).astype(np.uint8) * 255
        transformed_mask = preprocessor.transform_mask(raw_mask, tx)

        # Find lesion center and size
        binary = (transformed_mask > 127).astype(np.uint8)
        coords = cv2.findNonZero(binary)
        if coords is None:
            cx, cy = w // 2, h // 2
            box_size = min(h, w) // 2
        else:
            bx, by, bw, bh = cv2.boundingRect(coords)
            cx = bx + bw // 2
            cy = by + bh // 2
            box_size = max(bw, bh)

        # Crop generous region covering any sweep context_factor/rotation/offset
        source_size = _max_source_size(box_size)
        half = source_size // 2
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, x1 + source_size)
        y2 = min(h, y1 + source_size)
        x1 = max(0, x2 - source_size)
        y1 = max(0, y2 - source_size)

        source_crop = image_3ch[y1:y2, x1:x2].copy()
        mask_crop = transformed_mask[y1:y2, x1:x2].copy()

        sh, sw = source_crop.shape[:2]
        if sh < source_size or sw < source_size:
            img_canvas = np.zeros((source_size, source_size, 3), dtype=source_crop.dtype)
            mask_canvas = np.zeros((source_size, source_size), dtype=mask_crop.dtype)
            pad_y = (source_size - sh) // 2
            pad_x = (source_size - sw) // 2
            img_canvas[pad_y:pad_y + sh, pad_x:pad_x + sw] = source_crop
            mask_canvas[pad_y:pad_y + sh, pad_x:pad_x + sw] = mask_crop
            source_crop = img_canvas
            mask_crop = mask_canvas

        lesions.append({
            "source_crop": source_crop,
            "mask_crop": mask_crop,
            "box_size": box_size,
            "patient_id": sample["patient_id"],
            "side": sample["side"],
            "view": sample["view"],
            "density": sample["density"],
            "assessment": mask_info["assessment"],
            "pathology_label": mask_info.get("pathology_label", "UNKNOWN"),
            "assessment_label": mask_info.get("assessment_label", "UNKNOWN"),
        })

    return lesions


def preprocess_samples(
    samples: list[dict],
    representation: Representation = Representation.MMS_PSEUDO_COLOR,
    remove_pectoral: bool = False,
    num_workers: int = 0,
) -> list[dict]:
    """Preprocess all samples in parallel. Returns a flat list of lesion dicts.

    Each dict contains the full preprocessed image_3ch and transformed_mask,
    so patch extraction can use any context_factor/patch_size later.
    """
    if num_workers == 0:
        num_workers = min(os.cpu_count() or 4, 32)

    work = [(s, representation, remove_pectoral) for s in samples]

    all_lesions: list[dict] = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_preprocess_one_sample, w): i for i, w in enumerate(work)}
        done = 0
        for future in as_completed(futures):
            lesions = future.result()
            all_lesions.extend(lesions)
            done += 1
            if done % 100 == 0:
                elapsed = time.time() - t0
                print(f"  [preprocess] {done}/{len(samples)} images ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"  [preprocess] Done: {len(all_lesions)} lesions from {len(samples)} images in {elapsed:.1f}s")
    return all_lesions


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PatchDataset(Dataset):
    """
    Per-lesion patch dataset for standalone classifier training.

    Takes pre-processed lesion data (from preprocess_samples) and extracts
    patches on-the-fly with configurable context_factor, patch_size, and
    augmentation. No I/O in __getitem__.
    """

    def __init__(
        self,
        preprocessed_lesions: list[dict],
        patch_size: int = 224,
        context_factor: float = 1.5,
        label_source: str = "pathology",
        augment: bool = True,
        offset_jitter: float = 0.2,
        rotation_degrees: float = 15.0,
        seed: int = 42,
    ):
        self.patch_size = patch_size
        self.context_factor = context_factor
        self.augment = augment
        self.offset_jitter = offset_jitter if augment else 0.0
        self.rotation_degrees = rotation_degrees if augment else 0.0
        self.rng = random.Random(seed)

        # Filter out UNKNOWN labels
        label_key = "pathology_label" if label_source == "pathology" else "assessment_label"
        self.data = [l for l in preprocessed_lesions if l[label_key] != "UNKNOWN"]
        self.label_key = label_key

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, dict]:
        entry = self.data[idx]
        source_crop = entry["source_crop"]
        mask_crop = entry["mask_crop"]

        # extract_patch works on the source crop — the mask_crop lets it
        # find the bbox within the cropped region
        patch, crop_size = extract_patch(
            source_crop, mask_crop,
            patch_size=self.patch_size,
            context_factor=self.context_factor,
            offset_jitter=self.offset_jitter if self.augment else 0.0,
            rotation_degrees=self.rotation_degrees if self.augment else 0.0,
            rng=self.rng if self.augment else None,
        )

        if self.augment:
            patch = _augment_patch(patch, self.rng)

        label_str = entry[self.label_key]
        label = 1 if label_str == "MALIGNANT" else 0

        binary = (mask_crop > 127).astype(np.uint8)
        lesion_area_px = int(binary.sum())

        metadata = {
            "patient_id": entry["patient_id"],
            "side": entry["side"],
            "view": entry["view"],
            "density": entry["density"],
            "assessment": entry["assessment"],
            "pathology_label": entry["pathology_label"],
            "assessment_label": entry["assessment_label"],
            "lesion_area_px": lesion_area_px,
            "crop_size": crop_size,
        }

        patch_tensor = torch.from_numpy(patch.transpose(2, 0, 1).copy()).float()
        return patch_tensor, label, metadata


# ---------------------------------------------------------------------------
# DataLoader builders
# ---------------------------------------------------------------------------

def _collate(batch):
    images, labels, metas = zip(*batch)
    return torch.stack(images), torch.tensor(labels, dtype=torch.long), list(metas)


def build_patch_dataloaders(
    preprocessed_train: list[dict],
    preprocessed_val: list[dict],
    preprocessed_test: list[dict],
    patch_size: int = 224,
    context_factor: float = 1.5,
    label_source: str = "pathology",
    batch_size: int = 16,
    seed: int = 42,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Build train, val, and test DataLoaders from pre-processed data."""

    train_ds = PatchDataset(
        preprocessed_train, patch_size=patch_size, context_factor=context_factor,
        label_source=label_source, augment=True, seed=seed,
    )
    val_ds = PatchDataset(
        preprocessed_val, patch_size=patch_size, context_factor=context_factor,
        label_source=label_source, augment=False, seed=seed,
    )
    test_ds = PatchDataset(
        preprocessed_test, patch_size=patch_size, context_factor=context_factor,
        label_source=label_source, augment=False, seed=seed,
    )

    print(f"[classify] Train: {len(train_ds)} patches")
    print(f"[classify] Val:   {len(val_ds)} patches")
    print(f"[classify] Test:  {len(test_ds)} patches")

    # Workers overlap CPU augmentation with GPU compute via COW fork
    num_workers = min(os.cpu_count() or 4, 8)
    loader_kw = dict(num_workers=num_workers, pin_memory=True, collate_fn=_collate,
                     prefetch_factor=4)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, **loader_kw,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, **loader_kw,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, **loader_kw,
    )

    return train_loader, val_loader, test_loader
