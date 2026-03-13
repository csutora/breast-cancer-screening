"""
Mammogram image preprocessing.

Handles orientation normalization, breast cropping, pectoral muscle removal,
denoising, CLAHE, pseudo-color / CLAHE-triple 3-channel conversion, and
letterbox resizing.

All spatial transforms are tracked so they can be replayed on masks.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

from config import PreprocessConfig, Representation


# ---------------------------------------------------------------------------
# Spatial-transform record (so masks can follow the same transforms)
# ---------------------------------------------------------------------------

@dataclass
class SpatialTransform:
    """Records spatial transforms applied to the image so masks can follow."""

    flipped_lr: bool = False
    crop_bbox: Optional[tuple[int, int, int, int]] = None  # (y1, x1, y2, x2)
    pad_top: int = 0
    pad_left: int = 0
    resized_to: Optional[tuple[int, int]] = None  # (H, W) after letterbox
    scale_factor: float = 1.0


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _otsu_breast_mask(gray: np.ndarray) -> np.ndarray:
    """Binary mask of the breast region via Otsu thresholding + morphology."""
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component (the breast)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    # Component 0 is background; find the largest among the rest
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    return (labels == largest).astype(np.uint8) * 255


def _breast_side(mask: np.ndarray) -> str:
    """Detect whether the breast is on the left or right side of the image."""
    h, w = mask.shape
    left_mass = mask[:, : w // 2].sum()
    right_mass = mask[:, w // 2 :].sum()
    return "LEFT" if left_mass > right_mass else "RIGHT"


# ---------------------------------------------------------------------------
# Preprocessing steps
# ---------------------------------------------------------------------------

def orient_breast(
    image: np.ndarray, breast_mask: np.ndarray, face_right: bool = True
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Flip image so the breast faces a consistent direction.

    Returns (oriented_image, oriented_mask, was_flipped).
    """
    side = _breast_side(breast_mask)
    need_flip = (face_right and side == "LEFT") or (not face_right and side == "RIGHT")
    if need_flip:
        return np.fliplr(image).copy(), np.fliplr(breast_mask).copy(), True
    return image, breast_mask, False


def remove_pectoral_muscle(
    image: np.ndarray, breast_mask: np.ndarray
) -> np.ndarray:
    """
    Basic pectoral muscle removal for MLO views.

    Uses a high-intensity threshold in the upper-left/right quadrant and
    masks it out.  This is a rough heuristic -- good enough as a baseline
    but not a polished solution.
    """
    h, w = image.shape[:2]
    roi_h, roi_w = h // 3, w // 3

    # Determine which corner the pectoral is in (breast-facing side, upper)
    side = _breast_side(breast_mask)
    if side == "RIGHT":
        roi = image[:roi_h, w - roi_w :]
    else:
        roi = image[:roi_h, :roi_w]

    # Threshold bright region in the ROI
    thresh_val = np.percentile(roi[roi > 0], 80) if roi.any() else 200
    _, pec_mask = cv2.threshold(roi, int(thresh_val), 255, cv2.THRESH_BINARY)

    # Dilate to be conservative
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    pec_mask = cv2.dilate(pec_mask, kernel, iterations=2)

    # Apply to the correct corner of the image
    out = image.copy()
    if side == "RIGHT":
        out[:roi_h, w - roi_w :][pec_mask > 0] = 0
    else:
        out[:roi_h, :roi_w][pec_mask > 0] = 0
    return out


def tight_crop_breast(
    image: np.ndarray, breast_mask: np.ndarray, margin: int = 10
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Crop image tightly around the breast tissue.

    Returns (cropped_image, (y1, x1, y2, x2)).
    """
    coords = cv2.findNonZero(breast_mask)
    if coords is None:
        h, w = image.shape[:2]
        return image, (0, 0, h, w)
    x, y, bw, bh = cv2.boundingRect(coords)
    y1 = max(0, y - margin)
    x1 = max(0, x - margin)
    y2 = min(image.shape[0], y + bh + margin)
    x2 = min(image.shape[1], x + bw + margin)
    return image[y1:y2, x1:x2], (y1, x1, y2, x2)


def denoise(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply median filter for denoising."""
    if kernel_size <= 0:
        return image
    return cv2.medianBlur(image, kernel_size)


def apply_clahe(
    image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8
) -> np.ndarray:
    """Apply CLAHE contrast enhancement."""
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit, tileGridSize=(tile_size, tile_size)
    )
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return clahe.apply(image)


def make_pseudo_color(image: np.ndarray, se_radius: int = 15) -> np.ndarray:
    """
    Pseudo-color 3-channel representation (Min et al. 2020).

    Channels:
        0: Original grayscale (after CLAHE, done before calling this)
        1: Top-hat transform  (bright structures on dark background)
        2: Black-hat transform (dark structures on bright background)
    """
    se = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * se_radius + 1, 2 * se_radius + 1)
    )
    tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, se)
    blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, se)
    return np.stack([image, tophat, blackhat], axis=-1)  # (H, W, 3)


def make_clahe_triple(
    gray: np.ndarray,
    clips: tuple[float, float, float] = (1.0, 2.0, 3.0),
    tile_size: int = 8,
) -> np.ndarray:
    """
    CLAHE-tripling: three CLAHE parameterizations as three channels.
    """
    channels = []
    for clip in clips:
        channels.append(apply_clahe(gray, clip_limit=clip, tile_size=tile_size))
    return np.stack(channels, axis=-1)  # (H, W, 3)


def letterbox_resize(
    image: np.ndarray,
    target_size: tuple[int, int],
    fill_value: int = 0,
) -> tuple[np.ndarray, float, int, int]:
    """
    Resize with aspect-ratio preservation + zero-padding (letterbox).

    Args:
        image: (H, W) or (H, W, C)
        target_size: (target_H, target_W)

    Returns:
        (resized_padded, scale_factor, pad_top, pad_left)
    """
    th, tw = target_size
    h, w = image.shape[:2]
    scale = min(th / h, tw / w)
    new_h, new_w = int(h * scale), int(w * scale)

    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    pad_top = (th - new_h) // 2
    pad_left = (tw - new_w) // 2

    if image.ndim == 3:
        canvas = np.full((th, tw, image.shape[2]), fill_value, dtype=image.dtype)
    else:
        canvas = np.full((th, tw), fill_value, dtype=image.dtype)

    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
    return canvas, scale, pad_top, pad_left


def minmax_normalize(image: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1] float32."""
    img = image.astype(np.float32)
    lo, hi = img.min(), img.max()
    if hi - lo < 1e-6:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Main preprocessor
# ---------------------------------------------------------------------------

class MammogramPreprocessor:
    """
    Full preprocessing pipeline for a single mammogram.

    Applies spatial + intensity transforms to the image and tracks all
    spatial ops so that masks can be transformed consistently via
    `transform_mask()`.
    """

    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg

    def __call__(
        self,
        image: np.ndarray,
        view: str = "CC",
    ) -> tuple[np.ndarray, SpatialTransform]:
        """
        Preprocess a grayscale mammogram.

        Args:
            image: (H, W) uint8 or uint16 grayscale
            view:  'CC' or 'MLO'

        Returns:
            processed: (H, W, 3) float32 in [0, 1]
            tx:        SpatialTransform record for mask replay
        """
        cfg = self.cfg
        tx = SpatialTransform()

        # Ensure uint8 grayscale
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(
                np.uint8
            )

        # Breast mask for spatial ops
        breast_mask = _otsu_breast_mask(image)
        breast_mask = _largest_connected_component(breast_mask)

        # 1. Orient breast (flip so it faces right)
        if cfg.orient_breast:
            image, breast_mask, flipped = orient_breast(image, breast_mask)
            tx.flipped_lr = flipped

        # 2. Pectoral muscle removal (MLO only)
        if cfg.remove_pectoral and view.upper() == "MLO":
            image = remove_pectoral_muscle(image, breast_mask)

        # 3. Tight crop around breast
        if cfg.tight_crop:
            image, bbox = tight_crop_breast(image, breast_mask)
            tx.crop_bbox = bbox

        # 4. Denoise
        image = denoise(image, cfg.denoise_kernel)

        # 5. Build 3-channel representation
        if cfg.representation == Representation.PSEUDO_COLOR:
            clahe_img = apply_clahe(image, cfg.clahe_clip_limit, cfg.clahe_tile_size)
            image_3ch = make_pseudo_color(clahe_img, cfg.morpho_se_radius)
        else:
            image_3ch = make_clahe_triple(
                image, cfg.clahe_triple_clips, cfg.clahe_tile_size
            )

        # 6. Min-max normalize
        image_3ch = minmax_normalize(image_3ch)

        # 7. Letterbox resize
        if cfg.target_size is not None:
            image_3ch, scale, pad_top, pad_left = letterbox_resize(
                image_3ch, cfg.target_size
            )
            tx.resized_to = cfg.target_size
            tx.scale_factor = scale
            tx.pad_top = pad_top
            tx.pad_left = pad_left

        return image_3ch, tx

    def transform_mask(
        self, mask: np.ndarray, tx: SpatialTransform
    ) -> np.ndarray:
        """
        Apply the same spatial transforms that were applied to the image.

        Args:
            mask: (H, W) binary uint8 mask from the original image
            tx:   SpatialTransform from preprocessing the image

        Returns:
            (H', W') binary uint8 mask in the same coordinate frame as
            the preprocessed image
        """
        if tx.flipped_lr:
            mask = np.fliplr(mask).copy()
        if tx.crop_bbox is not None:
            y1, x1, y2, x2 = tx.crop_bbox
            mask = mask[y1:y2, x1:x2]
        if tx.resized_to is not None:
            th, tw = tx.resized_to
            h, w = mask.shape[:2]
            scale = tx.scale_factor
            new_h, new_w = int(h * scale), int(w * scale)
            mask = cv2.resize(
                mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST
            )
            canvas = np.zeros((th, tw), dtype=mask.dtype)
            canvas[tx.pad_top : tx.pad_top + new_h, tx.pad_left : tx.pad_left + new_w] = mask
            mask = canvas
        return mask
