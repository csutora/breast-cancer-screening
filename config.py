"""
Configuration for the mammogram preprocessing + dataset pipeline.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Representation(Enum):
    """3-channel representation strategy for grayscale mammograms."""

    CLAHE_TRIPLE = "clahe_triple"
    PSEUDO_COLOR = "pseudo_color"


class ClassificationTarget(Enum):
    """What label to use for benign/malignant classification."""

    PATHOLOGY = "pathology"  # ground-truth from biopsy
    ASSESSMENT = "assessment"  # BI-RADS from radiologist


@dataclass
class PreprocessConfig:
    """Image preprocessing parameters."""

    # --- 3-channel representation ---
    representation: Representation = Representation.PSEUDO_COLOR

    # CLAHE parameters (used in both modes)
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8

    # For CLAHE_TRIPLE mode: three different clip limits for the three channels
    clahe_triple_clips: tuple[float, float, float] = (1.0, 2.0, 3.0)

    # For PSEUDO_COLOR mode: morphological structuring element radius (px)
    morpho_se_radius: int = 15

    # --- Spatial preprocessing ---
    # Median filter kernel size for denoising (0 = skip)
    denoise_kernel: int = 3

    # Target size for the Mask R-CNN input (H, W). Images are letterboxed.
    # None = no resize (pass through at original resolution)
    target_size: Optional[tuple[int, int]] = (1024, 800)

    # Whether to normalize breast orientation (all face right)
    orient_breast: bool = True

    # Whether to crop tightly around the breast tissue
    tight_crop: bool = True

    # Whether to attempt basic pectoral muscle removal on MLO views
    remove_pectoral: bool = True


@dataclass
class DatasetConfig:
    """Dataset and training configuration."""

    # --- Paths ---
    data_root: str = "./data/cbis-ddsm"  # contains images/ and masks/
    train_csv: str = "./meta/cbis-ddsm/mass_case_description_train_set.csv"
    test_csv: str = "./meta/cbis-ddsm/mass_case_description_test_set.csv"

    # --- Preprocessing ---
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)

    # --- Classification target ---
    classification_target: ClassificationTarget = ClassificationTarget.PATHOLOGY

    # --- Augmentation (training only) ---
    augment: bool = True
    aug_hflip: bool = True
    aug_vflip: bool = True
    aug_rotation_degrees: float = 15.0
    aug_contrast_jitter: float = 0.15
    aug_gaussian_noise_std: float = 0.02

    # --- DataLoader ---
    batch_size: int = 2  # mammograms are large, keep small
    num_workers: int = 4
    pin_memory: bool = True
