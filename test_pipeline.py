"""
Quick smoke test for the mammogram pipeline.

Run from the project root (where config.py / preprocess.py / dataset.py live):
    python test_pipeline.py --data_root ./cbis_ddsm

Checks:
  1. Preprocessing on a synthetic image (no data needed)
  2. Filename parsing regexes
  3. If --data_root is provided and exists: sample index building,
     dataset __getitem__, and a single batch from the dataloader
"""

import argparse
import os
import sys
import traceback

import cv2
import numpy as np

from config import DatasetConfig, PreprocessConfig, Representation
from preprocess import MammogramPreprocessor


def test_preprocessing():
    """Test both representations on a synthetic mammogram."""
    print("=" * 60)
    print("1. Preprocessing (synthetic image)")
    print("=" * 60)

    # Fake mammogram: bright blob (breast) with a brighter spot (mass)
    fake = np.zeros((2000, 1500), dtype=np.uint8)
    fake[200:1800, 100:1200] = 120
    fake[800:1000, 500:700] = 200

    fake_mask = np.zeros((2000, 1500), dtype=np.uint8)
    fake_mask[800:1000, 500:700] = 255

    for rep in Representation:
        cfg = PreprocessConfig(target_size=(512, 400), representation=rep)
        proc = MammogramPreprocessor(cfg)

        for view in ("CC", "MLO"):
            result, tx = proc(fake, view=view)
            mask_out = proc.transform_mask(fake_mask, tx)

            assert result.ndim == 3 and result.shape[2] == 3, \
                f"Expected (H,W,3), got {result.shape}"
            assert result.shape[:2] == (512, 400), \
                f"Expected (512,400), got {result.shape[:2]}"
            assert -1e-6 <= result.min() and result.max() <= 1.0 + 1e-6, \
                f"Range [{result.min()}, {result.max()}] outside [0,1]"
            assert mask_out.shape == (512, 400), \
                f"Mask shape {mask_out.shape} != image spatial dims"
            assert (mask_out > 0).sum() > 0, \
                "Mask is empty after transform"

            print(f"  {rep.value:15s} | {view} | "
                  f"img {result.shape} [{result.min():.2f}-{result.max():.2f}] | "
                  f"mask nonzero={int((mask_out > 0).sum()):>6d}  ✓")

    # Edge case: no-resize mode
    cfg_noresize = PreprocessConfig(target_size=None)
    proc_nr = MammogramPreprocessor(cfg_noresize)
    result_nr, tx_nr = proc_nr(fake, view="CC")
    assert result_nr.ndim == 3, "No-resize output should still be 3-channel"
    print(f"  {'(no resize)':15s} | CC | img {result_nr.shape}  ✓")

    print()


def test_filename_parsing():
    """Test the regex patterns for image and mask filenames."""
    print("=" * 60)
    print("2. Filename parsing")
    print("=" * 60)

    from dataset import _parse_image_filename, _parse_mask_filename

    # Image filenames
    good_imgs = [
        ("P_00001_LEFT_CC_density_3.png",
         {"patient_id": "P_00001", "side": "LEFT", "view": "CC", "density": 3}),
        ("P_01234_RIGHT_MLO_density_1.png",
         {"patient_id": "P_01234", "side": "RIGHT", "view": "MLO", "density": 1}),
    ]
    for fname, expected in good_imgs:
        parsed = _parse_image_filename(fname)
        assert parsed is not None, f"Failed to parse: {fname}"
        for k, v in expected.items():
            assert parsed[k] == v, f"{fname}: {k} = {parsed[k]}, expected {v}"
        print(f"  img  {fname:45s} ✓")

    # Mask filenames
    good_masks = [
        ("P_00001_LEFT_CC_density_3_mass_0.png",
         {"patient_id": "P_00001", "side": "LEFT", "view": "CC",
          "density": 3, "abnormality_type": "mass", "mask_index": 0}),
        ("P_00500_RIGHT_MLO_density_2_calc_2.png",
         {"patient_id": "P_00500", "side": "RIGHT", "view": "MLO",
          "density": 2, "abnormality_type": "calc", "mask_index": 2}),
    ]
    for fname, expected in good_masks:
        parsed = _parse_mask_filename(fname)
        assert parsed is not None, f"Failed to parse: {fname}"
        for k, v in expected.items():
            assert parsed[k] == v, f"{fname}: {k} = {parsed[k]}, expected {v}"
        print(f"  mask {fname:45s} ✓")

    # Should return None for non-matching names
    for bad in ["readme.txt", "P_00001_LEFT.png", "random_file.png"]:
        assert _parse_image_filename(bad) is None
        assert _parse_mask_filename(bad) is None
        print(f"  skip {bad:45s} ✓")

    print()


def test_real_data(cfg: "DatasetConfig"):
    """Test sample index building, dataset, and dataloader on real data."""
    print("=" * 60)
    print("3. Real data")
    print("=" * 60)

    import torch
    from dataset import build_sample_index, CBISDDSMDataset, build_dataloaders

    # --- Sample index ---
    train_samples = build_sample_index(os.path.join(cfg.data_root, "train"), cfg.train_csv)
    test_samples = build_sample_index(os.path.join(cfg.data_root, "test"), cfg.test_csv)
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Test samples:  {len(test_samples)}")
    assert len(train_samples) > 0, "No training samples found"
    assert len(test_samples) > 0, "No test samples found"

    # Check no patient overlap
    train_pids = {s["patient_id"] for s in train_samples}
    test_pids = {s["patient_id"] for s in test_samples}
    overlap = train_pids & test_pids
    assert len(overlap) == 0, f"Patient leakage! Overlap: {overlap}"
    print(f"  Patient overlap: 0  ✓")

    # --- Single sample (use smaller target_size + no aug for speed) ---
    from dataclasses import replace
    test_cfg = replace(
        cfg,
        preprocess=replace(cfg.preprocess, target_size=(512, 400)),
        augment=False,
        num_workers=0,
    )
    ds = CBISDDSMDataset(train_samples[:5], test_cfg, train=False)
    img, tgt = ds[0]

    print(f"\n  Sample 0:")
    print(f"    Image:     {img.shape}, dtype={img.dtype}, "
          f"range=[{img.min():.2f}, {img.max():.2f}]")
    print(f"    Boxes:     {tgt['boxes'].shape}")
    print(f"    Masks:     {tgt['masks'].shape}")
    print(f"    Labels:    {tgt['labels'].tolist()}")
    print(f"    Pathology: {tgt['pathology']}")
    print(f"    BI-RADS:   {tgt['assessment'].tolist()}")

    assert img.shape[0] == 3, "Image should be (3, H, W)"
    assert img.dtype == torch.float32
    n_masses = tgt["boxes"].shape[0]
    if n_masses > 0:
        assert tgt["masks"].shape[0] == n_masses
        assert tgt["masks"].shape[1:] == img.shape[1:]
        assert (tgt["labels"] == 1).all(), "All labels should be 1 (mass)"
        # Boxes should be within image bounds
        h, w = img.shape[1], img.shape[2]
        assert (tgt["boxes"][:, 0] >= 0).all() and (tgt["boxes"][:, 2] <= w).all()
        assert (tgt["boxes"][:, 1] >= 0).all() and (tgt["boxes"][:, 3] <= h).all()
    print(f"    Consistency checks  ✓")

    # --- DataLoader batch ---
    from torch.utils.data import DataLoader
    from dataset import _collate_fn

    small_ds = CBISDDSMDataset(train_samples[:4], test_cfg, train=True)
    loader = DataLoader(small_ds, batch_size=2, collate_fn=_collate_fn, num_workers=0)
    images, targets = next(iter(loader))

    print(f"\n  Batch:")
    print(f"    Batch size: {len(images)}")
    for i, (im, tg) in enumerate(zip(images, targets)):
        print(f"    [{i}] img {im.shape}, {tg['boxes'].shape[0]} masses")
    print(f"    Collation  ✓")

    print()


def main():
    parser = argparse.ArgumentParser(description="Smoke test the mammogram pipeline")
    parser.add_argument("--data_root", default=None,
                        help="Path to CBIS-DDSM root (with train/ and test/ subdirs)")
    parser.add_argument("--train_csv", default=None)
    parser.add_argument("--test_csv", default=None)
    args = parser.parse_args()

    # Build config — use defaults from DatasetConfig, override with CLI args
    cfg_kwargs = {}
    if args.data_root is not None:
        cfg_kwargs["data_root"] = args.data_root
    if args.train_csv is not None:
        cfg_kwargs["train_csv"] = args.train_csv
    if args.test_csv is not None:
        cfg_kwargs["test_csv"] = args.test_csv
    cfg = DatasetConfig(**cfg_kwargs)

    passed, failed = 0, 0

    # Tests that don't need real data
    for test_fn in [test_preprocessing, test_filename_parsing]:
        try:
            test_fn()
            passed += 1
        except Exception:
            failed += 1
            traceback.print_exc()
            print()

    # Test with real data (if available)
    train_images_dir = os.path.join(cfg.data_root, "train", "images")
    if os.path.isdir(train_images_dir) and len(os.listdir(train_images_dir)) > 0:
        try:
            test_real_data(cfg)
            passed += 1
        except Exception:
            failed += 1
            traceback.print_exc()
            print()
    else:
        print("=" * 60)
        print(f"3. Real data — SKIPPED ('{train_images_dir}' not found)")
        print(f"   data_root = '{cfg.data_root}'")
        print("   Run again once the download finishes.")
        print("=" * 60)
        print()

    # Summary
    print("=" * 60)
    total = passed + failed
    if failed == 0:
        print(f"All {passed}/{total} test groups passed ✓")
    else:
        print(f"{failed}/{total} test groups FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
