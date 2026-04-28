"""
PyTorch Dataset for the Balloon dataset (VIA polygon format).

Outputs the same Mask R-CNN compatible target dicts as CBISDDSMDataset:
    {
        "boxes":    FloatTensor[N, 4]  (x1, y1, x2, y2),
        "labels":   Int64Tensor[N]     (1 = balloon),
        "masks":    UInt8Tensor[N, H, W],
        "image_id": Int64Tensor[1],
        "area":     FloatTensor[N],
        "iscrowd":  Int64Tensor[N],
    }
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class BalloonDataset(Dataset):
    def __init__(self, image_dir: str, annotation_json: str, target_size: tuple[int, int] = (512, 512)):
        """
        Args:
            image_dir:        path to folder containing the jpg images
            annotation_json:  path to the VIA JSON file (train.json / test.json)
            target_size:      (H, W) to resize images and masks to
        """
        self.image_dir = image_dir
        self.target_size = target_size

        with open(annotation_json) as f:
            via = json.load(f)

        # Keep only entries that have at least one region
        self.samples = [v for v in via.values() if v["regions"]]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        entry = self.samples[idx]
        image_path = os.path.join(self.image_dir, entry["filename"])

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        # Resize image
        image = cv2.resize(image, (self.target_size[1], self.target_size[0]))

        scale_x = self.target_size[1] / orig_w
        scale_y = self.target_size[0] / orig_h

        boxes, binary_masks, areas = [], [], []

        for region in entry["regions"].values():
            shape = region["shape_attributes"]
            if shape["name"] != "polygon":
                continue

            xs = np.array(shape["all_points_x"], dtype=np.float32)
            ys = np.array(shape["all_points_y"], dtype=np.float32)

            # Scale polygon points to target size
            xs = (xs * scale_x).astype(np.int32)
            ys = (ys * scale_y).astype(np.int32)

            # Rasterize polygon to binary mask
            mask = np.zeros((self.target_size[0], self.target_size[1]), dtype=np.uint8)
            cv2.fillPoly(mask, [np.stack([xs, ys], axis=1)], 1)

            if mask.sum() == 0:
                continue

            coords = cv2.findNonZero(mask)
            x, y, bw, bh = cv2.boundingRect(coords)
            if bw < 2 or bh < 2:
                continue

            boxes.append([x, y, x + bw, y + bh])
            binary_masks.append(mask)
            areas.append(float(mask.sum()))

        n = len(boxes)
        h, w = self.target_size

        if n == 0:
            target = {
                "boxes":    torch.zeros((0, 4), dtype=torch.float32),
                "labels":   torch.zeros((0,), dtype=torch.int64),
                "masks":    torch.zeros((0, h, w), dtype=torch.uint8),
                "image_id": torch.tensor([idx], dtype=torch.int64),
                "area":     torch.zeros((0,), dtype=torch.float32),
                "iscrowd":  torch.zeros((0,), dtype=torch.int64),
            }
        else:
            target = {
                "boxes":    torch.as_tensor(boxes, dtype=torch.float32),
                "labels":   torch.ones((n,), dtype=torch.int64),
                "masks":    torch.as_tensor(np.stack(binary_masks), dtype=torch.uint8),
                "image_id": torch.tensor([idx], dtype=torch.int64),
                "area":     torch.as_tensor(areas, dtype=torch.float32),
                "iscrowd":  torch.zeros((n,), dtype=torch.int64),
            }

        # (H, W, 3) float32 -> (3, H, W) normalized to [0, 1]
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        return image_tensor, target


def _collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
