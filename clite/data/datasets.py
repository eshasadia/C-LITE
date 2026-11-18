# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#
# Dataset classes used by train_uni2_distill.py, split out into their own
# module so they can be imported independently (e.g. from other scripts,
# or reused by the distillation stage without dragging in training code).

import os
import sys
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from PIL import Image
from torch.utils.data import Dataset

import random
t


class TCGACellCropDataset1(Dataset):
    """
    Loads 64×64 cell-centric crops from Zarr stores produced by
    the extraction pipeline.

    Each Zarr store lives at:
        <zarr_root>/<slide_id>.zarr/
            images  → (N, 64, 64, 3)  uint8
            types   → (N,)            int64

    Args:
        zarr_root:  path to the directory containing all .zarr stores
        slides:     optional list of slide_id 
        transform:  optional torchvision transform applied to PIL patches
    """

    ZARR_ROOT = ""
    PATCH_SIZE = 64

    def __init__(self, zarr_root=None, slides=None, transform=None):
        self.zarr_root = Path(zarr_root or self.ZARR_ROOT)
        self.transform = transform
        self.patch_size = self.PATCH_SIZE

        # Discover Zarr stores
        if slides is not None:
            zarr_paths = [self.zarr_root / f"{s}.zarr" for s in slides]
            zarr_paths = [p for p in zarr_paths if p.exists()]
        else:
            zarr_paths = sorted(self.zarr_root.glob("*.zarr"))

        if len(zarr_paths) == 0:
            raise RuntimeError(f"No .zarr stores found under {self.zarr_root}")

        # Open all stores and build a flat index: (store_idx, cell_idx)
        self.stores = []
        self.samples = []

        for zp in zarr_paths:
            root = zarr.open(str(zp), mode="r")
            img_ds = root["images"]
            type_ds = root["types"]
            store_idx = len(self.stores)
            self.stores.append({"images": img_ds, "types": type_ds, "name": zp.stem})

            n = img_ds.shape[0]
            for i in range(n):
                self.samples.append((store_idx, i))

        print(
            f"[TCGACellCropDataset] {len(zarr_paths)} slides, "
            f"{len(self.samples)} cells total"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        store_idx, cell_idx = self.samples[idx]
        store = self.stores[store_idx]

        patch = np.array(store["images"][cell_idx])       # (64, 64, 3) uint8
        patch = Image.fromarray(patch)

        if self.transform is not None:
            patch = self.transform(patch)

        return patch, ()  # DINO ignores labels


