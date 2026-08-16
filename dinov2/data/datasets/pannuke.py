# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#
# --- Added for prototype-conditioned self-distillation on PanNuke ---
"""
PanNuke dataset for dinov2, following the same ExtendedVisionDataset pattern
as CHAMMI_CP (dinov2/data/datasets/cell_dino/chammi_cp.py) exactly, so it
plugs into the existing make_dataset() / dataset-string loading machinery
with no other changes needed to the data pipeline.

Expects the on-disk format produced by preprocess_pannuke_for_dinov2.py:
    <root>/pannuke_metadata.csv   -- columns: file_path,label,fold
    <root>/fold1/*.png, <root>/fold2/*.png, <root>/fold3/*.png
"""
import csv
import os
from enum import Enum
from typing import Any, Callable, Optional, Union

import numpy as np

from .extended import ExtendedVisionDataset
from .decoders import DecoderType

METADATA_FILE = "/tia/phd-data/u5552013/ImmunoDINO_baselines_comparison/continual_self_distillation/PanNuke_dinov2_format/pannuke_metadata.csv"


class _Split(Enum):
    FOLD1 = "1"
    FOLD2 = "2"
    FOLD3 = "3"
    TRAIN_12 = "1,2"  # convenience: folds 1+2 combined, for standard PanNuke train/val protocol


def _load_file_paths_and_targets(root: str, split: "PanNuke.Split"):
    allowed_folds = set(split.value.split(","))
    image_paths, labels = [], []
    with open(os.path.join(root, METADATA_FILE)) as metadata:
        reader = csv.DictReader(metadata)
        for row in reader:
            if row["fold"] in allowed_folds:
                image_paths.append(row["file_path"])
                labels.append(int(row["label"]))
    return image_paths, labels


class PanNuke(ExtendedVisionDataset):
    """PanNuke nucleus-crop dataset, following the CHAMMI_CP integration
    pattern. Labels are 1-5 (Neoplastic, Inflammatory, Connective, Dead,
    Epithelial), matching data_pannuke.py's convention elsewhere in this
    project (0 reserved, unused since crops are never background-centered).
    """

    Split = Union[_Split]

    def __init__(
        self,
        *,
        split: "PanNuke.Split",
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_decoder_type: DecoderType = DecoderType.ImageDataDecoder,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            root, transforms, transform, target_transform,
            image_decoder_type=image_decoder_type, **kwargs,
        )
        self.split = split
        self.root = root
        self._image_paths, self._targets = _load_file_paths_and_targets(root, split)

    def get_image_relpath(self, index: int) -> str:
        return self._image_paths[index]

    def get_image_data(self, index: int) -> bytes:
        image_full_path = os.path.join(self.root, self.get_image_relpath(index))
        with open(image_full_path, mode="rb") as f:
            return f.read()

    def get_target(self, index: int) -> Any:
        return self._targets[index]

    def get_targets(self) -> np.ndarray:
        return np.array(self._targets)

    def __len__(self) -> int:
        return len(self._image_paths)