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

sys.path.append('/tia/phd-data/u5552013/ImmunoDINO_datasets/PathoCellbench/Pathology-Foundation-Model-Benchmark/src')
from benchmark.dataset.lmdb_dataset import LMDBDataset


class PreExtractedCellDataset(Dataset):
    """
    Generic dataset for pre-extracted cell crops.
    Expects folder structure:
        root/
            ClassA/Images/*.png
            ClassB/Images/*.png
            ...
    """
    IMG_EXTENSIONS = (".png", ".jpg", ".jpeg")

    def __init__(self, root, transform=None, dataset_name="Dataset"):
        self.root = Path(root)
        self.transform = transform
        self.dataset_name = dataset_name
        self.samples = []
        self.class_counts = Counter()

        # Load from all class subdirectories
        for class_folder in sorted(self.root.iterdir()):
            if not class_folder.is_dir():
                continue

            images_dir = class_folder / "Images"
            if not images_dir.exists():
                continue

            for img_path in sorted(images_dir.iterdir()):
                if img_path.suffix.lower() in self.IMG_EXTENSIONS:
                    self.samples.append(img_path)
                    self.class_counts[class_folder.name] += 1

        if len(self.samples) == 0:
            raise RuntimeError(f"No crops found in {root}")

        print(f"\n[{self.dataset_name}] Loaded {len(self.samples)} cell crops:")
        for cls, count in sorted(self.class_counts.items()):
            print(f"  {cls}: {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, ()


class MIDOGCellDataset(Dataset):
    """
    MIDOG CSV-based dataset for H&E cell crops.
    Returns (image, ()) for DINO training.
    """

    def __init__(self, csv_file, root_dir, transform=None, dataset_name="MIDOG"):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.dataset_name = dataset_name

        # Label mapping (only for stats, not used by DINO)
        self.label_map = {'Atypical': 0, 'Normal': 1}
        self.df['label'] = self.df['final_label'].map(self.label_map)

        # Build image paths
        self.df['image_path'] = (
            self.root_dir + '/' +
            self.df['final_label'] + '/' +
            self.df['dataset_uid']
        )

        # Store samples
        self.samples = self.df['image_path'].tolist()
        self.labels = self.df['label'].tolist()
        self.groups = self.df['slide'].tolist()

        # Stats
        self.class_counts = Counter(self.df['final_label'])

        if len(self.samples) == 0:
            raise RuntimeError("No MIDOG samples found")

        print(f"\n[{self.dataset_name}] Loaded {len(self.samples)} samples:")
        for cls, count in sorted(self.class_counts.items()):
            print(f"  {cls}: {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, ()  # IMPORTANT for DINO


# ---------------------------------------------------------------------------
# TCGA cell-crop datasets
#
# The original module defined near-identical classes (TCGACellCropDataset,
# TCGACellCropDataset1, Balanced_TCGACellCropDataset, ...1, ...2) that only
# differed in which ZARR_ROOT / METADATA_CSV they pointed at. That logic is
# now centralized in two base classes below; the original names are kept as
# thin subclasses (just setting the paths) so existing imports/usages don't
# need to change.
# ---------------------------------------------------------------------------

_TCGA_CLASS_NAMES = {
    0: "noLabel",
    1: "Neoplastic",
    2: "Inflammatory",
    3: "ConnectiveSoftTissue",
    4: "Necroses",
    5: "nonNeoplastic",
}
_TCGA_PATCH_SIZE = 64


class TCGACellCropDatasetBase(Dataset):
    """
    Loads 64x64 cell-centric crops from Zarr stores produced by
    the Stage 2 extraction pipeline.

    Each Zarr store lives at:
        <zarr_root>/<slide_id>.zarr/
            images  -> (N, 64, 64, 3)  uint8
            types   -> (N,)            int64

    Subclasses just set ZARR_ROOT as a default; it can also be passed
    explicitly to __init__.

    Args:
        zarr_root:  path to the directory containing all .zarr stores
                    (falls back to the subclass's ZARR_ROOT)
        slides:     optional list of slide_id strings to include;
                    if None, all .zarr dirs under zarr_root are used
        transform:  optional torchvision transform applied to PIL patches
    """

    ZARR_ROOT = None
    PATCH_SIZE = _TCGA_PATCH_SIZE
    CLASS_NAMES = _TCGA_CLASS_NAMES

    def __init__(self, zarr_root=None, slides=None, transform=None):
        self.zarr_root = Path(zarr_root or self.ZARR_ROOT)
        self.transform = transform
        self.patch_size = self.PATCH_SIZE

        zarr_paths = self._discover_zarr_paths(slides)
        if len(zarr_paths) == 0:
            raise RuntimeError(f"No .zarr stores found under {self.zarr_root}")

        self.stores = []
        self.samples = []

        for zp in zarr_paths:
            root = zarr.open(str(zp), mode="r")
            img_ds = root["images"]
            type_ds = root["types"]
            store_idx = len(self.stores)
            self.stores.append({"images": img_ds, "types": type_ds, "name": zp.stem})

            n = img_ds.shape[0]
            self.samples.extend((store_idx, i) for i in range(n))

        print(
            f"[{type(self).__name__}] {len(zarr_paths)} slides, "
            f"{len(self.samples)} cells total"
        )

    def _discover_zarr_paths(self, slides):
        if slides is not None:
            zarr_paths = [self.zarr_root / f"{s}.zarr" for s in slides]
            return [p for p in zarr_paths if p.exists()]
        return sorted(self.zarr_root.glob("*.zarr"))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        store_idx, cell_idx = self.samples[idx]
        store = self.stores[store_idx]

        patch = np.array(store["images"][cell_idx])  # (64, 64, 3) uint8
        patch = Image.fromarray(patch)

        if self.transform is not None:
            patch = self.transform(patch)

        return patch, ()  # DINO ignores labels


class TCGACellCropDataset1(TCGACellCropDatasetBase):
    ZARR_ROOT = "/tia/lab-share/esha/TCGA_data/tcga-zarr1"


class TCGACellCropDataset(TCGACellCropDatasetBase):
    ZARR_ROOT = "/tia/lab-share/esha/TCGA_data/tcga-zarr2"

class TCGACellCropDataset2(TCGACellCropDatasetBase):
    ZARR_ROOT = "/tia/lab-share/esha/TCGA_data/tcga-zarr3"
class BalancedTCGACellCropDatasetBase(Dataset):
    """
    Balanced TCGA Zarr dataset with an internal metadata CSV mapping
    slide -> cancer type. Ensures equal sampling across cancer types
    (e.g. LUAD, GBMLGG) by capping every type to the smallest type's
    cell count (or to max_per_type, whichever is smaller).

    Subclasses just set ZARR_ROOT / METADATA_CSV as defaults; both can
    also be passed explicitly to __init__.
    """

    ZARR_ROOT = None
    METADATA_CSV = None
    PATCH_SIZE = _TCGA_PATCH_SIZE
    CLASS_NAMES = _TCGA_CLASS_NAMES

    def __init__(
        self,
        zarr_root=None,
        metadata_csv=None,
        slides=None,
        transform=None,
        max_per_type=None,
        seed=42,
    ):
        self.zarr_root = Path(zarr_root or self.ZARR_ROOT)
        self.transform = transform
        self.max_per_type = max_per_type
        self.seed = seed
        random.seed(seed)
        meta = pd.read_csv(metadata_csv or self.METADATA_CSV)
        slide_to_type = dict(zip(meta["slide"], meta["type"]))
        if slides is not None:
            zarr_paths = [self.zarr_root / f"{s}.zarr" for s in slides]
            zarr_paths = [p for p in zarr_paths if p.exists()]
        else:
            zarr_paths = sorted(self.zarr_root.glob("*.zarr"))

        if len(zarr_paths) == 0:
            raise RuntimeError(f"No .zarr stores found under {self.zarr_root}")

        self.stores = []
        type_to_samples = defaultdict(list)  # type -> list of (store_idx, cell_idx)

        for zp in zarr_paths:
            slide_id = zp.stem
            cancer_type = slide_to_type.get(slide_id, None)

            # skip slides not in CSV
            if cancer_type is None:
                continue

            root = zarr.open(str(zp), mode="r")
            img_ds = root["images"]
            type_ds = root["types"]

            store_idx = len(self.stores)
            self.stores.append({
                "images": img_ds,
                "types": type_ds,
                "slide": slide_id,
                "type": cancer_type,
            })

            n = img_ds.shape[0]
            type_to_samples[cancer_type].extend((store_idx, i) for i in range(n))

        if len(type_to_samples) == 0:
            raise RuntimeError("No valid samples found after CSV filtering")
        all_types = list(type_to_samples.keys())
        min_count = min(len(v) for v in type_to_samples.values())

        if self.max_per_type is not None:
            min_count = min(min_count, self.max_per_type)

        self.samples = []
        for t in all_types:
            samples_t = type_to_samples[t]
            random.shuffle(samples_t)
            self.samples.extend(samples_t[:min_count])

        random.shuffle(self.samples)

        print(
            f"[{type(self).__name__}] "
            f"Types: {len(all_types)} | "
            f"Samples/type: {min_count} | "
            f"Total samples: {len(self.samples)}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        store_idx, cell_idx = self.samples[idx]
        store = self.stores[store_idx]

        patch = np.array(store["images"][cell_idx])  # (64,64,3)
        patch = Image.fromarray(patch)

        if self.transform is not None:
            patch = self.transform(patch)

        return patch, ()


class Balanced_TCGACellCropDataset(BalancedTCGACellCropDatasetBase):
    ZARR_ROOT = "/tia/lab-share/esha/TCGA_data/tcga-zarr1"
    METADATA_CSV = "/tia/phd-data/u5552013/TCGA_data/TCGA_slides.csv"


class Balanced_TCGACellCropDataset1(BalancedTCGACellCropDatasetBase):
    ZARR_ROOT = "/tia/lab-share/esha/TCGA_data/tcga-zarr2"
    METADATA_CSV = "/tia/phd-data/u5552013/TCGA_data/TCGA_new_5slides_per_type.csv"


class Balanced_TCGACellCropDataset2(BalancedTCGACellCropDatasetBase):
    ZARR_ROOT = "/tia/lab-share/esha/TCGA_data/tcga-zarr3"
    METADATA_CSV = "/tia/phd-data/u5552013/TCGA_data/TCGA_new_5slides_per_type1.csv"
    
class Balanced_TCGACellCropDataset3(BalancedTCGACellCropDatasetBase):
    ZARR_ROOT = "/tia/lab-share/esha/TCGA_data/tcga-zarr4"
    METADATA_CSV = "/tia/phd-data/u5552013/TCGA_data/TCGA_new_5slides_per_type2.csv"

class PathoCellbenchDataset(Dataset):
    """
    Self-contained dataset for 64x64 cell-centric patches from
    PathoCellbench LMDB tiles, including all cells,
    supporting train/valid/test splits.
    """
    BASE_PATH = '/tia/phd-data/u5552013/ImmunoDINO_datasets/PathoCellbench/data'
    DS_PATH = os.path.join(BASE_PATH, 'phenocell')
    DATASET_PATH = os.path.join(DS_PATH, 'phenocell_lmdb')
    SPLIT_CSV = os.path.join(DS_PATH, 'splits', 'phenocell_dataset_split.csv')
    LABEL_DICT_JSON = os.path.join(DS_PATH, 'label_dict.json')

    PATCH_SIZE = 64

    def __init__(self, splits=("train", "valid", "test"), transform=None):
        """
        Args:
            splits: tuple of splits to include ("train", "valid", "test")
            transform: optional transform to apply to PIL patches
        """
        self.transform = transform
        self.patch_size = self.PATCH_SIZE
        self.splits = splits

        # Load label dict
        with open(self.LABEL_DICT_JSON) as f:
            label_dict = json.load(f)
        self.label_dict = {int(k): v for k, v in label_dict.items() if int(k) != 0}  # remove background

        # Load split CSV
        split_df = pd.read_csv(self.SPLIT_CSV)

        # Store LMDB datasets per split
        self.lmdb_datasets = {}
        for split in splits:
            include_fovs = split_df[split_df["train_test_val_split"] == split]["sample_name"].tolist()
            self.lmdb_datasets[split] = LMDBDataset(
                path=self.DATASET_PATH,
                include_sample_names=include_fovs
            )

        # Precompute all cell instances across splits
        self.samples = []
        for split in splits:
            dataset = self.lmdb_datasets[split]
            for idx, el in enumerate(dataset):
                inst_mask = np.squeeze(el["instance_mask"])
                sem_mask = np.squeeze(el["semantic_mask"])

                inst_ids = np.unique(inst_mask)
                inst_ids = inst_ids[inst_ids > 0]

                for inst_id in inst_ids:
                    cell_mask = (inst_mask == inst_id)

                    sem_vals = sem_mask[cell_mask]
                    sem_vals = sem_vals[sem_vals > 0]
                    if len(sem_vals) == 0:
                        continue

                    class_idx = int(np.bincount(sem_vals).argmax())
                    if class_idx not in self.label_dict:
                        continue

                    self.samples.append((split, idx, inst_id))

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def crop_with_padding(arr, cy, cx, patch_size):
        half = patch_size // 2
        if arr.ndim == 3:
            _, H, W = arr.shape
        else:
            H, W = arr.shape
        y1, y2 = cy - half, cy + half
        x1, x2 = cx - half, cx + half
        y1_src, y2_src = max(0, y1), min(H, y2)
        x1_src, x2_src = max(0, x1), min(W, x2)

        if arr.ndim == 3:
            crop = arr[:, y1_src:y2_src, x1_src:x2_src]
            out = np.zeros((arr.shape[0], patch_size, patch_size), dtype=arr.dtype)
            y_off = half - (cy - y1_src)
            x_off = half - (cx - x1_src)
            out[:, y_off:y_off+crop.shape[1], x_off:x_off+crop.shape[2]] = crop
        else:
            crop = arr[y1_src:y2_src, x1_src:x2_src]
            out = np.zeros((patch_size, patch_size), dtype=arr.dtype)
            y_off = half - (cy - y1_src)
            x_off = half - (cx - x1_src)
            out[y_off:y_off+crop.shape[0], x_off:x_off+crop.shape[1]] = crop
        return out

    @staticmethod
    def normalize(img):
        img = img.astype(np.float32)
        if img.max() == 0:
            return img
        return img / img.max()

    def __getitem__(self, idx):
        split, tile_idx, inst_id = self.samples[idx]
        el = self.lmdb_datasets[split][tile_idx]
        img = el["image"]  # [C,H,W]
        inst_mask = np.squeeze(el["instance_mask"])

        cell_mask = (inst_mask == inst_id)
        ys, xs = np.where(cell_mask)
        cy, cx = int(np.mean(ys)), int(np.mean(xs))

        patch = self.crop_with_padding(img, cy, cx, self.patch_size)
        patch = np.transpose(self.normalize(patch), (1, 2, 0))  # HWC
        patch = (patch * 255).astype(np.uint8)
        patch = Image.fromarray(patch)

        if self.transform:
            patch = self.transform(patch)

        return patch, ()  # DINO ignores labels


class ImmunoctoHEDataset(Dataset):
    """
    Loads HE images from:
      root/CRCxx/HE/*.png (or jpg, tif)
    Masks are ignored.
    """

    IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []

        for crc_dir in sorted(self.root.iterdir()):
            he_dir = crc_dir / "HE"
            if not he_dir.is_dir():
                continue
            for img_path in he_dir.iterdir():
                if img_path.suffix.lower() in self.IMG_EXTENSIONS:
                    self.samples.append(img_path)

        if len(self.samples) == 0:
            raise RuntimeError(f"No HE images found under {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, ()  # target must be empty tuple for DINO


class CONICCellDataset(Dataset):
    """
    Loads images directly from the specified CONIC class folders.
    Returns (image, ()) for DINO.
    """

    IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    _CLASSES = ("Connective", "Eosinophil", "Epithelial", "Lymphocyte", "Neutrophil", "Plasma")
    _BASE = "/tia/phd-data/u5552013/ImmunoDINO_datasets/CONIC"

    def __init__(self, transform=None):
        self.transform = transform
        self.samples = []

        image_dirs = [
            Path(self._BASE) / split / cls / "Images"
            for split in ("train", "test")
            for cls in self._CLASSES
        ]

        for img_dir in image_dirs:
            if not img_dir.is_dir():
                print(f"Warning: {img_dir} does not exist, skipping.")
                continue

            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() in self.IMG_EXTENSIONS:
                    self.samples.append(img_path)

        if len(self.samples) == 0:
            raise RuntimeError("No images found in the provided CONIC folders.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, ()


class PanNukeCellDataset(Dataset):
    """
    Loads PanNuke cell patch images from hardcoded folders.

    Returns:
      img, ()   # for DINO
    """

    IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    _BASE = "/tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke"
    _FOLDS = ("Fold1", "Fold2", "Fold3")

    def __init__(self, transform=None):
        self.transform = transform
        self.samples = []
        self.labels = []
        self.sample_info = []  # (fold, class, path)

        self.class_names = [
            "Connective",
            "Dead",
            "Epithelial",
            "Inflammatory",
            "Neoplastic",
        ]

        self.class_to_idx = {cls: i for i, cls in enumerate(self.class_names)}

        self.fold_class_counts = {
            fold: {cls: 0 for cls in self.class_names} for fold in self._FOLDS
        }

        image_dirs = [
            (fold, cls, Path(self._BASE) / f"{fold}_cell_patches" / cls)
            for fold in self._FOLDS
            for cls in self.class_names
        ]

        for fold, cls_name, img_dir in image_dirs:
            if not img_dir.is_dir():
                print(f"Warning: {img_dir} does not exist, skipping.")
                continue

            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() in self.IMG_EXTENSIONS:
                    self.samples.append(img_path)
                    self.labels.append(self.class_to_idx[cls_name])
                    self.sample_info.append((fold, cls_name, img_path))
                    self.fold_class_counts[fold][cls_name] += 1

        if len(self.samples) == 0:
            raise RuntimeError("No images found in the provided PanNuke folders.")

        # ===== SUMMARY =====
        print("\n[PanNukeCellDataset] Dataset summary:")
        grand_total = 0

        for fold in self._FOLDS:
            print(f"\n  {fold}:")
            fold_total = 0
            for cls_name in self.class_names:
                count = self.fold_class_counts[fold][cls_name]
                print(f"    {cls_name}: {count} images")
                fold_total += count
            print(f"    Total in {fold}: {fold_total}")
            grand_total += fold_total

        print(f"\n  GRAND TOTAL: {grand_total} images\n")

        print("First 5 sample paths:")
        for p in self.samples[:5]:
            print(" ", p)
        print()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, ()