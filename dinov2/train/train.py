
import argparse
import logging
import math
import os
import sys
import torch
import wandb
import json
import numpy as np
import pandas as pd
import sys
import tifffile
import cv2
from collections import Counter, defaultdict
import openslide
import zarr
from fvcore.common.checkpoint import PeriodicCheckpointer
from pathlib import Path
from PIL import Image
from pathlib import Path
from datasets import  PreExtractedCellDataset, MIDOGCellDataset,Balanced_TCGACellCropDataset, Balanced_TCGACellCropDataset1, Balanced_TCGACellCropDataset2, Balanced_TCGACellCropDataset3,ImmunoctoHEDataset, CONICCellDataset,PanNukeCellDataset
import math as _math
from functools import partial
sys.path.append(".")
import dinov2.distributed as distributed
from dinov2.data import (
    DataAugmentationDINO,
    DataAugmentationHEMA,
    MaskingGenerator,
    SamplerType,
    collate_data_and_cast,
    make_data_loader,
    make_dataset,)
import random
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger
from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.utils.config import setup
from dinov2.utils.utils import CosineScheduler, smooth_rank_measure
from torch.utils.data import Dataset
torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
logger = logging.getLogger("dinov2")
# torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
# torch.cuda.set_device(0)

sys.path.append('/tia/phd-data/u5552013/ImmunoDINO_datasets/PathoCellbench/Pathology-Foundation-Model-Benchmark/src')
from benchmark.dataset.lmdb_dataset import LMDBDataset

import os
import torch

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")
# device = torch.device("cuda:0")

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
        from collections import Counter
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


class TCGACellCropDataset(Dataset):
    """
    Loads 64×64 cell-centric crops from Zarr stores produced by
    the Stage 2 extraction pipeline.

    Each Zarr store lives at:
        <zarr_root>/<slide_id>.zarr/
            images  → (N, 64, 64, 3)  uint8
            types   → (N,)            int64

    Args:
        zarr_root:  path to the directory containing all .zarr stores
        slides:     optional list of slide_id strings to include;
                    if None, all .zarr dirs under zarr_root are used
        transform:  optional torchvision transform applied to PIL patches
    """

    ZARR_ROOT = "/tia/lab-share/esha/TCGA_data/tcga-zarr2"
    PATCH_SIZE = 64

    CLASS_NAMES = {
        0: "noLabel",
        1: "Neoplastic",
        2: "Inflammatory",
        3: "ConnectiveSoftTissue",
        4: "Necroses",
        5: "nonNeoplastic",
    }

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

class TCGACellCropDataset1(Dataset):
    """
    Loads 64×64 cell-centric crops from Zarr stores produced by
    the Stage 2 extraction pipeline.

    Each Zarr store lives at:
        <zarr_root>/<slide_id>.zarr/
            images  → (N, 64, 64, 3)  uint8
            types   → (N,)            int64

    Args:
        zarr_root:  path to the directory containing all .zarr stores
        slides:     optional list of slide_id strings to include;
                    if None, all .zarr dirs under zarr_root are used
        transform:  optional torchvision transform applied to PIL patches
    """

    ZARR_ROOT = "/tia/lab-share/esha/TCGA_data/tcga-zarr1"
    PATCH_SIZE = 64

    CLASS_NAMES = {
        0: "noLabel",
        1: "Neoplastic",
        2: "Inflammatory",
        3: "ConnectiveSoftTissue",
        4: "Necroses",
        5: "nonNeoplastic",
    }

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

class Balanced_TCGACellCropDataset1(Dataset):
    """
    Balanced TCGA Zarr dataset with internal metadata CSV.

    Ensures equal sampling across cancer types (e.g., LUAD, GBMLGG).
    """

    # -------------------------
    # Paths / config
    # -------------------------
    ZARR_ROOT = "/tia/web-public/tcga-zarr1"

    METADATA_CSV = "/tia/phd-data/u5552013/TCGA_data/TCGA_new_5slides_per_type.csv"

    PATCH_SIZE = 64

    CLASS_NAMES = {
        0: "noLabel",
        1: "Neoplastic",
        2: "Inflammatory",
        3: "ConnectiveSoftTissue",
        4: "Necroses",
        5: "nonNeoplastic",
    }

    def __init__(
        self,
        zarr_root=None,
        slides=None,
        transform=None,
        max_per_type=None,
        seed=42
    ):
        self.zarr_root = Path(zarr_root or self.ZARR_ROOT)
        self.transform = transform
        self.max_per_type = max_per_type
        self.seed = seed

        random.seed(seed)

        # -------------------------
        # Load metadata CSV (inside class)
        # -------------------------
        meta = pd.read_csv(self.METADATA_CSV)

        # slide → cancer type mapping
        slide_to_type = dict(zip(meta["slide"], meta["type"]))

        # -------------------------
        # Load Zarr stores
        # -------------------------
        if slides is not None:
            zarr_paths = [self.zarr_root / f"{s}.zarr" for s in slides]
            zarr_paths = [p for p in zarr_paths if p.exists()]
        else:
            zarr_paths = sorted(self.zarr_root.glob("*.zarr"))

        if len(zarr_paths) == 0:
            raise RuntimeError(f"No .zarr stores found under {self.zarr_root}")

        self.stores = []

        # type → list of (store_idx, cell_idx)
        type_to_samples = defaultdict(list)

        # -------------------------
        # Build index
        # -------------------------
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
                "type": cancer_type
            })

            n = img_ds.shape[0]

            for i in range(n):
                type_to_samples[cancer_type].append((store_idx, i))

        if len(type_to_samples) == 0:
            raise RuntimeError("No valid samples found after CSV filtering")

        # -------------------------
        # BALANCING STEP
        # -------------------------
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
            f"[Balanced Dataset] "
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


class Balanced_TCGACellCropDataset(Dataset):
    """
    Balanced TCGA Zarr dataset with internal metadata CSV.

    Ensures equal sampling across cancer types (e.g., LUAD, GBMLGG).
    """

    # -------------------------
    # Paths / config
    # -------------------------
    ZARR_ROOT = "/tia/web-public/tcga-zarr"

    METADATA_CSV = "/tia/phd-data/u5552013/TCGA_data/TCGA_slides.csv"

    PATCH_SIZE = 64

    CLASS_NAMES = {
        0: "noLabel",
        1: "Neoplastic",
        2: "Inflammatory",
        3: "ConnectiveSoftTissue",
        4: "Necroses",
        5: "nonNeoplastic",
    }

    def __init__(
        self,
        zarr_root=None,
        slides=None,
        transform=None,
        max_per_type=None,
        seed=42
    ):
        self.zarr_root = Path(zarr_root or self.ZARR_ROOT)
        self.transform = transform
        self.max_per_type = max_per_type
        self.seed = seed

        random.seed(seed)

        # -------------------------
        # Load metadata CSV (inside class)
        # -------------------------
        meta = pd.read_csv(self.METADATA_CSV)

        # slide → cancer type mapping
        slide_to_type = dict(zip(meta["slide"], meta["type"]))

        # -------------------------
        # Load Zarr stores
        # -------------------------
        if slides is not None:
            zarr_paths = [self.zarr_root / f"{s}.zarr" for s in slides]
            zarr_paths = [p for p in zarr_paths if p.exists()]
        else:
            zarr_paths = sorted(self.zarr_root.glob("*.zarr"))

        if len(zarr_paths) == 0:
            raise RuntimeError(f"No .zarr stores found under {self.zarr_root}")

        self.stores = []

        # type → list of (store_idx, cell_idx)
        type_to_samples = defaultdict(list)

        # -------------------------
        # Build index
        # -------------------------
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
                "type": cancer_type
            })

            n = img_ds.shape[0]

            for i in range(n):
                type_to_samples[cancer_type].append((store_idx, i))

        if len(type_to_samples) == 0:
            raise RuntimeError("No valid samples found after CSV filtering")

        # -------------------------
        # BALANCING STEP
        # -------------------------
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
            f"[Balanced Dataset] "
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
                    # if cell_mask.sum() < 10:
                    #     continue

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

    def __init__(self, transform=None):
        self.transform = transform
        self.samples = []

        image_dirs = [
            Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/CONIC/train/Connective/Images"),
            Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/CONIC/train/Eosinophil/Images"),
            Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/CONIC/train/Epithelial/Images"),
            Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/CONIC/train/Lymphocyte/Images"),
            Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/CONIC/train/Neutrophil/Images"),
            Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/CONIC/train/Plasma/Images"),
            Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/CONIC/test/Connective/Images"),
            Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/CONIC/test/Eosinophil/Images"),
            Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/CONIC/test/Epithelial/Images"),
            Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/CONIC/test/Lymphocyte/Images"),
            Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/CONIC/test/Neutrophil/Images"),
            Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/CONIC/test/Plasma/Images"),
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

        # hardcoded counts dictionary
        self.fold_class_counts = {
            "Fold1": {cls: 0 for cls in self.class_names},
            "Fold2": {cls: 0 for cls in self.class_names},
            "Fold3": {cls: 0 for cls in self.class_names},
        }

        image_dirs = [
            # ===== Fold1 =====
            ("Fold1", "Connective",   Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke/Fold1_cell_patches/Connective")),
            ("Fold1", "Dead",         Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke/Fold1_cell_patches/Dead")),
            ("Fold1", "Epithelial",   Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke/Fold1_cell_patches/Epithelial")),
            ("Fold1", "Inflammatory", Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke/Fold1_cell_patches/Inflammatory")),
            ("Fold1", "Neoplastic",   Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke/Fold1_cell_patches/Neoplastic")),

            # ===== Fold2 =====
            ("Fold2", "Connective",   Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke/Fold2_cell_patches/Connective")),
            ("Fold2", "Dead",         Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke/Fold2_cell_patches/Dead")),
            ("Fold2", "Epithelial",   Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke/Fold2_cell_patches/Epithelial")),
            ("Fold2", "Inflammatory", Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke/Fold2_cell_patches/Inflammatory")),
            ("Fold2", "Neoplastic",   Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke/Fold2_cell_patches/Neoplastic")),

            # ===== Fold3 =====
            ("Fold3", "Connective",   Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke/Fold3_cell_patches/Connective")),
            ("Fold3", "Dead",         Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke/Fold3_cell_patches/Dead")),
            ("Fold3", "Epithelial",   Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke/Fold3_cell_patches/Epithelial")),
            ("Fold3", "Inflammatory", Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke/Fold3_cell_patches/Inflammatory")),
            ("Fold3", "Neoplastic",   Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke/Fold3_cell_patches/Neoplastic")),
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

        for fold in ["Fold1", "Fold2", "Fold3"]:
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

def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument(
        "--config-file", default="./dinov2/configs/train/custom.yaml", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )
    parser.add_argument("--name", type=str, default="debug", help="Name of the run for logging")
    parser.add_argument("--local-rank", type=int, help="Variable for distributed computing.")

    return parser


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


def build_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=int(cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH),
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[
        : cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH
    ] = 0  # mimicking the original schedules

    logger.info("Schedulers ready.")

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier


def do_test(cfg, model, iteration):
    new_state_dict = model.teacher.state_dict()

    if distributed.is_main_process():
        iterstring = str(iteration)
        eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)
        os.makedirs(eval_dir, exist_ok=True)
        # save teacher checkpoint
        teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
        torch.save({"teacher": new_state_dict}, teacher_ckp_path)


def do_train(cfg, model, resume=False):
    model.train()
    inputs_dtype = torch.half
    fp16_scaler = model.fp16_scaler  # for mixed precision training

    # setup optimizer

    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)

    # checkpointer
    checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)
    if cfg.MODEL.WEIGHTS:
        ckpt = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume)
    elif resume and checkpointer.has_checkpoint():
        ckpt = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume)
    else:
        logger.info("No MODEL.WEIGHTS and no existing checkpoint in output_dir -- "
                     "starting fresh from the pretrained backbone loaded in SSLMetaArch.")
        ckpt = {}
    start_iter = ckpt.get("iteration", -1) + 1 if isinstance(ckpt, dict) else 0
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period=3 * OFFICIAL_EPOCH_LENGTH,
        max_iter=max_iter,
        max_to_keep=3,
    )

    # setup data preprocessing

    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )
    if cfg.data_transform == "default":
        data_transform = DataAugmentationDINO(
            cfg.crops.global_crops_scale,
            cfg.crops.local_crops_scale,
            cfg.crops.local_crops_number,
            global_crops_size=cfg.crops.global_crops_size,
            local_crops_size=cfg.crops.local_crops_size,
        )
    elif cfg.data_transform == "hema":
        data_transform = DataAugmentationHEMA(
            cfg.crops.global_crops_scale,
            cfg.crops.local_crops_scale,
            cfg.crops.local_crops_number,
            global_crops_size=cfg.crops.global_crops_size,
            local_crops_size=cfg.crops.local_crops_size,
        )

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )
    dataset1 = ImmunoctoHEDataset(
            root=cfg.train.dataset_path,
            transform=data_transform,)
    dataset2 = CONICCellDataset(transform=data_transform)
    dataset3 = PanNukeCellDataset(transform=data_transform)
    dataset5 = PreExtractedCellDataset(
        root="/tia/phd-data/u5552013/ImmunoDINO_datasets/PUMA",
        transform=data_transform,
        dataset_name="PUMA"
    )
    dataset6 = PreExtractedCellDataset(
        root="/tia/phd-data/u5552013/ImmunoDINO_datasets/Monkey",
        transform=data_transform,
        dataset_name="Monkey"
    )
    dataset7 = MIDOGCellDataset(
    csv_file="/tia/lab-share/esha/MIDOG/Classification/Final/seven.csv",
    root_dir="/tia/lab-share/esha/MIDOG/Classification/Final",
    transform=data_transform,
    dataset_name="MIDOG")
    dataset8 = Balanced_TCGACellCropDataset(transform=data_transform)
    dataset9 = Balanced_TCGACellCropDataset1(transform=data_transform)
    dataset10 = Balanced_TCGACellCropDataset2(transform=data_transform)
    dataset11 = Balanced_TCGACellCropDataset3(transform=data_transform)
    dataset =  torch.utils.data.ConcatDataset([dataset1, dataset2, dataset3, dataset5,dataset6, dataset7, dataset8,dataset9, dataset10, dataset11])
    # Print dataset sizes
    print("\n=== Dataset sizes before training ===")
    print(f"ImmunoctoHE: {len(dataset1)} samples")
    print(f"CONIC:       {len(dataset2)} samples")
    print(f"PanNuke:     {len(dataset3)} samples")
    print(f"PUMA:   {len(dataset5)} samples")
    print(f"MONKEY:   {len(dataset6)} samples")
    print(f"MIDOG:   {len(dataset7)} samples")
    print(f"TCGA:   {len(dataset8)} samples")
    print(f"TCGA1:   {len(dataset9)} samples")
    print(f"TCGA2:   {len(dataset10)} samples")
    print(f"TCGA3:   {len(dataset11)} samples")
    # + len(dataset4)
    total_samples = len(dataset1) + len(dataset2) + len(dataset3)  + len(dataset5) +len(dataset6) + len(dataset7)+len(dataset8)+len(dataset9)+len(dataset10)+len(dataset11)
    print(f"TOTAL samples: {total_samples}\n")
    sampler_type = SamplerType.DISTRIBUTED
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        seed=start_iter,  # TODO: Fix this -- cfg.train.seed
        sampler_type=sampler_type,
        sampler_advance=0,  # TODO (qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # training loop

    iteration = start_iter

    logger.info("Starting training from iteration {}".format(start_iter))
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    header = "Training"

    batch_collection = []
    total_tokens_collected = 0

    for data in metric_logger.log_every(
        data_loader,
        10,
        header,
        max_iter,
        start_iter,
    ):
        current_batch_size = (data["collated_global_crops"].shape[0] / 2) * torch.cuda.device_count()
        if iteration > max_iter:
            return

        # apply schedules

        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        mom = momentum_schedule[iteration]
        teacher_temp = teacher_temp_schedule[iteration]
        last_layer_lr = last_layer_lr_schedule[iteration]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # compute losses

        optimizer.zero_grad(set_to_none=True)
        loss_dict, class_tokens = model.forward_backward(data, teacher_temp=teacher_temp)

        if fp16_scaler is not None:
            if cfg.optim.clip_grad:
                fp16_scaler.unscale_(optimizer)
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            if cfg.optim.clip_grad:
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            optimizer.step()

        # perform teacher EMA update

        model.update_teacher(mom)

        # logging

        if distributed.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
        loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

        if math.isnan(sum(loss_dict_reduced.values())):
            logger.info("NaN detected")
            raise AssertionError
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(current_batch_size=current_batch_size)
        metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)

        wandb.log(
            {
                "lr": lr,
                "wd": wd,
                "mom": mom,
                "last_layer_lr": last_layer_lr,
                "current_batch_size": current_batch_size,
                "total_loss": losses_reduced,
                **loss_dict_reduced,
            }
        )

        # compute smooth rank measure

        desired_tokens = 2500
        tokens_needed = desired_tokens - total_tokens_collected
        batch_size = class_tokens.shape[0]  # Current batch size from class_tokens shape

        if tokens_needed > 0 and iteration % 1000 < (int(desired_tokens / batch_size) + 1):
            # If the whole batch can be added without exceeding 1000 tokens
            if batch_size <= tokens_needed:
                batch_collection.append(class_tokens.detach())
                total_tokens_collected += batch_size
            else:
                # Add only the required number of tokens from the batch
                batch_collection.append(class_tokens.detach()[:tokens_needed, :])
                total_tokens_collected += tokens_needed

            print(f"{tokens_needed} tokens needed, {total_tokens_collected }tokens collected")

        # Once desired_tokens are collected, process them
        if total_tokens_collected == desired_tokens:
            embedding_matrix = torch.cat(batch_collection, dim=0)
            smooth_rank = smooth_rank_measure(embedding_matrix)  # Assuming this function is defined elsewhere
            wandb.log({"smooth_rank": smooth_rank})
            # Reset for the next tokens
            batch_collection = []
            total_tokens_collected = 0

        if cfg.evaluation.eval_period_iterations > 0 and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0:
            do_test(cfg, model, f"training_{iteration}")
            torch.cuda.synchronize()
        periodic_checkpointer.step(iteration)

        iteration = iteration + 1
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    cfg = setup(args)

    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    
    model.prepare_for_distributed_training()

    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        eval_checkpointer = FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
        if cfg.MODEL.WEIGHTS:
            eval_ckpt = eval_checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
        elif not args.no_resume and eval_checkpointer.has_checkpoint():
            eval_ckpt = eval_checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
        else:
            eval_ckpt = {}
        iteration = eval_ckpt.get("iteration", -1) + 1 if isinstance(eval_ckpt, dict) else 0
        return do_test(cfg, model, f"manual_{iteration}")

    do_train(cfg, model, resume=not args.no_resume)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    name = "dino_p4_vit5"  # updated: ViT-5-Small backbone, 96/48 crops (patch16)
    args.output_dir = os.path.join(args.output_dir, name)
    wandb.init(
    entity="eshasadia2015-university-of-warwick",  # your own account
    project="dinov2",
    name=name,
    mode="online",
    config=args)
    main(args)
