"""
Unified dataset classes for feature extraction across multiple cell classification datasets.

Contains dataset classes for:
- ImmunoctoFeatureDataset: Immunocto dataset with patient-based splits
- CONICCellDataset: CONIC dataset with class-based folder structure
- PanNukeCellDataset: PanNuke dataset with fold-based splits
- PathoCellbenchDataset: PathoCellbench dataset with LMDB storage
- PreExtractedCellDataset: Generic dataset for pre-extracted cell crops
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path
import h5py  
import numpy as np
import pandas as pd
import torch
import tqdm
from PIL import Image
from torch.utils.data import Dataset

# PathoCellbench requires special import
try:
    sys.path.append('/tia/phd-data/u5552013/ImmunoDINO_datasets/PathoCellbench/Pathology-Foundation-Model-Benchmark/src')
    from benchmark.dataset.lmdb_dataset import LMDBDataset
    PATHOCELLBENCH_AVAILABLE = True
except ImportError:
    PATHOCELLBENCH_AVAILABLE = False
    print("Warning: PathoCellbench LMDB dataset not available")


# ============================================================================
# IMMUNOCTO DATASET
# ============================================================================

class ImmunoctoFeatureDataset(Dataset):
    """
    Dataset for Immunocto cell classification.
    
    Expected structure:
        root/
            {patient}/
                HE/
                    {mask_filename}.png
    
    Args:
        df: DataFrame with columns ['patient', 'mask', 'class']
        root: Path to root data directory
        transform: Optional transform to apply
    
    Returns:
        image, label (int), unique_id
    """
    
    def __init__(self, df, root, transform=None):
        self.df = df
        self.root = root
        self.transform = transform
        
        # Define classes and integer mapping
        self.classes = ["bcell", "cd4", "cd8", "macrophage", "other"]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        print(f"\n[ImmunoctoFeatureDataset] Loaded {len(self.df)} samples")
        print(f"Classes: {self.classes}")
        print(f"Class distribution:")
        for cls in self.classes:
            count = (self.df["class"] == cls).sum()
            print(f"  {cls}: {count}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.root, row["patient"], "HE", row["mask"])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Convert image to RGB
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        # Convert string label to integer
        label_str = row.get("class", None)
        if label_str is None:
            raise ValueError(f"No class label found for index {idx}")
        label = self.class_to_idx[label_str]

        # Return image, label, and unique identifier
        unique_id = f"{row['patient']}_{row['mask']}"
        return image, label, unique_id
# ============================================================
# ADD TO eval_datasets.py
# ============================================================


class NuCLSDataset(Dataset):
    """
    Loads NuCLS nucleus crop images from pre-extracted class folders.

    Structure from extract_nucls_crops.py:
        root_dir/
            super_classification/fold_1/train/{class}/
            main_classification/fold_1/train/{class}/
            raw_classification/fold_1/train/{class}/

    Args:
        transform   : torchvision transform
        fold        : int, 1–5
        split       : 'train' or 'test'
        label_type  : 'super' | 'main' | 'raw'
        root_dir    : OUTPUT_ROOT from extract_nucls_crops.py
        splits_dir  : path to train_test_splits folder
        keep_classes: optional list of class names to keep, e.g.
                      ["nonTIL_stromal", "sTIL", "tumor_any"]
                      All others are dropped and labels are remapped
                      to contiguous 0-based indices.
    """

    LABEL_TYPE_MAP = {
        "super": "super_classification",
        "main":  "main_classification",
        "raw":   "raw_classification",
    }
    IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    def __init__(
        self,
        transform=None,
        fold=1,
        split="train",
        label_type="super",
        root_dir="/tia/phd-data/u5552013/ImmunoDINO_datasets/NuCLS/crops",
        splits_dir="/tia/phd-data/u5552013/ImmunoDINO_datasets/NuCLS/NuCLS/single_rater/train_test_splits",
        keep_classes=None,
    ):
        assert split in ("train", "test")
        assert fold in (1, 2, 3, 4, 5)
        assert label_type in self.LABEL_TYPE_MAP, \
            f"label_type must be one of {list(self.LABEL_TYPE_MAP)}"

        self.transform   = transform
        self.split       = split
        self.fold        = fold
        self.label_type  = label_type
        self.root_dir    = Path(root_dir)
        self.splits_dir  = Path(splits_dir)

        label_col      = self.LABEL_TYPE_MAP[label_type]
        self.split_dir = self.root_dir / label_col / f"fold_{fold}" / split

        if not self.split_dir.exists():
            raise RuntimeError(f"Split directory not found: {self.split_dir}")

        # Discover classes from disk
        self.classes = sorted([d.name for d in self.split_dir.iterdir() if d.is_dir()])
        if not self.classes:
            raise RuntimeError(f"No class subdirectories found in {self.split_dir}")

        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.class_counts = {c: 0 for c in self.classes}

        # Load slide names for this fold/split
        fold_csv    = self.splits_dir / f"fold_{fold}_{split}.csv"
        slide_names = set(pd.read_csv(fold_csv)["slide_name"].tolist())

        self.samples = []
        for cls_name in self.classes:
            for img_path in sorted((self.split_dir / cls_name).iterdir()):
                if img_path.suffix.lower() not in self.IMG_EXTENSIONS:
                    continue
                slide = img_path.stem.split("_cx")[0]
                if slide not in slide_names:
                    continue
                self.samples.append((cls_name, img_path, self.class_to_idx[cls_name]))
                self.class_counts[cls_name] += 1

        if not self.samples:
            raise RuntimeError(
                f"No images matched for fold={fold} split='{split}' "
                f"label_type='{label_type}' in {self.split_dir}"
            )

        # ── Optional class filtering + contiguous label remapping ──────────
        if keep_classes is not None:
            keep_set = set(keep_classes)
            unknown  = keep_set - set(self.classes)
            if unknown:
                raise ValueError(
                    f"keep_classes contains unknown classes: {unknown}. "
                    f"Available: {self.classes}"
                )

            # Filter
            self.samples = [(c, p, l) for c, p, l in self.samples if c in keep_set]
            if not self.samples:
                raise RuntimeError(f"No samples remain after filtering to {keep_classes}")

            # Rebuild classes preserving discovery order, then remap labels
            self.classes      = [c for c in self.classes if c in keep_set]
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
            self.class_counts = {c: 0 for c in self.classes}
            self.samples      = [
                (c, p, self.class_to_idx[c]) for c, p, _ in self.samples
            ]
            for c, _, _ in self.samples:
                self.class_counts[c] += 1

        # ── Summary ────────────────────────────────────────────────────────
        print(f"\n[NuCLSDataset] label_type='{label_type}' | fold={fold} | split='{split}'")
        if keep_classes is not None:
            print(f"  Filtered to classes: {self.classes}")
            print(f"  Label mapping: {self.class_to_idx}")
        total = 0
        for cls_name in self.classes:
            count = self.class_counts[cls_name]
            print(f"  {cls_name:30s}: {count:>6,}")
            total += count
        print(f"  {'TOTAL':30s}: {total:>6,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cls_name, img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        unique_id = f"fold{self.fold}_{self.split}_{cls_name}_{img_path.stem}"
        return image, label, unique_id
class HESTCellDataset(Dataset):
    """
    Gene expression regression dataset from HEST pre-extracted patches.
    Reads directly from patches/{sample_id}.h5 and st/{sample_id}.h5ad.
    No WSI needed.

    Args:
        patches_h5:  path to patches/{sample_id}.h5
        adata_path:  path to st/{sample_id}.h5ad
        gene_list:   list of gene name strings to predict (top-50)
        n_cells:     spots to subsample per slide (default 10000)
        transform:   model-specific transform from get_transforms()
        seed:        for reproducible subsampling

    Returns:
        image (tensor), expr_vec (float32 tensor [n_genes]), unique_id (str)
    """

    def __init__(self, patches_h5, adata_path, gene_list,
                 n_cells=10000, transform=None, seed=42):
        import scanpy as sc

        self.transform  = transform
        self.gene_list  = gene_list
        self.sample_id  = Path(patches_h5).stem
        self.patches_h5 = str(patches_h5)

        # Load AnnData for expression
        adata = sc.read_h5ad(adata_path)

        # Only keep genes present in this slide
        available = [g for g in gene_list if g in adata.var_names]
        missing   = len(gene_list) - len(available)
        if missing > 0:
            print(f"  [{self.sample_id}] {missing} genes missing, "
                  f"using {len(available)}/{len(gene_list)}")
        self.available_genes = available

        # Build barcode -> expression dict
        # Pad to full gene_list length with zeros for missing genes
        # so all slides return same-shaped vectors and can be batched together
        expr = adata[:, available].X
        if hasattr(expr, 'toarray'):
            expr = expr.toarray()
        expr = expr.astype(np.float32)

        gene_to_col = {g: i for i, g in enumerate(gene_list)}

        barcode_to_expr = {}
        for i, bc in enumerate(adata.obs_names):
            full_vec = np.zeros(len(gene_list), dtype=np.float32)
            for j, gene in enumerate(available):
                full_vec[gene_to_col[gene]] = expr[i, j]
            barcode_to_expr[bc] = full_vec

        # Load barcodes from patches h5
        with h5py.File(self.patches_h5, "r") as hf:
            raw_barcodes = hf["barcode"][:]          # (N, 1)
            n_total      = hf["img"].shape[0]

        # Flatten (N,1) -> (N,) then decode bytes
        raw_barcodes = raw_barcodes[:, 0] if raw_barcodes.ndim == 2 else raw_barcodes
        all_barcodes = [
            bc.decode("utf-8") if isinstance(bc, bytes) else str(bc)
            for bc in raw_barcodes
        ]

        # Keep only patches with matching expression
        valid_indices = [
            i for i, bc in enumerate(all_barcodes)
            if bc in barcode_to_expr
        ]

        if len(valid_indices) == 0:
            raise RuntimeError(
                f"[{self.sample_id}] No barcodes matched between "
                f"patches h5 and st h5ad."
            )

        # Subsample
        rng    = np.random.default_rng(seed)
        n      = min(n_cells, len(valid_indices))
        chosen = rng.choice(len(valid_indices), size=n, replace=False)

        self.patch_indices = [valid_indices[i] for i in chosen]
        self.barcodes      = [all_barcodes[i] for i in self.patch_indices]
        self.expr          = np.stack(
            [barcode_to_expr[bc] for bc in self.barcodes]
        )                                             # (N, n_genes)

        # Dummy classes so existing pipeline doesn't break
        self.classes = gene_list

        print(f"\n[HESTCellDataset] {self.sample_id}: "
              f"{len(self.patch_indices)} spots matched, "
              f"{len(available)}/{len(gene_list)} genes "
              f"(zeros padded for missing)")

    def __len__(self):
        return len(self.patch_indices)

    def __getitem__(self, idx):
        patch_idx = self.patch_indices[idx]

        # Lazy load single patch — no full file in RAM
        with h5py.File(self.patches_h5, "r") as hf:
            img_arr = hf["img"][patch_idx]           # (224, 224, 3) uint8

        image = Image.fromarray(img_arr.astype(np.uint8)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        expr_vec  = torch.tensor(self.expr[idx], dtype=torch.float32)
        unique_id = f"{self.sample_id}_{self.barcodes[idx]}"

        return image, expr_vec, unique_id


class MIDOGDataset(Dataset):
    """
    MIDOG classification dataset

    Expected structure:
    root/
    ├── Atypical/
    │   ├── img1.png
    │   ├── img2.png
    │   └── ...
    └── Normal/
        ├── img3.png
        ├── img4.png
        └── ...
    """

    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform

        self.classes = ["Atypical", "Normal"]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []

        valid_extensions = {
            ".png",
            ".jpg",
            ".jpeg",
            ".tif",
            ".tiff",
            ".bmp",
        }

        for cls in self.classes:
            cls_dir = self.root / cls

            if not cls_dir.exists():
                print(f"Warning: {cls_dir} does not exist")
                continue

            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() not in valid_extensions:
                    continue

                # IMPORTANT:
                # Match naming convention used by other datasets
                uid = f"cell_{cls}_{img_path.stem}"

                self.samples.append(
                    (
                        str(img_path),              # image path
                        self.class_to_idx[cls],     # label
                        uid                         # unique id
                    )
                )

        print(f"\nMIDOG Dataset Loaded")
        print(f"Classes: {self.classes}")
        print(f"Total samples: {len(self.samples)}")

        for cls in self.classes:
            count = sum(
                1 for _, label, _ in self.samples
                if label == self.class_to_idx[cls]
            )
            print(f"  {cls}: {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, uid = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label, uid
# ============================================================================
# CONIC DATASET
# ============================================================================

class CONICCellDataset(Dataset):
    """
    Loads CONIC cell patch images from split/class folders.

    Expected structure:
        root/
            train/
                Connective/Images/
                Eosinophil/Images/
                Epithelial/Images/
                Lymphocyte/Images/
                Neutrophil/Images/
                Plasma/Images/
            val/   (optional)
                ...
            test/
                ...

    Args:
        transform: Optional transform to apply
        split: One of ("train", "val", "test")
        root_dir: Path to CONIC dataset root
    
    Returns:
        image, label (int), unique_id
    """

    IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    def __init__(self, transform=None, split="train", 
                 root_dir="/tia/phd-data/u5552013/ImmunoDINO_datasets/CONIC"):
        self.transform = transform
        self.split = split
        self.root_dir = Path(root_dir)
        self.samples = []

        self.classes = [
            "Connective",
            "Eosinophil",
            "Epithelial",
            "Lymphocyte",
            "Neutrophil",
            "Plasma",
        ]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.class_counts = {cls: 0 for cls in self.classes}

        image_dirs = [
            (cls_name, self.root_dir / split / cls_name / "Images")
            for cls_name in self.classes
        ]

        for cls_name, img_dir in image_dirs:
            if not img_dir.is_dir():
                print(f"Warning: {img_dir} does not exist, skipping.")
                continue

            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() in self.IMG_EXTENSIONS:
                    label = self.class_to_idx[cls_name]
                    self.samples.append((cls_name, img_path, label))
                    self.class_counts[cls_name] += 1

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found for split='{split}' in {self.root_dir}")

        # Summary
        print(f"\n[CONICCellDataset] Dataset summary for split='{self.split}':")
        total = 0
        for cls_name in self.classes:
            count = self.class_counts[cls_name]
            print(f"  {cls_name}: {count} images")
            total += count
        print(f"  TOTAL: {total} images\n")

        print("First 5 sample paths:")
        for sample in self.samples[:5]:
            print(" ", sample[1])
        print()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cls_name, img_path, label = self.samples[idx]

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Unique identifier
        unique_id = f"{self.split}_{cls_name}_{img_path.name}"

        return image, label, unique_id


# ============================================================================
# PANNUKE DATASET
# ============================================================================

class PanNukeCellDataset(Dataset):
    """
    Loads PanNuke cell patch images from hardcoded fold folders.

    Expected structure:
        /tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke/
            Fold1_cell_patches/
                Connective/
                Dead/
                Epithelial/
                Inflammatory/
                Neoplastic/
            Fold2_cell_patches/
                ...
            Fold3_cell_patches/
                ...

    Args:
        transform: Optional transform to apply
        folds: List of folds to include, e.g., ["Fold1", "Fold2", "Fold3"]
    
    Returns:
        image, label (int), unique_id
    """

    IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    def __init__(self, transform=None, folds=None):
        self.transform = transform
        self.samples = []

        self.classes = [
            "Connective",
            "Dead",
            "Epithelial",
            "Inflammatory",
            "Neoplastic",
        ]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        print("Class to ID mapping:", self.class_to_idx)
        
        if folds is None:
            folds = ["Fold1", "Fold2", "Fold3"]
        self.folds = set(folds)

        self.fold_class_counts = {
            "Fold1": {cls: 0 for cls in self.classes},
            "Fold2": {cls: 0 for cls in self.classes},
            "Fold3": {cls: 0 for cls in self.classes},
        }

        image_dirs = [
            # Fold1
            ("Fold1", "Connective",   Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke/Fold1_cell_patches/Connective")),
            ("Fold1", "Dead",         Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke/Fold1_cell_patches/Dead")),
            ("Fold1", "Epithelial",   Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke/Fold1_cell_patches/Epithelial")),
            ("Fold1", "Inflammatory", Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke/Fold1_cell_patches/Inflammatory")),
            ("Fold1", "Neoplastic",   Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke/Fold1_cell_patches/Neoplastic")),

            # Fold2
            ("Fold2", "Connective",   Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke/Fold2_cell_patches/Connective")),
            ("Fold2", "Dead",         Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke/Fold2_cell_patches/Dead")),
            ("Fold2", "Epithelial",   Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke/Fold2_cell_patches/Epithelial")),
            ("Fold2", "Inflammatory", Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke/Fold2_cell_patches/Inflammatory")),
            ("Fold2", "Neoplastic",   Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke/Fold2_cell_patches/Neoplastic")),

            # Fold3
            ("Fold3", "Connective",   Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke/Fold3_cell_patches/Connective")),
            ("Fold3", "Dead",         Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke/Fold3_cell_patches/Dead")),
            ("Fold3", "Epithelial",   Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke/Fold3_cell_patches/Epithelial")),
            ("Fold3", "Inflammatory", Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke/Fold3_cell_patches/Inflammatory")),
            ("Fold3", "Neoplastic",   Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke/Fold3_cell_patches/Neoplastic")),
        ]

        for fold, cls_name, img_dir in image_dirs:
            if fold not in self.folds:
                continue

            if not img_dir.is_dir():
                print(f"Warning: {img_dir} does not exist, skipping.")
                continue

            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() in self.IMG_EXTENSIONS:
                    label = self.class_to_idx[cls_name]
                    self.samples.append((fold, cls_name, img_path, label))
                    self.fold_class_counts[fold][cls_name] += 1

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found for folds: {self.folds}")

        # Summary
        print(f"\n[PanNukeCellDataset] Dataset summary for {sorted(self.folds)}:")
        grand_total = 0

        for fold in ["Fold1", "Fold2", "Fold3"]:
            if fold not in self.folds:
                continue
            print(f"\n  {fold}:")
            fold_total = 0
            for cls_name in self.classes:
                count = self.fold_class_counts[fold][cls_name]
                print(f"    {cls_name}: {count} images")
                fold_total += count
            print(f"    Total in {fold}: {fold_total}")
            grand_total += fold_total

        print(f"\n  GRAND TOTAL: {grand_total} images\n")

        print("First 5 sample paths:")
        for sample in self.samples[:5]:
            print(" ", sample[2])
        print()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fold, cls_name, img_path, label = self.samples[idx]

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Unique identifier
        unique_id = f"{fold}_{cls_name}_{img_path.name}"

        return image, label, unique_id


# ============================================================================
# PATHOCELLBENCH DATASET
# ============================================================================

class PathoCellbenchDataset(Dataset):
    """
    Self-contained dataset for 64x64 cell-centric patches from
    PathoCellbench LMDB tiles, including all cells,
    supporting train/valid/test splits.

    Args:
        split: One of ("train", "valid", "test")
        transform: Optional transform to apply to PIL patches
        base_root: phenocell root directory
    
    Returns:
        image, label (int), unique_id
    """

    PATCH_SIZE = 64

    def __init__(self, split="train", transform=None, 
                 base_root="/tia/phd-data/u5552013/ImmunoDINO_datasets/PathoCellbench/data/phenocell"):
        if not PATHOCELLBENCH_AVAILABLE:
            raise RuntimeError("PathoCellbench LMDB dataset not available. Check import path.")
        
        self.transform = transform
        self.patch_size = self.PATCH_SIZE
        self.split = split

        self.base_root = Path(base_root)
        self.dataset_path = self.base_root / "phenocell_lmdb"
        self.split_csv = self.base_root / "splits" / "phenocell_dataset_split.csv"
        self.label_dict_json = self.base_root / "label_dict.json"

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"LMDB dataset path not found: {self.dataset_path}")
        if not self.split_csv.exists():
            raise FileNotFoundError(f"Split CSV not found: {self.split_csv}")
        if not self.label_dict_json.exists():
            raise FileNotFoundError(f"Label dict not found: {self.label_dict_json}")

        # Load label dict
        with open(self.label_dict_json) as f:
            label_dict = json.load(f)
        self.label_dict = {int(k): v for k, v in label_dict.items() if int(k) != 0}  # remove background
        self.inv_label_dict = {v: k for k, v in self.label_dict.items()}

        # Load split CSV
        split_df = pd.read_csv(self.split_csv)

        include_fovs = split_df[split_df["train_test_val_split"] == split]["sample_name"].tolist()
        if len(include_fovs) == 0:
            raise RuntimeError(f"No FOVs found for split='{split}' in {self.split_csv}")

        self.lmdb_dataset = LMDBDataset(
            path=str(self.dataset_path),
            include_sample_names=include_fovs
        )

        # Precompute all cell instances for this split
        self.samples = []
        self.class_counts = {}

        print(f"\n[Indexing PathoCellbench split='{split}'] This may take some time...")

        for tile_idx, el in enumerate(tqdm.tqdm(self.lmdb_dataset, desc=f"Indexing {split}")):
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

                self.samples.append((tile_idx, int(inst_id), class_idx))
                class_name = self.label_dict[class_idx]
                self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid cell instances found for split='{split}'")

        # Summary
        print(f"\n[PathoCellbenchDataset] Dataset summary for split='{self.split}':")
        total = 0
        for cls_name, count in sorted(self.class_counts.items()):
            print(f"  {cls_name}: {count} cells")
            total += count
        print(f"  TOTAL: {total} cells\n")

        print("First 5 samples:")
        for sample in self.samples[:5]:
            print(" ", sample)
        print()

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
            out[:, y_off:y_off + crop.shape[1], x_off:x_off + crop.shape[2]] = crop
        else:
            crop = arr[y1_src:y2_src, x1_src:x2_src]
            out = np.zeros((patch_size, patch_size), dtype=arr.dtype)
            y_off = half - (cy - y1_src)
            x_off = half - (cx - x1_src)
            out[y_off:y_off + crop.shape[0], x_off:x_off + crop.shape[1]] = crop

        return out

    @staticmethod
    def normalize(img):
        img = img.astype(np.float32)
        if img.max() == 0:
            return img
        return img / img.max()

    def __getitem__(self, idx):
        tile_idx, inst_id, class_idx = self.samples[idx]
        el = self.lmdb_dataset[tile_idx]

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

        unique_id = f"{self.split}_tile{tile_idx}_inst{inst_id}"

        return patch, class_idx, unique_id


# ============================================================================
# GENERIC PRE-EXTRACTED CELL DATASET
# ============================================================================

class PreExtractedCellDataset(Dataset):
    """
    Generic dataset for pre-extracted cell crops.

    Expected folder structure:
        root/
            ClassA/Images/*.png
            ClassB/Images/*.png
            ...

    Args:
        root: Path to dataset root
        transform: Optional transform to apply
        dataset_name: Name for logging purposes
    
    Returns:
        image, label (int), unique_id
    """

    IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    def __init__(self, root, transform=None, dataset_name="Dataset"):
        self.root = Path(root)
        self.transform = transform
        self.dataset_name = dataset_name
        self.samples = []
        self.class_counts = Counter()

        if not self.root.exists():
            raise RuntimeError(f"Dataset root does not exist: {self.root}")

        self.classes = sorted(
            [p.name for p in self.root.iterdir() if p.is_dir() and (p / "Images").exists()]
        )

        if len(self.classes) == 0:
            raise RuntimeError(f"No class folders with 'Images/' found in {self.root}")

        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}

        print(f"\n[{self.dataset_name}] Class index mapping:")
        for idx in sorted(self.idx_to_class):
            print(f"  class{idx} -> {self.idx_to_class[idx]}")

        for class_folder in sorted(self.root.iterdir()):
            if not class_folder.is_dir():
                continue

            images_dir = class_folder / "Images"
            if not images_dir.exists():
                continue

            class_name = class_folder.name
            label = self.class_to_idx[class_name]

            for img_path in sorted(images_dir.iterdir()):
                if img_path.suffix.lower() in self.IMG_EXTENSIONS:
                    self.samples.append((img_path, label, class_name))
                    self.class_counts[class_name] += 1

        if len(self.samples) == 0:
            raise RuntimeError(f"No crops found in {self.root}")

        print(f"\n[{self.dataset_name}] Loaded {len(self.samples)} cell crops from: {self.root}")
        print(f"[{self.dataset_name}] Classes ({len(self.classes)}): {self.classes}")
        for cls, count in sorted(self.class_counts.items()):
            print(f"  {cls}: {count}")

        print("\nFirst 5 sample paths:")
        for sample in self.samples[:5]:
            print(" ", sample[0])
        print()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, class_name = self.samples[idx]

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        unique_id = f"{class_name}_{img_path.stem}"

        return img, label, unique_id