import argparse
import os
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import torch
import tqdm
from models.return_model_dinov3 import get_models, get_transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
# Import all dataset classes from the unified file
from eval_datasets import (
    ImmunoctoFeatureDataset,
    CONICCellDataset,
    PanNukeCellDataset,
    PathoCellbenchDataset,
    PreExtractedCellDataset,
    MIDOGDataset,
    HESTCellDataset,  
    NuCLSDataset
)

parser = argparse.ArgumentParser(description="Unified feature extraction for multiple datasets")

parser.add_argument(
    "--model_name",
    help="name of model",
    type=str,
    required=True,
)
# Supported models:
# ctranspath, resnet50, resnet50_full, owkin
# dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
# celldino, dinov3_vits16, virchow2, cellvitpp, atrous_convnext_cell

parser.add_argument(
    "--dataset",
    help="name of dataset: immunocto, conic, pannuke, pathocellbench, preextracted",
    type=str,
    required=True,
    choices=["immunocto", "conic", "pannuke", "pathocellbench", "preextracted", "midog", "hest", "nucls"],
)

parser.add_argument(
    "--data_dir",
    "--data-dir",
    help="path to root data directory (for immunocto, conic, pathocellbench, preextracted)",
    type=str,
    default=None,
)

parser.add_argument(
    "--csv_path",
    "--csv-path",
    help="path to CSV file with data (for immunocto only)",
    type=str,
    default=None,
)

parser.add_argument(
    "--checkpoint",
    help="path to checkpoint",
    default=None,
    type=str,
)

parser.add_argument(
    "--hf_token",
    "--hf-token",
    help="Hugging Face token for gated weights (e.g. dinov3_convnext). "
         "Falls back to the HF_TOKEN environment variable if not passed.",
    default=None,
    type=str,
)

parser.add_argument(
    "--save_dir",
    "--save-dir",
    "-s",
    help="path save directory",
    type=str,
    required=True,
)

parser.add_argument(
    "--batch_size",
    help="batch size for feature extraction",
    default=256,
    type=int,
)

parser.add_argument(
    "--num_workers",
    help="number of workers for data loading",
    default=2,
    type=int,
)
parser.add_argument(
    "--nucls_keep_classes",
    nargs="+",
    default=None,
    help="Subset of NuCLS classes to keep, e.g. nonTIL_stromal sTIL tumor_any. "
         "Default None keeps all classes."
)
parser.add_argument(
    "--seed",
    help="random seed for reproducible splits (preextracted only)",
    default=42,
    type=int,
)

parser.add_argument(
    "--hest_sample_ids",
    help="Comma-separated HEST sample IDs, e.g. NCBI783,NCBI785,TENX94,TENX95,TENX98",
    type=str,
    default=None,
)
parser.add_argument(
    "--hest_fold",
    help="Index of sample to hold out as test (leave-one-out, default 0)",
    type=int,
    default=0,
)
parser.add_argument(
    "--nucls_fold",
    help="NuCLS fold index (1-5 or 999)",
    type=int,
    default=1,
)
parser.add_argument(
    "--nucls_label_type",
    help="NuCLS label type: raw, main, or super",
    type=str,
    default="super",
)
def save_features_combined(feature_extractor, dataloader, save_dir, split_name, class_names=None):
    """Save all features in ONE file instead of millions of individual files - MUCH FASTER!"""
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_features = []
    all_labels = []
    all_ids = []
    
    with torch.no_grad():
        feature_extractor.eval()
        feature_extractor.to(device)
        
        for images, labels, unique_ids in tqdm.tqdm(dataloader, desc=f"Extracting {split_name} features"):
            images = images.to(device, non_blocking=True)
            # Using  MIXED PRECISION - FASTER!
            with autocast():
               batch_features = feature_extractor(images)
            
            # Handle different model output formats
            if hasattr(batch_features, "pooler_output"):
                batch_features = batch_features.pooler_output
            if isinstance(batch_features, tuple):
                batch_features = batch_features[0]
             # Convert to float32 for saving
            all_features.append(batch_features.float().cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_ids.extend(unique_ids)
    
    # Combine all batches
    all_features = np.vstack(all_features)
    all_labels = np.concatenate(all_labels)
    
    # Save everything in ONE file
    save_path = os.path.join(save_dir, f"{split_name}_features.h5")
    print(f"\nSaving {len(all_features)} features to {save_path}...")
    
    with h5py.File(save_path, "w") as hf:
        hf.create_dataset("features", data=all_features, compression="gzip")
        hf.create_dataset("labels", data=all_labels)
        hf.create_dataset("ids", data=np.array(all_ids, dtype=h5py.string_dtype(encoding="utf-8")))
        
        # Save class names if provided
        if class_names is not None:
            hf.create_dataset(
                "class_names",
                data=np.array(class_names, dtype=h5py.string_dtype(encoding="utf-8"))
            )
    
    print(f"✓ Saved {split_name} features: shape {all_features.shape}")
    return all_features.shape

def save_features_hest(feature_extractor, dataloader, save_dir, split_name):
    """
    Like save_features_combined but the 'label' slot holds a float32
    gene expression vector (N, n_genes) instead of an integer class label.
    Everything else is identical to save_features_combined.
    """
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_features = []
    all_expr     = []
    all_ids      = []

    with torch.no_grad():
        feature_extractor.eval()
        feature_extractor.to(device)

        for images, expr_vecs, unique_ids in tqdm.tqdm(
            dataloader, desc=f"Extracting {split_name} features"
        ):
            images = images.to(device, non_blocking=True)
            with autocast():
                batch_features = feature_extractor(images)

            if hasattr(batch_features, "pooler_output"):
                batch_features = batch_features.pooler_output
            if isinstance(batch_features, tuple):
                batch_features = batch_features[0]

            all_features.append(batch_features.float().cpu().numpy())
            all_expr.append(expr_vecs.numpy())   # (B, n_genes) float32
            all_ids.extend(unique_ids)

    all_features = np.vstack(all_features)   # (N, feat_dim)
    all_expr     = np.vstack(all_expr)       # (N, n_genes)

    save_path = os.path.join(save_dir, f"{split_name}_features.h5")
    print(f"\nSaving {len(all_features)} HEST features to {save_path}...")

    with h5py.File(save_path, "w") as hf:
        hf.create_dataset("features", data=all_features, compression="gzip")
        hf.create_dataset("labels",   data=all_expr,     compression="gzip")
        hf.create_dataset("ids", data=np.array(
            all_ids, dtype=h5py.string_dtype(encoding="utf-8")
        ))

    print(f"✓ Saved {split_name}: features {all_features.shape}, "
          f"expr {all_expr.shape}")
    return all_features.shape

def setup_nucls_datasets(args, transform):
    """
    Returns (train_dataset, None, test_dataset) for the requested fold/label_type.
    Called once per (fold, label_type) combination.
    """
    fold       = getattr(args, "nucls_fold",       1)
    label_type = getattr(args, "nucls_label_type", "super")

    train_dataset = dataset = NuCLSDataset(
                        fold=args.nucls_fold,
                        split="train",
                        label_type=args.nucls_label_type,
                        transform=transform,
                        keep_classes=args.nucls_keep_classes,   # ← add this
                    )
    test_dataset  = dataset = NuCLSDataset(
            fold=args.nucls_fold,
            split="test",
            label_type=args.nucls_label_type,
            transform=transform,
            keep_classes=args.nucls_keep_classes,   # ← add this
        )
    return train_dataset, None, test_dataset

def setup_immunocto_datasets(args, transform):
    """Setup Immunocto dataset with patient-based splits."""
    if args.csv_path is None or args.data_dir is None:
        raise ValueError("--csv_path and --data_dir are required for immunocto dataset")
    
    # Load CSV
    csv_path = Path(args.csv_path)
    if csv_path.is_dir():
        possible_csvs = list(csv_path.glob("*.csv"))
        if len(possible_csvs) == 0:
            raise FileNotFoundError(f"No CSV files found in directory: {csv_path}")
        elif len(possible_csvs) == 1:
            csv_path = possible_csvs[0]
            print(f"Found CSV file: {csv_path}")
        else:
            print(f"Multiple CSV files found in {csv_path}:")
            for i, csv_file in enumerate(possible_csvs):
                print(f"  {i}: {csv_file.name}")
            raise ValueError(f"Multiple CSV files found. Please specify the exact CSV file path.")
    
    print(f"Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Map class names
    CLASS_NAME_MAP = {
        "CD4+ T cells": "cd4", 
        "CD8+ T cells": "cd8",
        "B cells": "bcell", 
        "Macrophages": "macrophage", 
        "Others": "other"
    }
    df["class"] = df["class"].map(lambda x: CLASS_NAME_MAP.get(x, x))
    
    # Train/Val/Test Split
    test_df = df[df["patient"] == "CRC01"]
    train_val_df = df[df["patient"] != "CRC01"]
    
    train_patients, val_patients = train_test_split(
        train_val_df["patient"].unique(), test_size=5, random_state=42
    )
    
    train_df = train_val_df[train_val_df["patient"].isin(train_patients)]
    val_df = train_val_df[train_val_df["patient"].isin(val_patients)]
    
    # Filter missing files
    def filter_missing_files(df, root):
        valid = []
        for i, row in df.iterrows():
            img = os.path.join(root, row["patient"], "HE", row["mask"])
            if os.path.exists(img):
                valid.append(i)
        return df.loc[valid].reset_index(drop=True)
    
    train_df = filter_missing_files(train_df, args.data_dir)
    val_df = filter_missing_files(val_df, args.data_dir)
    test_df = filter_missing_files(test_df, args.data_dir)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create datasets
    train_dataset = ImmunoctoFeatureDataset(train_df, args.data_dir, transform=transform)
    val_dataset = ImmunoctoFeatureDataset(val_df, args.data_dir, transform=transform)
    test_dataset = ImmunoctoFeatureDataset(test_df, args.data_dir, transform=transform)
    
    return train_dataset, val_dataset, test_dataset


def setup_conic_datasets(args, transform):
    """Setup CONIC dataset with train/val/test splits."""
    if args.data_dir is None:
        args.data_dir = "/tia/phd-data/u5552013/ImmunoDINO_datasets/CONIC"
    
    train_dataset = CONICCellDataset(transform=transform, split="train", root_dir=args.data_dir)
    
    # Check if val split exists
    val_root = Path(args.data_dir) / "val"
    if val_root.exists():
        val_dataset = CONICCellDataset(transform=transform, split="val", root_dir=args.data_dir)
    else:
        print("No 'val' split found. Using test split as validation.")
        val_dataset = CONICCellDataset(transform=transform, split="test", root_dir=args.data_dir)
    
    test_dataset = CONICCellDataset(transform=transform, split="test", root_dir=args.data_dir)
    
    return train_dataset, val_dataset, test_dataset


def setup_pannuke_datasets(args, transform):
    """Setup PanNuke dataset with fold-based splits."""
    train_dataset = PanNukeCellDataset(transform=transform, folds=["Fold1"])
    val_dataset = PanNukeCellDataset(transform=transform, folds=["Fold2"])
    test_dataset = PanNukeCellDataset(transform=transform, folds=["Fold3"])
    
    return train_dataset, val_dataset, test_dataset


def setup_pathocellbench_datasets(args, transform):
    """Setup PathoCellbench dataset with predefined splits."""
    if args.data_dir is None:
        args.data_dir = "/tia/phd-data/u5552013/ImmunoDINO_datasets/PathoCellbench/data/phenocell"
    
    train_dataset = PathoCellbenchDataset(split="train", transform=transform, base_root=args.data_dir)
    val_dataset = PathoCellbenchDataset(split="valid", transform=transform, base_root=args.data_dir)
    test_dataset = PathoCellbenchDataset(split="test", transform=transform, base_root=args.data_dir)
    
    return train_dataset, val_dataset, test_dataset


def setup_preextracted_datasets(args, transform):
    """Setup generic pre-extracted cell dataset with stratified split."""
    from collections import defaultdict
    import random
    from torch.utils.data import Subset
    
    if args.data_dir is None:
        raise ValueError("--data_dir is required for preextracted dataset")
    
    # Load full dataset
    full_dataset = PreExtractedCellDataset(
        root=args.data_dir,
        transform=transform,
        dataset_name="FullDataset"
    )
    
    # Stratified split
    random.seed(args.seed)
    
    class_to_indices = defaultdict(list)
    for idx, (_, label, _) in enumerate(full_dataset.samples):
        class_to_indices[label].append(idx)
    
    train_indices, val_indices, test_indices = [], [], []
    train_ratio, val_ratio, test_ratio = 0.8, 0.1, 0.1
    
    print("\n=== Stratified Split Summary ===")
    for label, indices in class_to_indices.items():
        random.shuffle(indices)
        n = len(indices)
        
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        train_indices.extend(train_idx)
        val_indices.extend(val_idx)
        test_indices.extend(test_idx)
        
        class_name = full_dataset.classes[label]
        print(f"{class_name}: total={n}, train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    
    random.shuffle(train_indices)
    random.shuffle(val_indices)
    random.shuffle(test_indices)
    
    print(f"\nTotal split sizes:")
    print(f"  Train: {len(train_indices)}")
    print(f"  Val:   {len(val_indices)}")
    print(f"  Test:  {len(test_indices)}")
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    # Store class names for saving
    train_dataset.classes = full_dataset.classes
    val_dataset.classes = full_dataset.classes
    test_dataset.classes = full_dataset.classes
    
    return train_dataset, val_dataset, test_dataset
def compute_top50_genes(adata_paths):
    """
    LEMON protocol: rank genes by mean expression per slide,
    combine ranks across slides, return top-50 gene names.

    Args:
        adata_paths: list of paths to st/{sample_id}.h5ad files
    """
    import scanpy as sc

    combined_ranks = {}
    for path in adata_paths:
        adata     = sc.read_h5ad(path)
        expr      = adata.X
        if hasattr(expr, 'toarray'):
            expr  = expr.toarray()
        mean_expr = np.mean(expr, axis=0)
        genes     = list(adata.var_names)
        ranks     = pd.Series(mean_expr, index=genes).rank(ascending=False)
        for gene, rank in ranks.items():
            combined_ranks[gene] = combined_ranks.get(gene, 0) + rank

    top50 = sorted(combined_ranks, key=combined_ranks.get)[:50]
    print(f"\n[HEST] Top-50 genes: {top50[:3]} ... {top50[-3:]}")
    return top50

def setup_hest_datasets(args, transform):
    """
    Setup HEST dataset using pre-extracted patches (no WSI needed).
    Reads from patches/{id}.h5 and st/{id}.h5ad directly.

    --hest_sample_ids  comma-separated sample IDs
    --hest_fold        index into sample_ids to hold out as test (default 0)
    --data_dir         path to HEST tissue folder, e.g. HEST/breast/
                       must contain patches/ and st/ subfolders

    Train/val split: 90/10 of non-test slides, seeded with args.seed.
    """
    import scanpy as sc

    if args.data_dir is None:
        raise ValueError("--data_dir required for hest dataset")
    if args.hest_sample_ids is None:
        raise ValueError(
            "--hest_sample_ids required, "
            "e.g. NCBI783,NCBI785,TENX94,TENX95,TENX98"
        )

    data_dir   = Path(args.data_dir)
    sample_ids = [s.strip() for s in args.hest_sample_ids.split(",")]
    fold_idx   = args.hest_fold

    if fold_idx >= len(sample_ids):
        raise ValueError(
            f"--hest_fold {fold_idx} out of range "
            f"for {len(sample_ids)} sample IDs"
        )

    test_id   = sample_ids[fold_idx]
    train_ids = [s for s in sample_ids if s != test_id]

    print(f"\n[HEST] LOO fold {fold_idx}: "
          f"test={test_id}, train={train_ids}")

    # Resolve file paths
    def get_paths(sid):
        patches_h5  = data_dir / "patches" / f"{sid}.h5"
        adata_path  = data_dir / "st"      / f"{sid}.h5ad"
        if not patches_h5.exists():
            raise FileNotFoundError(f"patches not found: {patches_h5}")
        if not adata_path.exists():
            raise FileNotFoundError(f"st not found: {adata_path}")
        return patches_h5, adata_path

    # Compute top-50 genes across ALL slides (train + test)
    all_adata_paths = [get_paths(sid)[1] for sid in sample_ids]
    gene_list       = compute_top50_genes(all_adata_paths)

    # Build one dataset per training slide
    train_slide_datasets = []
    for sid in train_ids:
        patches_h5, adata_path = get_paths(sid)
        train_slide_datasets.append(
            HESTCellDataset(
                patches_h5=patches_h5,
                adata_path=adata_path,
                gene_list=gene_list,
                n_cells=10000,
                transform=transform,
                seed=args.seed,
            )
        )

    # Build test dataset
    patches_h5, adata_path = get_paths(test_id)
    test_dataset = HESTCellDataset(
        patches_h5=patches_h5,
        adata_path=adata_path,
        gene_list=gene_list,
        n_cells=10000,
        transform=transform,
        seed=args.seed,
    )

    # Combine training slides then 90/10 train/val split
    from torch.utils.data import ConcatDataset
    full_train = ConcatDataset(train_slide_datasets)
    n_val      = max(1, int(len(full_train) * 0.1))
    n_train    = len(full_train) - n_val

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    # Attach gene_list and classes so existing pipeline doesn't break
    for ds in (train_dataset, val_dataset, test_dataset):
        ds.gene_list = gene_list
        ds.classes   = gene_list

    print(f"\n[HEST] Final split — "
          f"train: {len(train_dataset)}, "
          f"val: {len(val_dataset)}, "
          f"test: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset
    
def setup_midog_datasets(args, transform):
    """Setup MIDOG dataset with stratified train/val/test split."""
    from collections import defaultdict
    import random
    from torch.utils.data import Subset

    if args.data_dir is None:
        args.data_dir = "/tia/lab-share/esha/MIDOG/Classification/Final"

    # Load full dataset
    full_dataset = MIDOGDataset(
        root=args.data_dir,
        transform=transform
    )

    random.seed(args.seed)

    # Stratified split by class
    class_to_indices = defaultdict(list)

    for idx, (_, label, _) in enumerate(full_dataset.samples):
        class_to_indices[label].append(idx)

    train_indices = []
    val_indices = []
    test_indices = []

    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    print("\n=== MIDOG Stratified Split Summary ===")

    for label, indices in class_to_indices.items():
        random.shuffle(indices)

        n = len(indices)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        train_indices.extend(train_idx)
        val_indices.extend(val_idx)
        test_indices.extend(test_idx)

        class_name = full_dataset.classes[label]

        print(
            f"{class_name}: "
            f"total={n}, "
            f"train={len(train_idx)}, "
            f"val={len(val_idx)}, "
            f"test={len(test_idx)}"
        )

    random.shuffle(train_indices)
    random.shuffle(val_indices)
    random.shuffle(test_indices)

    print(f"\nTotal split sizes:")
    print(f"  Train: {len(train_indices)}")
    print(f"  Val:   {len(val_indices)}")
    print(f"  Test:  {len(test_indices)}")

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    # preserve class names
    train_dataset.classes = full_dataset.classes
    val_dataset.classes = full_dataset.classes
    test_dataset.classes = full_dataset.classes

    return train_dataset, val_dataset, test_dataset

def main(args):
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Get transform
    transform = get_transforms(args.model_name)
    
    # Setup datasets based on dataset type
    print(f"\n=== Setting up {args.dataset} dataset ===")
    if args.dataset == "immunocto":
        train_dataset, val_dataset, test_dataset = setup_immunocto_datasets(args, transform)
    elif args.dataset == "conic":
        train_dataset, val_dataset, test_dataset = setup_conic_datasets(args, transform)
    elif args.dataset == "pannuke":
        train_dataset, val_dataset, test_dataset = setup_pannuke_datasets(args, transform)
    elif args.dataset == "pathocellbench":
        train_dataset, val_dataset, test_dataset = setup_pathocellbench_datasets(args, transform)
    elif args.dataset == "preextracted":
        train_dataset, val_dataset, test_dataset = setup_preextracted_datasets(args, transform)
    elif args.dataset == "midog":
        train_dataset, val_dataset, test_dataset = setup_midog_datasets(args, transform)
    elif args.dataset == "hest":
        train_dataset, val_dataset, test_dataset = setup_hest_datasets(args, transform)
    elif args.dataset == "nucls":
        train_dataset, val_dataset, test_dataset = setup_nucls_datasets(args, transform)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    ) if val_dataset is not None else None
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=False,
    )
    
    # Load model
    print(f"\n=== Loading model: {args.model_name} ===")
    if args.checkpoint:
        print(f"Using checkpoint: {args.checkpoint}")
    feature_extractor = get_models(
        args.model_name, saved_model_path=args.checkpoint, hf_token=args.hf_token
    )

    # Skip max-autotune compilation for cellvitpp: ViT-H at 1024x1024 makes
    # autotune compilation extremely slow / memory-hungry for little benefit
    # during a single feature-extraction pass. All other models compile as before.
    # if args.model_name.lower() != "cellvitpp":
    #     feature_extractor = torch.compile(feature_extractor, mode="max-autotune")

    # Create save directories
    folder_name = args.model_name
    
    # Add dataset-specific suffix if using checkpoint with dinov2
    if args.model_name.split("_")[0] == "dinov2" and args.checkpoint is not None:
        dataset_suffixes = {
            "immunocto": "_celldino",
            "conic": "_immunodino_conic",
            "pannuke": "_immunodino_pannuke",
            "pathocellbench": "_immunodino_pathocellbench",
            "preextracted": "_immunodino"
        }
        folder_name += dataset_suffixes.get(args.dataset, "_finetuned")

    # Same idea for the atrous DINOv3-ConvNeXt-Tiny/BYOT checkpoints from
    # run_pannuke_pipeline.py -- readable dataset-suffixed folder names
    # instead of just the bare model name, mirroring the dinov2_* case above.
    if args.model_name.lower() == "atrous_convnext_cell" and args.checkpoint is not None:
        dataset_suffixes = {
            "immunocto": "_atrousconvnext_immunocto",
            "conic": "_atrousconvnext_conic",
            "pannuke": "_atrousconvnext_pannuke",
            "pathocellbench": "_atrousconvnext_pathocellbench",
            "preextracted": "_atrousconvnext_preextracted",
            "midog": "_atrousconvnext_midog",
            "hest": "_atrousconvnext_hest",
            "nucls": "_atrousconvnext_nucls",
        }
        folder_name += dataset_suffixes.get(args.dataset, "_atrousconvnext_finetuned")
    
    if args.checkpoint is not None:
        folder_name = f"{folder_name}_{Path(args.checkpoint).parent.name}_{Path(args.checkpoint).stem}"
    
    # output_base = Path(args.save_dir) / args.dataset / folder_name
    if args.dataset == "nucls":
        output_base = (Path(args.save_dir) / args.dataset
                       / args.nucls_label_type / f"fold_{args.nucls_fold}"
                       / folder_name)
    else:
        output_base = Path(args.save_dir) / args.dataset / folder_name
    train_output = output_base / "train_data"
    val_output = output_base / "val_data"
    test_output = output_base / "test_data"
    
    # Get class names if available
    class_names = None
    if hasattr(train_dataset, 'classes'):
        class_names = train_dataset.classes
    elif hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'classes'):
        # For Subset datasets
        class_names = train_dataset.dataset.classes
    

    # --- feature saving block (replace existing block) ---
    print("\n=== Extracting Features ===")
    if args.dataset == "hest":
        save_features_hest(feature_extractor, train_loader, train_output, "train")
        save_features_hest(feature_extractor, val_loader,   val_output,   "val")
        save_features_hest(feature_extractor, test_loader,  test_output,  "test")
    else:
        save_features_combined(feature_extractor, train_loader, train_output, "train", class_names)
        if val_loader is not None:
            save_features_combined(feature_extractor, val_loader, val_output, "val", class_names)
        test_split_name = "valid" if args.dataset == "pathocellbench" else "test"
        save_features_combined(feature_extractor, test_loader, test_output, test_split_name, class_names)
    # # Extract and save features
    # print("\n=== Extracting Features ===")
    # save_features_combined(feature_extractor, train_loader, train_output, "train", class_names)
    # save_features_combined(feature_extractor, val_loader, val_output, "val", class_names)
    
    # # Use "valid" for pathocellbench, "test" for others
    # test_split_name = "valid" if args.dataset == "pathocellbench" else "test"
    # save_features_combined(feature_extractor, test_loader, test_output, test_split_name, class_names)
    
    print(f"\n=== Feature Extraction Complete ===")
    print(f"Features saved to: {output_base}")
    
    if class_names:
        print("\n=== Final Class Mapping Summary ===")
        for idx, class_name in enumerate(class_names):
            print(f"  class{idx} -> {class_name}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)