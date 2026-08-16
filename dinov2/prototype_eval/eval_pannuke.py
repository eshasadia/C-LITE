"""
eval_pannuke.py -- Linear probe + k-NN evaluation of a DST-DINOv2 teacher
checkpoint on PanNuke 5-class nuclei classification.

Standalone: only needs dst_vision_transformer.py alongside this file, plus
torch / sklearn / PIL / numpy. Does NOT import the dinov2 package or FSDP --
this is single-process feature extraction + a classical classifier on top,
not distributed training, so none of that machinery is needed here.

Usage:
    python eval_pannuke.py \
        --checkpoint /path/to/teacher_checkpoint.pth \
        --pannuke-root /tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke \
        --train-folds Fold1 Fold2 --test-folds Fold3 \
        --output pannuke_eval_report.json

IMPORTANT CAVEAT (read before trusting the numbers):
Preprocessing here uses ImageNet-style normalization (mean/std below), which
is the common dinov2 default -- but your training pipeline used
DataAugmentationHEMA (stain-aware augmentation), and I don't have its exact
normalization constants. If HEMA normalizes differently, these features will
be systematically off-distribution relative to what the backbone was trained
on, and both linear-probe and k-NN numbers will look worse than the model
actually is. Check dinov2/data/augmentations.py (or wherever HEMA is
defined) for its eval-time / non-augmented normalization and update
NORM_MEAN / NORM_STD below to match before trusting these results.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parent))
from dinov2.models.vision_transformer import DualBranchDSTViT, DST_ARCH_PRESETS  # noqa: E402

# See CAVEAT above -- verify against your actual training-time normalization.
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

CLASS_NAMES = ["Connective", "Dead", "Epithelial", "Inflammatory", "Neoplastic"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


# ---------------------------------------------------------------------------
# Checkpoint loading (standalone copy of the logic in dst_ssl_meta_arch.py's
# load_backbone_checkpoint -- duplicated here rather than imported, so this
# script has no dependency on dinov2/fsdp/etc.)
# ---------------------------------------------------------------------------
def load_teacher_backbone(backbone: nn.Module, checkpoint_path: str):
    chkpt = torch.load(checkpoint_path, map_location="cpu")

    if "teacher" in chkpt:
        raw_sd = chkpt["teacher"]
    elif "model" in chkpt:
        raw_sd = chkpt["model"]
    elif "backbone" in chkpt:
        raw_sd = {k.replace("_fsdp_wrapped_module.", ""): v for k, v in chkpt["backbone"].items()}
    else:
        raw_sd = chkpt

    backbone_sd = {}
    for k, v in raw_sd.items():
        k = k.replace("_fsdp_wrapped_module.", "")
        if k.startswith("teacher.backbone."):
            backbone_sd[k[len("teacher.backbone."):]] = v
        elif k.startswith("backbone."):
            backbone_sd[k[len("backbone."):]] = v
        elif k.startswith("dino_head.") or k.startswith("ibot_head."):
            continue  # heads not needed for feature extraction
        else:
            backbone_sd[k] = v

    target_sd = backbone.state_dict()
    filtered_sd, skipped = {}, []
    for k, v in backbone_sd.items():
        if k in target_sd and target_sd[k].shape == v.shape:
            filtered_sd[k] = v
        else:
            skipped.append(k)

    missing, unexpected = backbone.load_state_dict(filtered_sd, strict=False)
    print(f"Loaded {len(filtered_sd)}/{len(backbone_sd)} backbone tensors "
          f"(skipped {len(skipped)}, missing {len(missing)}, unexpected {len(unexpected)})")
    if len(filtered_sd) == 0:
        raise RuntimeError(
            "0 tensors loaded from checkpoint -- backbone architecture almost certainly "
            "doesn't match the checkpoint. Double check --embed-dim/--depth/--num-heads/"
            "--micro-patch-size/--macro-patch-size against your training config."
        )
    if skipped:
        print(f"  skipped (first 10): {skipped[:10]}")
    return missing, unexpected


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PanNukeEvalDataset(Dataset):
    """
    Labeled PanNuke dataset for evaluation, restricted to given folds.
    Expects the same directory layout as PanNukeCellDataset in train_dst.py:
        <root>/Fold{1,2,3}_cell_patches/<ClassName>/*.png
    """

    def __init__(self, root, folds, img_size=64, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []  # (path, label_idx)

        for fold in folds:
            for cls_name in CLASS_NAMES:
                img_dir = self.root / f"{fold}_cell_patches" / cls_name
                if not img_dir.is_dir():
                    print(f"Warning: {img_dir} does not exist, skipping.")
                    continue
                for img_path in sorted(img_dir.iterdir()):
                    if img_path.suffix.lower() in IMG_EXTENSIONS:
                        self.samples.append((img_path, CLASS_TO_IDX[cls_name]))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found under {root} for folds {folds}")

        counts = np.bincount([lbl for _, lbl in self.samples], minlength=len(CLASS_NAMES))
        print(f"[PanNukeEvalDataset] folds={folds}: {len(self.samples)} samples, "
              f"per-class: {dict(zip(CLASS_NAMES, counts.tolist()))}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_eval_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_features(backbone, loader, device):
    backbone.eval()
    feats, labels = [], []
    for imgs, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        out = backbone(imgs, is_training=True)  # need dict form to get cls token explicitly
        cls = out["x_norm_clstoken"]
        feats.append(cls.cpu().numpy())
        labels.append(lbls.numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def run_linear_probe(train_feats, train_labels, test_feats, test_labels):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(train_feats)
    Xtr = scaler.transform(train_feats)
    Xte = scaler.transform(test_feats)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, train_labels)
    preds = clf.predict(Xte)

    return {
        "accuracy": float(accuracy_score(test_labels, preds)),
        "macro_f1": float(f1_score(test_labels, preds, average="macro")),
        "confusion_matrix": confusion_matrix(test_labels, preds).tolist(),
    }


def run_knn(train_feats, train_labels, test_feats, test_labels, k_values=(10, 20, 50)):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, f1_score

    results = {}
    for k in k_values:
        clf = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        clf.fit(train_feats, train_labels)
        preds = clf.predict(test_feats)
        results[f"k={k}"] = {
            "accuracy": float(accuracy_score(test_labels, preds)),
            "macro_f1": float(f1_score(test_labels, preds, average="macro")),
        }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate DST-DINOv2 teacher checkpoint on PanNuke")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--pannuke-root", type=str,
                         default="/tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke")
    parser.add_argument("--train-folds", nargs="+", default=["Fold1", "Fold2"])
    parser.add_argument("--test-folds", nargs="+", default=["Fold3"])
    parser.add_argument("--arch", type=str, default="dst_dinov2_vits", choices=list(DST_ARCH_PRESETS.keys()))
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--micro-patch-size", type=int, default=4)
    parser.add_argument("--macro-patch-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output", type=str, default="pannuke_eval_report.json")
    parser.add_argument("--knn-k", nargs="+", type=int, default=[10, 20, 50])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preset = DST_ARCH_PRESETS[args.arch]

    print(f"Building {args.arch} backbone: img_size={args.img_size} "
          f"micro_patch={args.micro_patch_size} macro_patch={args.macro_patch_size} {preset}")
    backbone = DualBranchDSTViT(
        img_size=args.img_size,
        micro_patch_size=args.micro_patch_size,
        macro_patch_size=args.macro_patch_size,
        embed_dim=preset["embed_dim"],
        depth=preset["depth"],
        num_heads=preset["num_heads"],
        drop_path_rate=0.0,  # eval: no stochastic depth
    ).to(device)

    load_teacher_backbone(backbone, args.checkpoint)

    transform = build_eval_transform(args.img_size)
    train_ds = PanNukeEvalDataset(args.pannuke_root, args.train_folds, args.img_size, transform)
    test_ds = PanNukeEvalDataset(args.pannuke_root, args.test_folds, args.img_size, transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("Extracting train features...")
    train_feats, train_labels = extract_features(backbone, train_loader, device)
    print(f"  {train_feats.shape[0]} samples, dim={train_feats.shape[1]}")

    print("Extracting test features...")
    test_feats, test_labels = extract_features(backbone, test_loader, device)
    print(f"  {test_feats.shape[0]} samples, dim={test_feats.shape[1]}")

    print("Running linear probe...")
    linear_results = run_linear_probe(train_feats, train_labels, test_feats, test_labels)
    print(f"  Linear probe accuracy: {linear_results['accuracy']:.4f}, "
          f"macro-F1: {linear_results['macro_f1']:.4f}")

    print("Running k-NN...")
    knn_results = run_knn(train_feats, train_labels, test_feats, test_labels, tuple(args.knn_k))
    for k, r in knn_results.items():
        print(f"  {k}: accuracy={r['accuracy']:.4f}, macro-F1={r['macro_f1']:.4f}")

    report = {
        "checkpoint": args.checkpoint,
        "arch": args.arch,
        "train_folds": args.train_folds,
        "test_folds": args.test_folds,
        "n_train": int(train_feats.shape[0]),
        "n_test": int(test_feats.shape[0]),
        "feature_dim": int(train_feats.shape[1]),
        "class_names": CLASS_NAMES,
        "linear_probe": linear_results,
        "knn": knn_results,
    }
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {args.output}")


if __name__ == "__main__":
    main()