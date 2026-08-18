"""
Linear probe evaluation in the spirit of the original DINOv2 eval_linear.py,
but adapted to read features you've already cached to HDF5 with your
extract_features.py pipeline (instead of running the backbone live inside
the training loop)

Usage:
    python linear_probe_original_style.py \
        --train-features path/to/train_features.h5 \
        --val-features   path/to/val_features.h5 \
        --test-features  path/to/test_features.h5 \
        --output-dir     runs/linear_probe \
        --max-iter 12500 --batch-size 128
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class HDF5FeatureDataset(Dataset):
    """Loads a combined features/labels/ids HDF5 file (as produced by
    save_features_combined in extract_features.py) fully into memory."""

    def __init__(self, h5_path: str):
        with h5py.File(h5_path, "r") as hf:
            self.features = hf["features"][:].astype(np.float32)
            self.labels = hf["labels"][:].astype(np.int64)
            self.class_names = None
            if "class_names" in hf:
                self.class_names = [
                    c.decode("utf-8") if isinstance(c, bytes) else c
                    for c in hf["class_names"][:]
                ]
        if self.features.ndim > 2:
            self.features = self.features.reshape(self.features.shape[0], -1)
        self.labels = self.labels.flatten()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# --------------------------------------------------------------------------- #
# Model (mirrors dinov2's LinearClassifier / AllClassifiers, but for
# already-fixed feature vectors instead of x_tokens_list)
# --------------------------------------------------------------------------- #
class LinearClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        return self.linear(x)


class AllClassifiers(nn.Module):
    def __init__(self, classifiers_dict):
        super().__init__()
        self.classifiers_dict = nn.ModuleDict(classifiers_dict)

    def forward(self, x):
        return {k: v(x) for k, v in self.classifiers_dict.items()}

    def __len__(self):
        return len(self.classifiers_dict)


def scale_lr(lr: float, batch_size: int) -> float:
    """Original DINOv2 linear-scaling rule (no distributed world size here)."""
    return lr * batch_size / 256.0


def setup_linear_classifiers(in_dim, num_classes, learning_rates, batch_size, device):
    classifiers = {}
    param_groups = []
    for base_lr in learning_rates:
        lr = scale_lr(base_lr, batch_size)
        key = f"lr_{lr:.6f}".replace(".", "_")
        clf = LinearClassifier(in_dim, num_classes).to(device)
        classifiers[key] = clf
        param_groups.append({"params": clf.parameters(), "lr": lr})
    return AllClassifiers(classifiers), param_groups


# --------------------------------------------------------------------------- #
# Eval
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(all_classifiers, data_loader, device):
    all_classifiers.eval()
    correct = {k: 0 for k in all_classifiers.classifiers_dict}
    total = 0
    for feats, labels in data_loader:
        feats, labels = feats.to(device), labels.to(device)
        outputs = all_classifiers(feats)
        for k, logits in outputs.items():
            preds = logits.argmax(dim=1)
            correct[k] += (preds == labels).sum().item()
        total += labels.size(0)
    all_classifiers.train()
    accuracies = {k: c / total for k, c in correct.items()}
    best_key = max(accuracies, key=accuracies.get)
    return accuracies, best_key


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def _infinite_loader(loader):
    """Cycle through the DataLoader forever, matching the original's
    infinite/resumable sampler behaviour."""
    while True:
        for batch in loader:
            yield batch


def train_linear_probe(
    train_ds,
    val_ds,
    test_ds: Optional[Dataset],
    output_dir: str,
    max_iter: int,
    batch_size: int,
    learning_rates: List[float],
    num_workers: int = 4,
    eval_period_iterations: int = 1250,
    checkpoint_period_iterations: int = 1250,
    log_every: int = 20,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_dim = train_ds.features.shape[1]
    num_classes = int(max(train_ds.labels.max(), val_ds.labels.max()) + 1)

    all_classifiers, param_groups = setup_linear_classifiers(
        in_dim, num_classes, learning_rates, batch_size, device
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iter, eta_min=0)
    criterion = nn.CrossEntropyLoss()

    metrics_path = os.path.join(output_dir, "results_eval_linear.json")
    best_val_acc = -1.0
    best_key = None

    print(f"Device: {device} | in_dim={in_dim} | num_classes={num_classes} "
          f"| classifiers={len(all_classifiers)} | max_iter={max_iter} "
          f"| batch_size={batch_size}")

    data_iter = _infinite_loader(train_loader)
    for iteration in range(max_iter):
        feats, labels = next(data_iter)
        feats, labels = feats.to(device), labels.to(device)

        outputs = all_classifiers(feats)
        losses = {k: criterion(v, labels) for k, v in outputs.items()}
        loss = sum(losses.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if iteration % log_every == 0:
            print(f"iter {iteration}/{max_iter} loss={loss.item():.4f} "
                  f"lr={optimizer.param_groups[0]['lr']:.6f}")

        do_eval = (iteration + 1) % eval_period_iterations == 0 or iteration == max_iter - 1
        if do_eval:
            accuracies, iter_best_key = evaluate(all_classifiers, val_loader, device)
            iter_best_acc = accuracies[iter_best_key]
            print(f"[val] iter {iteration}: best={iter_best_key} acc={iter_best_acc:.4f}")

            with open(metrics_path, "a") as f:
                f.write(json.dumps({
                    "iteration": iteration,
                    "val_accuracies": accuracies,
                    "best_classifier": {"name": iter_best_key, "accuracy": iter_best_acc},
                }) + "\n")

            if iter_best_acc > best_val_acc:
                best_val_acc = iter_best_acc
                best_key = iter_best_key

        if (iteration + 1) % checkpoint_period_iterations == 0 or iteration == max_iter - 1:
            torch.save(
                {
                    "iteration": iteration,
                    "classifiers_state_dict": all_classifiers.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                os.path.join(output_dir, "running_checkpoint_linear_eval.pt"),
            )

    if best_key is not None:
        torch.save(
            {
                "iteration": max_iter - 1,
                "best_classifier": best_key,
                "state_dict": all_classifiers.classifiers_dict[best_key].state_dict(),
            },
            os.path.join(output_dir, "best_linear_classifier.pt"),
        )

    print(f"\nBest classifier on val: {best_key} (acc={best_val_acc:.4f})")

    test_acc = None
    if test_ds is not None:
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_accuracies, _ = evaluate(all_classifiers, test_loader, device)
        test_acc = test_accuracies[best_key]
        print(f"Test accuracy (using best-on-val classifier {best_key}): {test_acc:.4f}")

        with open(metrics_path, "a") as f:
            f.write(json.dumps({
                "final": True,
                "best_classifier": {"name": best_key, "val_accuracy": best_val_acc, "test_accuracy": test_acc},
            }) + "\n")

    return {"best_classifier": best_key, "val_accuracy": best_val_acc, "test_accuracy": test_acc}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def get_args_parser():
    parser = argparse.ArgumentParser(description="Original-style DINOv2 linear probe on cached HDF5 features")
    parser.add_argument("--train-features", required=True, type=str)
    parser.add_argument("--val-features", required=True, type=str)
    parser.add_argument("--test-features", default=None, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--max-iter", type=int, default=12500,
                         help="Total SGD iterations, as in the paper (default 12500).")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-period-iterations", type=int, default=1250)
    parser.add_argument("--checkpoint-period-iterations", type=int, default=1250)
    parser.add_argument(
        "--learning-rates", nargs="+", type=float,
        default=[0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005,
                 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5],
        help="Grid of base learning rates, exactly as in the paper "
             "(scaled internally by batch_size/256).",
    )
    return parser


def main():
    args = get_args_parser().parse_args()

    print(f"Loading train features from {args.train_features}")
    train_ds = HDF5FeatureDataset(args.train_features)
    print(f"Loading val features from {args.val_features}")
    val_ds = HDF5FeatureDataset(args.val_features)
    test_ds = None
    if args.test_features:
        print(f"Loading test features from {args.test_features}")
        test_ds = HDF5FeatureDataset(args.test_features)

    results = train_linear_probe(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        output_dir=args.output_dir,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        learning_rates=args.learning_rates,
        num_workers=args.num_workers,
        eval_period_iterations=args.eval_period_iterations,
        checkpoint_period_iterations=args.checkpoint_period_iterations,
    )

    print("\nFinal results:", results)


if __name__ == "__main__":
    main()