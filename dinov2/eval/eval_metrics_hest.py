import argparse
import os
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import umap
import wandb
from sklearn.linear_model import RidgeCV
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE

parser = argparse.ArgumentParser(description="Feature extraction")
os.environ["WANDB__SERVICE_WAIT"] = "300"

parser.add_argument("--knn", help="perform knn or not", default=True, type=bool)
parser.add_argument(
    "--logistic_regression", "--logistic-regression", "-log",
    help="perform logistic regression or not", default=True, type=bool,
)
parser.add_argument("--umap", help="perform umap or not", default=True, type=bool)
parser.add_argument(
    "--path_folder", "--path-folder", "-p",
    help="path to folder containing subfolders with training, val and test data",
    default="", type=str,
)
parser.add_argument("--save_dir", "--save-dir", "-s", help="specify where to save the umap", default="", type=str)
parser.add_argument("--dataset", help="specify the dataset name", default="", type=str)
parser.add_argument("--model_name", help="specify the model name", default="dinov2_finetuned", type=str)
parser.add_argument("--checkpoint", help="path to checkpoint", default=None, type=str)
parser.add_argument("--log_embeddings", help="log feature embeddings to wandb", default=False, type=bool)
parser.add_argument(
    "--topk_values", help="comma-separated list of k values for top-k accuracy (e.g. 1,5,10)",
    default="1,5,10", type=str,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_hest_features(file_path):
    """Load HEST features where labels are float32 expr vectors, not ints."""
    with h5py.File(file_path, "r") as hf:
        features = hf["features"][:]
        labels   = hf["labels"][:]      # (N, 50) float32
        ids      = hf["ids"][:] if "ids" in hf else None
    return features, labels, ids


def get_data_hest(args):
    """
    Mirrors get_data() exactly but returns expr matrices instead of int labels.
    Returns (train_data, train_expr, test_data, test_expr) as numpy arrays.
    """
    train_file = os.path.join(args.path_folder, "train_data", "train_features.h5")
    val_file   = os.path.join(args.path_folder, "val_data",   "val_features.h5")
    test_file  = os.path.join(args.path_folder, "test_data",  "test_features.h5")

    print(f"Loading train: {train_file}")
    train_feat, train_expr, _ = load_hest_features(train_file)
    print(f"Loading val:   {val_file}")
    val_feat,   val_expr,   _ = load_hest_features(val_file)
    print(f"Loading test:  {test_file}")
    test_feat,  test_expr,  _ = load_hest_features(test_file)

    # Combine train + val (mirrors get_data exactly)
    train_data = np.vstack([train_feat, val_feat])
    train_expr = np.vstack([train_expr, val_expr])

    if train_data.ndim > 2:
        train_data = train_data.reshape(train_data.shape[0], -1)
    if test_feat.ndim > 2:
        test_feat  = test_feat.reshape(test_feat.shape[0], -1)

    print(f"  train: {train_data.shape}, expr: {train_expr.shape}")
    print(f"  test:  {test_feat.shape},  expr: {test_expr.shape}")

    return train_data, train_expr, test_feat, test_expr


def train_and_evaluate_ridge(train_data, train_expr, test_data, test_expr,
                              dataset, save_dir):
    """
    Ridge regression on frozen embeddings for HEST gene expression prediction.
    Reports per-gene Pearson correlation coefficient (PCC) — the LEMON metric.
    Mirrors train_and_evaluate_logistic_regression in structure.
    """
    wandb.init(
        entity="eshasadia2015-university-of-warwick",
        project="ridge_regression",
        name=f"{dataset}_{Path(save_dir).name}",
    )

    wandb.config.update({
        "num_features":  train_data.shape[1],
        "num_genes":     train_expr.shape[1],
        "train_samples": len(train_data),
        "test_samples":  len(test_data),
    })

    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
    ridge.fit(train_data, train_expr)

    pred_expr  = ridge.predict(test_data)       # (N_test, n_genes)
    n_genes    = test_expr.shape[1]

    per_gene_pcc = []
    for g in range(n_genes):
        try:
            r, _ = pearsonr(test_expr[:, g], pred_expr[:, g])
        except Exception:
            r = float("nan")
        per_gene_pcc.append(r)
    per_gene_pcc = np.array(per_gene_pcc)

    valid      = per_gene_pcc[~np.isnan(per_gene_pcc)]
    mean_pcc   = float(np.mean(valid))   if len(valid) else float("nan")
    median_pcc = float(np.median(valid)) if len(valid) else float("nan")

    print(f"\n=== Ridge Regression (HEST) ===")
    print(f"  Mean PCC:   {mean_pcc:.4f}")
    print(f"  Median PCC: {median_pcc:.4f}")
    print(f"  Best PCC:   {float(np.nanmax(per_gene_pcc)):.4f}")
    print(f"  Worst PCC:  {float(np.nanmin(per_gene_pcc)):.4f}")

    metrics = {
        "Mean_PCC":   mean_pcc,
        "Median_PCC": median_pcc,
        "Best_PCC":   float(np.nanmax(per_gene_pcc)),
        "Worst_PCC":  float(np.nanmin(per_gene_pcc)),
        "Std_PCC":    float(np.nanstd(per_gene_pcc)),
    }
    wandb.log(metrics)

    # Per-gene bar chart
    plt.figure(figsize=(14, 5))
    plt.bar(range(n_genes), per_gene_pcc, alpha=0.8, color="steelblue")
    plt.axhline(y=mean_pcc, color="crimson", linestyle="--",
                label=f"Mean PCC: {mean_pcc:.4f}")
    plt.xlabel("Gene index")
    plt.ylabel("Pearson r")
    plt.title("Per-Gene Pearson Correlation (HEST)")
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    wandb.log({"per_gene_pcc": wandb.Image(plt)})
    plt.close()

    # Save CSVs
    os.makedirs(save_dir, exist_ok=True)
    pd.DataFrame({
        "gene_idx":  range(n_genes),
        "pearson_r": per_gene_pcc,
    }).to_csv(
        os.path.join(save_dir, f"{Path(save_dir).name}_hest_per_gene_pcc.csv"),
        index=False,
    )
    pd.DataFrame({k: [v] for k, v in metrics.items()}).to_csv(
        os.path.join(save_dir, f"{Path(save_dir).name}_hest_metrics.csv"),
        index=False,
    )

    wandb.finish()
    return metrics
def load_combined_features(file_path):
    """Load features from a single combined HDF5 file."""
    with h5py.File(file_path, "r") as hf:
        features = hf["features"][:]
        labels = hf["labels"][:]
        ids = hf["ids"][:] if "ids" in hf else None
    return features, labels, ids


def get_data(args):
    if args.checkpoint is not None:
        args.path_folder = (
            Path(args.path_folder)
            / args.dataset
            / f"{args.model_name}_{Path(args.checkpoint).parent.name}_{Path(args.checkpoint).stem}"
        )

    train_dir = os.path.join(args.path_folder, "train_data")
    val_dir   = os.path.join(args.path_folder, "val_data")
    test_dir  = os.path.join(args.path_folder, "test_data")

    train_file = os.path.join(train_dir, "train_features.h5")
    val_file   = os.path.join(val_dir,   "val_features.h5")
    test_file  = os.path.join(test_dir,  "test_features.h5")

    print(f"Loading training data from:   {train_file}")
    train_features, train_labels, train_ids = load_combined_features(train_file)

    print(f"Loading validation data from: {val_file}")
    val_features, val_labels, val_ids = load_combined_features(val_file)

    print(f"Loading test data from:       {test_file}")
    test_features, test_labels, test_ids = load_combined_features(test_file)

    # Combine train + val
    train_data   = np.vstack([train_features, val_features])
    train_labels = np.concatenate([train_labels, val_labels])

    if len(train_data.shape) > 2:
        train_data = train_data.reshape(train_data.shape[0], -1)
    if len(test_features.shape) > 2:
        test_features = test_features.reshape(test_features.shape[0], -1)

    test_data    = test_features
    test_labels  = test_labels.flatten()
    train_labels = train_labels.flatten()

    return train_data, train_labels, test_data, test_labels


def test_data_creation():
    num_samples, num_features = 100, 100
    synthetic_data   = np.random.rand(num_samples, num_features)
    synthetic_labels = np.random.randint(0, 2, size=num_samples)
    return train_test_split(synthetic_data, synthetic_labels, test_size=0.2)


# ---------------------------------------------------------------------------
# Core metric helpers
# ---------------------------------------------------------------------------

def topk_accuracy(y_true, y_pred_proba, k):
    """
    Fraction of samples where the true label falls within the top-k predicted classes.
    Works for any k and any number of classes.
    """
    # argsort descending → shape (N, C); take first k columns
    top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]
    correct = sum(y_true[i] in top_k_preds[i] for i in range(len(y_true)))
    return correct / len(y_true)


def compute_topk_accuracies(y_true, y_pred_proba, k_values):
    """Return dict of top-k accuracies for every k in k_values (capped at num_classes)."""
    num_classes = y_pred_proba.shape[1]
    results = {}
    for k in k_values:
        k_capped = min(k, num_classes)
        results[f"Top{k_capped}_Accuracy"] = topk_accuracy(y_true, y_pred_proba, k_capped)
    return results


# def compute_auroc(y_true, y_pred_proba):
#     """
#     Weighted one-vs-rest AUROC for multi-class; standard AUROC for binary.
#     Returns float or np.nan if computation fails.
#     """
#     num_classes = y_pred_proba.shape[1]
#     try:
#         if num_classes == 2:
#             return roc_auc_score(y_true, y_pred_proba[:, 1])
#         else:
#             y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
#             return roc_auc_score(y_true_bin, y_pred_proba, multi_class="ovr", average="weighted")
#     except ValueError as e:
#         print(f"[AUROC] Could not compute: {e}")
#         return float("nan")
def normalize_proba_shape(y_pred_proba):
    """
    Ensures probability matrix always has shape:
        binary      -> (N,2)
        multiclass  -> (N,C)
    """

    y_pred_proba = np.asarray(y_pred_proba)

    # Case: (N,)
    if y_pred_proba.ndim == 1:
        y_pred_proba = np.column_stack([
            1 - y_pred_proba,
            y_pred_proba
        ])

    # Case: (N,1)
    elif y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 1:
        pos_probs = y_pred_proba[:, 0]
        y_pred_proba = np.column_stack([
            1 - pos_probs,
            pos_probs
        ])

    return y_pred_proba
def compute_auroc(y_true, y_pred_proba):
    """
    Robust AUROC for binary and multiclass classification.
    """

    y_pred_proba = normalize_proba_shape(y_pred_proba)

    num_classes = len(np.unique(y_true))

    try:
        # Binary
        if num_classes == 2:

            return roc_auc_score(
                y_true,
                y_pred_proba[:, 1]
            )

        # Multiclass
        else:

            y_true_bin = label_binarize(
                y_true,
                classes=np.unique(y_true)
            )

            return roc_auc_score(
                y_true_bin,
                y_pred_proba,
                multi_class="ovr",
                average="weighted"
            )

    except ValueError as e:
        print(f"[AUROC] Could not compute: {e}")
        return float("nan")
def compute_per_class_auroc(y_true, y_pred_proba):
    """
    Robust per-class AUROC for binary + multiclass datasets.
    """

    y_pred_proba = normalize_proba_shape(y_pred_proba)

    classes = np.unique(y_true)
    num_classes = len(classes)

    y_true_bin = label_binarize(y_true, classes=classes)

    per_class = {}

    # Binary classification
    if num_classes == 2:

        try:
            auc = roc_auc_score(
                y_true_bin[:, 0],
                y_pred_proba[:, 1]
            )

            per_class[f"class_{classes[1]}_auroc"] = auc

        except ValueError:
            per_class[f"class_{classes[1]}_auroc"] = float("nan")

    # Multiclass
    else:

        for i in range(num_classes):

            try:

                # Skip invalid classes
                if len(np.unique(y_true_bin[:, i])) < 2:
                    per_class[f"class_{classes[i]}_auroc"] = float("nan")
                    continue

                auc = roc_auc_score(
                    y_true_bin[:, i],
                    y_pred_proba[:, i]
                )

                per_class[f"class_{classes[i]}_auroc"] = auc

            except ValueError:
                per_class[f"class_{classes[i]}_auroc"] = float("nan")

    return per_class
# def compute_per_class_auroc(y_true, y_pred_proba):
#     """Per-class AUROC (one-vs-rest). Returns dict {class_X_auroc: value}."""
#     num_classes = y_pred_proba.shape[1]
#     y_true_bin  = label_binarize(y_true, classes=np.arange(num_classes))
#     per_class   = {}
#     for i in range(num_classes):
#         try:
#             per_class[f"class_{i}_auroc"] = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
#         except ValueError:
#             per_class[f"class_{i}_auroc"] = float("nan")
#     return per_class


def compute_precision_recall(y_true, y_pred):
    """
    Macro / weighted / per-class precision and recall.
    Returns flat dict ready for wandb.log().
    """
    macro_precision    = precision_score(y_true, y_pred, average="macro",    zero_division=0)
    weighted_precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_recall       = recall_score(   y_true, y_pred, average="macro",    zero_division=0)
    weighted_recall    = recall_score(   y_true, y_pred, average="weighted", zero_division=0)

    per_class_prec = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_rec  = recall_score(   y_true, y_pred, average=None, zero_division=0)

    unique_classes = np.unique(y_true)
    per_class_dict = {}
    for cls, p, r in zip(unique_classes, per_class_prec, per_class_rec):
        per_class_dict[f"class_{cls}_precision"] = p
        per_class_dict[f"class_{cls}_recall"]    = r

    return {
        "Macro_Precision":    macro_precision,
        "Weighted_Precision": weighted_precision,
        "Macro_Recall":       macro_recall,
        "Weighted_Recall":    weighted_recall,
        **per_class_dict,
    }


def compute_f1_scores(y_true, y_pred):
    """Macro / weighted / per-class F1. Returns structured dict."""
    macro_f1    = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    per_class   = f1_score(y_true, y_pred, average=None)

    unique_classes = np.unique(y_true)
    per_class_dict = {f"class_{cls}_f1": f1 for cls, f1 in zip(unique_classes, per_class)}

    return {
        "macro_f1":          macro_f1,
        "weighted_f1":       weighted_f1,
        "per_class_f1":      per_class_dict,
        "per_class_f1_values": per_class,
    }


def compute_std_metrics(f1_results, precision_recall_results):
    """
    Standard deviations across per-class scores.
    Returns dict of std metrics for wandb.
    """
    per_class_f1   = list(f1_results["per_class_f1"].values())
    per_class_prec = [v for k, v in precision_recall_results.items() if k.endswith("_precision") and k.startswith("class_")]
    per_class_rec  = [v for k, v in precision_recall_results.items() if k.endswith("_recall")    and k.startswith("class_")]

    return {
        "Std_PerClass_F1":        np.std(per_class_f1),
        "Std_PerClass_Precision": np.std(per_class_prec),
        "Std_PerClass_Recall":    np.std(per_class_rec),
        "Std_MacroWeighted_F1":   np.std([f1_results["macro_f1"], f1_results["weighted_f1"]]),
    }


def compute_cohen_kappa(y_true, y_pred):
    """
    Cohen's kappa: agreement corrected for chance. Range −1 to 1; >0.8 = strong.
    Also returns quadratic-weighted kappa (penalises large disagreements more —
    useful when classes are ordinal, e.g. severity grades).
    """
    kappa          = cohen_kappa_score(y_true, y_pred, weights=None)
    kappa_quad     = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    kappa_lin_wt   = cohen_kappa_score(y_true, y_pred, weights="linear")
    return {
        "Cohen_Kappa":           kappa,
        "Cohen_Kappa_Quadratic": kappa_quad,
        "Cohen_Kappa_Linear_Wt": kappa_lin_wt,
    }

def compute_map(y_true, y_pred_proba):
    """
    Mean Average Precision (mAP).
    Works for binary and multiclass datasets.
    """

    y_pred_proba = normalize_proba_shape(y_pred_proba)

    classes = np.unique(y_true)
    num_classes = len(classes)

    y_true_bin = label_binarize(y_true, classes=classes)

    per_class_ap = {}
    ap_values = []

    # Binary
    if num_classes == 2:

        try:
            ap = average_precision_score(
                y_true_bin[:, 0],
                y_pred_proba[:, 1]
            )
        except ValueError:
            ap = float("nan")

        per_class_ap[f"class_{classes[1]}_AP"] = ap
        ap_values.append(ap)

    # Multiclass
    else:

        for i in range(num_classes):

            try:

                if len(np.unique(y_true_bin[:, i])) < 2:
                    ap = float("nan")
                else:
                    ap = average_precision_score(
                        y_true_bin[:, i],
                        y_pred_proba[:, i]
                    )

            except ValueError:
                ap = float("nan")

            per_class_ap[f"class_{classes[i]}_AP"] = ap
            ap_values.append(ap)

    valid = [v for v in ap_values if not np.isnan(v)]

    mean_ap = np.mean(valid) if valid else float("nan")

    return {"mAP": mean_ap, **per_class_ap}, np.array(ap_values)
# def compute_map(y_true, y_pred_proba):
#     """
#     Mean Average Precision (mAP): macro-average of per-class AP (area under PR curve).
#     Per-class AP logged individually so you can spot which classes are hardest.
#     Returns (flat_dict_for_wandb, np.array of per-class AP values for plotting).
#     """
#     num_classes = y_pred_proba.shape[1]
#     y_true_bin  = label_binarize(y_true, classes=np.arange(num_classes))

#     per_class_ap = {}
#     ap_values    = []
#     for i in range(num_classes):
#         try:
#             ap = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
#         except ValueError:
#             ap = float("nan")
#         per_class_ap[f"class_{i}_AP"] = ap
#         ap_values.append(ap)

#     valid = [v for v in ap_values if not np.isnan(v)]
#     mean_ap = np.mean(valid) if valid else float("nan")

#     return {"mAP": mean_ap, **per_class_ap}, np.array(ap_values)


def build_full_metrics(y_true, y_pred, y_pred_proba, k_values=(1, 5, 10)):
    """
    Single entry-point that assembles every metric into a flat dict for wandb.log().
    """
    f1_results              = compute_f1_scores(y_true, y_pred)
    pr_results              = compute_precision_recall(y_true, y_pred)
    std_results             = compute_std_metrics(f1_results, pr_results)
    topk_results            = compute_topk_accuracies(y_true, y_pred_proba, k_values)
    auroc                   = compute_auroc(y_true, y_pred_proba)
    per_cls_auroc           = compute_per_class_auroc(y_true, y_pred_proba)
    kappa_results           = compute_cohen_kappa(y_true, y_pred)
    map_results, ap_values  = compute_map(y_true, y_pred_proba)
    accuracy                = accuracy_score(y_true, y_pred)
    balanced_acc            = balanced_accuracy_score(y_true, y_pred)

    metrics = {
        "Accuracy":         accuracy,
        "Balanced_Acc":     balanced_acc,
        "AUROC":            auroc,
        "Macro_F1":         f1_results["macro_f1"],
        "Weighted_F1":      f1_results["weighted_f1"],
        **f1_results["per_class_f1"],
        **pr_results,
        **std_results,
        **topk_results,
        **per_cls_auroc,
        **kappa_results,
        **map_results,
    }
    return metrics, f1_results, ap_values


# ---------------------------------------------------------------------------
# Wandb visualisation helpers
# ---------------------------------------------------------------------------

def log_confusion_matrix(y_true, y_pred, class_names=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    if class_names is not None:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
    else:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.close()
    return cm


def log_roc_curves(y_true, y_pred_proba, num_classes):
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
    plt.figure(figsize=(10, 8))
    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
    else:
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            auc = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
            plt.plot(fpr, tpr, label=f"Class {i} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves"); plt.legend(loc="lower right"); plt.grid(True, alpha=0.3)
    wandb.log({"roc_curves": wandb.Image(plt)})
    plt.close()


def log_f1_scores_plot(f1_results):
    per_class_f1 = f1_results["per_class_f1_values"]
    classes = list(range(len(per_class_f1)))
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, per_class_f1, alpha=0.8, color="steelblue")
    plt.axhline(y=f1_results["macro_f1"],    color="red",   linestyle="--",
                label=f'Macro F1: {f1_results["macro_f1"]:.4f}')
    plt.axhline(y=f1_results["weighted_f1"], color="green", linestyle="--",
                label=f'Weighted F1: {f1_results["weighted_f1"]:.4f}')
    for bar, f1 in zip(bars, per_class_f1):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{f1:.3f}", ha="center", va="bottom", fontsize=10)
    plt.xlabel("Class"); plt.ylabel("F1 Score"); plt.title("Per-Class F1 Scores")
    plt.xticks(classes, [f"Class {c}" for c in classes])
    plt.legend(); plt.grid(True, alpha=0.3, axis="y"); plt.ylim(0, 1.05); plt.tight_layout()
    wandb.log({"f1_scores_per_class": wandb.Image(plt)})
    plt.close()


def log_topk_accuracy_plot(topk_results):
    """Bar chart of top-k accuracies."""
    labels = list(topk_results.keys())
    values = list(topk_results.values())
    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color="mediumpurple", alpha=0.85)
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{v:.4f}", ha="center", va="bottom", fontsize=11)
    plt.xlabel("k"); plt.ylabel("Accuracy"); plt.title("Top-k Accuracy")
    plt.ylim(0, 1.05); plt.grid(True, alpha=0.3, axis="y"); plt.tight_layout()
    wandb.log({"topk_accuracy": wandb.Image(plt)})
    plt.close()


def log_precision_recall_plot(pr_results):
    """Grouped bar chart: per-class precision and recall."""
    classes = sorted(set(
        int(k.split("_")[1])
        for k in pr_results
        if k.startswith("class_") and k.endswith("_precision")
    ))
    precs = [pr_results[f"class_{c}_precision"] for c in classes]
    recs  = [pr_results[f"class_{c}_recall"]    for c in classes]

    x = np.arange(len(classes)); width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, precs, width, label="Precision", alpha=0.8, color="steelblue")
    ax.bar(x + width / 2, recs,  width, label="Recall",    alpha=0.8, color="coral")
    ax.set_xlabel("Class"); ax.set_ylabel("Score")
    ax.set_title("Per-Class Precision and Recall")
    ax.set_xticks(x); ax.set_xticklabels([f"Class {c}" for c in classes])
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    wandb.log({"precision_recall_per_class": wandb.Image(plt)})
    plt.close()


def log_per_class_auroc_plot(per_cls_auroc):
    """Bar chart of per-class AUROC scores."""
    classes = sorted(per_cls_auroc.keys())
    values  = [per_cls_auroc[c] for c in classes]
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, values, alpha=0.8, color="darkorange")
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    plt.xlabel("Class"); plt.ylabel("AUROC"); plt.title("Per-Class AUROC (OvR)")
    plt.ylim(0, 1.05); plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis="y"); plt.tight_layout()
    wandb.log({"auroc_per_class": wandb.Image(plt)})
    plt.close()


def log_feature_importance(model, feature_names=None, top_k=20):
    if not hasattr(model, "coef_"):
        return
    coef = np.abs(model.coef_)
    importance = np.mean(coef, axis=0) if coef.shape[0] > 1 else coef[0]
    top_indices  = np.argsort(importance)[-top_k:][::-1]
    top_importance = importance[top_indices]
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importance))]
    top_features = [feature_names[i] for i in top_indices]
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(top_features)), top_importance)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel("Absolute Coefficient Value"); plt.title(f"Top {top_k} Most Important Features")
    plt.tight_layout()
    wandb.log({"feature_importance": wandb.Image(plt)})
    plt.close()
    importance_df = pd.DataFrame({"Feature": top_features, "Importance": top_importance})
    wandb.log({"feature_importance_table": wandb.Table(dataframe=importance_df)})


def log_prediction_distribution(y_true, y_pred, y_pred_proba):
    max_probs = np.max(y_pred_proba, axis=1)
    correct_mask = (y_true == y_pred)
    correct_probs   = max_probs[correct_mask]
    incorrect_probs = max_probs[~correct_mask]
    plt.figure(figsize=(10, 6))
    plt.hist(correct_probs,   bins=50, alpha=0.5, label="Correct Predictions",   color="green")
    plt.hist(incorrect_probs, bins=50, alpha=0.5, label="Incorrect Predictions", color="red")
    plt.xlabel("Prediction Confidence"); plt.ylabel("Count")
    plt.title("Prediction Confidence Distribution"); plt.legend(); plt.grid(True, alpha=0.3)
    wandb.log({"prediction_confidence": wandb.Image(plt)})
    plt.close()
    wandb.log({
        "avg_correct_confidence":    np.mean(correct_probs),
        "avg_incorrect_confidence":  np.mean(incorrect_probs) if len(incorrect_probs) else float("nan"),
        "median_correct_confidence": np.median(correct_probs),
        "median_incorrect_confidence": np.median(incorrect_probs) if len(incorrect_probs) else float("nan"),
    })


def log_class_distribution(train_labels, test_labels):
    unique_classes = np.unique(np.concatenate([train_labels, test_labels]))
    train_counts = [np.sum(train_labels == c) for c in unique_classes]
    test_counts  = [np.sum(test_labels  == c) for c in unique_classes]
    x = np.arange(len(unique_classes)); width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, train_counts, width, label="Train", alpha=0.8)
    ax.bar(x + width / 2, test_counts,  width, label="Test",  alpha=0.8)
    ax.set_xlabel("Class"); ax.set_ylabel("Count"); ax.set_title("Class Distribution")
    ax.set_xticks(x); ax.set_xticklabels([f"Class {c}" for c in unique_classes])
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    wandb.log({"class_distribution": wandb.Image(plt)})
    plt.close()
    dist_df = pd.DataFrame({
        "Class":            unique_classes,
        "Train_Count":      train_counts,
        "Test_Count":       test_counts,
        "Train_Percentage": [c / sum(train_counts) * 100 for c in train_counts],
        "Test_Percentage":  [c / sum(test_counts)  * 100 for c in test_counts],
    })
    wandb.log({"class_distribution_table": wandb.Table(dataframe=dist_df)})


def log_per_class_metrics(y_true, y_pred, y_pred_proba):
    report = classification_report(y_true, y_pred, output_dict=True)
    classes = sorted([k for k in report if k not in ["accuracy", "macro avg", "weighted avg"]])
    metrics_data = [
        {"Class": cls, "Precision": report[cls]["precision"],
         "Recall": report[cls]["recall"], "F1-Score": report[cls]["f1-score"],
         "Support": report[cls]["support"]}
        for cls in classes if isinstance(report[cls], dict)
    ]
    df = pd.DataFrame(metrics_data)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, col, title in zip(
        axes.flat,
        ["Precision", "Recall", "F1-Score", "Support"],
        ["Precision per Class", "Recall per Class", "F1-Score per Class", "Support per Class"],
    ):
        ax.bar(df["Class"], df[col])
        ax.set_title(title); ax.set_ylabel(col)
        ax.tick_params(axis="x", rotation=45); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    wandb.log({"per_class_metrics": wandb.Image(plt)})
    plt.close()


def log_std_summary(std_results):
    """Horizontal bar chart for standard deviation metrics."""
    labels = list(std_results.keys())
    values = list(std_results.values())
    plt.figure(figsize=(8, 4))
    bars = plt.barh(labels, values, color="slategray", alpha=0.8)
    for bar, v in zip(bars, values):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                 f"{v:.4f}", va="center", fontsize=10)
    plt.xlabel("Standard Deviation"); plt.title("Metric Standard Deviations")
    plt.tight_layout()
    wandb.log({"std_metrics": wandb.Image(plt)})
    plt.close()


def log_map_plot(ap_values, mean_ap):
    """Bar chart of per-class Average Precision with mAP reference line."""
    classes = list(range(len(ap_values)))
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, ap_values, alpha=0.8, color="mediumseagreen")
    plt.axhline(y=mean_ap, color="crimson", linestyle="--",
                label=f"mAP: {mean_ap:.4f}")
    for bar, v in zip(bars, ap_values):
        if not np.isnan(v):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    plt.xlabel("Class"); plt.ylabel("Average Precision")
    plt.title("Per-Class Average Precision (AP) and mAP")
    plt.xticks(classes, [f"Class {c}" for c in classes])
    plt.legend(); plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3, axis="y"); plt.tight_layout()
    wandb.log({"map_per_class": wandb.Image(plt)})
    plt.close()


def log_kappa_summary(kappa_results):
    """Bar chart of the three kappa variants with reference thresholds."""
    labels = list(kappa_results.keys())
    values = list(kappa_results.values())
    plt.figure(figsize=(7, 4))
    colors = ["steelblue" if v >= 0.8 else "goldenrod" if v >= 0.6 else "tomato"
              for v in values]
    bars = plt.bar(labels, values, color=colors, alpha=0.85)
    # Reference lines
    for thresh, label, ls in [(0.8, "Strong (0.8)", "--"), (0.6, "Moderate (0.6)", ":")]:
        plt.axhline(y=thresh, color="gray", linestyle=ls, linewidth=1, label=label)
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{v:.4f}", ha="center", va="bottom", fontsize=10)
    plt.ylabel("Kappa"); plt.title("Cohen's Kappa variants")
    plt.ylim(min(0, min(values)) - 0.05, 1.05)
    plt.xticks(rotation=15); plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3, axis="y"); plt.tight_layout()
    wandb.log({"cohen_kappa": wandb.Image(plt)})
    plt.close()


# ---------------------------------------------------------------------------
# KNN
# ---------------------------------------------------------------------------

def perform_knn(train_data, train_labels, test_data, test_labels, save_dir, k_values=(1, 5, 10)):
    n_neighbors_values = [1, 20]

    for n_neighbors in n_neighbors_values:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(train_data, train_labels)

        test_predictions = knn.predict(test_data)
        test_proba       = knn.predict_proba(test_data)

        metrics, f1_results, ap_values = build_full_metrics(
            test_labels, test_predictions, test_proba, k_values=k_values
        )
        # Also compute and add topk std
        topk_vals = [v for k, v in metrics.items() if k.startswith("Top") and k.endswith("_Accuracy")]
        metrics["Std_TopK_Accuracy"] = np.std(topk_vals)

        # Console summary
        print(f"\n=== KNN  n_neighbors={n_neighbors} ===")
        for key in ["Accuracy", "Balanced_Acc", "AUROC", "mAP",
                    "Macro_F1", "Weighted_F1", "Macro_Precision", "Macro_Recall",
                    "Cohen_Kappa", "Cohen_Kappa_Quadratic",
                    "Std_PerClass_F1", "Std_PerClass_Precision", "Std_PerClass_Recall",
                    *[k for k in metrics if k.startswith("Top")]]:
            print(f"  {key}: {metrics.get(key, 'N/A'):.4f}")

        run_name = f"{n_neighbors}NN_{Path(save_dir).name}"
        wandb.init(entity="eshasadia2015-university-of-warwick", project="knn", name=run_name)
        wandb.log({"n_neighbors": n_neighbors, **metrics})

        # Classification report
        report    = classification_report(test_labels, test_predictions, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        wandb.log({"Classification Report": wandb.Table(dataframe=report_df)})

        # Visualisations
        num_classes   = len(np.unique(test_labels))
        topk_only     = {k: v for k, v in metrics.items() if k.startswith("Top") and k.endswith("_Accuracy")}
        per_cls_auroc = {k: v for k, v in metrics.items() if k.endswith("_auroc")}
        pr_only       = {k: v for k, v in metrics.items()
                         if k.startswith("class_") and (k.endswith("_precision") or k.endswith("_recall"))}
        std_only      = {k: v for k, v in metrics.items() if k.startswith("Std_")}
        kappa_only    = {k: v for k, v in metrics.items() if k.startswith("Cohen_Kappa")}

        log_confusion_matrix(test_labels, test_predictions)
        log_roc_curves(test_labels, test_proba, num_classes)
        log_f1_scores_plot(f1_results)
        log_topk_accuracy_plot(topk_only)
        log_precision_recall_plot({**pr_only})
        log_per_class_auroc_plot(per_cls_auroc)
        log_map_plot(ap_values, metrics["mAP"])
        log_kappa_summary(kappa_only)
        log_std_summary(std_only)
        log_prediction_distribution(test_labels, test_predictions, test_proba)
        log_per_class_metrics(test_labels, test_predictions, test_proba)

        wandb.finish()

        # Save metrics CSV
        os.makedirs(save_dir, exist_ok=True)
        metrics_csv = {k: [v] for k, v in metrics.items()}
        pd.DataFrame(metrics_csv).to_csv(
            os.path.join(save_dir, f"{Path(save_dir).name}_{n_neighbors}NN_metrics.csv"), index=False
        )


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------

def create_umap(data, labels, save_dir, filename_addon="train"):
    reducer   = umap.UMAP()
    umap_data = reducer.fit_transform(data)

    umap_dir = os.path.join(save_dir, "umaps")
    os.makedirs(umap_dir, exist_ok=True)

    for size in [(48, 32), (36, 24), (24, 16), (12, 8), (6, 4)]:
        plt.figure(figsize=size, dpi=300)
        plt.scatter(umap_data[:, 0], umap_data[:, 1], c=labels, s=0.1, cmap="Spectral")
        plt.colorbar(); plt.title("UMAP")
        image_filename = f"umap_visualization_{Path(save_dir).name}_{size[0]}x{size[1]}_{filename_addon}.png"
        plt.savefig(os.path.join(umap_dir, image_filename)); plt.close()

    if wandb.run is not None:
        plt.figure(figsize=(12, 8))
        plt.scatter(umap_data[:, 0], umap_data[:, 1], c=labels, s=1, cmap="Spectral", alpha=0.6)
        plt.colorbar(label="Class"); plt.title(f"UMAP Visualization - {filename_addon}")
        plt.xlabel("UMAP 1"); plt.ylabel("UMAP 2")
        wandb.log({f"umap_{filename_addon}": wandb.Image(plt)})
        plt.close()


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

def train_and_evaluate_logistic_regression(
    train_data, train_labels, test_data, test_labels, dataset, save_dir,
    max_iter=1000, k_values=(1, 5, 10)
):
    wandb.init(
        entity="eshasadia2015-university-of-warwick",
        project="logistic_regression",
        name=f"{dataset}_{Path(save_dir).name}",
    )

    log_class_distribution(train_labels, test_labels)

    M   = train_data.shape[1]
    C   = len(np.unique(train_labels))
    l2  = 100 / (M * C)

    wandb.config.update({
        "num_features":  M, "num_classes": C, "l2_reg_coef": l2,
        "C": 1 / l2, "max_iter": max_iter,
        "train_samples": len(train_labels), "test_samples": len(test_labels),
    })

    # logistic_reg = LogisticRegression(
    #     C=1 / l2, max_iter=max_iter, multi_class="multinomial", solver="lbfgs"
    # )
    logistic_reg = LogisticRegression(
        C=1 / l2,
        max_iter=max_iter,
        solver="lbfgs"
    )
    logistic_reg.fit(train_data, train_labels)

    test_predictions = logistic_reg.predict(test_data)
    predicted_proba  = logistic_reg.predict_proba(test_data)

    metrics, f1_results, ap_values = build_full_metrics(
        test_labels, test_predictions, predicted_proba, k_values=k_values
    )
    topk_vals = [v for k, v in metrics.items() if k.startswith("Top") and k.endswith("_Accuracy")]
    metrics["Std_TopK_Accuracy"] = np.std(topk_vals)

    loss = log_loss(test_labels, predicted_proba)
    metrics["Final_Loss"] = loss

    # Console summary
    print(f"\n=== Logistic Regression ===")
    for key in ["Final_Loss", "Accuracy", "Balanced_Acc", "AUROC", "mAP",
                "Macro_F1", "Weighted_F1", "Macro_Precision", "Macro_Recall",
                "Cohen_Kappa", "Cohen_Kappa_Quadratic",
                "Std_PerClass_F1", "Std_PerClass_Precision", "Std_PerClass_Recall",
                *[k for k in metrics if k.startswith("Top")]]:
        print(f"  {key}: {metrics.get(key, 'N/A'):.4f}")

    print("\nPer-class F1:", f1_results["per_class_f1"])
    report    = classification_report(test_labels, test_predictions, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print(classification_report(test_labels, test_predictions))

    # Save single metrics CSV
    os.makedirs(save_dir, exist_ok=True)
    pd.DataFrame({k: [v] for k, v in metrics.items()}).to_csv(
        os.path.join(save_dir, f"{Path(save_dir).name}_logreg_metrics.csv"), index=False
    )

    # Log to wandb
    wandb.log({**metrics, "Classification Report": wandb.Table(dataframe=report_df)})

    # Visualisations
    topk_only     = {k: v for k, v in metrics.items() if k.startswith("Top") and k.endswith("_Accuracy")}
    per_cls_auroc = {k: v for k, v in metrics.items() if k.endswith("_auroc")}
    pr_only       = {k: v for k, v in metrics.items()
                     if k.startswith("class_") and (k.endswith("_precision") or k.endswith("_recall"))}
    std_only      = {k: v for k, v in metrics.items() if k.startswith("Std_")}
    kappa_only    = {k: v for k, v in metrics.items() if k.startswith("Cohen_Kappa")}

    log_confusion_matrix(test_labels, test_predictions)
    log_roc_curves(test_labels, predicted_proba, C)
    log_f1_scores_plot(f1_results)
    log_topk_accuracy_plot(topk_only)
    log_precision_recall_plot(pr_only)
    log_per_class_auroc_plot(per_cls_auroc)
    log_map_plot(ap_values, metrics["mAP"])
    log_kappa_summary(kappa_only)
    log_std_summary(std_only)
    log_feature_importance(logistic_reg, top_k=min(20, M))
    log_prediction_distribution(test_labels, test_predictions, predicted_proba)
    log_per_class_metrics(test_labels, test_predictions, predicted_proba)

    wandb.finish()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    k_values = [int(k) for k in args.topk_values.split(",")]

    # ---- HEST: regression path ----
    if args.dataset == "hest":
        train_data, train_expr, test_data, test_expr = get_data_hest(args)
        print("HEST data loaded")
        print(f"  train_data:  {train_data.shape}")
        print(f"  train_expr:  {train_expr.shape}")
        print(f"  test_data:   {test_data.shape}")
        print(f"  test_expr:   {test_expr.shape}")
        args.save_dir = Path(args.save_dir) / args.dataset / args.model_name
        train_and_evaluate_ridge(
            train_data, train_expr, test_data, test_expr,
            args.dataset, args.save_dir,
        )
        return

    # ---- All other datasets: existing classification path (unchanged) ----
    train_data, train_labels, test_data, test_labels = get_data(args)
    print("Data fully loaded")
    print(f"  train_data:   {train_data.shape}")
    print(f"  train_labels: {train_labels.shape}")
    print(f"  test_data:    {test_data.shape}")
    print(f"  test_labels:  {test_labels.shape}")

    print("\n=== Quick Class Check ===")
    _, _, train_ids = load_combined_features(
        os.path.join(args.path_folder, "train_data", "train_features.h5")
    )
    if train_ids is not None:
        for label in sorted(np.unique(train_labels)):
            idx        = np.where(train_labels == label)[0][0]
            sample_id  = train_ids[idx].decode("utf-8")
            class_name = sample_id.split("_")[1]
            print(f"  Label {label} → {class_name}")

    if args.checkpoint is not None:
        args.model_name = (
            f"{args.model_name}_{Path(args.checkpoint).parent.name}_{Path(args.checkpoint).stem}"
        )
    args.save_dir = Path(args.save_dir) / args.dataset / args.model_name

    if args.logistic_regression:
        train_and_evaluate_logistic_regression(
            train_data, train_labels, test_data, test_labels,
            args.dataset, args.save_dir, max_iter=1000, k_values=k_values,
        )
        print("logistic_regression done")

    if args.umap:
        wandb.init(
            entity="eshasadia2015-university-of-warwick",
            project="umap_visualization",
            name=f"{args.dataset}_{args.model_name}_umap",
        )
        create_umap(train_data, train_labels, args.save_dir)
        create_umap(test_data,  test_labels,  args.save_dir, "test")
        wandb.finish()
        print("umap done")

    if args.knn:
        perform_knn(
            train_data, train_labels, test_data, test_labels,
            args.save_dir, k_values=k_values,
        )
        print("knn done")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)