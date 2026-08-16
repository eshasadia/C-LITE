import argparse
import os
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import umap
import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
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
parser.add_argument("--logistic_regression", "--logistic-regression", "-log", help="perform logistic regression or not", default=True, type=bool)
parser.add_argument("--umap", help="perform umap or not", default=True, type=bool)
parser.add_argument("--path_folder", "--path-folder", "-p", help="path to folder containing subfolders with training, val and test data", default="", type=str)
parser.add_argument("--save_dir", "--save-dir", "-s", help="specify where to save the umap", default="", type=str)
parser.add_argument("--dataset", help="specify the dataset name", default="", type=str)
parser.add_argument("--model_name", help="specify the model name", default="dinov2_finetuned", type=str)
parser.add_argument("--checkpoint", help="path to checkpoint", default=None, type=str)
parser.add_argument("--log_embeddings", help="log feature embeddings to wandb", default=False, type=bool)
parser.add_argument("--class_names", help="comma-separated class names in label order e.g. Healthy,NPM1,PML::RARA", default=None, type=str)


def load_combined_features(file_path):
    """Load features from a single combined HDF5 file"""
    with h5py.File(file_path, "r") as hf:
        features = hf["features"][:]
        labels = hf["labels"][:]
        ids = hf["ids"][:] if "ids" in hf else None
    return features, labels, ids


def get_data(args):
    if args.checkpoint is not None:
        args.path_folder = (
            Path(args.path_folder) / args.dataset
            / f"{args.model_name}_{Path(args.checkpoint).parent.name}_{Path(args.checkpoint).stem}"
        )

    train_dir = os.path.join(args.path_folder, "train_data")
    val_dir   = os.path.join(args.path_folder, "val_data")
    test_dir  = os.path.join(args.path_folder, "test_data")

    print(f"Loading training data from:   {os.path.join(train_dir, 'train_features.h5')}")
    train_features, train_labels, train_ids = load_combined_features(os.path.join(train_dir, "train_features.h5"))

    print(f"Loading validation data from: {os.path.join(val_dir, 'val_features.h5')}")
    val_features, val_labels, val_ids = load_combined_features(os.path.join(val_dir, "val_features.h5"))

    print(f"Loading test data from:       {os.path.join(test_dir, 'test_features.h5')}")
    test_features, test_labels, test_ids = load_combined_features(os.path.join(test_dir, "test_features.h5"))

    train_data   = np.vstack([train_features, val_features])
    train_labels = np.concatenate([train_labels, val_labels])

    if len(train_data.shape)    > 2: train_data    = train_data.reshape(train_data.shape[0], -1)
    if len(test_features.shape) > 2: test_features = test_features.reshape(test_features.shape[0], -1)

    test_data    = test_features
    test_labels  = test_labels.flatten()
    train_labels = train_labels.flatten()

    return train_data, train_labels, test_data, test_labels


def test_data_creation():
    num_samples  = 100
    num_features = 100
    synthetic_data   = np.random.rand(num_samples, num_features)
    synthetic_labels = np.random.randint(0, 2, size=num_samples)
    train_data, test_data, train_labels, test_labels = train_test_split(synthetic_data, synthetic_labels, test_size=0.2)
    return train_data, train_labels, test_data, test_labels


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def compute_f1_scores(y_true, y_pred):
    macro_f1    = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    per_class_f1 = f1_score(y_true, y_pred, average=None)
    unique_classes = np.unique(y_true)
    per_class_f1_dict = {f"class_{cls}_f1": f1 for cls, f1 in zip(unique_classes, per_class_f1)}
    return {
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class_f1': per_class_f1_dict,
        'per_class_f1_values': per_class_f1,
    }


def log_confusion_matrix(y_true, y_pred, class_names=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    if class_names is not None:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.close()
    return cm


def log_roc_curves(y_true, y_pred_proba, num_classes):
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
    plt.figure(figsize=(10, 8))
    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    else:
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
            plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curves'); plt.legend(loc="lower right"); plt.grid(True, alpha=0.3)
    wandb.log({"roc_curves": wandb.Image(plt)})
    plt.close()


def log_f1_scores_plot(f1_results):
    per_class_f1 = f1_results['per_class_f1_values']
    classes = list(range(len(per_class_f1)))
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, per_class_f1, alpha=0.8, color='steelblue')
    plt.axhline(y=f1_results['macro_f1'],    color='red',   linestyle='--', label=f'Macro F1: {f1_results["macro_f1"]:.4f}')
    plt.axhline(y=f1_results['weighted_f1'], color='green', linestyle='--', label=f'Weighted F1: {f1_results["weighted_f1"]:.4f}')
    for bar, f1 in zip(bars, per_class_f1):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{f1:.3f}', ha='center', va='bottom', fontsize=10)
    plt.xlabel('Class'); plt.ylabel('F1 Score'); plt.title('Per-Class F1 Scores')
    plt.xticks(classes, [f'Class {c}' for c in classes])
    plt.legend(); plt.grid(True, alpha=0.3, axis='y'); plt.ylim(0, 1.05); plt.tight_layout()
    wandb.log({"f1_scores_per_class": wandb.Image(plt)})
    plt.close()


def log_feature_importance(model, feature_names=None, top_k=20):
    if hasattr(model, 'coef_'):
        coef = np.abs(model.coef_)
        importance = np.mean(coef, axis=0) if coef.shape[0] > 1 else coef[0]
        top_indices   = np.argsort(importance)[-top_k:][::-1]
        top_importance = importance[top_indices]
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(importance))]
        top_features = [feature_names[i] for i in top_indices]
        plt.figure(figsize=(12, 6))
        plt.barh(range(len(top_features)), top_importance)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Absolute Coefficient Value'); plt.title(f'Top {top_k} Most Important Features')
        plt.tight_layout()
        wandb.log({"feature_importance": wandb.Image(plt)})
        plt.close()
        importance_df = pd.DataFrame({'Feature': top_features, 'Importance': top_importance})
        wandb.log({"feature_importance_table": wandb.Table(dataframe=importance_df)})


def log_prediction_distribution(y_true, y_pred, y_pred_proba):
    max_probs = np.max(y_pred_proba, axis=1)
    correct_mask    = (y_true == y_pred)
    correct_probs   = max_probs[correct_mask]
    incorrect_probs = max_probs[~correct_mask]
    plt.figure(figsize=(10, 6))
    plt.hist(correct_probs,   bins=50, alpha=0.5, label='Correct Predictions',   color='green')
    plt.hist(incorrect_probs, bins=50, alpha=0.5, label='Incorrect Predictions', color='red')
    plt.xlabel('Prediction Confidence'); plt.ylabel('Count')
    plt.title('Prediction Confidence Distribution'); plt.legend(); plt.grid(True, alpha=0.3)
    wandb.log({"prediction_confidence": wandb.Image(plt)})
    plt.close()
    wandb.log({
        "avg_correct_confidence":    np.mean(correct_probs),
        "avg_incorrect_confidence":  np.mean(incorrect_probs),
        "median_correct_confidence": np.median(correct_probs),
        "median_incorrect_confidence": np.median(incorrect_probs),
    })


def log_class_distribution(train_labels, test_labels):
    unique_classes = np.unique(np.concatenate([train_labels, test_labels]))
    train_counts = [np.sum(train_labels == c) for c in unique_classes]
    test_counts  = [np.sum(test_labels  == c) for c in unique_classes]
    x = np.arange(len(unique_classes)); width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, train_counts, width, label='Train', alpha=0.8)
    ax.bar(x + width/2, test_counts,  width, label='Test',  alpha=0.8)
    ax.set_xlabel('Class'); ax.set_ylabel('Count'); ax.set_title('Class Distribution')
    ax.set_xticks(x); ax.set_xticklabels([f'Class {c}' for c in unique_classes])
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    wandb.log({"class_distribution": wandb.Image(plt)})
    plt.close()
    dist_df = pd.DataFrame({
        'Class': unique_classes,
        'Train_Count': train_counts, 'Test_Count': test_counts,
        'Train_Percentage': [c/sum(train_counts)*100 for c in train_counts],
        'Test_Percentage':  [c/sum(test_counts) *100 for c in test_counts],
    })
    wandb.log({"class_distribution_table": wandb.Table(dataframe=dist_df)})


def log_per_class_metrics(y_true, y_pred, y_pred_proba):
    report  = classification_report(y_true, y_pred, output_dict=True)
    classes = sorted([k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']])
    metrics_data = []
    for cls in classes:
        if isinstance(report[cls], dict):
            metrics_data.append({
                'Class': cls,
                'Precision': report[cls]['precision'],
                'Recall':    report[cls]['recall'],
                'F1-Score':  report[cls]['f1-score'],
                'Support':   report[cls]['support'],
            })
    df = pd.DataFrame(metrics_data)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, (col, title) in zip(axes.flatten(), [
        ('Precision', 'Precision per Class'), ('Recall', 'Recall per Class'),
        ('F1-Score',  'F1-Score per Class'),  ('Support', 'Support per Class'),
    ]):
        ax.bar(df['Class'], df[col])
        ax.set_title(title); ax.set_ylabel(col)
        ax.tick_params(axis='x', rotation=45); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    wandb.log({"per_class_metrics": wandb.Image(plt)})
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# UMAP  (updated)
# ─────────────────────────────────────────────────────────────────────────────

def create_umap(data, labels, save_dir, filename_addon="train", class_names=None):
    """
    Creates:
      1. Central UMAP coloured by class with centroid annotations.
      2. One satellite UMAP per class (that class highlighted, rest greyed out).

    Parameters
    ----------
    data          : np.ndarray  (N, D)
    labels        : np.ndarray  (N,)  integer class labels
    save_dir      : str
    filename_addon: str
    class_names   : dict | None   {int_label: str_name}
                    e.g. {0: "Healthy", 1: "NPM1", 2: "PML::RARA"}
                    If None, labels are shown as "Class 0", "Class 1", …
    """

    umap_dir = os.path.join(save_dir, "umaps")
    os.makedirs(umap_dir, exist_ok=True)

    # Palettes for satellite plots (light = all patches of class, dark = random ~10% highlight)
    LIGHT = ["#AEC6E8", "#F4B88A", "#A8D8A8", "#C5B4E3", "#F4A6A0",
             "#FFFACD", "#B2DFDB", "#FFCCBC", "#D7CCC8", "#CFD8DC"]
    DARK  = ["#1F4E79", "#7B3300", "#1B5E20", "#4A148C", "#7B0000",
             "#F57F17", "#004D40", "#BF360C", "#4E342E", "#263238"]

    unique_classes = np.unique(labels)
    n_classes      = len(unique_classes)
    cmap           = plt.cm.get_cmap("tab20", n_classes)
    cluster_colors = {c: cmap(i) for i, c in enumerate(unique_classes)}

    def _name(cl):
        return class_names.get(int(cl), f"Class {cl}") if class_names else f"Class {cl}"

    # ── Fit UMAP once ────────────────────────────────────────────────────────
    print(f"Fitting UMAP [{filename_addon}] …")
    reducer   = umap.UMAP(random_state=42, n_neighbors=30, min_dist=0.1)
    umap_data = reducer.fit_transform(data)

    # ── 1. Central UMAP ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 10))

    for cl in unique_classes:
        mask  = labels == cl
        color = cluster_colors[cl]
        ax.scatter(umap_data[mask, 0], umap_data[mask, 1],
                   c=[color], s=2, alpha=0.5, linewidths=0, label=_name(cl))
        # Centroid annotation
        cx, cy = umap_data[mask, 0].mean(), umap_data[mask, 1].mean()
        ax.text(cx, cy, _name(cl), fontsize=9, ha="center", va="center",
                color="dimgray", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.6, linewidth=0))

    ax.set_title(f"UMAP – {filename_addon}", fontsize=14)
    ax.axis("off")
    ax.legend(markerscale=5, fontsize=8, loc="upper left",
              framealpha=0.7, title="Class")

    central_path = os.path.join(umap_dir, f"umap_central_{Path(save_dir).name}_{filename_addon}.png")
    fig.savefig(central_path, dpi=200, bbox_inches="tight")
    if wandb.run is not None:
        wandb.log({f"umap_central_{filename_addon}": wandb.Image(fig)})
    plt.close(fig)
    print(f"Saved central UMAP → {central_path}")

    # ── 2. Satellite UMAPs – one per class ───────────────────────────────────
    ncols = min(3, n_classes)
    nrows = int(np.ceil(n_classes / ncols))

    fig2, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).flatten() if n_classes > 1 else np.array([axes])

    rng = np.random.default_rng(42)

    for ax_idx, cl in enumerate(unique_classes):
        ax      = axes[ax_idx]
        mask_cl = labels == cl
        idx     = int(cl) % len(LIGHT)
        light   = LIGHT[idx]
        dark    = DARK[idx]

        # All other patches – grey background
        ax.scatter(umap_data[~mask_cl, 0], umap_data[~mask_cl, 1],
                   c="#D0D0D0", s=1, alpha=0.3, linewidths=0)

        # All patches of this class – light colour
        ax.scatter(umap_data[mask_cl, 0], umap_data[mask_cl, 1],
                   c=light, s=2, alpha=0.6, linewidths=0,
                   label="All patches of class")

        # Random ~10% of patches – dark colour (mimics "random patient" in reference)
        highlight_idx = rng.choice(
            np.where(mask_cl)[0],
            size=max(1, mask_cl.sum() // 10),
            replace=False,
        )
        ax.scatter(umap_data[highlight_idx, 0], umap_data[highlight_idx, 1],
                   c=dark, s=4, alpha=0.9, linewidths=0,
                   label="Random subset highlight")

        ax.set_title(_name(cl), fontsize=11, fontweight="bold")
        ax.axis("off")

        if ax_idx == 0:
            handles = [
                mpatches.Patch(color=light,     label="All patches of class"),
                mpatches.Patch(color=dark,      label="Random subset highlight"),
                mpatches.Patch(color="#D0D0D0", label="Other classes"),
            ]
            ax.legend(handles=handles, fontsize=7, loc="upper left", framealpha=0.7)

    # Hide any unused subplots
    for ax in axes[n_classes:]:
        ax.set_visible(False)

    fig2.suptitle(f"Per-Class UMAP – {filename_addon}", fontsize=14, y=1.01)
    fig2.tight_layout()

    satellite_path = os.path.join(umap_dir, f"umap_satellites_{Path(save_dir).name}_{filename_addon}.png")
    fig2.savefig(satellite_path, dpi=200, bbox_inches="tight")
    if wandb.run is not None:
        wandb.log({f"umap_satellites_{filename_addon}": wandb.Image(fig2)})
    plt.close(fig2)
    print(f"Saved satellite UMAPs → {satellite_path}")


# ─────────────────────────────────────────────────────────────────────────────
# KNN (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def perform_knn(train_data, train_labels, test_data, test_labels, save_dir):
    n_neighbors_values = [1, 20]
    for n_neighbors in n_neighbors_values:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(train_data, train_labels)
        test_predictions = knn.predict(test_data)
        test_proba       = knn.predict_proba(test_data)
        f1_results       = compute_f1_scores(test_labels, test_predictions)
        accuracy         = accuracy_score(test_labels, test_predictions)
        balanced_acc     = balanced_accuracy_score(test_labels, test_predictions)

        print(f"n_neighbors = {n_neighbors}")
        print(f"Accuracy: {accuracy}")
        print(f"Macro F1: {f1_results['macro_f1']:.4f}")
        print(f"Weighted F1: {f1_results['weighted_f1']:.4f}")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")

        run_name = f"{n_neighbors}NN_{Path(save_dir).name}"
        wandb.init(entity="eshasadia2015-university-of-warwick", project="knn", name=run_name)
        wandb.log({
            "n_neighbors": n_neighbors,
            "Accuracy": accuracy, "Balanced_Acc": balanced_acc,
            "Macro_F1": f1_results['macro_f1'], "Weighted_F1": f1_results['weighted_f1'],
            **f1_results['per_class_f1'],
        })

        report    = classification_report(test_labels, test_predictions, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        wandb.log({"Classification Report": wandb.Table(dataframe=report_df)})

        num_classes = len(np.unique(test_labels))
        log_confusion_matrix(test_labels, test_predictions)
        log_roc_curves(test_labels, test_proba, num_classes)
        log_prediction_distribution(test_labels, test_predictions, test_proba)
        log_per_class_metrics(test_labels, test_predictions, test_proba)
        log_f1_scores_plot(f1_results)
        wandb.finish()

        df_labels_to_save = pd.DataFrame({"True Labels": test_labels, "Predicted Labels": test_predictions})
        filename  = f"{Path(save_dir).name}_labels_and_predictions.csv"
        file_path = os.path.join(save_dir, filename)
        df_labels_to_save.to_csv(file_path, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Logistic Regression (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate_logistic_regression(
    train_data, train_labels, test_data, test_labels, dataset, save_dir, max_iter=1000
):
    wandb.init(
        entity="eshasadia2015-university-of-warwick",
        project="logistic_regression",
        name=f"{dataset}_{Path(save_dir).name}",
    )
    log_class_distribution(train_labels, test_labels)

    M = train_data.shape[1]
    C = len(np.unique(train_labels))
    l2_reg_coef = 100 / (M * C)

    wandb.config.update({
        "num_features": M, "num_classes": C, "l2_reg_coef": l2_reg_coef,
        "C": 1 / l2_reg_coef, "max_iter": max_iter,
        "train_samples": len(train_labels), "test_samples": len(test_labels),
    })

    logistic_reg = LogisticRegression(
        C=1 / l2_reg_coef, max_iter=max_iter,
        multi_class="multinomial", solver="lbfgs"
    )
    logistic_reg.fit(train_data, train_labels)

    test_predictions       = logistic_reg.predict(test_data)
    predicted_probabilities = logistic_reg.predict_proba(test_data)
    f1_results             = compute_f1_scores(test_labels, test_predictions)
    loss                   = log_loss(test_labels, predicted_probabilities)
    accuracy               = accuracy_score(test_labels, test_predictions)
    balanced_acc           = balanced_accuracy_score(test_labels, test_predictions)

    if C == 2:
        auroc = roc_auc_score(test_labels, predicted_probabilities[:, 1])
    else:
        y_true_bin = label_binarize(test_labels, classes=np.arange(C))
        auroc = roc_auc_score(y_true_bin, predicted_probabilities, multi_class='ovr', average='weighted')

    report    = classification_report(test_labels, test_predictions, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    os.makedirs(save_dir, exist_ok=True)
    df_labels_to_save = pd.DataFrame({"True Labels": test_labels, "Predicted Labels": test_predictions})
    df_labels_to_save.to_csv(os.path.join(save_dir, f"{Path(save_dir).name}_labels_and_predictions.csv"), index=False)

    predicted_probabilities_df = pd.DataFrame(
        predicted_probabilities,
        columns=[f"Probability Class {i}" for i in range(predicted_probabilities.shape[1])]
    )
    predicted_probabilities_df.to_csv(
        os.path.join(save_dir, f"{Path(save_dir).name}_predicted_probabilities_test.csv"), index=False
    )

    print(f"Final Loss: {loss}")
    print(f"Accuracy: {accuracy}")
    print(f"Macro F1: {f1_results['macro_f1']:.4f}")
    print(f"Weighted F1: {f1_results['weighted_f1']:.4f}")
    print(report)

    wandb.log({
        "Final Loss": loss, "Accuracy": accuracy, "Balanced_Acc": balanced_acc,
        "Macro_F1": f1_results['macro_f1'], "Weighted_F1": f1_results['weighted_f1'],
        "AUROC": auroc,
        **f1_results['per_class_f1'],
        "Classification Report": wandb.Table(dataframe=report_df),
    })

    log_confusion_matrix(test_labels, test_predictions)
    log_roc_curves(test_labels, predicted_probabilities, C)
    log_feature_importance(logistic_reg, top_k=min(20, M))
    log_prediction_distribution(test_labels, test_predictions, predicted_probabilities)
    log_per_class_metrics(test_labels, test_predictions, predicted_probabilities)
    log_f1_scores_plot(f1_results)
    wandb.finish()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    train_data, train_labels, test_data, test_labels = get_data(args)

    print("Data fully loaded")
    print("Shape of train_data:",   train_data.shape)
    print("Shape of train_labels:", train_labels.shape)
    print("Shape of test_data:",    test_data.shape)
    print("Shape of test_labels:",  test_labels.shape)

    # Quick class check
    print("\n=== Quick Class Check ===")
    _, _, train_ids = load_combined_features(
        os.path.join(args.path_folder, "train_data", "train_features.h5")
    )
    for label in sorted(np.unique(train_labels)):
        sample_idx  = np.where(train_labels == label)[0][0]
        if train_ids is not None:
            sample_id  = train_ids[sample_idx].decode('utf-8')
            class_name = sample_id.split('_')[1]
            print(f"Label {label} → {class_name}")
        else:
            print(f"Label {label}")

    # Parse optional class names from CLI  e.g. --class_names "Healthy,NPM1,PML::RARA"
    class_names = None
    if args.class_names:
        names = [n.strip() for n in args.class_names.split(",")]
        class_names = {i: name for i, name in enumerate(names)}
        print(f"Using class names: {class_names}")

    if args.checkpoint is not None:
        args.model_name = (
            f"{args.model_name}_{Path(args.checkpoint).parent.name}_{Path(args.checkpoint).stem}"
        )
    args.save_dir = Path(args.save_dir) / args.dataset / args.model_name

    if args.logistic_regression:
        train_and_evaluate_logistic_regression(
            train_data, train_labels, test_data, test_labels,
            args.dataset, args.save_dir, max_iter=1000
        )
        print("logistic_regression done")

    if args.umap:
        wandb.init(
            entity="eshasadia2015-university-of-warwick",
            project="umap_visualization",
            name=f"{args.dataset}_{args.model_name}_umap",
        )
        create_umap(train_data, train_labels, args.save_dir,
                    filename_addon="train", class_names=class_names)
        create_umap(test_data,  test_labels,  args.save_dir,
                    filename_addon="test",  class_names=class_names)
        wandb.finish()
        print("umap done")

    if args.knn:
        perform_knn(train_data, train_labels, test_data, test_labels, args.save_dir)
        print("knn done")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)