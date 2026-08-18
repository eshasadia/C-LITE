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

parser.add_argument(
    "--knn",
    help="perform knn or not",
    default=True,
    type=bool,
)
parser.add_argument(
    "--logistic_regression",
    "--logistic-regression",
    "-log",
    help="perform logistic regression or not",
    default=True,
    type=bool,
)
parser.add_argument(
    "--umap",
    help="perform umap or not",
    default=True,
    type=bool,
)
parser.add_argument(
    "--path_folder",
    "--path-folder",
    "-p",
    help="path to folder containing subfolders with training, val and test data",
    default="",
    type=str,
)
parser.add_argument(
    "--save_dir",
    "--save-dir",
    "-s",
    help="specify where to save the umap",
    default="",
    type=str,
)
parser.add_argument(
    "--dataset",
    help="specify the dataset name",
    default="",
    type=str,
)
parser.add_argument(
    "--model_name",
    help="specify the model name",
    default="dinov2_finetuned",
    type=str,
)
# only need the checkpoint name to determine the correct feature location / saving directory
parser.add_argument(
    "--checkpoint",
    help="path to checkpoint",
    default=None,
    type=str,
)
parser.add_argument(
    "--log_embeddings",
    help="log feature embeddings to wandb",
    default=False,
    type=bool,
)


def load_combined_features(file_path):
    """Load features from a single combined HDF5 file"""
    with h5py.File(file_path, "r") as hf:
        features = hf["features"][:]
        labels = hf["labels"][:]
        # IDs are optional, only load if they exist
        ids = hf["ids"][:] if "ids" in hf else None
    return features, labels, ids


def get_data(args):
    # Define the directories for train, validation, and test data
    if args.checkpoint is not None:
        args.path_folder = (
            Path(args.path_folder)
            / args.dataset
            / f"{args.model_name}_{Path(args.checkpoint).parent.name}_{Path(args.checkpoint).stem}"
        )

    train_dir = os.path.join(args.path_folder, "train_data")
    val_dir = os.path.join(args.path_folder, "val_data")
    test_dir = os.path.join(args.path_folder, "test_data")

    # Load training data from combined file
    train_file = os.path.join(train_dir, "train_features.h5")
    val_file = os.path.join(val_dir, "val_features.h5")
    test_file = os.path.join(test_dir, "test_features.h5")
    
    print(f"Loading training data from: {train_file}")
    train_features, train_labels, train_ids = load_combined_features(train_file)
    
    print(f"Loading validation data from: {val_file}")
    val_features, val_labels, val_ids = load_combined_features(val_file)
    
    print(f"Loading test data from: {test_file}")
    test_features, test_labels, test_ids = load_combined_features(test_file)
    
    # Combine train and validation data
    train_data = np.vstack([train_features, val_features])
    train_labels = np.concatenate([train_labels, val_labels])
    
    # Flatten if needed (features should already be 2D from extraction)
    if len(train_data.shape) > 2:
        train_data = train_data.reshape(train_data.shape[0], -1)
    if len(test_features.shape) > 2:
        test_features = test_features.reshape(test_features.shape[0], -1)
    
    test_data = test_features
    test_labels = test_labels.flatten()
    train_labels = train_labels.flatten()

    return train_data, train_labels, test_data, test_labels

def compute_f1_scores(y_true, y_pred):
    """
    Compute macro F1 (overall), weighted F1, and per-class F1 scores
    
    Returns:
        dict with macro_f1, weighted_f1, and per_class_f1 (dict)
    """
    # Macro F1 (average of per-class F1)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    # Weighted F1 (weighted by class support)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Per-class F1
    per_class_f1 = f1_score(y_true, y_pred, average=None)
    
    # Get class labels
    unique_classes = np.unique(y_true)
    per_class_f1_dict = {f"class_{cls}_f1": f1 for cls, f1 in zip(unique_classes, per_class_f1)}
    
    return {
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class_f1': per_class_f1_dict,
        'per_class_f1_values': per_class_f1  # For plotting
    }


def log_confusion_matrix(y_true, y_pred, class_names=None):
    """Create and log confusion matrix to wandb"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    if class_names is not None:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Log to wandb
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.close()
    
    return cm


def log_roc_curves(y_true, y_pred_proba, num_classes):
    """Create and log ROC curves for multi-class classification"""
    # Binarize the labels
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
    
    # Create ROC curve plot
    plt.figure(figsize=(10, 8))
    
    if num_classes == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    else:
        # Multi-class classification
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
            plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    wandb.log({"roc_curves": wandb.Image(plt)})
    plt.close()


def log_f1_scores_plot(f1_results):
    """Create bar plot of per-class F1 scores"""
    per_class_f1 = f1_results['per_class_f1_values']
    classes = list(range(len(per_class_f1)))
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, per_class_f1, alpha=0.8, color='steelblue')
    
    # Add horizontal line for macro F1
    plt.axhline(y=f1_results['macro_f1'], color='red', linestyle='--', 
                label=f'Macro F1: {f1_results["macro_f1"]:.4f}')
    
    # Add horizontal line for weighted F1
    plt.axhline(y=f1_results['weighted_f1'], color='green', linestyle='--', 
                label=f'Weighted F1: {f1_results["weighted_f1"]:.4f}')
    
    # Add value labels on bars
    for i, (bar, f1) in enumerate(zip(bars, per_class_f1)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{f1:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.title('Per-Class F1 Scores')
    plt.xticks(classes, [f'Class {c}' for c in classes])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    
    wandb.log({"f1_scores_per_class": wandb.Image(plt)})
    plt.close()


def log_feature_importance(model, feature_names=None, top_k=20):
    """Log feature importance for logistic regression"""
    if hasattr(model, 'coef_'):
        coef = np.abs(model.coef_)
        
        if coef.shape[0] > 1:  # Multi-class
            # Average importance across classes
            importance = np.mean(coef, axis=0)
        else:
            importance = coef[0]
        
        # Get top k features
        top_indices = np.argsort(importance)[-top_k:][::-1]
        top_importance = importance[top_indices]
        
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(importance))]
        
        top_features = [feature_names[i] for i in top_indices]
        
        # Create bar plot
        plt.figure(figsize=(12, 6))
        plt.barh(range(len(top_features)), top_importance)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Absolute Coefficient Value')
        plt.title(f'Top {top_k} Most Important Features')
        plt.tight_layout()
        
        wandb.log({"feature_importance": wandb.Image(plt)})
        plt.close()
        
        # Also log as table
        importance_df = pd.DataFrame({
            'Feature': top_features,
            'Importance': top_importance
        })
        wandb.log({"feature_importance_table": wandb.Table(dataframe=importance_df)})


def log_prediction_distribution(y_true, y_pred, y_pred_proba):
    """Log prediction confidence distribution"""
    # Get the maximum probability for each prediction
    max_probs = np.max(y_pred_proba, axis=1)
    
    # Separate correct and incorrect predictions
    correct_mask = (y_true == y_pred)
    correct_probs = max_probs[correct_mask]
    incorrect_probs = max_probs[~correct_mask]
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(correct_probs, bins=50, alpha=0.5, label='Correct Predictions', color='green')
    plt.hist(incorrect_probs, bins=50, alpha=0.5, label='Incorrect Predictions', color='red')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Count')
    plt.title('Prediction Confidence Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    wandb.log({"prediction_confidence": wandb.Image(plt)})
    plt.close()
    
    # Log statistics
    wandb.log({
        "avg_correct_confidence": np.mean(correct_probs),
        "avg_incorrect_confidence": np.mean(incorrect_probs),
        "median_correct_confidence": np.median(correct_probs),
        "median_incorrect_confidence": np.median(incorrect_probs),
    })


def log_class_distribution(train_labels, test_labels):
    """Log class distribution for train and test sets"""
    unique_classes = np.unique(np.concatenate([train_labels, test_labels]))
    
    train_counts = [np.sum(train_labels == c) for c in unique_classes]
    test_counts = [np.sum(test_labels == c) for c in unique_classes]
    
    # Create grouped bar chart
    x = np.arange(len(unique_classes))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, train_counts, width, label='Train', alpha=0.8)
    ax.bar(x + width/2, test_counts, width, label='Test', alpha=0.8)
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title('Class Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Class {c}' for c in unique_classes])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    wandb.log({"class_distribution": wandb.Image(plt)})
    plt.close()
    
    # Log as table
    dist_df = pd.DataFrame({
        'Class': unique_classes,
        'Train_Count': train_counts,
        'Test_Count': test_counts,
        'Train_Percentage': [c/sum(train_counts)*100 for c in train_counts],
        'Test_Percentage': [c/sum(test_counts)*100 for c in test_counts],
    })
    wandb.log({"class_distribution_table": wandb.Table(dataframe=dist_df)})


def log_per_class_metrics(y_true, y_pred, y_pred_proba):
    """Log detailed per-class metrics"""
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Extract per-class metrics
    classes = sorted([k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']])
    
    metrics_data = []
    for cls in classes:
        if isinstance(report[cls], dict):
            metrics_data.append({
                'Class': cls,
                'Precision': report[cls]['precision'],
                'Recall': report[cls]['recall'],
                'F1-Score': report[cls]['f1-score'],
                'Support': report[cls]['support']
            })
    
    # Create grouped bar chart
    df = pd.DataFrame(metrics_data)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Precision
    axes[0, 0].bar(df['Class'], df['Precision'])
    axes[0, 0].set_title('Precision per Class')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Recall
    axes[0, 1].bar(df['Class'], df['Recall'])
    axes[0, 1].set_title('Recall per Class')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1-Score
    axes[1, 0].bar(df['Class'], df['F1-Score'])
    axes[1, 0].set_title('F1-Score per Class')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Support
    axes[1, 1].bar(df['Class'], df['Support'])
    axes[1, 1].set_title('Support per Class')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    wandb.log({"per_class_metrics": wandb.Image(plt)})
    plt.close()


def perform_knn(train_data, train_labels, test_data, test_labels, save_dir):
    # Define a range of values for n_neighbors to search
    n_neighbors_values = [1, 20]
    # n_neighbors_values = [1, 2, 5, 10, 20, 50, 100, 500]
    # n_neighbors_values = [1, 2, 3, 4, 5] # -> for testing

    for n_neighbors in n_neighbors_values:
        # Initialize a KNeighborsClassifier with the current n_neighbors
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)

        # Fit the KNN classifier to the training data
        knn.fit(train_data, train_labels)

        # Predict labels for the test data
        test_predictions = knn.predict(test_data)
        test_proba = knn.predict_proba(test_data)

        # Compute F1 scores
        f1_results = compute_f1_scores(test_labels, test_predictions)

        # Evaluate the classifier
        accuracy = accuracy_score(test_labels, test_predictions)
        balanced_acc = balanced_accuracy_score(test_labels, test_predictions)

        print(f"n_neighbors = {n_neighbors}")
        print(f"Accuracy: {accuracy}")
        print(f"Macro F1: {f1_results['macro_f1']:.4f}")
        print(f"Weighted F1: {f1_results['weighted_f1']:.4f}")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print("standard deviation of F1 scores across classes:", np.std(list(f1_results['per_class_f1'].values())))
        print("standard deviation overall F1 scores:", np.std([f1_results['macro_f1'], f1_results['weighted_f1']]) )

        run_name = f"{n_neighbors}NN_{Path(save_dir).name}"

        # If you want to log the results with Weights & Biases (wandb), you can initialize a wandb run:
        wandb.init(
            entity="eshasadia2015-university-of-warwick",
            project="knn",
            name=run_name,
        )

        # Log metrics including F1 scores
        wandb.log({
            "n_neighbors": n_neighbors,
            "Accuracy": accuracy,
            "Balanced_Acc": balanced_acc,
            "Macro_F1": f1_results['macro_f1'],
            "Weighted_F1": f1_results['weighted_f1'],
            **f1_results['per_class_f1']  # Log individual class F1 scores
        })

        ## Calculate the classification report
        report = classification_report(test_labels, test_predictions, output_dict=True)

        print(f"report: {report}")

        # Convert the report to a Pandas DataFrame for logging
        report_df = pd.DataFrame(report).transpose()

        # Log the final loss, accuracy, and classification report using wandb.log
        wandb.log({"Classification Report": wandb.Table(dataframe=report_df)})

        # Log additional visualizations
        num_classes = len(np.unique(test_labels))
        log_confusion_matrix(test_labels, test_predictions)
        log_roc_curves(test_labels, test_proba, num_classes)
        log_prediction_distribution(test_labels, test_predictions, test_proba)
        log_per_class_metrics(test_labels, test_predictions, test_proba)
        log_f1_scores_plot(f1_results)

        # Finish the wandb run
        wandb.finish()

        df_labels_to_save = pd.DataFrame({"True Labels": test_labels, "Predicted Labels": test_predictions})
        filename = f"{Path(save_dir).name}_labels_and_predictions.csv"
        file_path = os.path.join(save_dir, filename)
        # Save the DataFrame to CSV file
        df_labels_to_save.to_csv(file_path, index=False)


def create_umap(data, labels, save_dir, filename_addon="train"):
    # Create a UMAP model and fit it to your data
    # reducer = umap.UMAP(random_state=42)
    reducer = umap.UMAP()
    umap_data = reducer.fit_transform(data)

    # Specify the directory for saving the images

    umap_dir = os.path.join(save_dir, "umaps")
    os.makedirs(umap_dir, exist_ok=True)

    # Loop through different figure sizes
    figure_sizes = [(48, 32), (36, 24), (24, 16), (12, 8), (6, 4)]  # Add more sizes as needed

    for size in figure_sizes:
        # Create a scatter plot with the specified size
        plt.figure(figsize=size, dpi=300)
        plt.scatter(umap_data[:, 0], umap_data[:, 1], c=labels, s=0.1, cmap="Spectral")
        plt.colorbar()
        plt.title("UMAP")

        # Specify the filename with the size information
        image_filename = f"umap_visualization_{Path(save_dir).name}_{size[0]}x{size[1]}_{filename_addon}.png"

        # Save the UMAP visualization as an image in the specified directory
        plt.savefig(os.path.join(umap_dir, image_filename))
        plt.close()  # Close the figure to free memory
    
    # Also log to wandb if in active run
    if wandb.run is not None:
        plt.figure(figsize=(12, 8))
        plt.scatter(umap_data[:, 0], umap_data[:, 1], c=labels, s=1, cmap="Spectral", alpha=0.6)
        plt.colorbar(label='Class')
        plt.title(f"UMAP Visualization - {filename_addon}")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        wandb.log({f"umap_{filename_addon}": wandb.Image(plt)})
        plt.close()


def train_and_evaluate_logistic_regression(
    train_data, train_labels, test_data, test_labels, dataset, save_dir, max_iter=1000
):
    # Initialize wandb
    wandb.init(
        entity="eshasadia2015-university-of-warwick",
        project="logistic_regression",
        name=f"{dataset}_{Path(save_dir).name}",
    )

    # Log dataset info
    log_class_distribution(train_labels, test_labels)

    M = train_data.shape[1]
    C = len(np.unique(train_labels))  # Dynamically get number of classes
    l2_reg_coef = 100 / (M * C)

    # Log hyperparameters
    wandb.config.update({
        "num_features": M,
        "num_classes": C,
        "l2_reg_coef": l2_reg_coef,
        "C": 1 / l2_reg_coef,
        "max_iter": max_iter,
        "train_samples": len(train_labels),
        "test_samples": len(test_labels),
    })

    # Initialize the logistic regression model with L-BFGS solver
    logistic_reg = LogisticRegression(C=1 / l2_reg_coef, max_iter=max_iter, multi_class="multinomial", solver="lbfgs")

    logistic_reg.fit(train_data, train_labels)

    # Evaluate the model on the test data
    test_predictions = logistic_reg.predict(test_data)
    predicted_probabilities = logistic_reg.predict_proba(test_data)
    
    # Compute F1 scores
    f1_results = compute_f1_scores(test_labels, test_predictions)
    
    loss = log_loss(test_labels, predicted_probabilities)
    accuracy = accuracy_score(test_labels, test_predictions)
    balanced_acc = balanced_accuracy_score(test_labels, test_predictions)
    
    # Calculate AUROC
    if C == 2:
        auroc = roc_auc_score(test_labels, predicted_probabilities[:, 1])
    else:
        y_true_bin = label_binarize(test_labels, classes=np.arange(C))
        auroc = roc_auc_score(y_true_bin, predicted_probabilities, multi_class='ovr', average='weighted')
    
    report = classification_report(test_labels, test_predictions, output_dict=True)

    df_labels_to_save = pd.DataFrame({"True Labels": test_labels, "Predicted Labels": test_predictions})
    filename = f"{Path(save_dir).name}_labels_and_predictions.csv"
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, filename)
    # Save the DataFrame to CSV file
    df_labels_to_save.to_csv(file_path, index=False)

    predicted_probabilities_df = pd.DataFrame(
        predicted_probabilities, columns=[f"Probability Class {i}" for i in range(predicted_probabilities.shape[1])]
    )
    predicted_probabilities_filename = f"{Path(save_dir).name}_predicted_probabilities_test.csv"
    predicted_probabilities_file_path = os.path.join(save_dir, predicted_probabilities_filename)
    predicted_probabilities_df.to_csv(predicted_probabilities_file_path, index=False)

    # Convert the report to a Pandas DataFrame for logging
    report_df = pd.DataFrame(report).transpose()

    # some prints
    print(f"Final Loss: {loss}")
    print(f"Accuracy: {accuracy}")
    print(f"Macro F1: {f1_results['macro_f1']:.4f}")
    print(f"Weighted F1: {f1_results['weighted_f1']:.4f}")
    print("Per-class F1 scores:")
    for class_name, f1 in f1_results['per_class_f1'].items():
        print(f"  {class_name}: {f1:.4f}")
    print(report)

    # Log the final loss, accuracy, and classification report using wandb.log
    wandb.log({
        "Final Loss": loss,
        "Accuracy": accuracy,
        "Balanced_Acc": balanced_acc,
        "Macro_F1": f1_results['macro_f1'],
        "Weighted_F1": f1_results['weighted_f1'],
        "AUROC": auroc,
        **f1_results['per_class_f1'],  # Log individual class F1 scores
        "Classification Report": wandb.Table(dataframe=report_df),
    })

    # Log additional visualizations
    log_confusion_matrix(test_labels, test_predictions)
    log_roc_curves(test_labels, predicted_probabilities, C)
    log_feature_importance(logistic_reg, top_k=min(20, M))
    log_prediction_distribution(test_labels, test_predictions, predicted_probabilities)
    log_per_class_metrics(test_labels, test_predictions, predicted_probabilities)
    log_f1_scores_plot(f1_results)

    # Finish the wandb run
    wandb.finish()


def main(args):
    train_data, train_labels, test_data, test_labels = get_data(args)
    print("data fully loaded")
    print("Shape of train_data:", train_data.shape)
    print("Shape of train_labels:", train_labels.shape)
    print("Shape of test_data:", test_data.shape)
    print("Shape of test_labels:", test_labels.shape)
    print("\n=== Quick Class Check ===")
    _, train_ids = load_combined_features(os.path.join(args.path_folder, "train_data", "train_features.h5"))[1:3]
    for label in sorted(np.unique(train_labels)):
        sample_idx = np.where(train_labels == label)[0][0]
        sample_id = train_ids[sample_idx].decode('utf-8')
        class_name = sample_id.split('_')[1]
        print(f"Label {label} → {class_name}")
    run_name = f"{Path(args.path_folder).name}"
    save_directory = Path(args.save_dir) / args.dataset / run_name

    if args.checkpoint is not None:
        args.model_name = f"{args.model_name}_{Path(args.checkpoint).parent.name}_{Path(args.checkpoint).stem}"
    args.save_dir = Path(args.save_dir) / args.dataset / args.model_name

    if args.logistic_regression:
        train_and_evaluate_logistic_regression(
            train_data, train_labels, test_data, test_labels, args.dataset, args.save_dir, max_iter=1000
        )
        print("logistic_regression done")

    if args.umap:
        # Create a temporary wandb run for UMAP logging
        wandb.init(
            entity="eshasadia2015-university-of-warwick",
            project="umap_visualization",
            name=f"{args.dataset}_{args.model_name}_umap",
        )
        
        create_umap(train_data, train_labels, args.save_dir)
        create_umap(test_data, test_labels, args.save_dir, "test")
        
        wandb.finish()
        print("umap done")

    if args.knn:
        perform_knn(train_data, train_labels, test_data, test_labels, args.save_dir)
        print("knn done")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)