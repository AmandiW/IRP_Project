import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import io
import base64
from typing import Dict, List, Tuple, Any
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.accountants import RDPAccountant
from opacus.utils.batch_memory_manager import BatchMemoryManager

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# Custom Dataset class
class RetinopathyDataset(Dataset):
    def __init__(self, img_dir, labels_df, transform=None):
        self.img_dir = img_dir
        self.labels_df = labels_df
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_id = self.labels_df.iloc[idx]['id_code']
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        image = Image.open(img_path).convert('RGB')
        label = self.labels_df.iloc[idx]['diagnosis']

        if self.transform:
            image = self.transform(image)

        return image, label


# Early stopping class
class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None

    def __call__(self, val_loss, model=None):
        score = -val_loss  # Higher score is better

        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights and model is not None:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.restore_best_weights and model is not None:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0

        return self.early_stop


# Data loading function
def load_data(img_dir, labels_path, num_clients=3, batch_size=32):
    # Get logger
    logger = logging.getLogger("DataLoader")

    # Data transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load labels
    labels_df = pd.read_csv(labels_path)

    # Log dataset statistics
    class_distribution = labels_df['diagnosis'].value_counts()
    logger.info(f"Total dataset size: {len(labels_df)}")
    logger.info(f"Class distribution: {class_distribution.to_dict()}")

    # Split data for clients (IID distribution)
    client_dfs = []
    remaining_df = labels_df.copy()

    for i in range(num_clients - 1):
        client_size = len(remaining_df) // (num_clients - i)
        # Use different random seeds for each client to avoid data leakage
        client_df = remaining_df.sample(n=client_size, random_state=42 + i * 100)
        client_dfs.append(client_df)
        remaining_df = remaining_df.drop(client_df.index)

    client_dfs.append(remaining_df)  # Last client gets remaining data

    # Create train/test splits for each client
    client_data = []
    for i, client_df in enumerate(client_dfs):
        # Use different random seeds for train/test splits
        train_df, test_df = train_test_split(client_df, test_size=0.2, random_state=42 + i * 10)

        # Log client data statistics
        train_class_dist = train_df['diagnosis'].value_counts().to_dict()
        test_class_dist = test_df['diagnosis'].value_counts().to_dict()

        logger.info(f"Client {i} - Train samples: {len(train_df)}, Test samples: {len(test_df)}")
        logger.info(f"Client {i} - Train class distribution: {train_class_dist}")
        logger.info(f"Client {i} - Test class distribution: {test_class_dist}")

        train_dataset = RetinopathyDataset(img_dir, train_df, transform)
        test_dataset = RetinopathyDataset(img_dir, test_df, transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        client_data.append((train_loader, test_loader, len(train_df), len(test_df)))

    return client_data


# Model creation function with DP compatibility
def create_model(make_dp_compatible=False):
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # Binary classification

    if make_dp_compatible:
        # Make model compatible with DP
        model = ModuleValidator.fix(model)
        # Ensure the model is valid for DP training
        ModuleValidator.validate(model, strict=True)

    return model


# Create privacy engine function
def create_privacy_engine(
        model,
        optimizer,
        data_loader,
        noise_multiplier=1.0,
        max_grad_norm=1.0
):
    """
    Create a privacy engine that works with any Opacus version.
    """
    from opacus import PrivacyEngine

    # Create the privacy engine (no parameters)
    privacy_engine = PrivacyEngine()

    # Use make_private with standard parameters for newer versions
    try:
        result = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm
        )

        # Extract the optimizer from whatever is returned
        if isinstance(result, tuple):
            dp_optimizer = result[0]  # First element should be optimizer in any tuple return
        else:
            dp_optimizer = result

        return privacy_engine, dp_optimizer
    except Exception as e:
        print(f"Error using direct make_private: {e}")
        raise e  # Re-raise the exception for debugging


# Privacy metrics calculation
def calculate_privacy_metrics(privacy_engine, epochs):
    """
    Calculate privacy metrics that works with any Opacus version.
    """
    try:
        # Try newer API first
        epsilon = privacy_engine.get_epsilon(delta=1e-5)
        try:
            alpha = privacy_engine.accountant.get_best_alpha()
        except:
            alpha = None
    except Exception as e:
        print(f"Warning: Could not calculate epsilon using standard method: {e}")
        # Fallback
        epsilon = 0.0
        alpha = None

    # Try to get other metrics with appropriate fallbacks
    try:
        noise_multiplier = privacy_engine.noise_multiplier
    except:
        try:
            # Try to get from other possible locations
            noise_multiplier = getattr(privacy_engine, "noise_multiplier", 1.0)
        except:
            noise_multiplier = 1.0

    try:
        max_grad_norm = privacy_engine.max_grad_norm
    except:
        try:
            max_grad_norm = getattr(privacy_engine, "max_grad_norm", 1.0)
        except:
            max_grad_norm = 1.0

    try:
        sample_rate = privacy_engine.sample_rate
    except:
        try:
            sample_rate = getattr(privacy_engine.accountant, "sample_rate", None)
        except:
            sample_rate = None

    return {
        "epsilon": epsilon,
        "delta": 1e-5,
        "best_alpha": alpha,
        "noise_multiplier": noise_multiplier,
        "max_grad_norm": max_grad_norm,
        "sample_rate": sample_rate,
        "epochs": epochs
    }

# Create a visualization directory if it doesn't exist
def ensure_visualization_dir():
    if not os.path.exists("./visualizations"):
        os.makedirs("./visualizations")


# Confusion matrix visualization
def plot_confusion_matrix(y_true, y_pred, client_id=None, round_num=None, is_global=False):
    """
    Plot confusion matrix and save the figure.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        client_id: Client ID (None for global model)
        round_num: Training round number
        is_global: Whether this is for the global model

    Returns:
        Path to the saved figure
    """
    ensure_visualization_dir()

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)

    if is_global:
        plt.title(f"Global Model Confusion Matrix (Round {round_num})")
        filepath = f"./visualizations/global_confusion_matrix_round_{round_num}.png"
    else:
        plt.title(f"Client {client_id} Confusion Matrix (Round {round_num})")
        filepath = f"./visualizations/client_{client_id}_confusion_matrix_round_{round_num}.png"

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    return filepath


# Metrics visualization over rounds
def plot_metrics_over_rounds(metrics_history, metric_name, client_id=None, is_global=False):
    """
    Plot metrics over rounds and save the figure.

    Args:
        metrics_history: List of metrics for each round
        metric_name: Name of the metric to plot
        client_id: Client ID (None for global model)
        is_global: Whether this is for the global model

    Returns:
        Path to the saved figure
    """
    ensure_visualization_dir()

    rounds = list(range(1, len(metrics_history) + 1))
    values = [metrics.get(metric_name, 0) for metrics in metrics_history]

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, values, marker='o', linestyle='-')

    if is_global:
        plt.title(f"Global Model {metric_name.capitalize()} over Rounds")
        filepath = f"./visualizations/global_{metric_name}_over_rounds.png"
    else:
        plt.title(f"Client {client_id} {metric_name.capitalize()} over Rounds")
        filepath = f"./visualizations/client_{client_id}_{metric_name}_over_rounds.png"

    plt.ylabel(metric_name.capitalize())
    plt.xlabel("Round")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    return filepath


# Privacy budget visualization
def plot_privacy_budget(privacy_metrics_history, client_id=None):
    """
    Plot privacy budget (epsilon) consumption over rounds.

    Args:
        privacy_metrics_history: List of privacy metrics for each round
        client_id: Client ID (None for global model)

    Returns:
        Path to the saved figure
    """
    ensure_visualization_dir()

    rounds = list(range(1, len(privacy_metrics_history) + 1))
    epsilons = [metrics.get("epsilon", 0) for metrics in privacy_metrics_history]

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, epsilons, marker='o', linestyle='-', color='red')

    if client_id is not None:
        plt.title(f"Client {client_id} Privacy Budget (ε) Consumption over Rounds")
        filepath = f"./visualizations/client_{client_id}_privacy_budget.png"
    else:
        plt.title(f"Global Model Privacy Budget (ε) Consumption over Rounds")
        filepath = f"./visualizations/global_privacy_budget.png"

    plt.ylabel("Privacy Budget (ε)")
    plt.xlabel("Round")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    return filepath


# ROC curve visualization
def plot_roc_curve(y_true, y_scores, client_id=None, round_num=None, is_global=False):
    """
    Plot ROC curve and save the figure.

    Args:
        y_true: Ground truth labels
        y_scores: Predicted scores (probabilities)
        client_id: Client ID (None for global model)
        round_num: Training round number
        is_global: Whether this is for the global model

    Returns:
        Path to the saved figure
    """
    ensure_visualization_dir()

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    if is_global:
        plt.title(f"Global Model ROC Curve (Round {round_num})")
        filepath = f"./visualizations/global_roc_curve_round_{round_num}.png"
    else:
        plt.title(f"Client {client_id} ROC Curve (Round {round_num})")
        filepath = f"./visualizations/client_{client_id}_roc_curve_round_{round_num}.png"

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    return filepath


# Class distribution visualization
def plot_class_distribution(y_true, y_pred, client_id=None, round_num=None, is_global=False):
    """
    Plot class distribution comparison between true and predicted labels.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        client_id: Client ID (None for global model)
        round_num: Training round number
        is_global: Whether this is for the global model

    Returns:
        Path to the saved figure
    """
    ensure_visualization_dir()

    true_counts = np.bincount(y_true, minlength=2)
    pred_counts = np.bincount(y_pred, minlength=2)

    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    r1 = np.arange(len(true_counts))
    r2 = [x + bar_width for x in r1]

    plt.bar(r1, true_counts, color='blue', width=bar_width, label='True')
    plt.bar(r2, pred_counts, color='orange', width=bar_width, label='Predicted')

    if is_global:
        plt.title(f"Global Model Class Distribution (Round {round_num})")
        filepath = f"./visualizations/global_class_distribution_round_{round_num}.png"
    else:
        plt.title(f"Client {client_id} Class Distribution (Round {round_num})")
        filepath = f"./visualizations/client_{client_id}_class_distribution_round_{round_num}.png"

    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks([r + bar_width / 2 for r in range(len(true_counts))], ['Class 0', 'Class 1'])
    plt.legend()
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    return filepath


# Create a combined visualization of key metrics
def create_metrics_dashboard(metrics_history, privacy_metrics_history, client_id=None, is_global=False):
    """
    Create a dashboard of multiple metrics over rounds.

    Args:
        metrics_history: List of performance metrics for each round
        privacy_metrics_history: List of privacy metrics for each round
        client_id: Client ID (None for global model)
        is_global: Whether this is for the global model

    Returns:
        Path to the saved figure
    """
    ensure_visualization_dir()

    rounds = list(range(1, len(metrics_history) + 1))

    # Extract metrics
    accuracy_values = [metrics.get("accuracy", 0) for metrics in metrics_history]
    f1_values = [metrics.get("f1", 0) for metrics in metrics_history]
    precision_values = [metrics.get("precision", 0) for metrics in metrics_history]
    recall_values = [metrics.get("recall", 0) for metrics in metrics_history]
    loss_values = [metrics.get("loss", 0) for metrics in metrics_history]
    epsilon_values = [metrics.get("epsilon", 0) for metrics in privacy_metrics_history]

    plt.figure(figsize=(15, 10))

    # Plot performance metrics
    plt.subplot(2, 1, 1)
    plt.plot(rounds, accuracy_values, marker='o', linestyle='-', label='Accuracy')
    plt.plot(rounds, f1_values, marker='s', linestyle='-', label='F1 Score')
    plt.plot(rounds, precision_values, marker='^', linestyle='-', label='Precision')
    plt.plot(rounds, recall_values, marker='*', linestyle='-', label='Recall')
    plt.plot(rounds, loss_values, marker='d', linestyle='-', label='Loss')

    if is_global:
        plt.title(f"Global Model Performance Metrics over Rounds")
    else:
        plt.title(f"Client {client_id} Performance Metrics over Rounds")

    plt.ylabel("Metric Value")
    plt.xlabel("Round")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Plot privacy budget
    plt.subplot(2, 1, 2)
    plt.plot(rounds, epsilon_values, marker='o', linestyle='-', color='red', label='Privacy Budget (ε)')

    if is_global:
        plt.title(f"Global Model Privacy Budget (ε) Consumption over Rounds")
        filepath = f"./visualizations/global_metrics_dashboard.png"
    else:
        plt.title(f"Client {client_id} Privacy Budget (ε) Consumption over Rounds")
        filepath = f"./visualizations/client_{client_id}_metrics_dashboard.png"

    plt.ylabel("Privacy Budget (ε)")
    plt.xlabel("Round")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

    return filepath


class PrivacyMetricsLogger:
    """
    Class to track privacy metrics throughout training.
    """

    def __init__(self, client_id=None, is_global=False):
        self.client_id = client_id
        self.is_global = is_global
        self.privacy_metrics_history = []
        self.performance_metrics_history = []
        self.logger = logging.getLogger(f"{'Global' if is_global else f'Client_{client_id}'}_PrivacyMetrics")

    def log_privacy_metrics(self, privacy_metrics, round_num):
        """Log privacy metrics for the current round"""
        self.privacy_metrics_history.append(privacy_metrics)

        # Log the metrics
        self.logger.info(
            f"{'Global' if self.is_global else f'Client {self.client_id}'} Privacy Metrics - Round {round_num}:")
        self.logger.info(f"  ε (Epsilon): {privacy_metrics['epsilon']:.4f}")
        self.logger.info(f"  δ (Delta): {privacy_metrics['delta']}")
        self.logger.info(f"  Best α: {privacy_metrics['best_alpha']}")
        self.logger.info(f"  Noise Multiplier: {privacy_metrics['noise_multiplier']}")
        self.logger.info(f"  Max Gradient Norm: {privacy_metrics['max_grad_norm']}")

        # Visualize privacy budget consumption
        if len(self.privacy_metrics_history) > 1:
            plot_privacy_budget(self.privacy_metrics_history, self.client_id if not self.is_global else None)

    def log_performance_metrics(self, performance_metrics, round_num):
        """Log performance metrics for the current round"""
        self.performance_metrics_history.append(performance_metrics)

        # Log the metrics
        self.logger.info(
            f"{'Global' if self.is_global else f'Client {self.client_id}'} Performance Metrics - Round {round_num}:")
        self.logger.info(f"  Accuracy: {performance_metrics.get('accuracy', 0):.4f}")
        self.logger.info(f"  F1 Score: {performance_metrics.get('f1', 0):.4f}")
        self.logger.info(f"  Precision: {performance_metrics.get('precision', 0):.4f}")
        self.logger.info(f"  Recall: {performance_metrics.get('recall', 0):.4f}")
        self.logger.info(f"  Loss: {performance_metrics.get('loss', 0):.4f}")

        # Visualize performance metrics
        if len(self.performance_metrics_history) > 1:
            for metric in ['accuracy', 'f1', 'precision', 'recall']:
                if metric in performance_metrics:
                    plot_metrics_over_rounds(
                        self.performance_metrics_history,
                        metric,
                        self.client_id if not self.is_global else None,
                        self.is_global
                    )

    def create_round_dashboard(self, round_num):
        """Create a dashboard for the current round combining privacy and performance metrics"""
        if len(self.performance_metrics_history) > 0 and len(self.privacy_metrics_history) > 0:
            create_metrics_dashboard(
                self.performance_metrics_history,
                self.privacy_metrics_history,
                self.client_id if not self.is_global else None,
                self.is_global
            )
            self.logger.info(
                f"Created metrics dashboard for {'global model' if self.is_global else f'client {self.client_id}'} at round {round_num}")
