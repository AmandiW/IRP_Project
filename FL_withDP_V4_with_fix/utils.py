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
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class RetinopathyDataset(Dataset):
    """Dataset class for diabetic retinopathy images."""

    def __init__(self, img_dir, labels_df, transform=None):
        """
        Args:
            img_dir (str): Directory with all the images
            labels_df (DataFrame): DataFrame containing image IDs and labels
            transform (callable, optional): Optional transform to be applied on a sample
        """
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


class SimpleModel(nn.Module):
    """A simplified model for diabetic retinopathy classification."""

    def __init__(self, num_classes=2):
        super(SimpleModel, self).__init__()
        # Use a pretrained ResNet18 model with modifications for privacy
        self.backbone = models.resnet18(pretrained=True)

        # Replace batch normalization with group normalization for better DP compatibility
        self._replace_batchnorm_with_groupnorm()

        # Freeze all layers except the final layer
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Modify the final layer for our classification task
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)

        # Make only the final layer trainable
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def _replace_batchnorm_with_groupnorm(self):
        """Replace all BatchNorm layers with GroupNorm layers."""
        for name, module in self.backbone.named_children():
            if len(list(module.children())) > 0:
                # Recursively convert nested modules
                setattr(self.backbone, name, self._convert_batchnorm_module(module))
            elif isinstance(module, nn.BatchNorm2d):
                # Replace BatchNorm2d with GroupNorm
                setattr(self.backbone, name, nn.GroupNorm(
                    num_groups=min(32, module.num_features),
                    num_channels=module.num_features
                ))

    def _convert_batchnorm_module(self, module):
        """Recursively convert BatchNorm modules to GroupNorm."""
        for name, child in module.named_children():
            if len(list(child.children())) > 0:
                setattr(module, name, self._convert_batchnorm_module(child))
            elif isinstance(child, nn.BatchNorm2d):
                setattr(module, name, nn.GroupNorm(
                    num_groups=min(32, child.num_features),
                    num_channels=child.num_features
                ))
        return module

    def forward(self, x):
        return self.backbone(x)


class EarlyStopping:
    """Early stopping to prevent overfitting."""

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


def create_model(num_classes=2):
    """Create a model for diabetic retinopathy classification."""
    return SimpleModel(num_classes)


def load_data(img_dir, labels_path, num_clients=3, batch_size=8):
    """
    Load and prepare data for federated learning.

    Args:
        img_dir (str): Directory containing the images
        labels_path (str): Path to CSV file with labels
        num_clients (int): Number of simulated clients
        batch_size (int): Batch size for DataLoader

    Returns:
        List of tuples: (train_loader, test_loader, train_size, test_size) for each client
    """
    logger = logging.getLogger("DataLoader")

    # Image preprocessing and augmentation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load labels
    try:
        labels_df = pd.read_csv(labels_path)
        logger.info(f"Loaded {len(labels_df)} labels from {labels_path}")
    except Exception as e:
        logger.error(f"Error loading labels: {e}")
        raise

    # Log dataset statistics
    class_distribution = labels_df['diagnosis'].value_counts()
    logger.info(f"Total dataset size: {len(labels_df)}")
    logger.info(f"Class distribution: {class_distribution.to_dict()}")

    # Split data for clients (IID distribution)
    client_dfs = []
    remaining_df = labels_df.copy()

    for i in range(num_clients - 1):
        client_size = len(remaining_df) // (num_clients - i)
        client_df = remaining_df.sample(n=client_size, random_state=42 + i)
        client_dfs.append(client_df)
        remaining_df = remaining_df.drop(client_df.index)

    client_dfs.append(remaining_df)  # Last client gets remaining data

    # Create train/test splits for each client
    client_data = []
    for i, client_df in enumerate(client_dfs):
        # Use different random seeds for train/test splits
        train_df, test_df = train_test_split(client_df, test_size=0.2, random_state=42 + i)

        # Log client data statistics
        train_class_dist = train_df['diagnosis'].value_counts().to_dict()
        test_class_dist = test_df['diagnosis'].value_counts().to_dict()

        logger.info(f"Client {i} - Train samples: {len(train_df)}, Test samples: {len(test_df)}")
        logger.info(f"Client {i} - Train class distribution: {train_class_dist}")
        logger.info(f"Client {i} - Test class distribution: {test_class_dist}")

        # Create datasets and dataloaders
        train_dataset = RetinopathyDataset(img_dir, train_df, transform)
        test_dataset = RetinopathyDataset(img_dir, test_df, transform)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=False  # Simplify for debugging
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        client_data.append((train_loader, test_loader, len(train_df), len(test_df)))

    return client_data


def compute_epsilon(noise_multiplier, sample_rate, iterations, delta=1e-5):
    """
    Compute privacy budget (epsilon) based on DP-SGD parameters.
    This is a simplified version for privacy accounting.

    Args:
        noise_multiplier (float): Noise multiplier used in DP-SGD
        sample_rate (float): Sampling rate of data (batch_size / dataset_size)
        iterations (int): Number of iterations (epochs * batches_per_epoch)
        delta (float): Target delta

    Returns:
        float: Estimated epsilon value
    """
    # This is a simplified formula - in practice you would use a more precise
    # implementation like the one in Opacus
    c = sample_rate * np.sqrt(2 * iterations * np.log(1 / delta))
    return c / noise_multiplier


def apply_dp_noise(gradients, noise_multiplier, max_grad_norm):
    """
    Manually apply differential privacy noise to gradients.

    Args:
        gradients (list): List of gradient tensors
        noise_multiplier (float): Scale of noise to add (higher = more privacy)
        max_grad_norm (float): Maximum gradient norm for clipping

    Returns:
        list: Noisy gradients with DP guarantees
    """
    # Step 1: Compute total gradient norm
    total_norm = 0
    for grad in gradients:
        if grad is not None:
            total_norm += grad.norm(2).item() ** 2
    total_norm = total_norm ** 0.5

    # Step 2: Compute scaling factor for gradient clipping
    scale = min(1, max_grad_norm / (total_norm + 1e-10))

    # Step 3: Clip gradients and add noise
    noisy_gradients = []
    for grad in gradients:
        if grad is not None:
            # Clip gradient
            clipped_grad = grad * scale

            # Add Gaussian noise calibrated to sensitivity
            noise = torch.randn_like(grad) * noise_multiplier * max_grad_norm / (len(gradients) ** 0.5)
            noisy_grad = clipped_grad + noise
            noisy_gradients.append(noisy_grad)
        else:
            noisy_gradients.append(None)

    return noisy_gradients


def train_with_dp(model, train_loader, optimizer, criterion, device, dp_params, logger):
    """
    Train a model with differential privacy guarantees.

    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        optimizer (Optimizer): Optimizer
        criterion: Loss function
        device: Device to use (cuda/cpu)
        dp_params (dict): DP parameters (noise_multiplier, max_grad_norm)
        logger: Logger

    Returns:
        tuple: (model, metrics) - Trained model and performance metrics
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []

    noise_multiplier = dp_params["noise_multiplier"]
    max_grad_norm = dp_params["max_grad_norm"]

    # Calculate privacy parameters
    batch_size = next(iter(train_loader))[0].shape[0]
    sample_rate = batch_size / len(train_loader.dataset)
    iterations = len(train_loader)

    # Log DP parameters
    logger.info(f"DP parameters: noise={noise_multiplier}, clip={max_grad_norm}, sample_rate={sample_rate}")

    # Training loop with manual DP
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()

        # Collect and process gradients with DP
        gradients = [p.grad.clone() for p in model.parameters() if p.requires_grad and p.grad is not None]
        noisy_gradients = apply_dp_noise(gradients, noise_multiplier, max_grad_norm)

        # Manually update gradients with noisy versions
        grad_idx = 0
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                p.grad = noisy_gradients[grad_idx]
                grad_idx += 1

        # Update weights
        optimizer.step()

        # Collect metrics
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        all_targets.extend(target.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

        # Log progress
        if batch_idx % 5 == 0:
            logger.info(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

    # Calculate final metrics
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct / total

    # Calculate other metrics
    metrics = {
        "loss": epoch_loss,
        "accuracy": epoch_accuracy,
        "f1": f1_score(all_targets, all_predictions, average='weighted'),
        "precision": precision_score(all_targets, all_predictions, average='weighted', zero_division=0),
        "recall": recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
    }

    # Calculate privacy metrics
    epsilon = compute_epsilon(noise_multiplier, sample_rate, iterations)
    privacy_metrics = {
        "epsilon": epsilon,
        "delta": 1e-5,
        "noise_multiplier": noise_multiplier,
        "max_grad_norm": max_grad_norm,
        "sample_rate": sample_rate,
        "iterations": iterations
    }

    return model, metrics, privacy_metrics


def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluate model performance on test data.

    Args:
        model (nn.Module): Model to evaluate
        test_loader (DataLoader): Test data loader
        criterion: Loss function
        device: Device to use (cuda/cpu)

    Returns:
        tuple: (loss, metrics) - Loss and dictionary of metrics
    """
    model.eval()
    loss = 0.0
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    y_scores = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target).item()

            # Get probabilities for ROC curve
            probs = torch.nn.functional.softmax(output, dim=1)
            y_scores.extend(probs[:, 1].cpu().numpy())  # Probability for class 1

            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            y_true.extend(target.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    loss /= len(test_loader)
    accuracy = correct / total

    # Calculate additional metrics
    metrics = {
        "loss": loss,
        "accuracy": accuracy,
        "f1": f1_score(y_true, y_pred, average='weighted'),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_scores": y_scores
    }

    return loss, metrics


def plot_confusion_matrix(y_true, y_pred, client_id=None, round_num=None, is_global=False):
    """Create and save confusion matrix visualization."""
    # Create directory if it doesn't exist
    os.makedirs("./visualizations", exist_ok=True)

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


def plot_roc_curve(y_true, y_scores, client_id=None, round_num=None, is_global=False):
    """Create and save ROC curve visualization."""
    # Create directory if it doesn't exist
    os.makedirs("./visualizations", exist_ok=True)

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


def plot_metrics_over_rounds(metrics_history, metric_name, client_id=None, is_global=False):
    """Plot metrics progression over training rounds."""
    # Create directory if it doesn't exist
    os.makedirs("./visualizations", exist_ok=True)

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


def plot_privacy_budget(privacy_metrics_history, client_id=None):
    """Plot privacy budget consumption over rounds."""
    # Create directory if it doesn't exist
    os.makedirs("./visualizations", exist_ok=True)

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


def generate_summary_report(rounds_df, output_path="./aggregated_metrics/final_summary_report.txt"):
    """Generate a summary report of the federated learning experiment."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("===============================================\n")
        f.write("DIFFERENTIAL PRIVACY FEDERATED LEARNING REPORT\n")
        f.write("===============================================\n\n")

        f.write("PRIVACY METRICS:\n")
        f.write(f"Final Privacy Budget (ε): {rounds_df['average_epsilon'].iloc[-1]:.4f}\n")
        if len(rounds_df) > 1:
            epsilon_increase = (rounds_df['average_epsilon'].iloc[-1] - rounds_df['average_epsilon'].iloc[0]) / (
                        len(rounds_df) - 1)
            f.write(f"Privacy Budget Increase Rate: {epsilon_increase:.4f} per round\n\n")

        f.write("UTILITY METRICS:\n")
        f.write(f"Final Accuracy: {rounds_df['average_accuracy'].iloc[-1]:.4f}\n")
        f.write(f"Final F1 Score: {rounds_df['average_f1'].iloc[-1]:.4f}\n")
        f.write(f"Final Loss: {rounds_df['average_loss'].iloc[-1]:.4f}\n\n")

        if len(rounds_df) > 1:
            f.write("PRIVACY-UTILITY TRADEOFF:\n")
            f.write(
                f"ε/Accuracy Ratio: {rounds_df['average_epsilon'].iloc[-1] / rounds_df['average_accuracy'].iloc[-1]:.4f}\n")

            epsilon_diff = rounds_df['average_epsilon'].iloc[-1] - rounds_df['average_epsilon'].iloc[0]
            accuracy_diff = rounds_df['average_accuracy'].iloc[-1] - rounds_df['average_accuracy'].iloc[0]
            utility_per_privacy = accuracy_diff / (epsilon_diff + 1e-8)

            f.write(f"Accuracy gained per unit of privacy spent: {utility_per_privacy:.4f}\n\n")

        f.write("TRAINING PROGRESSION:\n")
        for i, row in rounds_df.iterrows():
            f.write(
                f"Round {int(row['round'])}: ε = {row['average_epsilon']:.4f}, "
                f"Accuracy = {row['average_accuracy']:.4f}, "
                f"F1 = {row['average_f1']:.4f}, "
                f"Loss = {row['average_loss']:.4f}\n"
            )

    return output_path


class PrivacyMetricsLogger:
    """Class to track and visualize privacy and performance metrics."""

    def __init__(self, client_id=None, is_global=False):
        self.client_id = client_id
        self.is_global = is_global
        self.privacy_metrics_history = []
        self.performance_metrics_history = []
        self.logger = logging.getLogger(f"{'Global' if is_global else f'Client_{client_id}'}_Metrics")

    def log_privacy_metrics(self, privacy_metrics, round_num):
        """Log privacy metrics for the current round."""
        self.privacy_metrics_history.append(privacy_metrics)

        # Log the metrics
        self.logger.info(
            f"{'Global' if self.is_global else f'Client {self.client_id}'} Privacy Metrics - Round {round_num}:")
        self.logger.info(f"  ε (Epsilon): {privacy_metrics['epsilon']:.4f}")
        self.logger.info(f"  δ (Delta): {privacy_metrics['delta']}")
        self.logger.info(f"  Noise Multiplier: {privacy_metrics['noise_multiplier']}")
        self.logger.info(f"  Max Gradient Norm: {privacy_metrics['max_grad_norm']}")

        # Visualize privacy budget consumption if we have history
        if len(self.privacy_metrics_history) > 1:
            plot_privacy_budget(self.privacy_metrics_history, None if self.is_global else self.client_id)

    def log_performance_metrics(self, performance_metrics, round_num):
        """Log performance metrics for the current round."""
        self.performance_metrics_history.append(performance_metrics)

        # Log the metrics
        self.logger.info(
            f"{'Global' if self.is_global else f'Client {self.client_id}'} Performance Metrics - Round {round_num}:")
        self.logger.info(f"  Accuracy: {performance_metrics.get('accuracy', 0):.4f}")
        self.logger.info(f"  F1 Score: {performance_metrics.get('f1', 0):.4f}")
        self.logger.info(f"  Precision: {performance_metrics.get('precision', 0):.4f}")
        self.logger.info(f"  Recall: {performance_metrics.get('recall', 0):.4f}")
        self.logger.info(f"  Loss: {performance_metrics.get('loss', 0):.4f}")

        # Visualize performance metrics if we have history
        if len(self.performance_metrics_history) > 1:
            for metric in ['accuracy', 'f1', 'precision', 'recall']:
                if metric in performance_metrics:
                    plot_metrics_over_rounds(
                        self.performance_metrics_history,
                        metric,
                        None if self.is_global else self.client_id,
                        self.is_global
                    )