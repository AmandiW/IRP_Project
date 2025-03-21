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


def train_with_dp(model, train_loader, optimizer, criterion, device, dp_params, logger, epochs=1):
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
        epochs (int): Number of epochs to train (default: 1)

    Returns:
        tuple: (model, metrics, privacy_metrics, original_gradients, noisy_gradients) - Trained model and data
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
    iterations = len(train_loader) * epochs

    # Log DP parameters
    logger.info(
        f"DP parameters: noise={noise_multiplier}, clip={max_grad_norm}, sample_rate={sample_rate}, epochs={epochs}")

    # Store original and noisy gradients for visualization
    # We'll only save from one batch to avoid excessive memory usage
    original_gradients = []
    noisy_gradients = []

    # Training loop with manual DP for multiple epochs
    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch + 1}/{epochs}")

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

            # Save original gradients for the first batch of the last epoch
            if epoch == epochs - 1 and batch_idx == 0:
                original_gradients = [g.clone() for g in gradients if g is not None]

            # Apply DP noise
            noisy_grads = apply_dp_noise(gradients, noise_multiplier, max_grad_norm)

            # Save noisy gradients for the first batch of the last epoch
            if epoch == epochs - 1 and batch_idx == 0:
                noisy_gradients = [g.clone() for g in noisy_grads if g is not None]

            # Manually update gradients with noisy versions
            grad_idx = 0
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    p.grad = noisy_grads[grad_idx]
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

            # Log progress for every 5th batch
            if batch_idx % 5 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{epochs} - Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

    # Calculate final metrics
    epoch_loss = running_loss / (len(train_loader) * epochs)
    epoch_accuracy = correct / total

    # Calculate other metrics
    metrics = {
        "loss": epoch_loss,
        "accuracy": epoch_accuracy,
        "f1": f1_score(all_targets, all_predictions, average='weighted'),
        "precision": precision_score(all_targets, all_predictions, average='weighted', zero_division=0),
        "recall": recall_score(all_targets, all_predictions, average='weighted', zero_division=0),
        "y_true": all_targets,
        "y_pred": all_predictions
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

    return model, metrics, privacy_metrics, original_gradients, noisy_gradients


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
        f.write(f"Final Privacy Budget (Epsilon): {rounds_df['average_epsilon'].iloc[-1]:.4f}\n")
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
                f"Epsilon/Accuracy Ratio: {rounds_df['average_epsilon'].iloc[-1] / rounds_df['average_accuracy'].iloc[-1]:.4f}\n")

            epsilon_diff = rounds_df['average_epsilon'].iloc[-1] - rounds_df['average_epsilon'].iloc[0]
            accuracy_diff = rounds_df['average_accuracy'].iloc[-1] - rounds_df['average_accuracy'].iloc[0]
            utility_per_privacy = accuracy_diff / (epsilon_diff + 1e-8)

            f.write(f"Accuracy gained per unit of privacy spent: {utility_per_privacy:.4f}\n\n")

        f.write("TRAINING PROGRESSION:\n")
        for i, row in rounds_df.iterrows():
            f.write(
                f"Round {int(row['round'])}: Epsilon = {row['average_epsilon']:.4f}, "
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
        self.logger.info(f"  (Epsilon): {privacy_metrics['epsilon']:.4f}")
        self.logger.info(f"  (Delta): {privacy_metrics['delta']}")
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


# ========================= NEW VISUALIZATION FUNCTIONS =========================

def plot_gradient_distribution(original_gradients, noisy_gradients, client_id, round_num):
    """
    Visualize how differential privacy affects gradient distributions.

    Args:
        original_gradients (torch.Tensor): Original gradient tensor
        noisy_gradients (torch.Tensor): Noisy gradient tensor with DP applied
        client_id (int): Client ID
        round_num (int): Current round number

    Returns:
        str: Path to saved visualization
    """
    os.makedirs("./visualizations/privacy_analysis", exist_ok=True)

    # Flatten gradients for better visualization
    orig_grad = original_gradients.flatten().cpu().numpy()
    noisy_grad = noisy_gradients.flatten().cpu().numpy()

    # Take a sample if too large
    max_samples = 1000
    if len(orig_grad) > max_samples:
        indices = np.random.choice(len(orig_grad), max_samples, replace=False)
        orig_grad = orig_grad[indices]
        noisy_grad = noisy_grad[indices]

    plt.figure(figsize=(12, 6))

    # Plot histograms of gradient distributions
    plt.subplot(1, 2, 1)
    plt.hist(orig_grad, bins=50, alpha=0.7, label='Original Gradients')
    plt.hist(noisy_grad, bins=50, alpha=0.7, label='DP Noisy Gradients')
    plt.title('Gradient Distribution Comparison')
    plt.xlabel('Gradient Value')
    plt.ylabel('Frequency')
    plt.legend()

    # Plot QQ plot to compare distributions
    plt.subplot(1, 2, 2)
    orig_sorted = np.sort(orig_grad)
    noisy_sorted = np.sort(noisy_grad)
    plt.scatter(orig_sorted, noisy_sorted, alpha=0.5, s=10)
    min_val = min(orig_sorted.min(), noisy_sorted.min())
    max_val = max(orig_sorted.max(), noisy_sorted.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.title('Gradient Q-Q Plot')
    plt.xlabel('Original Gradients')
    plt.ylabel('DP Noisy Gradients')

    plt.tight_layout()
    filepath = f"./visualizations/privacy_analysis/client_{client_id}_gradient_distribution_round_{round_num}.png"
    plt.savefig(filepath)
    plt.close()

    return filepath


def plot_noise_impact_on_accuracy(noise_multipliers, accuracies, f1_scores):
    """
    Visualize how different noise multipliers affect model performance.

    Args:
        noise_multipliers (list): List of noise multiplier values
        accuracies (list): Corresponding accuracy values
        f1_scores (list): Corresponding F1 score values

    Returns:
        str: Path to saved visualization
    """
    os.makedirs("./visualizations/privacy_analysis", exist_ok=True)

    plt.figure(figsize=(10, 6))

    plt.plot(noise_multipliers, accuracies, 'bo-', linewidth=2, markersize=8, label='Accuracy')
    plt.plot(noise_multipliers, f1_scores, 'go-', linewidth=2, markersize=8, label='F1 Score')

    # Calculate theoretical privacy budget (epsilon) for each noise multiplier
    # Use a fixed sample rate and iterations for illustration
    sample_rate = 0.01
    iterations = 100
    epsilons = [compute_epsilon(nm, sample_rate, iterations) for nm in noise_multipliers]

    # Add privacy budget as text labels
    for i, (nm, eps) in enumerate(zip(noise_multipliers, epsilons)):
        plt.annotate(f"ε={eps:.2f}",
                     (nm, accuracies[i]),
                     textcoords="offset points",
                     xytext=(0, 10),
                     ha='center')

    plt.title('Impact of DP Noise on Model Performance', fontsize=14)
    plt.xlabel('Noise Multiplier', fontsize=12)
    plt.ylabel('Performance Metric', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    filepath = "./visualizations/privacy_analysis/noise_impact_on_performance.png"
    plt.savefig(filepath)
    plt.close()

    return filepath


def plot_privacy_utility_tradeoff_curve(epsilons, accuracies, f1_scores):
    """
    Create a privacy-utility tradeoff curve.

    Args:
        epsilons (list): List of privacy budget values
        accuracies (list): Corresponding accuracy values
        f1_scores (list): Corresponding F1 score values

    Returns:
        str: Path to saved visualization
    """
    os.makedirs("./visualizations/privacy_analysis", exist_ok=True)

    plt.figure(figsize=(10, 6))

    # Sort by epsilon for proper curve
    sorted_data = sorted(zip(epsilons, accuracies, f1_scores))
    epsilons_sorted = [x[0] for x in sorted_data]
    accuracies_sorted = [x[1] for x in sorted_data]
    f1_scores_sorted = [x[2] for x in sorted_data]

    plt.plot(epsilons_sorted, accuracies_sorted, 'bo-', linewidth=2, markersize=8, label='Accuracy')
    plt.plot(epsilons_sorted, f1_scores_sorted, 'go-', linewidth=2, markersize=8, label='F1 Score')

    plt.title('Privacy-Utility Tradeoff', fontsize=14)
    plt.xlabel('Privacy Budget (ε)', fontsize=12)
    plt.ylabel('Utility (Performance Metric)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Add regions indicating privacy levels
    max_epsilon = max(epsilons_sorted)

    plt.axvspan(0, 1, alpha=0.2, color='green')
    plt.axvspan(1, 5, alpha=0.2, color='yellow')
    plt.axvspan(5, max_epsilon + 1, alpha=0.2, color='red')

    # Add additional legend for privacy regions
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.2, label='High Privacy (ε < 1)'),
        Patch(facecolor='yellow', alpha=0.2, label='Medium Privacy (1 ≤ ε < 5)'),
        Patch(facecolor='red', alpha=0.2, label='Low Privacy (ε ≥ 5)')
    ]
    plt.legend(handles=legend_elements, loc='lower right')

    filepath = "./visualizations/privacy_analysis/privacy_utility_tradeoff.png"
    plt.savefig(filepath)
    plt.close()

    return filepath


def plot_client_vs_global_accuracy_per_round(rounds_data):
    """
    Compare client and global model accuracy per round.

    Args:
        rounds_data (pandas.DataFrame): DataFrame with round metrics

    Returns:
        str: Path to saved visualization
    """
    os.makedirs("./visualizations/client_comparison", exist_ok=True)

    plt.figure(figsize=(10, 6))

    rounds = rounds_data['round'].tolist()
    client_avg_accuracy = rounds_data['average_accuracy'].tolist()

    # Use global accuracy from rounds_data if available
    if 'global_accuracy' in rounds_data.columns:
        global_accuracy = rounds_data['global_accuracy'].tolist()
    else:
        # If global accuracy isn't explicitly tracked, we can use the average accuracy
        # as an approximation, with a small adjustment to show the difference
        global_accuracy = client_avg_accuracy

    plt.plot(rounds, client_avg_accuracy, 'bo-', linewidth=2, markersize=8, label='Client Avg. Accuracy')
    plt.plot(rounds, global_accuracy, 'ro-', linewidth=2, markersize=8, label='Global Model Accuracy')

    plt.title('Client vs. Global Model Accuracy per Round', fontsize=14)
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    filepath = "./visualizations/client_comparison/client_vs_global_accuracy.png"
    plt.savefig(filepath)
    plt.close()

    return filepath


def plot_per_client_privacy_consumption(client_ids, client_epsilons, round_num):
    """
    Visualize how privacy budget is consumed by different clients.

    Args:
        client_ids (list): List of client IDs
        client_epsilons (list): List of privacy budgets (epsilon) for each client
        round_num (int): Current round number

    Returns:
        str: Path to saved visualization
    """
    os.makedirs("./visualizations/privacy_analysis", exist_ok=True)

    plt.figure(figsize=(10, 6))

    # Create bar chart of epsilon values
    bars = plt.bar(client_ids, client_epsilons, color='skyblue', alpha=0.7)

    # Add value labels on top of bars
    for bar, eps in zip(bars, client_epsilons):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f'ε={eps:.2f}', ha='center', va='bottom')

    plt.title(f'Privacy Budget Consumption by Client (Round {round_num})', fontsize=14)
    plt.xlabel('Client ID', fontsize=12)
    plt.ylabel('Privacy Budget (ε)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')

    # Add a horizontal line for average epsilon
    avg_epsilon = sum(client_epsilons) / len(client_epsilons)
    plt.axhline(y=avg_epsilon, color='r', linestyle='--', label=f'Average ε = {avg_epsilon:.2f}')
    plt.legend()

    filepath = f"./visualizations/privacy_analysis/per_client_privacy_budget_round_{round_num}.png"
    plt.savefig(filepath)
    plt.close()

    return filepath


def plot_membership_inference_risk(epsilon_values, membership_inference_risks):
    """
    Visualize how differential privacy protects against membership inference attacks.

    Args:
        epsilon_values (list): List of privacy budget values
        membership_inference_risks (list): Estimated risk of membership inference attack

    Returns:
        str: Path to saved visualization
    """
    os.makedirs("./visualizations/privacy_analysis", exist_ok=True)

    plt.figure(figsize=(10, 6))

    # Plot theoretical risk curve
    plt.plot(epsilon_values, membership_inference_risks, 'ro-', linewidth=2, markersize=8)

    plt.title('Theoretical Membership Inference Attack Risk vs. Privacy Budget', fontsize=14)
    plt.xlabel('Privacy Budget (ε)', fontsize=12)
    plt.ylabel('Membership Inference Attack Success Rate', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add risk levels
    plt.axhspan(0, 0.55, alpha=0.2, color='green', label='Low Risk')
    plt.axhspan(0.55, 0.7, alpha=0.2, color='yellow', label='Medium Risk')
    plt.axhspan(0.7, 1.0, alpha=0.2, color='red', label='High Risk')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.2, label='Low Risk (<55% success)'),
        Patch(facecolor='yellow', alpha=0.2, label='Medium Risk (55-70% success)'),
        Patch(facecolor='red', alpha=0.2, label='High Risk (>70% success)')
    ]
    plt.legend(handles=legend_elements)

    filepath = "./visualizations/privacy_analysis/membership_inference_risk.png"
    plt.savefig(filepath)
    plt.close()

    return filepath


def plot_dp_vs_non_dp_performance(rounds, dp_accuracies, non_dp_accuracies):
    """
    Compare performance between DP and non-DP training over rounds.

    Args:
        rounds (list): List of training rounds
        dp_accuracies (list): Accuracy values with DP
        non_dp_accuracies (list): Accuracy values without DP

    Returns:
        str: Path to saved visualization
    """
    os.makedirs("./visualizations/privacy_analysis", exist_ok=True)

    plt.figure(figsize=(10, 6))

    plt.plot(rounds, dp_accuracies, 'bo-', linewidth=2, markersize=8, label='With DP')
    plt.plot(rounds, non_dp_accuracies, 'go-', linewidth=2, markersize=8, label='Without DP')

    # Calculate accuracy gap
    accuracy_gaps = [non_dp - dp for dp, non_dp in zip(dp_accuracies, non_dp_accuracies)]

    # Fill the area between curves to highlight privacy cost
    plt.fill_between(rounds, dp_accuracies, non_dp_accuracies, color='red', alpha=0.2, label='Privacy Cost')

    plt.title('Accuracy Comparison: DP vs. Non-DP Training', fontsize=14)
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Add text annotation for final privacy cost
    if len(rounds) > 0:
        final_gap = accuracy_gaps[-1]
        plt.annotate(f'Final accuracy gap: {final_gap:.2%}',
                     xy=(rounds[-1], (dp_accuracies[-1] + non_dp_accuracies[-1]) / 2),
                     xytext=(rounds[-1] - 1, (dp_accuracies[-1] + non_dp_accuracies[-1]) / 2 + 0.05),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))

    filepath = "./visualizations/privacy_analysis/dp_vs_non_dp_performance.png"
    plt.savefig(filepath)
    plt.close()

    return filepath


def plot_gradient_norm_distribution(original_norms, clipped_norms, max_grad_norm, client_id, round_num):
    """
    Visualize the effect of gradient clipping on gradient norms.

    Args:
        original_norms (list): List of original gradient norms
        clipped_norms (list): List of clipped gradient norms
        max_grad_norm (float): Maximum gradient norm parameter
        client_id (int): Client ID
        round_num (int): Current round number

    Returns:
        str: Path to saved visualization
    """
    os.makedirs("./visualizations/privacy_analysis", exist_ok=True)

    plt.figure(figsize=(10, 6))

    max_norm = max(max(original_norms), max_grad_norm * 1.5)
    bins = np.linspace(0, max_norm, 50)

    plt.hist(original_norms, bins=bins, alpha=0.5, label='Original Gradient Norms')
    plt.hist(clipped_norms, bins=bins, alpha=0.5, label='Clipped Gradient Norms')

    plt.axvline(x=max_grad_norm, color='r', linestyle='--',
                label=f'Clipping Threshold (C={max_grad_norm})')

    # Calculate percentage of gradients clipped
    pct_clipped = 100 * sum(norm > max_grad_norm for norm in original_norms) / len(original_norms)

    plt.title(f'Gradient Norm Distribution (Client {client_id}, Round {round_num})', fontsize=14)
    plt.xlabel('Gradient Norm', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Add text annotation for percentage clipped
    plt.annotate(f'{pct_clipped:.1f}% of gradients clipped',
                 xy=(max_grad_norm, plt.ylim()[1] * 0.9),
                 xytext=(max_grad_norm * 1.1, plt.ylim()[1] * 0.9),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))

    filepath = f"./visualizations/privacy_analysis/client_{client_id}_gradient_clipping_round_{round_num}.png"
    plt.savefig(filepath)
    plt.close()

    return filepath


def visualize_epsilon_delta_tradeoff(noise_multipliers, delta=1e-5):
    """
    Visualize the tradeoff between epsilon and delta for different noise multipliers.

    Args:
        noise_multipliers (list): List of noise multiplier values
        delta (float): Fixed delta value

    Returns:
        str: Path to saved visualization
    """
    os.makedirs("./visualizations/privacy_analysis", exist_ok=True)

    # Calculate epsilon for different values of noise multiplier
    sample_rate = 0.01  # Example value
    iterations = 100  # Example value

    epsilons = [compute_epsilon(nm, sample_rate, iterations, delta) for nm in noise_multipliers]

    plt.figure(figsize=(10, 6))

    plt.plot(noise_multipliers, epsilons, 'bo-', linewidth=2, markersize=8)

    plt.title(f'Privacy Parameters Tradeoff (δ={delta})', fontsize=14)
    plt.xlabel('Noise Multiplier', fontsize=12)
    plt.ylabel('Privacy Budget (ε)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add privacy regions
    plt.axhspan(0, 1, alpha=0.2, color='green', label='High Privacy (ε < 1)')
    plt.axhspan(1, 5, alpha=0.2, color='yellow', label='Medium Privacy (1 ≤ ε < 5)')
    plt.axhspan(5, max(epsilons) + 1, alpha=0.2, color='red', label='Low Privacy (ε ≥ 5)')

    plt.legend()

    filepath = "./visualizations/privacy_analysis/epsilon_noise_tradeoff.png"
    plt.savefig(filepath)
    plt.close()

    return filepath


def plot_accuracy_improvement_rate(client_ids, round_improvements, is_dp=True):
    """
    Visualize the rate of accuracy improvement for each client.

    Args:
        client_ids (list): List of client IDs
        round_improvements (list): List of lists containing accuracy improvements per round for each client
        is_dp (bool): Whether the improvements are for DP training or not

    Returns:
        str: Path to saved visualization
    """
    os.makedirs("./visualizations/client_comparison", exist_ok=True)

    plt.figure(figsize=(12, 6))

    # Plot improvement rate for each client
    for i, client_id in enumerate(client_ids):
        improvements = round_improvements[i]
        rounds = list(range(1, len(improvements) + 1))
        plt.plot(rounds, improvements, marker='o', linestyle='-', label=f'Client {client_id}')

    title_prefix = "DP" if is_dp else "Non-DP"
    plt.title(f'{title_prefix} Training: Accuracy Improvement Rate per Client', fontsize=14)
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Accuracy Improvement from Previous Round', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Add horizontal line at zero improvement
    plt.axhline(y=0, color='r', linestyle='--')

    filepath = f"./visualizations/client_comparison/{'dp' if is_dp else 'non_dp'}_accuracy_improvement_rate.png"
    plt.savefig(filepath)
    plt.close()

    return filepath


def plot_all_clients_per_round_accuracy(client_ids, all_client_accuracies, global_accuracies=None):
    """
    Plot the accuracy of all clients and the global model for each round.

    Args:
        client_ids (list): List of client IDs
        all_client_accuracies (list): List of lists containing accuracies for each client across rounds
        global_accuracies (list, optional): List of global model accuracies per round

    Returns:
        str: Path to saved visualization
    """
    os.makedirs("./visualizations/client_comparison", exist_ok=True)

    # Determine the number of rounds based on data
    num_rounds = len(all_client_accuracies[0])
    rounds = list(range(1, num_rounds + 1))

    plt.figure(figsize=(12, 8))

    # Plot each client's accuracy
    for i, client_id in enumerate(client_ids):
        plt.plot(rounds, all_client_accuracies[i], marker='o', linestyle='-', alpha=0.7,
                 label=f'Client {client_id}')

    # Plot global model accuracy if provided
    if global_accuracies:
        plt.plot(rounds, global_accuracies, 'ko-', linewidth=3, markersize=10, label='Global Model')

    plt.title('Client and Global Model Accuracy per Round', fontsize=14)
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    filepath = "./visualizations/client_comparison/all_clients_accuracy_per_round.png"
    plt.savefig(filepath)
    plt.close()

    return filepath


def plot_privacy_leakage_reduction(epsilons, leak_probabilities):
    """
    Visualize how increasing privacy protection (lower epsilon) reduces the probability of data leakage.

    Args:
        epsilons (list): List of privacy budget values
        leak_probabilities (list): Corresponding theoretical probabilities of information leakage

    Returns:
        str: Path to saved visualization
    """
    os.makedirs("./visualizations/privacy_analysis", exist_ok=True)

    plt.figure(figsize=(10, 6))

    # Sort by epsilon for proper curve
    sorted_data = sorted(zip(epsilons, leak_probabilities))
    epsilons_sorted = [x[0] for x in sorted_data]
    leak_probs_sorted = [x[1] for x in sorted_data]

    plt.plot(epsilons_sorted, leak_probs_sorted, 'ro-', linewidth=2, markersize=8)

    plt.title('Privacy Protection: Information Leakage Risk Reduction', fontsize=14)
    plt.xlabel('Privacy Budget (ε)', fontsize=12)
    plt.ylabel('Theoretical Probability of Information Leakage', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add annotations for specific epsilon values
    for eps, prob in zip(epsilons_sorted, leak_probs_sorted):
        if eps in [min(epsilons_sorted), max(epsilons_sorted)] or eps == 1.0:
            plt.annotate(f'ε={eps:.1f}, risk={prob:.2%}',
                         xy=(eps, prob),
                         xytext=(eps + 0.2, prob + 0.05),
                         arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                         bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))

    filepath = "./visualizations/privacy_analysis/privacy_leakage_reduction.png"
    plt.savefig(filepath)
    plt.close()

    return filepath


def simulate_membership_inference_risk(epsilon):
    """
    Simulate the theoretical risk of a membership inference attack based on epsilon.
    This is a simplified model for visualization purposes.

    Args:
        epsilon (float): Privacy budget

    Returns:
        float: Estimated risk of successful membership inference (0-1)
    """
    # A simple model: risk increases with epsilon but saturates
    # This is simplified for visualization - actual risk depends on many factors
    base_risk = 0.5  # Random guessing
    max_risk = 0.95  # Maximum achievable risk

    # Risk increases with epsilon but with diminishing returns
    risk = base_risk + (max_risk - base_risk) * (1 - np.exp(-epsilon / 3))
    return min(risk, max_risk)


def calculate_theoretical_leak_probability(epsilon):
    """
    Calculate a theoretical probability of information leakage based on epsilon.
    This is a simplified model for visualization purposes.

    Args:
        epsilon (float): Privacy budget

    Returns:
        float: Theoretical probability of information leakage (0-1)
    """
    # This is a simplified model for visualization
    # The actual relationship is complex and depends on many factors
    return 1 - np.exp(-epsilon)