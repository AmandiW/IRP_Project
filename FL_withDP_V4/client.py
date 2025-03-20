import flwr as fl
import torch
from utils import create_model, load_data, EarlyStopping, create_privacy_engine, calculate_privacy_metrics
from utils import plot_confusion_matrix, plot_metrics_over_rounds, plot_privacy_budget, plot_roc_curve
from utils import plot_class_distribution, PrivacyMetricsLogger
from collections import OrderedDict
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
import logging
import numpy as np
import os

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class RetinopathyClient(fl.client.NumPyClient):
    def __init__(self, client_id, train_loader, test_loader, train_size, test_size, dp_params=None):
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_size = train_size
        self.test_size = test_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_model(make_dp_compatible=True).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())

        # Differential privacy parameters
        self.dp_params = dp_params or {
            "noise_multiplier": 1.0,  # Higher noise = more privacy but less accuracy
            "max_grad_norm": 1.0,  # Lower norm = more privacy but less accuracy
            "delta": 1e-5,  # Target delta
        }

        # Privacy engine (will be initialized during training)
        self.privacy_engine = None

        # Current round tracking
        self.current_round = 0

        # Metrics tracking
        self.privacy_metrics_logger = PrivacyMetricsLogger(client_id=client_id)
        self.metrics_history = []
        self.privacy_metrics_history = []

        # Early stopping initialization
        self.early_stopping = EarlyStopping(patience=3, min_delta=0.001)

        # Get logger
        self.logger = logging.getLogger(f"Client_{client_id}")
        self.logger.info(f"Client {self.client_id} initialized on device: {self.device}")
        self.logger.info(
            f"Client {self.client_id} - Training set size: {self.train_size}, Test set size: {self.test_size}")
        self.logger.info(f"Client {self.client_id} - DP parameters: {self.dp_params}")

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.current_round = config.get("server_round", self.current_round + 1)
        self.logger.info(f"\nClient {self.client_id} - Starting training round {self.current_round}")
        self.logger.info(f"Client {self.client_id} - Training on {self.train_size} samples")
        self.set_parameters(parameters)

        # Reset early stopping for this round
        self.early_stopping = EarlyStopping(patience=3, min_delta=0.001)
        stop_early = False

        # Setup differential privacy
        # Calculate sample rate based on batch size and dataset size
        batch_size = next(iter(self.train_loader))[0].shape[0]
        sample_rate = batch_size / self.train_size

        # Create new optimizer for this round (required for DP integration)
        optimizer = torch.optim.Adam(self.model.parameters())

        # Initialize privacy engine with the simpler attach method
        self.privacy_engine, optimizer = create_privacy_engine(
            model=self.model,
            optimizer=optimizer,
            data_loader=self.train_loader,
            noise_multiplier=self.dp_params["noise_multiplier"],
            max_grad_norm=self.dp_params["max_grad_norm"]
        )

        # Then use this optimizer in the training loop

        # Use the DP optimizer
        # optimizer = dp_optimizer

        # Training
        self.model.train()
        for epoch in range(2):
            running_loss = 0.0
            correct = 0
            total = 0
            all_targets = []
            all_predictions = []

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                all_targets.extend(target.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

                if batch_idx % 10 == 0:
                    self.logger.info(
                        f"Client {self.client_id} - Epoch {epoch} - Batch {batch_idx}/{len(self.train_loader)} - Loss: {loss.item():.4f}")

            epoch_loss = running_loss / len(self.train_loader)
            epoch_accuracy = correct / total

            # Calculate other metrics
            f1 = f1_score(all_targets, all_predictions, average='weighted')
            precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
            recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)

            # Count class predictions
            unique, counts = np.unique(all_predictions, return_counts=True)
            pred_distribution = dict(zip(unique, counts))

            self.logger.info(
                f"Client {self.client_id} - Epoch {epoch} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, "
                f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            self.logger.info(f"Client {self.client_id} - Epoch {epoch} - Prediction distribution: {pred_distribution}")

            # Get privacy metrics
            privacy_metrics = calculate_privacy_metrics(self.privacy_engine, epoch + 1)
            self.logger.info(
                f"Client {self.client_id} - Epoch {epoch} - Privacy budget (ε): {privacy_metrics['epsilon']:.4f}")

            # Visualize confusion matrix
            plot_confusion_matrix(
                all_targets,
                all_predictions,
                client_id=self.client_id,
                round_num=self.current_round
            )

            # Visualize class distribution
            plot_class_distribution(
                all_targets,
                all_predictions,
                client_id=self.client_id,
                round_num=self.current_round
            )

            # Evaluate on validation data for early stopping
            val_loss, val_metrics = self.evaluate_for_early_stopping()

            # Check early stopping
            if self.early_stopping(val_loss, self.model):
                self.logger.info(f"Client {self.client_id} - Early stopping at epoch {epoch}")
                stop_early = True

                # Restore best weights if configured
                if self.early_stopping.restore_best_weights and self.early_stopping.best_weights is not None:
                    self.model.load_state_dict(self.early_stopping.best_weights)
                break

        # Calculate and log final privacy metrics
        privacy_metrics = calculate_privacy_metrics(self.privacy_engine, 2)  # 2 epochs
        self.privacy_metrics_history.append(privacy_metrics)
        self.privacy_metrics_logger.log_privacy_metrics(privacy_metrics, self.current_round)

        # Log training performance metrics
        performance_metrics = {
            "loss": epoch_loss,
            "accuracy": epoch_accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }
        self.metrics_history.append(performance_metrics)
        self.privacy_metrics_logger.log_performance_metrics(performance_metrics, self.current_round)

        # Create dashboard for this round
        self.privacy_metrics_logger.create_round_dashboard(self.current_round)

        # Save model after training
        if not os.path.exists("./saved_models"):
            os.makedirs("./saved_models")
        torch.save(self.model.state_dict(),
                   f"./saved_models/client_{self.client_id}_model_round_{self.current_round}.pth")
        self.logger.info(f"Client {self.client_id} - Model saved after round {self.current_round}")

        return self.get_parameters(config={}), self.train_size, {
            "privacy_epsilon": privacy_metrics["epsilon"],
            "privacy_max_grad_norm": privacy_metrics["max_grad_norm"],
            "privacy_noise_multiplier": privacy_metrics["noise_multiplier"],
            "accuracy": epoch_accuracy,
            "f1": f1,
            "loss": epoch_loss
        }

    def evaluate_for_early_stopping(self):
        """Evaluate model for early stopping purposes"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        y_scores = []

        with torch.no_grad():
            for data, target in self.test_loader:  # Using test set as validation
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()

                # For metrics
                probs = torch.nn.functional.softmax(output, dim=1)
                y_scores.extend(probs[:, 1].cpu().numpy())  # Probability for class 1

                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                y_true.extend(target.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        val_loss /= len(self.test_loader)
        val_accuracy = correct / total

        # Calculate additional metrics
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)

        metrics = {
            "accuracy": val_accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "loss": val_loss
        }

        self.logger.info(f"Client {self.client_id} - Validation: Loss={val_loss:.4f}, Accuracy={val_accuracy:.4f}")

        return val_loss, metrics

    def evaluate(self, parameters, config):
        self.current_round = config.get("server_round", self.current_round)
        self.logger.info(f"\nClient {self.client_id} - Starting evaluation for round {self.current_round}")
        self.logger.info(f"Client {self.client_id} - Evaluating on {self.test_size} samples")
        self.set_parameters(parameters)

        # Evaluation
        self.model.eval()
        loss = 0.0
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        y_scores = []

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += self.criterion(output, target).item()

                # Get probabilities for ROC curve
                probs = torch.nn.functional.softmax(output, dim=1)
                y_scores.extend(probs[:, 1].cpu().numpy())  # Probability for class 1

                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                y_true.extend(target.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        loss /= len(self.test_loader)
        accuracy = correct / total

        # Calculate additional metrics
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)

        # Count class distribution in predictions
        unique, counts = np.unique(y_pred, return_counts=True)
        pred_distribution = dict(zip(unique, counts))

        # Count true class distribution
        unique_true, counts_true = np.unique(y_true, return_counts=True)
        true_distribution = dict(zip(unique_true, counts_true))

        # Print evaluation metrics
        self.logger.info(f"\nClient {self.client_id} - Evaluation Results for Round {self.current_round}:")
        self.logger.info(f"Loss: {loss:.4f}")
        self.logger.info(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        self.logger.info(f"True class distribution: {true_distribution}")
        self.logger.info(f"Prediction distribution: {pred_distribution}")
        self.logger.info("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        self.logger.info(f"\n{cm}")
        self.logger.info("\nClassification Report:")
        cr = classification_report(y_true, y_pred)
        self.logger.info(f"\n{cr}")

        # Create visualizations
        plot_confusion_matrix(
            y_true,
            y_pred,
            client_id=self.client_id,
            round_num=self.current_round
        )

        plot_roc_curve(
            y_true,
            y_scores,
            client_id=self.client_id,
            round_num=self.current_round
        )

        plot_class_distribution(
            y_true,
            y_pred,
            client_id=self.client_id,
            round_num=self.current_round
        )

        # Log performance metrics
        performance_metrics = {
            "loss": loss,
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }

        self.metrics_history.append(performance_metrics)
        self.privacy_metrics_logger.log_performance_metrics(performance_metrics, self.current_round)

        # Calculate privacy metrics if available
        if hasattr(self, 'privacy_engine') and self.privacy_engine is not None:
            privacy_metrics = calculate_privacy_metrics(self.privacy_engine, 2)  # 2 epochs
            self.privacy_metrics_logger.log_privacy_metrics(privacy_metrics, self.current_round)

        # Create dashboard
        self.privacy_metrics_logger.create_round_dashboard(self.current_round)

        return loss, self.test_size, {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "privacy_epsilon": self.privacy_metrics_history[-1]["epsilon"] if self.privacy_metrics_history else 0.0
        }


def start_client(client_id, dp_params=None):
    """Function to start a client with specific ID"""
    # Default DP parameters if not provided
    if dp_params is None:
        dp_params = {
            "noise_multiplier": 1.0,
            "max_grad_norm": 1.0,
            "delta": 1e-5
        }

    # Load data
    client_data = load_data(
        img_dir="D:/FYP_Data/combined_images",
        labels_path="D:/FYP_Data/cleaned_valid_image_labels.csv"
    )
    train_loader, test_loader, train_size, test_size = client_data[client_id]

    logger = logging.getLogger(f"Client_{client_id}")
    # Log client dataset sizes
    logger.info(f"Client {client_id} - Starting with {train_size} training samples, {test_size} testing samples")
    logger.info(f"Client {client_id} - Using DP parameters: {dp_params}")

    # Start client with DP parameters
    client = RetinopathyClient(
        client_id,
        train_loader,
        test_loader,
        train_size,
        test_size,
        dp_params=dp_params
    )
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    import sys

    # Use the logging setup from main
    from main import setup_logging

    setup_logging()

    # Parse client ID from command line arguments
    client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    # Parse DP parameters if provided
    dp_params = None
    if len(sys.argv) > 2:
        noise_multiplier = float(sys.argv[2])
        max_grad_norm = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
        dp_params = {
            "noise_multiplier": noise_multiplier,
            "max_grad_norm": max_grad_norm,
            "delta": 1e-5
        }

    start_client(client_id, dp_params)