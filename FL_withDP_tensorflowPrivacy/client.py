import flwr as fl
import torch
import logging
import numpy as np
import os
from collections import OrderedDict
from utils import (
    create_model,
    train_with_dp,
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_gradient_distribution,
    plot_gradient_norm_distribution,
    PrivacyMetricsLogger,
    EarlyStopping,
    # Additional visualization functions
    compute_dp_sgd_privacy_budget,
    simulate_membership_inference_risk,
    plot_epsilon_composition,
    visualize_attack_risk_reduction
)


class RetinopathyClient(fl.client.NumPyClient):
    """Flower client for federated learning with differential privacy."""

    def __init__(self, client_id, train_loader, test_loader, train_size, test_size, dp_params=None):
        """
        Initialize the client.

        Args:
            client_id (int): Unique identifier for this client
            train_loader (DataLoader): Training data loader
            test_loader (DataLoader): Test data loader
            train_size (int): Number of training samples
            test_size (int): Number of test samples
            dp_params (dict): Differential privacy parameters
        """
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_size = train_size
        self.test_size = test_size

        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create model and optimizer
        self.model = create_model().to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()

        # Get only trainable parameters for optimizer
        self.optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=0.001
        )

        # Differential privacy parameters
        self.dp_params = dp_params or {
            "noise_multiplier": 1.0,  # Higher = more privacy, less accuracy
            "max_grad_norm": 1.0,  # Lower = more privacy, less accuracy
            "delta": 1e-5,  # Fixed delta value for privacy accounting
            "epochs": 1  # Number of epochs per round
        }

        # Current round tracking
        self.current_round = 0

        # Metrics tracking
        self.metrics_logger = PrivacyMetricsLogger(client_id=client_id)
        self.metrics_history = []
        self.privacy_metrics_history = []

        # Accuracy history for improvement rate calculation
        self.accuracy_history = []

        # Early stopping
        self.early_stopping = EarlyStopping(patience=3, min_delta=0.001)

        # Set up logger
        self.logger = logging.getLogger(f"Client_{client_id}")
        self.logger.info(f"Client {self.client_id} initialized on device: {self.device}")
        self.logger.info(f"Client {self.client_id} - Train size: {self.train_size}, Test size: {self.test_size}")
        self.logger.info(f"Client {self.client_id} - DP parameters: {self.dp_params}")

        # Verify that only the final layer parameters are trainable
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model has {trainable_params} trainable parameters out of {total_params} total parameters")

        # Create additional privacy visualizations at initialization
        self._create_privacy_visualizations()

    def _create_privacy_visualizations(self):
        """Create client-specific privacy visualizations."""
        # Ensure directory exists
        os.makedirs("./visualizations/privacy_analysis", exist_ok=True)

        # Calculate sample rate for this client's dataset
        batch_size = next(iter(self.train_loader))[0].shape[0]
        sample_rate = batch_size / self.train_size

        # Create epsilon composition visualization
        plot_epsilon_composition(
            num_rounds=10,  # Show for 10 rounds
            noise_multiplier=self.dp_params["noise_multiplier"],
            sample_rate=sample_rate,
            delta=self.dp_params["delta"]
        )

        # Create attack risk reduction visualization
        noise_values = np.linspace(0.5, 3.0, 10)
        visualize_attack_risk_reduction(noise_values, self.dp_params["delta"])

        self.logger.info(f"Client {self.client_id} - Created additional privacy visualizations")

    def get_parameters(self, config):
        """Get model parameters in NumPy format."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Set model parameters from NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

        # Re-freeze backbone parameters after loading
        for name, param in self.model.named_parameters():
            if 'backbone.fc' not in name:  # If not in the final layer
                param.requires_grad = False
            else:
                param.requires_grad = True

    def fit(self, parameters, config):
        """
        Train the model on local data with differential privacy.

        Args:
            parameters: Model parameters from the server
            config: Configuration from the server

        Returns:
            tuple: Updated parameters, number of examples, metrics
        """
        # Update round number and set parameters
        self.current_round = config.get("server_round", self.current_round + 1)
        self.logger.info(f"\nClient {self.client_id} - Starting training round {self.current_round}")
        self.set_parameters(parameters)

        # Reset early stopping for this round
        self.early_stopping = EarlyStopping(patience=3, min_delta=0.001)

        # Get number of epochs from dp_params
        epochs = self.dp_params.get("epochs", 1)
        self.logger.info(f"Client {self.client_id} - Training for {epochs} epochs this round using DP-SGD")

        # Train model with differential privacy
        # Now also returning the original and noisy gradients for visualization
        self.model, metrics, privacy_metrics, original_gradients, noisy_gradients = train_with_dp(
            model=self.model,
            train_loader=self.train_loader,
            optimizer=self.optimizer,
            criterion=self.criterion,
            device=self.device,
            dp_params=self.dp_params,
            logger=self.logger,
            epochs=epochs
        )

        # Store metrics
        self.metrics_history.append(metrics)
        self.privacy_metrics_history.append(privacy_metrics)

        # Store accuracy for improvement rate calculation
        self.accuracy_history.append(metrics["accuracy"])

        # Log metrics
        self.metrics_logger.log_performance_metrics(metrics, self.current_round)
        self.metrics_logger.log_privacy_metrics(privacy_metrics, self.current_round)

        # Create directory for storing client-specific data
        os.makedirs("./aggregated_metrics", exist_ok=True)

        # Save client metrics to file for later analysis
        # This will be used by main.py for creating client performance visualizations
        try:
            import pandas as pd
            client_metrics = {
                'client_id': self.client_id,
                'round': self.current_round,
                'accuracy': metrics["accuracy"],
                'f1': metrics["f1"],
                'loss': metrics["loss"],
                'epsilon': privacy_metrics["epsilon"],
                'noise_multiplier': privacy_metrics["noise_multiplier"],
                'max_grad_norm': privacy_metrics["max_grad_norm"]
            }

            client_metrics_df = pd.DataFrame([client_metrics])
            client_metrics_file = "./aggregated_metrics/client_performance.csv"

            if os.path.exists(client_metrics_file):
                existing_df = pd.read_csv(client_metrics_file)
                # Check if this exact entry already exists
                if not ((existing_df['client_id'] == self.client_id) &
                        (existing_df['round'] == self.current_round)).any():
                    # Only append if it doesn't exist
                    updated_df = pd.concat([existing_df, client_metrics_df], ignore_index=True)
                    updated_df.to_csv(client_metrics_file, index=False)
            else:
                client_metrics_df.to_csv(client_metrics_file, index=False)

            self.logger.info(f"Client {self.client_id} metrics saved for round {self.current_round}")
        except Exception as e:
            self.logger.error(f"Error saving client metrics: {e}")

        # Evaluate for early stopping
        val_loss, val_metrics = self.evaluate_for_early_stopping()

        # Check early stopping
        if self.early_stopping(val_loss, self.model):
            self.logger.info(f"Client {self.client_id} - Early stopping triggered")

            # Restore best weights if configured
            if self.early_stopping.restore_best_weights and self.early_stopping.best_weights is not None:
                self.model.load_state_dict(self.early_stopping.best_weights)

        # Create visualizations
        y_true = metrics.get("y_true", [])
        y_pred = metrics.get("y_pred", [])

        if len(y_true) > 0 and len(y_pred) > 0:
            plot_confusion_matrix(
                y_true=y_true,
                y_pred=y_pred,
                client_id=self.client_id,
                round_num=self.current_round
            )

        # Add visualization of gradient distributions if we have both original and noisy gradients
        # Ensure directories exist
        os.makedirs("./visualizations/privacy_analysis", exist_ok=True)

        try:
            if original_gradients is not None and noisy_gradients is not None and isinstance(original_gradients,
                                                                                             list) and isinstance(
                    noisy_gradients, list) and len(original_gradients) > 0 and len(noisy_gradients) > 0:
                # Ensure we're comparing corresponding layers
                for i in range(min(len(original_gradients), len(noisy_gradients))):
                    if i < len(original_gradients) and i < len(noisy_gradients) and original_gradients[
                        i] is not None and noisy_gradients[i] is not None:
                        # Use the first gradient tensor for visualization
                        plot_gradient_distribution(
                            original_gradients[i:i + 1],  # Pass as a list to avoid tensor boolean ambiguity
                            noisy_gradients[i:i + 1],  # Pass as a list to avoid tensor boolean ambiguity
                            self.client_id,
                            self.current_round
                        )
                        # Only use one layer for visualization
                        break

                # Calculate gradient norms for visualization
                try:
                    original_norms = [g.norm().item() for g in original_gradients if g is not None]
                    clipped_norms = [min(g.norm().item(), self.dp_params["max_grad_norm"])
                                     for g in original_gradients if g is not None]

                    if len(original_norms) > 0 and len(clipped_norms) > 0:
                        plot_gradient_norm_distribution(
                            original_norms,
                            clipped_norms,
                            self.dp_params["max_grad_norm"],
                            self.client_id,
                            self.current_round
                        )
                except Exception as e:
                    self.logger.error(f"Error creating gradient norm visualization: {e}")
        except Exception as e:
            self.logger.error(f"Error creating gradient visualizations: {e}")

        # Calculate theoretical risk based on epsilon
        try:
            risk = simulate_membership_inference_risk(privacy_metrics["epsilon"])
            self.logger.info(f"Client {self.client_id} - Theoretical membership inference attack risk: {risk:.4f}")
        except Exception as e:
            self.logger.error(f"Error calculating theoretical privacy risk: {e}")

        # Save model checkpoint
        if not os.path.exists("./saved_models"):
            os.makedirs("./saved_models")

        torch.save(
            self.model.state_dict(),
            f"./saved_models/client_{self.client_id}_model_round_{self.current_round}.pth"
        )

        self.logger.info(f"Client {self.client_id} - Model saved for round {self.current_round}")

        # Calculate improvement from previous round if available
        improvement = 0
        if len(self.accuracy_history) > 1:
            improvement = self.accuracy_history[-1] - self.accuracy_history[-2]

        # Return updated parameters and metrics
        return self.get_parameters(config={}), self.train_size, {
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "loss": metrics["loss"],
            "privacy_epsilon": privacy_metrics["epsilon"],
            "privacy_delta": privacy_metrics["delta"],
            "privacy_noise_multiplier": privacy_metrics["noise_multiplier"],
            "privacy_max_grad_norm": privacy_metrics["max_grad_norm"],
            "accuracy_improvement": improvement
        }

    def evaluate_for_early_stopping(self):
        """Evaluate model for early stopping purposes."""
        loss, metrics = evaluate_model(
            model=self.model,
            test_loader=self.test_loader,
            criterion=self.criterion,
            device=self.device
        )

        self.logger.info(
            f"Client {self.client_id} - Validation: "
            f"Loss={loss:.4f}, Accuracy={metrics['accuracy']:.4f}"
        )

        return loss, metrics

    def evaluate(self, parameters, config):
        """
        Evaluate the model on local test data.

        Args:
            parameters: Model parameters from the server
            config: Configuration from the server

        Returns:
            tuple: Loss, number of examples, metrics
        """
        # Update round number and set parameters
        self.current_round = config.get("server_round", self.current_round)
        self.logger.info(f"\nClient {self.client_id} - Evaluating on round {self.current_round}")
        self.set_parameters(parameters)

        # Evaluate model
        loss, metrics = evaluate_model(
            model=self.model,
            test_loader=self.test_loader,
            criterion=self.criterion,
            device=self.device
        )

        # Create visualizations
        y_true = metrics.get("y_true", [])
        y_pred = metrics.get("y_pred", [])
        y_scores = metrics.get("y_scores", [])

        if len(y_true) > 0 and len(y_pred) > 0:
            plot_confusion_matrix(
                y_true=y_true,
                y_pred=y_pred,
                client_id=self.client_id,
                round_num=self.current_round
            )

            plot_roc_curve(
                y_true=y_true,
                y_scores=y_scores,
                client_id=self.client_id,
                round_num=self.current_round
            )

        # Log detailed results
        self.logger.info(f"Client {self.client_id} - Evaluation Results:")
        self.logger.info(f"  Loss: {loss:.4f}")
        self.logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"  F1 Score: {metrics['f1']:.4f}")
        self.logger.info(f"  Precision: {metrics['precision']:.4f}")
        self.logger.info(f"  Recall: {metrics['recall']:.4f}")

        # Return evaluation results
        return loss, self.test_size, {
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "privacy_epsilon": self.privacy_metrics_history[-1]["epsilon"] if self.privacy_metrics_history else 0.0
        }


def start_client(client_id, dp_params=None):
    """
    Start a federated learning client with the given ID.

    Args:
        client_id (int): Client ID
        dp_params (dict): Differential privacy parameters
    """
    # Set up logging
    logger = logging.getLogger(f"Client_{client_id}")

    # Default DP parameters if not provided
    if dp_params is None:
        dp_params = {
            "noise_multiplier": 1.0,
            "max_grad_norm": 1.0,
            "delta": 1e-5,
            "epochs": 1
        }

    try:
        # Get configuration for data loading from the main process
        distribution_type = os.environ.get("FL_DISTRIBUTION", "iid")
        alpha = float(os.environ.get("FL_ALPHA", "0.5"))
        num_clients = int(os.environ.get("FL_NUM_CLIENTS", "2"))

        # Load data for all clients
        from utils import load_data

        client_data = load_data(
            img_dir="D:/FYP_Data/combined_images",
            labels_path="D:/FYP_Data/cleaned_valid_image_labels.csv",
            num_clients=num_clients,
            distribution=distribution_type,
            alpha=alpha
        )

        # Ensure client_id is within range
        if client_id >= len(client_data):
            logger.error(f"Client ID {client_id} is out of range. Only {len(client_data)} clients available.")
            return

        train_loader, test_loader, train_size, test_size = client_data[client_id]

        # Log client dataset information
        logger.info(f"Client {client_id} - Loaded {train_size} training samples and {test_size} test samples")
        logger.info(f"Client {client_id} - Distribution type: {distribution_type}")
        if distribution_type == "non_iid":
            logger.info(f"Client {client_id} - Alpha value: {alpha}")
        logger.info(f"Client {client_id} - Using DP parameters: {dp_params}")

        # Create client
        client = RetinopathyClient(
            client_id=client_id,
            train_loader=train_loader,
            test_loader=test_loader,
            train_size=train_size,
            test_size=test_size,
            dp_params=dp_params
        )

        # Start client
        logger.info(f"Client {client_id} - Starting Flower client")
        fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)

    except Exception as e:
        logger.error(f"Client {client_id} - Error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    import sys
    from main import setup_logging

    # Set up logging
    setup_logging()

    # Parse command line arguments
    client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    # Parse DP parameters if provided
    dp_params = None
    if len(sys.argv) > 2:
        noise_multiplier = float(sys.argv[2])
        max_grad_norm = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
        epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        dp_params = {
            "noise_multiplier": noise_multiplier,
            "max_grad_norm": max_grad_norm,
            "delta": 1e-5,
            "epochs": epochs
        }

    # Start client
    start_client(client_id, dp_params)