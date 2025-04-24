import flwr as fl
import torch
import logging
import os
from collections import OrderedDict
from utils import (
    create_model,
    train_with_feature_specific_dp,
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_gradient_distribution,
    plot_gradient_norm_distribution,
    PrivacyMetricsLogger,
    EarlyStopping,
    # Privacy analysis functions
    compute_dp_sgd_privacy_budget,
    simulate_membership_inference_risk,
    plot_feature_specific_privacy_impact,
    visualize_feature_importance_heatmap,
    visualize_privacy_preservation_with_reconstruction,
    perform_membership_inference_attack, FocalLoss
)
import torch.nn.functional as F
import matplotlib.pyplot as plt


class RetinopathyClient(fl.client.NumPyClient):
    """Flower client for federated learning with differential privacy and feature-specific privacy."""

    def __init__(self, client_id, train_loader, test_loader, train_size, test_size,
                 model_config=None, dp_params=None):
        """
        Initialize the client.

        Args:
            client_id (int): Unique identifier for this client
            train_loader (DataLoader): Training data loader
            test_loader (DataLoader): Test data loader
            train_size (int): Number of training samples
            test_size (int): Number of test samples
            model_config (dict): Model configuration parameters
            dp_params (dict): Differential privacy parameters
        """
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_size = train_size
        self.test_size = test_size

        # Set default model config if not provided
        if model_config is None:
            model_config = {
                "model_name": "resnet18",
                "model_type": "resnet",
                "num_classes": 2
            }
        self.model_config = model_config

        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create model
        self.model = create_model(
            model_name=model_config.get("model_name", "resnet18"),
            model_type=model_config.get("model_type", "resnet"),
            num_classes=model_config.get("num_classes", 2)
        ).to(self.device)

        self.criterion = FocalLoss(alpha=0.25, gamma=2.0)

        # Get only trainable parameters for optimizer
        self.optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=0.001
        )

        # Differential privacy parameters
        self.dp_params = dp_params or {
            "noise_multiplier": 0.8,  # Noise multiplier for DP-SGD
            "max_grad_norm": 1.0,  # Maximum gradient norm for clipping
            "delta": 1e-5,  # Fixed delta value for privacy accounting
            "epochs": 3,  # Epochs per round
            "feature_specific": True  # Enable feature-specific DP using attention
        }

        # Current round tracking
        self.current_round = 0

        # Metrics tracking
        self.metrics_logger = PrivacyMetricsLogger(client_id=client_id)
        self.metrics_history = []
        self.privacy_metrics_history = []

        # Store accuracy history for improvement rate calculation
        self.accuracy_history = []

        # Early stopping
        self.early_stopping = EarlyStopping(patience=3, min_delta=0.001)

        # Set up logger
        self.logger = logging.getLogger(f"Client_{client_id}")
        self.logger.info(f"Client {self.client_id} initialized on device: {self.device}")
        self.logger.info(f"Client {self.client_id} - Train size: {self.train_size}, Test size: {self.test_size}")
        self.logger.info(f"Client {self.client_id} - DP parameters: {self.dp_params}")
        self.logger.info(f"Client {self.client_id} - Model: {self.model_config}")

        # Store global model parameters for FedProx
        self.global_model_params = None
        self.proximal_mu = 0.01  # Default value, will be updated as per user arguments

        # Verify that multiple layers are trainable
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Model has {trainable_params} trainable parameters out of {total_params} total parameters")

        # Add cumulative privacy budget tracking
        self.cumulative_epsilon = 0.0

    def get_parameters(self, config):
        """Get model parameters in NumPy format."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Set model parameters from NumPy arrays."""
        # Convert parameters to PyTorch tensors
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

        # Load parameters into model
        self.model.load_state_dict(state_dict, strict=True)

        # Store global model parameters for FedProx
        self.global_model_params = parameters

        # Specify which layers should be trainable
        for param in self.model.parameters():
            param.requires_grad = False

        # Make the final layer trainable
        if hasattr(self.model, 'backbone'):
            # For ResNet and DenseNet with CBAM
            if hasattr(self.model.backbone, 'fc'):
                # For ResNet
                for param in self.model.backbone.fc.parameters():
                    param.requires_grad = True

                # Make convolutional layers trainable
                if hasattr(self.model.backbone, 'layer4'):
                    for param in self.model.backbone.layer4.parameters():
                        param.requires_grad = True

                if hasattr(self.model.backbone, 'layer3'):
                    for param in self.model.backbone.layer3.parameters():
                        param.requires_grad = True
            elif hasattr(self.model.backbone, 'classifier'):
                # For DenseNet
                for param in self.model.backbone.classifier.parameters():
                    param.requires_grad = True

                # Make the last dense block trainable
                if hasattr(self.model.backbone, 'features'):
                    for name, module in self.model.backbone.features.named_children():
                        if 'denseblock4' in name:
                            for param in module.parameters():
                                param.requires_grad = True

        # Count and log which layers are trainable
        trainable_layers = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_layers.append(name)

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())

        self.logger.info(f"Client {self.client_id} - Trainable parameters: {trainable_params} of {total_params}")
        self.logger.info(f"Client {self.client_id} - Trainable layers: {len(trainable_layers)}")
        # Log a few example layer names to verify
        if trainable_layers:
            self.logger.info(f"Client {self.client_id} - Examples: {trainable_layers[:3]}")

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

        # Get FL strategy from config
        fl_strategy = config.get("strategy", "fedavg")
        if fl_strategy == "fedprox":
            self.proximal_mu = config.get("proximal_mu", 0.01)
            self.logger.info(f"Client {self.client_id} - Using FedProx with mu={self.proximal_mu}")
        else:
            self.logger.info(f"Client {self.client_id} - Using FedAvg strategy")

        # Reset early stopping for this round
        self.early_stopping = EarlyStopping(patience=5, min_delta=0.005)

        # Get number of epochs from dp_params
        epochs = self.dp_params.get("epochs", 3)
        self.logger.info(f"Client {self.client_id} - Training for {epochs} epochs this round")
        self.logger.info(
            f"Client {self.client_id} - Using feature-specific DP: {self.dp_params.get('feature_specific', True)}")

        # Train model with feature-specific differential privacy
        self.model, metrics, privacy_metrics, original_gradients, noisy_gradients = train_with_feature_specific_dp(
            model=self.model,
            train_loader=self.train_loader,
            optimizer=self.optimizer,
            criterion=self.criterion,
            device=self.device,
            dp_params=self.dp_params,
            logger=self.logger,
            epochs=epochs,
            global_model_params=self.global_model_params if fl_strategy == "fedprox" else None,
            proximal_mu=self.proximal_mu if fl_strategy == "fedprox" else 0.0
        )

        # Calculate and update cumulative privacy budget
        round_epsilon = privacy_metrics["epsilon"]
        self.cumulative_epsilon += round_epsilon
        self.logger.info(f"Client {self.client_id} - Round epsilon: {round_epsilon:.4f}, "
                         f"Cumulative epsilon: {self.cumulative_epsilon:.4f}")

        # Update privacy metrics with cumulative epsilon
        privacy_metrics["cumulative_epsilon"] = self.cumulative_epsilon

        # Create visualization directories
        os.makedirs("./visualizations/privacy_analysis", exist_ok=True)

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
        try:
            import pandas as pd
            client_metrics = {
                'client_id': self.client_id,
                'round': self.current_round,
                'accuracy': metrics["accuracy"],
                'f1': metrics["f1"],
                'loss': metrics["loss"],
                'epsilon': privacy_metrics["epsilon"],
                'cumulative_epsilon': self.cumulative_epsilon,
                'noise_multiplier': privacy_metrics["noise_multiplier"],
                'max_grad_norm': privacy_metrics["max_grad_norm"],
                'feature_specific': privacy_metrics.get("feature_specific", False)
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

        # Add visualization of gradient distributions since we have both original and noisy gradients
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
                            original_gradients[i:i + 1],
                            noisy_gradients[i:i + 1],
                            self.client_id,
                            self.current_round
                        )
                        # Only use one layer for visualization to prevent cluttering the output directory
                        break

                # Calculate gradient norms for visualization
                try:
                    original_norms = [g.norm().item() for g in original_gradients if g is not None]
                    clipped_norms = [min(g.norm().item(), self.dp_params["max_grad_norm"]) for g in original_gradients
                                     if g is not None]

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

        # Create feature-specific privacy visualizations if available
        if 'attention_maps' in privacy_metrics:
            try:
                plot_feature_specific_privacy_impact(
                    privacy_metrics['attention_maps'],
                    self.client_id,
                    self.current_round
                )
                self.logger.info(f"Client {self.client_id} - Created feature sensitivity visualization")

                # Create feature importance heatmap
                visualize_feature_importance_heatmap(self.model, self.test_loader, self.device)
                self.logger.info(f"Client {self.client_id} - Created feature importance heatmap")
            except Exception as e:
                self.logger.error(f"Error creating feature-specific privacy visualizations: {e}")

        # Create privacy-preserving image reconstruction visualization
        try:
            visualize_privacy_preservation_with_reconstruction(
                self.model,
                self.test_loader,
                self.device,
                self.dp_params["noise_multiplier"],
                self.client_id
            )
            self.logger.info(f"Client {self.client_id} - Created privacy reconstruction visualization")
        except Exception as e:
            self.logger.error(f"Error creating privacy reconstruction visualization: {e}")

        # Perform membership inference attack to evaluate privacy
        if self.current_round == config.get("num_rounds", 5):  # Only on final round to save time
            try:
                attack_accuracy, _ = perform_membership_inference_attack(
                    self.model,
                    self.train_loader,
                    self.test_loader,
                    self.device,
                    self.client_id
                )
                self.logger.info(
                    f"Client {self.client_id} - Membership inference attack accuracy: {attack_accuracy:.4f}")

                # Calculate theoretical risk based on epsilon for comparison
                theoretical_risk = simulate_membership_inference_risk(self.cumulative_epsilon)
                self.logger.info(
                    f"Client {self.client_id} - Theoretical membership inference risk: {theoretical_risk:.4f}")
                self.logger.info(
                    f"Client {self.client_id} - Actual vs theoretical risk difference: {(attack_accuracy - theoretical_risk):.4f}")
            except Exception as e:
                self.logger.error(f"Error performing membership inference attack: {e}")

        # Save model checkpoint
        if not os.path.exists("./saved_models"):
            os.makedirs("./saved_models")

        torch.save(
            self.model.state_dict(),
            f"./saved_models/client_{self.client_id}_model_round_{self.current_round}.pth"
        )

        self.logger.info(f"Client {self.client_id} - Model saved for round {self.current_round}")

        # Calculate improvement from previous round
        improvement = 0
        if len(self.accuracy_history) > 1:
            improvement = self.accuracy_history[-1] - self.accuracy_history[-2]

        # Return updated parameters and metrics
        return self.get_parameters(config={}), self.train_size, {
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "loss": metrics["loss"],
            "privacy_epsilon": privacy_metrics["epsilon"],
            "privacy_cumulative_epsilon": self.cumulative_epsilon,
            "privacy_delta": privacy_metrics["delta"],
            "privacy_noise_multiplier": privacy_metrics["noise_multiplier"],
            "privacy_max_grad_norm": privacy_metrics["max_grad_norm"],
            "accuracy_improvement": improvement,
            "feature_specific_privacy": self.dp_params.get("feature_specific", False)
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
            "privacy_epsilon": self.privacy_metrics_history[-1]["epsilon"] if self.privacy_metrics_history else 0.0,
            "privacy_cumulative_epsilon": self.cumulative_epsilon,
            "feature_specific_privacy": self.dp_params.get("feature_specific", False)
        }


def start_client(client_id, model_config=None, dp_params=None):
    """
    Start a federated learning client with the given ID.

    Args:
        client_id (int): Client ID
        model_config (dict): Model configuration parameters
        dp_params (dict): Differential privacy parameters
    """
    # Set up logging
    logger = logging.getLogger(f"Client_{client_id}")

    # Default DP parameters if not provided, will be changed if user provides arguments
    if dp_params is None:
        dp_params = {
            "noise_multiplier": 0.8,  # Noise multiplier for DP-SGD
            "max_grad_norm": 1.0,  # Maximum gradient norm for clipping
            "delta": 1e-5,  # Fixed delta value for privacy accounting
            "epochs": 3,  # Epochs per round
            "feature_specific": True  # Enable feature-specific DP using attention
        }

        # Vary parameters slightly for each client to create different privacy budgets
        # Add a small amount of variation based on client_id
    if "noise_multiplier" in dp_params:
        base_noise = dp_params["noise_multiplier"]
        dp_params["noise_multiplier"] = base_noise * (0.95 + 0.1 * (client_id % 3) / 2)
        logger.info(f"Adjusted noise multiplier for client {client_id}: {dp_params['noise_multiplier']:.4f}")

    # Default model config if not provided
    if model_config is None:
        model_config = {
            "model_name": "resnet18",
            "model_type": "resnet",
            "num_classes": 2
        }

    try:
        # Get configuration for data loading from the main process
        distribution_type = os.environ.get("FL_DISTRIBUTION", "iid")
        alpha = float(os.environ.get("FL_ALPHA", "0.5"))
        num_clients = int(os.environ.get("FL_NUM_CLIENTS", "2"))

        # Load data for all clients
        from utils import load_data

        # Use a configurable path for data loading, with fallback to default path
        img_dir = os.environ.get("FL_IMG_DIR", "E:/IRP_dataset_new/IRP_Final_Images")
        labels_path = os.environ.get("FL_LABELS_PATH", "E:/IRP_dataset_new/IRP_Final_Labels.csv")

        client_data = load_data(
            img_dir=img_dir,
            labels_path=labels_path,
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
        logger.info(f"Client {client_id} - Using model config: {model_config}")

        # Create client
        client = RetinopathyClient(
            client_id=client_id,
            train_loader=train_loader,
            test_loader=test_loader,
            train_size=train_size,
            test_size=test_size,
            model_config=model_config,
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
    model_config = None

    if len(sys.argv) > 2:
        noise_multiplier = float(sys.argv[2])
        max_grad_norm = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
        epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 3
        feature_specific = sys.argv[5].lower() == 'true' if len(sys.argv) > 5 else True

        dp_params = {
            "noise_multiplier": noise_multiplier,
            "max_grad_norm": max_grad_norm,
            "delta": 1e-5,
            "epochs": epochs,
            "feature_specific": feature_specific
        }

    # Parse model config if provided
    if len(sys.argv) > 6:
        model_type = sys.argv[6] if len(sys.argv) > 6 else "resnet"
        model_name = sys.argv[7] if len(sys.argv) > 7 else "resnet18"

        model_config = {
            "model_type": model_type,  # 'resnet' or 'densenet'
            "model_name": model_name,  # 'resnet18', 'resnet34', 'densenet121', etc.
            "num_classes": 2
        }

    # Start client
    start_client(client_id, model_config, dp_params)
