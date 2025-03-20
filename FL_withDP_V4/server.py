import flwr as fl
import torch
from utils import create_model, load_data
from utils import plot_confusion_matrix, plot_metrics_over_rounds, plot_privacy_budget, plot_roc_curve
from utils import plot_class_distribution, PrivacyMetricsLogger
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
import numpy as np
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class DPFedAvg(fl.server.strategy.FedAvg):
    """
    Extends Flower's FedAvg strategy to aggregate differential privacy metrics
    and handle privacy-preserving federated learning.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.privacy_metrics_history = []
        self.performance_metrics_history = []
        self.logger = logging.getLogger("DPFedAvg")
        self.privacy_metrics_logger = PrivacyMetricsLogger(is_global=True)

        # Create directory for aggregated metrics
        if not os.path.exists("./aggregated_metrics"):
            os.makedirs("./aggregated_metrics")

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate model weights and privacy metrics from client training results."""
        # Log round information
        self.logger.info(f"\nServer Round {server_round} - Aggregating training results")
        self.logger.info(f"Number of clients that succeeded: {len(results)}")
        self.logger.info(f"Number of clients that failed: {len(failures)}")

        # Extract privacy metrics from client results
        privacy_metrics = {
            "client_epsilons": [],
            "max_grad_norms": [],
            "noise_multipliers": []
        }

        for _, fit_res in results:
            metrics = fit_res.metrics
            if "privacy_epsilon" in metrics:
                privacy_metrics["client_epsilons"].append(metrics["privacy_epsilon"])
            if "privacy_max_grad_norm" in metrics:
                privacy_metrics["max_grad_norms"].append(metrics["privacy_max_grad_norm"])
            if "privacy_noise_multiplier" in metrics:
                privacy_metrics["noise_multipliers"].append(metrics["privacy_noise_multiplier"])

        # Compute average privacy metrics
        avg_privacy_metrics = {
            "epsilon": np.mean(privacy_metrics["client_epsilons"]) if privacy_metrics["client_epsilons"] else 0.0,
            "max_grad_norm": np.mean(privacy_metrics["max_grad_norms"]) if privacy_metrics["max_grad_norms"] else 0.0,
            "noise_multiplier": np.mean(privacy_metrics["noise_multipliers"]) if privacy_metrics[
                "noise_multipliers"] else 0.0,
            "delta": 1e-5,  # Fixed delta value
            "best_alpha": None,  # Not available from aggregation
            "sample_rate": None,  # Not available from aggregation
            "epochs": 2,  # Assuming 2 epochs for each client
        }

        # Log aggregated privacy metrics
        self.logger.info(f"Server Round {server_round} - Aggregated Privacy Metrics:")
        self.logger.info(f"  Average ε (Epsilon): {avg_privacy_metrics['epsilon']:.4f}")
        self.logger.info(f"  Average Max Gradient Norm: {avg_privacy_metrics['max_grad_norm']:.4f}")
        self.logger.info(f"  Average Noise Multiplier: {avg_privacy_metrics['noise_multiplier']:.4f}")

        # Store privacy metrics history
        self.privacy_metrics_history.append(avg_privacy_metrics)
        self.privacy_metrics_logger.log_privacy_metrics(avg_privacy_metrics, server_round)

        # Extract and average performance metrics
        performance_metrics = {
            "accuracies": [],
            "f1_scores": [],
            "losses": []
        }

        for _, fit_res in results:
            metrics = fit_res.metrics
            if "accuracy" in metrics:
                performance_metrics["accuracies"].append(metrics["accuracy"])
            if "f1" in metrics:
                performance_metrics["f1_scores"].append(metrics["f1"])
            if "loss" in metrics:
                performance_metrics["losses"].append(metrics["loss"])

        # Compute average performance metrics
        avg_performance_metrics = {
            "accuracy": np.mean(performance_metrics["accuracies"]) if performance_metrics["accuracies"] else 0.0,
            "f1": np.mean(performance_metrics["f1_scores"]) if performance_metrics["f1_scores"] else 0.0,
            "loss": np.mean(performance_metrics["losses"]) if performance_metrics["losses"] else 0.0
        }

        # Log aggregated performance metrics
        self.logger.info(f"Server Round {server_round} - Aggregated Performance Metrics:")
        self.logger.info(f"  Average Accuracy: {avg_performance_metrics['accuracy']:.4f}")
        self.logger.info(f"  Average F1 Score: {avg_performance_metrics['f1']:.4f}")
        self.logger.info(f"  Average Loss: {avg_performance_metrics['loss']:.4f}")

        # Store performance metrics history
        self.performance_metrics_history.append(avg_performance_metrics)
        self.privacy_metrics_logger.log_performance_metrics(avg_performance_metrics, server_round)

        # Create dashboard
        self.privacy_metrics_logger.create_round_dashboard(server_round)

        # Create and save metadata CSV for this round
        round_metadata = {
            "round": server_round,
            "num_clients": len(results),
            "average_epsilon": avg_privacy_metrics["epsilon"],
            "average_accuracy": avg_performance_metrics["accuracy"],
            "average_f1": avg_performance_metrics["f1"],
            "average_loss": avg_performance_metrics["loss"]
        }

        # Append to rounds metadata file
        rounds_df = pd.DataFrame([round_metadata])
        if os.path.exists("./aggregated_metrics/rounds_metadata.csv"):
            existing_df = pd.read_csv("./aggregated_metrics/rounds_metadata.csv")
            rounds_df = pd.concat([existing_df, rounds_df], ignore_index=True)

        rounds_df.to_csv("./aggregated_metrics/rounds_metadata.csv", index=False)

        # Call parent method to aggregate weights
        return super().aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate evaluation results from clients."""
        # Log round information
        self.logger.info(f"\nServer Round {server_round} - Aggregating evaluation results")
        self.logger.info(f"Number of clients that succeeded: {len(results)}")
        self.logger.info(f"Number of clients that failed: {len(failures)}")

        # Extract and average evaluation metrics
        metrics = {
            "accuracies": [],
            "f1_scores": [],
            "precisions": [],
            "recalls": [],
            "losses": [],
            "privacy_epsilons": []
        }

        # Collect metrics from all clients
        for _, eval_res in results:
            client_metrics = eval_res.metrics
            if "accuracy" in client_metrics:
                metrics["accuracies"].append(client_metrics["accuracy"])
            if "f1" in client_metrics:
                metrics["f1_scores"].append(client_metrics["f1"])
            if "precision" in client_metrics:
                metrics["precisions"].append(client_metrics["precision"])
            if "recall" in client_metrics:
                metrics["recalls"].append(client_metrics["recall"])
            if "privacy_epsilon" in client_metrics:
                metrics["privacy_epsilons"].append(client_metrics["privacy_epsilon"])

            # Store loss separately since it's part of the main result tuple
            metrics["losses"].append(eval_res.loss)

        # Compute average metrics
        avg_metrics = {
            "accuracy": np.mean(metrics["accuracies"]) if metrics["accuracies"] else 0.0,
            "f1": np.mean(metrics["f1_scores"]) if metrics["f1_scores"] else 0.0,
            "precision": np.mean(metrics["precisions"]) if metrics["precisions"] else 0.0,
            "recall": np.mean(metrics["recalls"]) if metrics["recalls"] else 0.0,
            "loss": np.mean(metrics["losses"]) if metrics["losses"] else 0.0,
            "privacy_epsilon": np.mean(metrics["privacy_epsilons"]) if metrics["privacy_epsilons"] else 0.0
        }

        # Log aggregated evaluation metrics
        self.logger.info(f"Server Round {server_round} - Aggregated Evaluation Metrics:")
        self.logger.info(f"  Average Accuracy: {avg_metrics['accuracy']:.4f}")
        self.logger.info(f"  Average F1 Score: {avg_metrics['f1']:.4f}")
        self.logger.info(f"  Average Precision: {avg_metrics['precision']:.4f}")
        self.logger.info(f"  Average Recall: {avg_metrics['recall']:.4f}")
        self.logger.info(f"  Average Loss: {avg_metrics['loss']:.4f}")
        self.logger.info(f"  Average Privacy Budget (ε): {avg_metrics['privacy_epsilon']:.4f}")

        # Call parent method to aggregate evaluation results
        return super().aggregate_evaluate(server_round, results, failures)


def get_evaluate_fn(test_loader):
    """Return an evaluation function for server-side evaluation."""
    logger = logging.getLogger("Server")
    privacy_metrics_logger = PrivacyMetricsLogger(is_global=True)
    metrics_history = []

    def evaluate(server_round, parameters, config):
        model = create_model(make_dp_compatible=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Update model with server parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)

        # Evaluation mode
        model.eval()
        loss = 0.0
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        y_scores = []
        criterion = torch.nn.CrossEntropyLoss()

        logger.info(f"Server-side evaluation - Round {server_round} - Testing on {len(test_loader.dataset)} samples")

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss += criterion(outputs, target).item()

                # Get probabilities for ROC curve
                probs = torch.nn.functional.softmax(outputs, dim=1)
                y_scores.extend(probs[:, 1].cpu().numpy())  # Probability for class 1

                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                y_true.extend(target.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        accuracy = correct / total
        loss /= len(test_loader)

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
        logger.info(f"\nServer-side evaluation - Round {server_round}")
        logger.info(f"Loss: {loss:.4f}")
        logger.info(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        logger.info(f"True class distribution: {true_distribution}")
        logger.info(f"Prediction distribution: {pred_distribution}")
        logger.info("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        logger.info(f"\n{cm}")
        logger.info("\nClassification Report:")
        cr = classification_report(y_true, y_pred)
        logger.info(f"\n{cr}")

        # Create visualizations
        plot_confusion_matrix(
            y_true,
            y_pred,
            round_num=server_round,
            is_global=True
        )

        plot_roc_curve(
            y_true,
            y_scores,
            round_num=server_round,
            is_global=True
        )

        plot_class_distribution(
            y_true,
            y_pred,
            round_num=server_round,
            is_global=True
        )

        # Store metrics for history
        performance_metrics = {
            "loss": loss,
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }

        metrics_history.append(performance_metrics)
        privacy_metrics_logger.log_performance_metrics(performance_metrics, server_round)

        # Visualize metrics over rounds if we have more than one round
        if len(metrics_history) > 1:
            for metric_name in ['accuracy', 'f1', 'precision', 'recall']:
                plot_metrics_over_rounds(metrics_history, metric_name, is_global=True)

        # Save model after each round
        if not os.path.exists("./saved_models"):
            os.makedirs("./saved_models")
        torch.save(model.state_dict(), f"./saved_models/global_model_round_{server_round}.pth")
        logger.info(f"Global model saved after round {server_round}")

        # Create comparison with client models if we have access to client metrics
        if os.path.exists("./aggregated_metrics/rounds_metadata.csv"):
            try:
                rounds_df = pd.read_csv("./aggregated_metrics/rounds_metadata.csv")
                if len(rounds_df) > 1:
                    # Plot global vs. client average metrics
                    plt.figure(figsize=(12, 8))
                    rounds = rounds_df['round'].tolist()

                    # Plot accuracy
                    plt.subplot(2, 2, 1)
                    plt.plot(rounds, rounds_df['average_accuracy'].tolist(), 'b-', label='Client Avg')
                    plt.plot([r for r in range(1, len(metrics_history) + 1)],
                             [m['accuracy'] for m in metrics_history], 'r-', label='Global')
                    plt.title('Accuracy')
                    plt.xlabel('Round')
                    plt.legend()

                    # Plot F1 score
                    plt.subplot(2, 2, 2)
                    plt.plot(rounds, rounds_df['average_f1'].tolist(), 'b-', label='Client Avg')
                    plt.plot([r for r in range(1, len(metrics_history) + 1)],
                             [m['f1'] for m in metrics_history], 'r-', label='Global')
                    plt.title('F1 Score')
                    plt.xlabel('Round')
                    plt.legend()

                    # Plot loss
                    plt.subplot(2, 2, 3)
                    plt.plot(rounds, rounds_df['average_loss'].tolist(), 'b-', label='Client Avg')
                    plt.plot([r for r in range(1, len(metrics_history) + 1)],
                             [m['loss'] for m in metrics_history], 'r-', label='Global')
                    plt.title('Loss')
                    plt.xlabel('Round')
                    plt.legend()

                    # Plot privacy budget
                    plt.subplot(2, 2, 4)
                    plt.plot(rounds, rounds_df['average_epsilon'].tolist(), 'r-')
                    plt.title('Privacy Budget (ε)')
                    plt.xlabel('Round')

                    plt.tight_layout()
                    plt.savefig(f"./visualizations/global_vs_client_metrics_round_{server_round}.png")
                    plt.close()

                    logger.info(f"Global vs. client metrics comparison saved for round {server_round}")
            except Exception as e:
                logger.error(f"Error creating comparison visualizations: {e}")

        return loss, {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }

    return evaluate


def main():
    logger = logging.getLogger("Server")

    # Load data for server-side evaluation
    client_data = load_data(
        img_dir="D:/FYP_Data/combined_images",
        labels_path="D:/FYP_Data/cleaned_valid_image_labels.csv"
    )

    # Create a combined test dataset from all clients for better server evaluation
    all_test_datasets = []
    for _, test_loader, _, test_size in client_data:
        all_test_datasets.append(test_loader.dataset)

    combined_test_dataset = torch.utils.data.ConcatDataset(all_test_datasets)
    combined_test_loader = torch.utils.data.DataLoader(
        combined_test_dataset, batch_size=32, shuffle=False
    )

    logger.info(f"Server-side evaluation - Combined test set size: {len(combined_test_dataset)}")

    # Define strategy using the custom DP-aware FedAvg
    strategy = DPFedAvg(
        fraction_fit=1.0,  # Sample 100% of clients for training
        fraction_evaluate=1.0,  # Sample 100% of clients for evaluation
        min_fit_clients=3,  # All 3 clients should participate in training
        min_evaluate_clients=3,  # All 3 clients should participate in evaluation
        min_available_clients=3,
        evaluate_fn=get_evaluate_fn(combined_test_loader),  # Pass the combined evaluation function
        on_fit_config_fn=lambda server_round: {
            "server_round": server_round,  # Pass server round to clients
        }
    )

    # Start server
    logger.info("Starting Flower server with DP-enabled strategy...")
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=2),  # Increased to 5 rounds
        strategy=strategy
    )


if __name__ == "__main__":
    # Use the logging setup from main
    from main import setup_logging

    setup_logging()

    main()