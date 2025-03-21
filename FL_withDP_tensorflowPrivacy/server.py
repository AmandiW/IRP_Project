import flwr as fl
import torch
import numpy as np
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
from utils import (
    create_model,
    load_data,
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_metrics_over_rounds,
    PrivacyMetricsLogger,
    generate_summary_report,
    # Enhanced privacy functions
    compute_dp_sgd_privacy_budget,
    plot_per_client_privacy_consumption,
    plot_privacy_utility_tradeoff_curve,
    plot_membership_inference_risk,
    plot_privacy_leakage_reduction,
    plot_all_clients_per_round_accuracy,
    simulate_membership_inference_risk,
    calculate_theoretical_leak_probability,
    plot_iid_vs_non_iid_performance,
    plot_epsilon_composition,
    visualize_attack_risk_reduction
)


class DPFedAvg(fl.server.strategy.FedAvg):
    """
    Extended version of FedAvg strategy with differential privacy metrics tracking.
    """

    def __init__(self, *args, num_rounds=10, **kwargs):
        super().__init__(*args, **kwargs)

        # Set up tracking
        self.privacy_metrics_history = []
        self.performance_metrics_history = []
        self.logger = logging.getLogger("DPFedAvg")
        self.metrics_logger = PrivacyMetricsLogger(is_global=True)

        # Store number of rounds for determining final round
        self.num_rounds = num_rounds

        # Create directory for metrics
        os.makedirs("./aggregated_metrics", exist_ok=True)

        # Create initial privacy visualizations
        self._create_initial_privacy_visualizations()

    def _create_initial_privacy_visualizations(self):
        """Create initial privacy visualizations at server startup."""
        os.makedirs("./visualizations/privacy_analysis", exist_ok=True)

        # Create epsilon composition visualization for different noise values
        noise_values = [0.5, 1.0, 1.5, 2.0]
        for noise in noise_values:
            plot_epsilon_composition(
                num_rounds=self.num_rounds,
                noise_multiplier=noise,
                sample_rate=0.01,  # Example value
                delta=1e-5
            )

        # Create privacy-attack risk visualization
        noise_multipliers = np.linspace(0.5, 3.0, 10)
        visualize_attack_risk_reduction(noise_multipliers)

        self.logger.info("Created initial server-side privacy visualizations")

    def aggregate_fit(self, server_round, results, failures):
        """
        Aggregate model weights and privacy metrics from client training results.

        Args:
            server_round (int): Current round number
            results (list): Results from client training
            failures (list): Failures from client training

        Returns:
            Optional[Parameters]: Aggregated parameters
        """
        # Log round information
        self.logger.info(f"\nServer Round {server_round} - Aggregating training results")
        self.logger.info(f"Number of clients that succeeded: {len(results)}")
        self.logger.info(f"Number of clients that failed: {len(failures)}")

        if not results:
            self.logger.warning(f"No results to aggregate, skipping round {server_round}")
            return None

        # Extract privacy metrics from client results
        privacy_metrics = {
            "client_epsilons": [],
            "max_grad_norms": [],
            "noise_multipliers": [],
            "client_ids": []
        }

        # Extract performance metrics
        performance_metrics = {
            "accuracies": [],
            "f1_scores": [],
            "losses": [],
            "improvement_rates": []
        }

        for client_idx, (_, fit_res) in enumerate(results):
            metrics = fit_res.metrics
            if "privacy_epsilon" in metrics:
                privacy_metrics["client_epsilons"].append(metrics["privacy_epsilon"])
                privacy_metrics["client_ids"].append(client_idx)
            if "privacy_max_grad_norm" in metrics:
                privacy_metrics["max_grad_norms"].append(metrics["privacy_max_grad_norm"])
            if "privacy_noise_multiplier" in metrics:
                privacy_metrics["noise_multipliers"].append(metrics["privacy_noise_multiplier"])

            # Collect performance metrics
            if "accuracy" in metrics:
                performance_metrics["accuracies"].append(metrics["accuracy"])
            if "f1" in metrics:
                performance_metrics["f1_scores"].append(metrics["f1"])
            if "loss" in metrics:
                performance_metrics["losses"].append(metrics["loss"])
            if "accuracy_improvement" in metrics:
                performance_metrics["improvement_rates"].append(metrics["accuracy_improvement"])

        # Compute average privacy metrics
        avg_privacy_metrics = {
            "epsilon": np.mean(privacy_metrics["client_epsilons"]) if privacy_metrics["client_epsilons"] else 0.0,
            "max_grad_norm": np.mean(privacy_metrics["max_grad_norms"]) if privacy_metrics["max_grad_norms"] else 0.0,
            "noise_multiplier": np.mean(privacy_metrics["noise_multipliers"]) if privacy_metrics[
                "noise_multipliers"] else 0.0,
            "delta": 1e-5,  # Fixed delta value
        }

        # Log aggregated privacy metrics
        self.logger.info(f"Server Round {server_round} - Aggregated Privacy Metrics:")
        self.logger.info(f"  Average Epsilon: {avg_privacy_metrics['epsilon']:.4f}")
        self.logger.info(f"  Average Max Gradient Norm: {avg_privacy_metrics['max_grad_norm']:.4f}")
        self.logger.info(f"  Average Noise Multiplier: {avg_privacy_metrics['noise_multiplier']:.4f}")

        # Calculate theoretical membership inference risk
        if avg_privacy_metrics['epsilon'] > 0:
            inference_risk = simulate_membership_inference_risk(avg_privacy_metrics['epsilon'])
            self.logger.info(f"  Theoretical Membership Inference Risk: {inference_risk:.4f}")

        # Store privacy metrics history
        self.privacy_metrics_history.append(avg_privacy_metrics)
        self.metrics_logger.log_privacy_metrics(avg_privacy_metrics, server_round)

        # Compute average performance metrics
        avg_performance_metrics = {
            "accuracy": np.mean(performance_metrics["accuracies"]) if performance_metrics["accuracies"] else 0.0,
            "f1": np.mean(performance_metrics["f1_scores"]) if performance_metrics["f1_scores"] else 0.0,
            "loss": np.mean(performance_metrics["losses"]) if performance_metrics["losses"] else 0.0,
            "improvement_rate": np.mean(performance_metrics["improvement_rates"]) if performance_metrics[
                "improvement_rates"] else 0.0
        }

        # Log aggregated performance metrics
        self.logger.info(f"Server Round {server_round} - Aggregated Performance Metrics:")
        self.logger.info(f"  Average Accuracy: {avg_performance_metrics['accuracy']:.4f}")
        self.logger.info(f"  Average F1 Score: {avg_performance_metrics['f1']:.4f}")
        self.logger.info(f"  Average Loss: {avg_performance_metrics['loss']:.4f}")
        self.logger.info(f"  Average Improvement Rate: {avg_performance_metrics['improvement_rate']:.4f}")

        # Store performance metrics history
        self.performance_metrics_history.append(avg_performance_metrics)
        self.metrics_logger.log_performance_metrics(avg_performance_metrics, server_round)

        # Create and save metadata for this round
        round_metadata = {
            "round": server_round,
            "num_clients": len(results),
            "average_epsilon": avg_privacy_metrics["epsilon"],
            "average_accuracy": avg_performance_metrics["accuracy"],
            "average_f1": avg_performance_metrics["f1"],
            "average_loss": avg_performance_metrics["loss"],
            "average_improvement_rate": avg_performance_metrics["improvement_rate"]
        }

        # Append to rounds metadata file
        rounds_df = pd.DataFrame([round_metadata])
        if os.path.exists("./aggregated_metrics/rounds_metadata.csv"):
            existing_df = pd.read_csv("./aggregated_metrics/rounds_metadata.csv")
            rounds_df = pd.concat([existing_df, rounds_df], ignore_index=True)

        rounds_df.to_csv("./aggregated_metrics/rounds_metadata.csv", index=False)

        # Create per-client privacy consumption visualization if we have data
        if len(privacy_metrics["client_epsilons"]) > 0 and len(privacy_metrics["client_ids"]) > 0:
            plot_per_client_privacy_consumption(
                privacy_metrics["client_ids"],
                privacy_metrics["client_epsilons"],
                server_round
            )

        # If this is the final round, create additional privacy-related visualizations
        if server_round == self.num_rounds:
            self.logger.info("Final round completed. Generating privacy analysis visualizations...")

            if os.path.exists("./aggregated_metrics/rounds_metadata.csv"):
                try:
                    rounds_df = pd.read_csv("./aggregated_metrics/rounds_metadata.csv")

                    # Generate privacy-utility tradeoff visualization
                    epsilons = rounds_df['average_epsilon'].tolist()
                    accuracies = rounds_df['average_accuracy'].tolist()
                    f1_scores = rounds_df['average_f1'].tolist()

                    plot_privacy_utility_tradeoff_curve(epsilons, accuracies, f1_scores)

                    # Generate membership inference risk visualization
                    epsilon_range = np.linspace(0.1, max(epsilons) * 1.5, 20)
                    inference_risks = [simulate_membership_inference_risk(eps) for eps in epsilon_range]
                    plot_membership_inference_risk(epsilon_range, inference_risks)

                    # Generate privacy leakage reduction visualization
                    leak_probs = [calculate_theoretical_leak_probability(eps) for eps in epsilon_range]
                    plot_privacy_leakage_reduction(epsilon_range, leak_probs)

                    self.logger.info("Privacy analysis visualizations created successfully")
                except Exception as e:
                    self.logger.error(f"Error creating privacy analysis visualizations: {e}")

        # Call parent method to aggregate weights
        return super().aggregate_fit(server_round, results, failures)

    def aggregate_evaluate(self, server_round, results, failures):
        """
        Aggregate evaluation results from clients.

        Args:
            server_round (int): Current round number
            results (list): Results from client evaluation
            failures (list): Failures from client evaluation

        Returns:
            Optional[Tuple[float, Dict[str, Scalar]]]: Loss and metrics
        """
        # Log round information
        self.logger.info(f"\nServer Round {server_round} - Aggregating evaluation results")
        self.logger.info(f"Number of clients that succeeded: {len(results)}")
        self.logger.info(f"Number of clients that failed: {len(failures)}")

        if not results:
            self.logger.warning(f"No results to aggregate, skipping round {server_round}")
            return None

        # Extract and average evaluation metrics
        metrics = {
            "accuracies": [],
            "f1_scores": [],
            "precisions": [],
            "recalls": [],
            "losses": [],
            "privacy_epsilons": [],
            "client_ids": []
        }

        # Collect metrics from all clients
        for client_idx, (_, eval_res) in enumerate(results):
            client_metrics = eval_res.metrics
            if "accuracy" in client_metrics:
                metrics["accuracies"].append(client_metrics["accuracy"])
                metrics["client_ids"].append(client_idx)
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
        self.logger.info(f"  Average Privacy Budget (Epsilon): {avg_metrics['privacy_epsilon']:.4f}")

        # Update the rounds metadata with global model accuracy
        if os.path.exists("./aggregated_metrics/rounds_metadata.csv"):
            try:
                rounds_df = pd.read_csv("./aggregated_metrics/rounds_metadata.csv")
                # Find the row for this round
                mask = rounds_df['round'] == server_round
                if any(mask):
                    # Update global accuracy
                    rounds_df.loc[mask, 'global_accuracy'] = avg_metrics['accuracy']
                    rounds_df.to_csv("./aggregated_metrics/rounds_metadata.csv", index=False)
            except Exception as e:
                self.logger.error(f"Error updating rounds metadata with global accuracy: {e}")

        # Call parent method to aggregate evaluation results
        return super().aggregate_evaluate(server_round, results, failures)


def get_evaluate_fn(test_loader, num_rounds):
    """
    Return a function for server-side evaluation.

    Args:
        test_loader (DataLoader): Test data loader
        num_rounds (int): Total number of rounds

    Returns:
        function: Evaluation function
    """
    logger = logging.getLogger("Server")
    metrics_logger = PrivacyMetricsLogger(is_global=True)
    metrics_history = []
    global_accuracies = []

    def evaluate(server_round, parameters, config):
        """
        Evaluate the global model on test data.

        Args:
            server_round (int): Current round number
            parameters (List[np.ndarray]): Model parameters
            config (Dict): Configuration

        Returns:
            Tuple[float, Dict[str, float]]: Loss and metrics
        """
        # Create and configure model
        model = create_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Load parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)

        # Evaluate model
        criterion = torch.nn.CrossEntropyLoss()
        loss, metrics = evaluate_model(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device
        )

        # Store global model accuracy for later visualization
        global_accuracies.append(metrics['accuracy'])

        # Log evaluation results
        logger.info(f"\nServer-side evaluation - Round {server_round}")
        logger.info(f"  Loss: {loss:.4f}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1 Score: {metrics['f1']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")

        # Create visualizations
        y_true = metrics.get("y_true", [])
        y_pred = metrics.get("y_pred", [])
        y_scores = metrics.get("y_scores", [])

        if len(y_true) > 0 and len(y_pred) > 0:
            plot_confusion_matrix(
                y_true=y_true,
                y_pred=y_pred,
                round_num=server_round,
                is_global=True
            )

            plot_roc_curve(
                y_true=y_true,
                y_scores=y_scores,
                round_num=server_round,
                is_global=True
            )

        # Store metrics for history
        performance_metrics = {
            "loss": loss,
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"]
        }

        metrics_history.append(performance_metrics)
        metrics_logger.log_performance_metrics(performance_metrics, server_round)

        # Visualize metrics over rounds if we have history
        if len(metrics_history) > 1:
            for metric_name in ['accuracy', 'f1', 'precision', 'recall']:
                plot_metrics_over_rounds(metrics_history, metric_name, is_global=True)

        # Save model after each round
        os.makedirs("./saved_models", exist_ok=True)
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
                    os.makedirs("./visualizations", exist_ok=True)
                    plt.savefig(f"./visualizations/global_vs_client_metrics_round_{server_round}.png")
                    plt.close()

                    logger.info(f"Global vs. client metrics comparison saved for round {server_round}")
            except Exception as e:
                logger.error(f"Error creating comparison visualizations: {e}")

        # If this is the final round, create a visualization showing all clients' accuracy progression
        if server_round == num_rounds:
            try:
                # Check if client performance data is available
                if os.path.exists("./aggregated_metrics/client_performance.csv"):
                    client_df = pd.read_csv("./aggregated_metrics/client_performance.csv")

                    # Get unique client IDs
                    client_ids = sorted(client_df['client_id'].unique())

                    # Collect accuracy data for each client
                    all_client_accuracies = []
                    for client_id in client_ids:
                        # Get this client's data, sorted by round
                        client_data = client_df[client_df['client_id'] == client_id].sort_values('round')
                        client_accuracies = client_data['accuracy'].tolist()

                        # Ensure we have data for all rounds (fill missing with NaN)
                        if len(client_accuracies) < num_rounds:
                            client_accuracies += [np.nan] * (num_rounds - len(client_accuracies))

                        all_client_accuracies.append(client_accuracies)

                    # Create the visualization
                    plot_all_clients_per_round_accuracy(client_ids, all_client_accuracies, global_accuracies)
                    logger.info("Created client and global model accuracy progression visualization")
            except Exception as e:
                logger.error(f"Error creating client accuracy progression visualization: {e}")

        return loss, {
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"]
        }

    return evaluate


def main(num_rounds=2, min_clients=2):
    """
    Start the Flower server for federated learning.

    Args:
        num_rounds (int): Number of federated learning rounds
        min_clients (int): Minimum number of available clients
    """
    logger = logging.getLogger("Server")
    logger.info("Starting Flower server for federated learning")
    logger.info(f"Server configured for {num_rounds} rounds with minimum {min_clients} clients")

    try:
        # Get configuration for data loading from environment variables
        distribution_type = os.environ.get("FL_DISTRIBUTION", "iid")
        alpha = float(os.environ.get("FL_ALPHA", "0.5"))
        num_clients = int(os.environ.get("FL_NUM_CLIENTS", "2"))

        logger.info(f"Data distribution type: {distribution_type}")
        if distribution_type == "non_iid":
            logger.info(f"Alpha parameter for Dirichlet distribution: {alpha}")
        logger.info(f"Number of clients: {num_clients}")

        # Load data for server-side evaluation
        client_data = load_data(
            img_dir="D:/FYP_Data/combined_images",
            labels_path="D:/FYP_Data/cleaned_valid_image_labels.csv",
            num_clients=num_clients,
            distribution=distribution_type,
            alpha=alpha
        )

        # Create a combined test dataset from all clients for better evaluation
        logger.info("Creating combined test dataset for server-side evaluation")
        all_test_datasets = []
        for _, test_loader, _, test_size in client_data:
            all_test_datasets.append(test_loader.dataset)

        combined_test_dataset = torch.utils.data.ConcatDataset(all_test_datasets)
        combined_test_loader = torch.utils.data.DataLoader(
            combined_test_dataset,
            batch_size=32,
            shuffle=False
        )

        logger.info(f"Server-side evaluation - Combined test set size: {len(combined_test_dataset)}")

        # Adjust min_clients to not exceed num_clients
        min_fit_clients = min(min_clients, num_clients)
        min_evaluate_clients = min(min_clients, num_clients)
        min_available_clients = min(min_clients, num_clients)

        # Define strategy using the custom DP-aware FedAvg
        strategy = DPFedAvg(
            fraction_fit=1.0,  # Sample 100% of clients for training
            fraction_evaluate=1.0,  # Sample 100% of clients for evaluation
            min_fit_clients=min_fit_clients,  # Number of clients required for training
            min_evaluate_clients=min_evaluate_clients,  # Number of clients required for evaluation
            min_available_clients=min_available_clients,  # Minimum number of available clients
            evaluate_fn=get_evaluate_fn(combined_test_loader, num_rounds),  # Server-side evaluation function
            on_fit_config_fn=lambda server_round: {
                "server_round": server_round,  # Pass round number to clients
            },
            num_rounds=num_rounds  # Pass number of rounds for final round detection
        )

        # Start server
        logger.info("Starting Flower server...")
        fl.server.start_server(
            server_address="localhost:8080",
            config=fl.server.ServerConfig(num_rounds=num_rounds),  # Number of federated learning rounds
            strategy=strategy
        )

        # Generate final summary report
        if os.path.exists("./aggregated_metrics/rounds_metadata.csv"):
            rounds_df = pd.read_csv("./aggregated_metrics/rounds_metadata.csv")
            report_path = generate_summary_report(rounds_df)
            logger.info(f"Final summary report generated: {report_path}")

    except Exception as e:
        logger.error(f"Error in server: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Set up logging
    from main import setup_logging

    setup_logging()

    # Get number of rounds and min clients from command line argument if provided
    import sys

    num_rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    min_clients = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    # Start server
    main(num_rounds, min_clients)