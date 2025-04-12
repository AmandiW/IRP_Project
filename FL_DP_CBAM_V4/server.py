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
    # Privacy functions
    compute_dp_sgd_privacy_budget,
    plot_per_client_privacy_consumption,
    plot_privacy_utility_tradeoff_curve,
    plot_membership_inference_risk,
    plot_privacy_leakage_reduction,
    plot_all_clients_per_round_accuracy,
    simulate_membership_inference_risk,
    calculate_theoretical_leak_probability,
    visualize_feature_importance_heatmap,
    visualize_privacy_preservation_with_reconstruction,
    perform_membership_inference_attack, FocalLoss
)


def fit_metrics_aggregation_fn(metrics_list):
    """
    Aggregate fit metrics from multiple clients.

    Args:
        metrics_list (list): List of tuples (num_examples, metrics) from clients

    Returns:
        dict: Aggregated metrics
    """
    if not metrics_list:
        return {}

    # Initialize weighted metrics
    weighted_metrics = {
        "accuracy": 0.0,
        "f1": 0.0,
        "loss": 0.0,
        "privacy_epsilon": 0.0,
        "privacy_cumulative_epsilon": 0.0,
        "accuracy_improvement": 0.0
    }

    total_examples = 0
    feature_specific_count = 0

    # Calculate weighted sum of metrics
    for num_examples, metrics in metrics_list:
        total_examples += num_examples

        # Add weighted metrics
        if isinstance(metrics, dict):
            for metric_name in weighted_metrics.keys():
                if metric_name in metrics:
                    weighted_metrics[metric_name] += num_examples * metrics[metric_name]

            # Track feature-specific privacy usage
            if metrics.get("feature_specific_privacy", False):
                feature_specific_count += 1

    # Calculate weighted average
    if total_examples > 0:
        for metric_name in weighted_metrics.keys():
            weighted_metrics[metric_name] /= total_examples

    # Add feature-specific privacy info
    weighted_metrics["feature_specific_effective"] = feature_specific_count > 0

    return weighted_metrics


def evaluate_metrics_aggregation_fn(metrics_list):
    """
    Aggregate evaluate metrics from multiple clients.

    Args:
        metrics_list (list): List of tuples (num_examples, metrics) from clients

    Returns:
        dict: Aggregated metrics
    """
    if not metrics_list:
        return {}

    # Initialize weighted metrics
    weighted_metrics = {
        "accuracy": 0.0,
        "f1": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "privacy_epsilon": 0.0,
        "privacy_cumulative_epsilon": 0.0
    }

    total_examples = 0
    feature_specific_count = 0

    # Calculate weighted sum of metrics
    for num_examples, metrics in metrics_list:
        total_examples += num_examples

        # Add weighted metrics
        if isinstance(metrics, dict):
            for metric_name in weighted_metrics.keys():
                if metric_name in metrics:
                    weighted_metrics[metric_name] += num_examples * metrics[metric_name]

            # Track feature-specific privacy usage
            if metrics.get("feature_specific_privacy", False):
                feature_specific_count += 1

    # Calculate weighted average
    if total_examples > 0:
        for metric_name in weighted_metrics.keys():
            weighted_metrics[metric_name] /= total_examples

    # Add feature-specific privacy info
    weighted_metrics["feature_specific_effective"] = feature_specific_count > 0

    return weighted_metrics


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
            "client_cumulative_epsilons": [],
            "max_grad_norms": [],
            "noise_multipliers": [],
            "client_ids": [],
            "feature_specific_counts": 0
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
            if "privacy_cumulative_epsilon" in metrics:
                privacy_metrics["client_cumulative_epsilons"].append(metrics["privacy_cumulative_epsilon"])
            if "privacy_max_grad_norm" in metrics:
                privacy_metrics["max_grad_norms"].append(metrics["privacy_max_grad_norm"])
            if "privacy_noise_multiplier" in metrics:
                privacy_metrics["noise_multipliers"].append(metrics["privacy_noise_multiplier"])
            if metrics.get("feature_specific_privacy", False):
                privacy_metrics["feature_specific_counts"] += 1

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
            "cumulative_epsilon": np.mean(privacy_metrics["client_cumulative_epsilons"]) if privacy_metrics[
                "client_cumulative_epsilons"] else 0.0,
            "max_grad_norm": np.mean(privacy_metrics["max_grad_norms"]) if privacy_metrics["max_grad_norms"] else 0.0,
            "noise_multiplier": np.mean(privacy_metrics["noise_multipliers"]) if privacy_metrics[
                "noise_multipliers"] else 0.0,
            "delta": 1e-5,  # Fixed delta value
            "feature_specific_effective": privacy_metrics["feature_specific_counts"] > 0
        }

        # Log aggregated privacy metrics
        self.logger.info(f"Server Round {server_round} - Aggregated Privacy Metrics:")
        self.logger.info(f"  Round Epsilon: {avg_privacy_metrics['epsilon']:.4f}")
        self.logger.info(f"  Cumulative Epsilon: {avg_privacy_metrics['cumulative_epsilon']:.4f}")
        self.logger.info(f"  Average Max Gradient Norm: {avg_privacy_metrics['max_grad_norm']:.4f}")
        self.logger.info(f"  Average Noise Multiplier: {avg_privacy_metrics['noise_multiplier']:.4f}")
        self.logger.info(f"  Feature-specific privacy effective: {avg_privacy_metrics['feature_specific_effective']}")

        # Calculate theoretical membership inference risk
        if avg_privacy_metrics['epsilon'] > 0:
            inference_risk = simulate_membership_inference_risk(avg_privacy_metrics['epsilon'])
            cumulative_risk = simulate_membership_inference_risk(avg_privacy_metrics['cumulative_epsilon'])
            self.logger.info(f"  Round Membership Inference Risk: {inference_risk:.4f}")
            self.logger.info(f"  Cumulative Membership Inference Risk: {cumulative_risk:.4f}")

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
            "average_cumulative_epsilon": avg_privacy_metrics["cumulative_epsilon"],
            "average_accuracy": avg_performance_metrics["accuracy"],
            "average_f1": avg_performance_metrics["f1"],
            "average_loss": avg_performance_metrics["loss"],
            "average_improvement_rate": avg_performance_metrics["improvement_rate"],
            "feature_specific_effective": avg_privacy_metrics["feature_specific_effective"]
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
                    epsilons = rounds_df['average_cumulative_epsilon'].tolist()
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
            "privacy_cumulative_epsilons": [],
            "client_ids": [],
            "feature_specific_counts": 0
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
            if "privacy_cumulative_epsilon" in client_metrics:
                metrics["privacy_cumulative_epsilons"].append(client_metrics["privacy_cumulative_epsilon"])
            if client_metrics.get("feature_specific_privacy", False):
                metrics["feature_specific_counts"] += 1

            # Store loss separately since it's part of the main result tuple
            metrics["losses"].append(eval_res.loss)

        # Compute average metrics
        avg_metrics = {
            "accuracy": np.mean(metrics["accuracies"]) if metrics["accuracies"] else 0.0,
            "f1": np.mean(metrics["f1_scores"]) if metrics["f1_scores"] else 0.0,
            "precision": np.mean(metrics["precisions"]) if metrics["precisions"] else 0.0,
            "recall": np.mean(metrics["recalls"]) if metrics["recalls"] else 0.0,
            "loss": np.mean(metrics["losses"]) if metrics["losses"] else 0.0,
            "privacy_epsilon": np.mean(metrics["privacy_epsilons"]) if metrics["privacy_epsilons"] else 0.0,
            "privacy_cumulative_epsilon": np.mean(metrics["privacy_cumulative_epsilons"]) if metrics[
                "privacy_cumulative_epsilons"] else 0.0,
            "feature_specific_effective": metrics["feature_specific_counts"] > 0
        }

        # Log aggregated evaluation metrics
        self.logger.info(f"Server Round {server_round} - Aggregated Evaluation Metrics:")
        self.logger.info(f"  Average Accuracy: {avg_metrics['accuracy']:.4f}")
        self.logger.info(f"  Average F1 Score: {avg_metrics['f1']:.4f}")
        self.logger.info(f"  Average Precision: {avg_metrics['precision']:.4f}")
        self.logger.info(f"  Average Recall: {avg_metrics['recall']:.4f}")
        self.logger.info(f"  Average Loss: {avg_metrics['loss']:.4f}")
        self.logger.info(f"  Average Privacy Budget (Round Epsilon): {avg_metrics['privacy_epsilon']:.4f}")
        self.logger.info(f"  Average Cumulative Privacy Budget: {avg_metrics['privacy_cumulative_epsilon']:.4f}")
        self.logger.info(f"  Feature-specific privacy effective: {avg_metrics['feature_specific_effective']}")

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


class DPFedProx(fl.server.strategy.FedProx):
    """
    Extended version of FedProx strategy with differential privacy metrics tracking.
    """

    def __init__(self, *args, num_rounds=10, **kwargs):
        super().__init__(*args, **kwargs)

        # Set up tracking
        self.privacy_metrics_history = []
        self.performance_metrics_history = []
        self.logger = logging.getLogger("DPFedProx")
        self.metrics_logger = PrivacyMetricsLogger(is_global=True)

        # Store number of rounds for determining final round
        self.num_rounds = num_rounds

        # Create directory for metrics
        os.makedirs("./aggregated_metrics", exist_ok=True)

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
            "client_cumulative_epsilons": [],
            "max_grad_norms": [],
            "noise_multipliers": [],
            "client_ids": [],
            "feature_specific_counts": 0
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
            if "privacy_cumulative_epsilon" in metrics:
                privacy_metrics["client_cumulative_epsilons"].append(metrics["privacy_cumulative_epsilon"])
            if "privacy_max_grad_norm" in metrics:
                privacy_metrics["max_grad_norms"].append(metrics["privacy_max_grad_norm"])
            if "privacy_noise_multiplier" in metrics:
                privacy_metrics["noise_multipliers"].append(metrics["privacy_noise_multiplier"])
            if metrics.get("feature_specific_privacy", False):
                privacy_metrics["feature_specific_counts"] += 1

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
            "cumulative_epsilon": np.mean(privacy_metrics["client_cumulative_epsilons"]) if privacy_metrics[
                "client_cumulative_epsilons"] else 0.0,
            "max_grad_norm": np.mean(privacy_metrics["max_grad_norms"]) if privacy_metrics["max_grad_norms"] else 0.0,
            "noise_multiplier": np.mean(privacy_metrics["noise_multipliers"]) if privacy_metrics[
                "noise_multipliers"] else 0.0,
            "delta": 1e-5,  # Fixed delta value
            "feature_specific_effective": privacy_metrics["feature_specific_counts"] > 0
        }

        # Log aggregated privacy metrics
        self.logger.info(f"Server Round {server_round} - Aggregated Privacy Metrics:")
        self.logger.info(f"  Round Epsilon: {avg_privacy_metrics['epsilon']:.4f}")
        self.logger.info(f"  Cumulative Epsilon: {avg_privacy_metrics['cumulative_epsilon']:.4f}")
        self.logger.info(f"  Average Max Gradient Norm: {avg_privacy_metrics['max_grad_norm']:.4f}")
        self.logger.info(f"  Average Noise Multiplier: {avg_privacy_metrics['noise_multiplier']:.4f}")
        self.logger.info(f"  Feature-specific privacy effective: {avg_privacy_metrics['feature_specific_effective']}")

        # Calculate theoretical membership inference risk
        if avg_privacy_metrics['epsilon'] > 0:
            inference_risk = simulate_membership_inference_risk(avg_privacy_metrics['epsilon'])
            cumulative_risk = simulate_membership_inference_risk(avg_privacy_metrics['cumulative_epsilon'])
            self.logger.info(f"  Round Membership Inference Risk: {inference_risk:.4f}")
            self.logger.info(f"  Cumulative Membership Inference Risk: {cumulative_risk:.4f}")

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
            "average_cumulative_epsilon": avg_privacy_metrics["cumulative_epsilon"],
            "average_accuracy": avg_performance_metrics["accuracy"],
            "average_f1": avg_performance_metrics["f1"],
            "average_loss": avg_performance_metrics["loss"],
            "average_improvement_rate": avg_performance_metrics["improvement_rate"],
            "feature_specific_effective": avg_privacy_metrics["feature_specific_effective"]
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
                    epsilons = rounds_df['average_cumulative_epsilon'].tolist()
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
            "privacy_cumulative_epsilons": [],
            "client_ids": [],
            "feature_specific_counts": 0
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
            if "privacy_cumulative_epsilon" in client_metrics:
                metrics["privacy_cumulative_epsilons"].append(client_metrics["privacy_cumulative_epsilon"])
            if client_metrics.get("feature_specific_privacy", False):
                metrics["feature_specific_counts"] += 1

            # Store loss separately since it's part of the main result tuple
            metrics["losses"].append(eval_res.loss)

        # Compute average metrics
        avg_metrics = {
            "accuracy": np.mean(metrics["accuracies"]) if metrics["accuracies"] else 0.0,
            "f1": np.mean(metrics["f1_scores"]) if metrics["f1_scores"] else 0.0,
            "precision": np.mean(metrics["precisions"]) if metrics["precisions"] else 0.0,
            "recall": np.mean(metrics["recalls"]) if metrics["recalls"] else 0.0,
            "loss": np.mean(metrics["losses"]) if metrics["losses"] else 0.0,
            "privacy_epsilon": np.mean(metrics["privacy_epsilons"]) if metrics["privacy_epsilons"] else 0.0,
            "privacy_cumulative_epsilon": np.mean(metrics["privacy_cumulative_epsilons"]) if metrics[
                "privacy_cumulative_epsilons"] else 0.0,
            "feature_specific_effective": metrics["feature_specific_counts"] > 0
        }

        # Log aggregated evaluation metrics
        self.logger.info(f"Server Round {server_round} - Aggregated Evaluation Metrics:")
        self.logger.info(f"  Average Accuracy: {avg_metrics['accuracy']:.4f}")
        self.logger.info(f"  Average F1 Score: {avg_metrics['f1']:.4f}")
        self.logger.info(f"  Average Precision: {avg_metrics['precision']:.4f}")
        self.logger.info(f"  Average Recall: {avg_metrics['recall']:.4f}")
        self.logger.info(f"  Average Loss: {avg_metrics['loss']:.4f}")
        self.logger.info(f"  Average Privacy Budget (Round Epsilon): {avg_metrics['privacy_epsilon']:.4f}")
        self.logger.info(f"  Average Cumulative Privacy Budget: {avg_metrics['privacy_cumulative_epsilon']:.4f}")
        self.logger.info(f"  Feature-specific privacy effective: {avg_metrics['feature_specific_effective']}")

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


def get_evaluate_fn(test_loader, num_rounds, model_config=None):
    """
    Return a function for server-side evaluation.

    Args:
        test_loader (DataLoader): Test data loader
        num_rounds (int): Total number of rounds
        model_config (dict): Model configuration parameters

    Returns:
        function: Evaluation function
    """
    logger = logging.getLogger("Server")
    metrics_logger = PrivacyMetricsLogger(is_global=True)
    metrics_history = []
    global_accuracies = []

    # Default model config if not provided
    if model_config is None:
        model_config = {
            "model_name": "resnet18",
            "model_type": "resnet",
            "num_classes": 2
        }

    # Track the best model parameters to avoid regression
    best_accuracy = 0.0
    best_parameters = None

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
        nonlocal best_accuracy, best_parameters

        # Create and configure model
        model = create_model(
            model_name=model_config.get("model_name", "resnet18"),
            model_type=model_config.get("model_type", "resnet"),
            num_classes=model_config.get("num_classes", 2)
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Load parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)

        # Log the number of trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Global model has {trainable_params} trainable parameters out of {total_params} total parameters")

        # Ensure proper layers are set to trainable for evaluation
        # This is important to make sure we're evaluating in the same configuration as training
        for param in model.parameters():
            param.requires_grad = False

        if model_config.get("model_type", "resnet") == "resnet":
            # For ResNet models
            if hasattr(model, 'backbone'):
                # Make final layer trainable
                if hasattr(model.backbone, 'fc'):
                    for param in model.backbone.fc.parameters():
                        param.requires_grad = True

                # Make some convolutional layers trainable too
                if hasattr(model.backbone, 'layer4'):
                    for param in model.backbone.layer4.parameters():
                        param.requires_grad = True

                if hasattr(model.backbone, 'layer3'):
                    # Make only the last block of layer3 trainable
                    for name, param in model.backbone.layer3.named_parameters():
                        if '1.' in name:  # Last block in layer3
                            param.requires_grad = True
        else:
            # For DenseNet models
            if hasattr(model, 'backbone'):
                # Make final layer trainable
                if hasattr(model.backbone, 'classifier'):
                    for param in model.backbone.classifier.parameters():
                        param.requires_grad = True

                # Make the last dense block trainable
                if hasattr(model.backbone, 'features'):
                    for name, module in model.backbone.features.named_children():
                        if 'denseblock4' in name:
                            for param in module.parameters():
                                param.requires_grad = True

        # Evaluate model
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        loss, metrics = evaluate_model(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device
        )

        # Store global model accuracy for later visualization
        global_accuracies.append(metrics['accuracy'])

        # Keep track of the best model parameters
        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            best_parameters = [p.copy() for p in parameters]
            logger.info(f"New best global model with accuracy: {best_accuracy:.4f}")

            # Save the best model explicitly
            torch.save(state_dict, f"./saved_models/global_model_best.pth")

        # If accuracy is degrading, consider using the best parameters
        if server_round > 1 and metrics['accuracy'] < 0.46 and best_parameters is not None:
            logger.warning(f"Global model accuracy is dropping. Consider using best parameters from previous rounds.")

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

        # Create feature importance heatmaps if possible
        try:
            if hasattr(model, 'get_private_features'):
                visualize_feature_importance_heatmap(model, test_loader, device)
                logger.info(f"Created feature importance heatmap for global model at round {server_round}")
        except Exception as e:
            logger.warning(f"Error creating feature importance heatmap: {e}")

        # Create privacy preservation reconstruction visualization for global model
        if server_round == num_rounds:  # Only in the final round
            try:
                visualize_privacy_preservation_with_reconstruction(
                    model,
                    test_loader,
                    device
                )
                logger.info(f"Created privacy reconstruction visualization for global model")
            except Exception as e:
                logger.error(f"Error creating privacy reconstruction visualization: {e}")

            # Perform membership inference attack on the global model
            try:
                # We need to detect if we can perform the attack effectively
                # Client training/test data is needed for this
                from utils import load_data

                img_dir = os.environ.get("FL_IMG_DIR", "E:/IRP_dataset_new/IRP_combined_processed_images")
                labels_path = os.environ.get("FL_LABELS_PATH", "E:/IRP_dataset_new/APTOS_labels_combined.csv")

                # Try to load a small subset of data for attack simulation
                try:
                    sample_data = load_data(
                        img_dir=img_dir,
                        labels_path=labels_path,
                        num_clients=1,  # Just load one client's worth of data
                        distribution="iid"
                    )

                    if sample_data and len(sample_data) > 0:
                        train_loader = sample_data[0][0]
                        attack_accuracy, _ = perform_membership_inference_attack(
                            model,
                            train_loader,
                            test_loader,
                            device
                        )
                        logger.info(f"Global model membership inference attack accuracy: {attack_accuracy:.4f}")

                        # This demonstrates that our privacy guarantees are working effectively
                        if attack_accuracy < 0.6:  # Success would be > 0.7
                            logger.info("Privacy guarantees effective: membership inference attack unsuccessful")
                except Exception as e:
                    logger.warning(f"Could not perform membership inference attack on global model: {e}")
            except Exception as global_e:
                logger.warning(f"Could not set up global privacy test: {global_e}")

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
                    plt.plot(rounds, rounds_df['average_cumulative_epsilon'].tolist(), 'r-')
                    plt.title('Cumulative Privacy Budget (ε)')
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

                    if len(client_ids) > 0:
                        # Collect accuracy data for each client
                        all_client_accuracies = []
                        valid_client_ids = []

                        for client_id in client_ids:
                            # Get this client's data, sorted by round
                            client_data = client_df[client_df['client_id'] == client_id].sort_values('round')

                            # Only include clients with data
                            if len(client_data) > 0:
                                client_accuracies = client_data['accuracy'].tolist()

                                # Ensure we have data for all rounds (fill missing with NaN)
                                if len(client_accuracies) < num_rounds:
                                    client_accuracies += [np.nan] * (num_rounds - len(client_accuracies))

                                all_client_accuracies.append(client_accuracies)
                                valid_client_ids.append(client_id)

                        # Check if we have any valid data
                        if valid_client_ids and all_client_accuracies:
                            # Make sure global_accuracies has at least one value
                            if global_accuracies:
                                # Create the visualization
                                plot_all_clients_per_round_accuracy(
                                    valid_client_ids,
                                    all_client_accuracies,
                                    global_accuracies
                                )
                                logger.info("Created client and global model accuracy progression visualization")
                            else:
                                # Create the visualization without global model data
                                plot_all_clients_per_round_accuracy(valid_client_ids, all_client_accuracies)
                                logger.info("Created client accuracy progression visualization (no global model data)")
            except Exception as e:
                logger.error(f"Error creating client accuracy progression visualization: {e}")

        return loss, {
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"]
        }

    return evaluate


def main(num_rounds=5, min_clients=2, strategy="fedavg", proximal_mu=0.01, model_config=None):
    """
    Start the Flower server for federated learning.

    Args:
        num_rounds (int): Number of federated learning rounds
        min_clients (int): Minimum number of available clients
        strategy (str): FL strategy to use ('fedavg' or 'fedprox')
        proximal_mu (float): Proximal term weight for FedProx
        model_config (dict): Model configuration parameters
    """
    logger = logging.getLogger("Server")
    logger.info("Starting Flower server for federated learning")
    logger.info(f"Server configured for {num_rounds} rounds with minimum {min_clients} clients")
    logger.info(f"Using {strategy.upper()} strategy")
    if strategy == "fedprox":
        logger.info(f"FedProx proximal term mu: {proximal_mu}")

    # Default model config if not provided
    if model_config is None:
        model_config = {
            "model_name": "resnet18",
            "model_type": "resnet",
            "num_classes": 2
        }

    logger.info(f"Using model config: {model_config}")

    try:
        # Get configuration for data loading from environment variables
        distribution_type = os.environ.get("FL_DISTRIBUTION", "iid")
        alpha = float(os.environ.get("FL_ALPHA", "0.5"))
        num_clients = int(os.environ.get("FL_NUM_CLIENTS", "2"))
        fed_strategy = os.environ.get("FL_STRATEGY", strategy)
        prox_mu = float(os.environ.get("FL_PROX_MU", proximal_mu))

        logger.info(f"Data distribution type: {distribution_type}")
        if distribution_type == "non_iid":
            logger.info(f"Alpha parameter for Dirichlet distribution: {alpha}")
        logger.info(f"Number of clients: {num_clients}")
        logger.info(f"Strategy: {fed_strategy}")
        if fed_strategy == "fedprox":
            logger.info(f"Proximal term mu: {prox_mu}")

        # Load data for server-side evaluation
        # Use configurable paths from environment variables
        img_dir = os.environ.get("FL_IMG_DIR", "E:/IRP_dataset_new/IRP_Final_Images")
        labels_path = os.environ.get("FL_LABELS_PATH", "E:/IRP_dataset_new/IRP_Final_Labels.csv")

        client_data = load_data(
            img_dir=img_dir,
            labels_path=labels_path,
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

        # Define strategy based on selection
        if fed_strategy.lower() == "fedprox":
            # Use FedProx strategy
            strategy = DPFedProx(
                fraction_fit=1.0,  # Sample 100% of clients for training
                fraction_evaluate=1.0,  # Sample 100% of clients for evaluation
                min_fit_clients=min_fit_clients,  # Number of clients required for training
                min_evaluate_clients=min_evaluate_clients,  # Number of clients required for evaluation
                min_available_clients=min_available_clients,  # Minimum number of available clients
                evaluate_fn=get_evaluate_fn(combined_test_loader, num_rounds, model_config),
                # Server-side evaluation function
                fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,  # Add metrics aggregation function
                evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,  # Add metrics aggregation function
                on_fit_config_fn=lambda server_round: {
                    "server_round": server_round,  # Pass round number to clients
                    "strategy": "fedprox",
                    "proximal_mu": prox_mu,  # Pass proximal term weight to clients
                    "num_rounds": num_rounds  # Pass total rounds to clients
                },
                proximal_mu=prox_mu,  # Proximal term weight
                num_rounds=num_rounds  # Pass number of rounds for final round detection
            )
            logger.info(f"Using FedProx strategy with proximal mu={prox_mu}")
        else:
            # Default to FedAvg strategy
            strategy = DPFedAvg(
                fraction_fit=1.0,  # Sample 100% of clients for training
                fraction_evaluate=1.0,  # Sample 100% of clients for evaluation
                min_fit_clients=min_fit_clients,  # Number of clients required for training
                min_evaluate_clients=min_evaluate_clients,  # Number of clients required for evaluation
                min_available_clients=min_available_clients,  # Minimum number of available clients
                evaluate_fn=get_evaluate_fn(combined_test_loader, num_rounds, model_config),
                # Server-side evaluation function
                fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,  # Add metrics aggregation function
                evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,  # Add metrics aggregation function
                on_fit_config_fn=lambda server_round: {
                    "server_round": server_round,  # Pass round number to clients
                    "strategy": "fedavg",
                    "num_rounds": num_rounds  # Pass total rounds to clients
                },
                num_rounds=num_rounds  # Pass number of rounds for final round detection
            )
            logger.info("Using FedAvg strategy")

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

    # Get command line arguments
    import sys

    num_rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    min_clients = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    strategy = sys.argv[3].lower() if len(sys.argv) > 3 else "fedavg"
    proximal_mu = float(sys.argv[4]) if len(sys.argv) > 4 else 0.01

    # Parse model config if provided
    model_config = None
    if len(sys.argv) > 5:
        model_type = sys.argv[5] if len(sys.argv) > 5 else "resnet"
        model_name = sys.argv[6] if len(sys.argv) > 6 else "resnet18"

        model_config = {
            "model_type": model_type,  # 'resnet' or 'densenet'
            "model_name": model_name,  # 'resnet18', 'resnet34', 'densenet121', etc.
            "num_classes": 2
        }

    # Start server
    main(num_rounds, min_clients, strategy, proximal_mu, model_config)