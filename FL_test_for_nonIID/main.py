import multiprocessing
import time
import logging
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import warnings

# Import server and client functionality
from server import main as server_main
from client import start_client

# Import new visualization functions
from utils import (
    plot_privacy_utility_tradeoff_curve,
    plot_noise_impact_on_accuracy,
    visualize_epsilon_delta_tradeoff,
    plot_membership_inference_risk,
    plot_privacy_leakage_reduction,
    plot_dp_vs_non_dp_performance,
    plot_all_clients_per_round_accuracy,
    plot_client_vs_global_accuracy_per_round,
    plot_accuracy_improvement_rate,
    simulate_membership_inference_risk,
    calculate_theoretical_leak_probability,
    plot_iid_vs_non_iid_performance
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="The NumPy module was reloaded")
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`",
    category=FutureWarning
)
warnings.filterwarnings(
    "ignore",
    message="Using a non-full backward hook when the forward contains multiple autograd Nodes is deprecated",
    category=FutureWarning
)
warnings.filterwarnings("ignore", module="opacus.*")


def setup_logging():
    """Set up centralized logging configuration."""
    # Create logs directory if it doesn't exist
    os.makedirs("./logs", exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("./logs/federated_learning.log"),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger("Main")


logger = setup_logging()


def create_directories():
    """Create necessary directories for results."""
    for dir_path in [
        "./logs",
        "./saved_models",
        "./visualizations",
        "./aggregated_metrics",
        "./visualizations/privacy_analysis",
        "./visualizations/client_comparison",
        "./visualizations/distribution_analysis"  # New directory for distribution comparisons
    ]:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Ensured directory exists: {dir_path}")


def generate_summary_visualizations(args):
    """Generate summary visualizations after all processes complete."""
    try:
        import pandas as pd

        # Check if metadata exists
        if os.path.exists("./aggregated_metrics/rounds_metadata.csv"):
            # Load rounds metadata
            rounds_df = pd.read_csv("./aggregated_metrics/rounds_metadata.csv")

            # Create privacy budget progression visualization
            plt.figure(figsize=(10, 6))
            plt.plot(rounds_df['round'], rounds_df['average_epsilon'], 'ro-', linewidth=2, markersize=8)
            plt.title('Privacy Budget (ε) Progression Over Training Rounds', fontsize=14)
            plt.xlabel('Round', fontsize=12)
            plt.ylabel('Privacy Budget (ε)', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig("./visualizations/final_privacy_budget_progression.png")
            plt.close()

            # Create performance metrics summary
            plt.figure(figsize=(12, 8))

            plt.subplot(2, 2, 1)
            plt.plot(rounds_df['round'], rounds_df['average_accuracy'], 'bo-', linewidth=2)
            plt.title('Average Accuracy', fontsize=12)
            plt.xlabel('Round', fontsize=10)
            plt.ylabel('Accuracy', fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)

            plt.subplot(2, 2, 2)
            plt.plot(rounds_df['round'], rounds_df['average_f1'], 'go-', linewidth=2)
            plt.title('Average F1 Score', fontsize=12)
            plt.xlabel('Round', fontsize=10)
            plt.ylabel('F1 Score', fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)

            plt.subplot(2, 2, 3)
            plt.plot(rounds_df['round'], rounds_df['average_loss'], 'ro-', linewidth=2)
            plt.title('Average Loss', fontsize=12)
            plt.xlabel('Round', fontsize=10)
            plt.ylabel('Loss', fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)

            plt.subplot(2, 2, 4)
            # Plotting privacy-utility tradeoff
            plt.scatter(rounds_df['average_epsilon'], rounds_df['average_accuracy'],
                        s=100, c=rounds_df['round'], cmap='viridis', alpha=0.8)
            plt.colorbar(label='Round')
            plt.title('Privacy-Utility Tradeoff', fontsize=12)
            plt.xlabel('Privacy Budget (ε)', fontsize=10)
            plt.ylabel('Accuracy', fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)

            plt.tight_layout()
            plt.savefig("./visualizations/final_performance_summary.png")
            plt.close()

            # Create a privacy-utility tradeoff visualization
            plt.figure(figsize=(10, 6))
            scatter = plt.scatter(rounds_df['average_epsilon'],
                                  rounds_df['average_accuracy'],
                                  s=rounds_df['average_f1'] * 100,
                                  c=rounds_df['round'],
                                  cmap='viridis',
                                  alpha=0.8)

            # Add round numbers as labels
            for i, row in rounds_df.iterrows():
                plt.annotate(f"R{int(row['round'])}",
                             (row['average_epsilon'], row['average_accuracy']),
                             xytext=(5, 5),
                             textcoords='offset points')

            plt.colorbar(scatter, label='Round')
            plt.title('Privacy-Utility Tradeoff (Size represents F1 Score)', fontsize=14)
            plt.xlabel('Privacy Budget (ε)', fontsize=12)
            plt.ylabel('Model Accuracy', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig("./visualizations/final_privacy_utility_tradeoff.png")
            plt.close()

            # Generate more advanced privacy-utility tradeoff visualization
            epsilons = rounds_df['average_epsilon'].tolist()
            accuracies = rounds_df['average_accuracy'].tolist()
            f1_scores = rounds_df['average_f1'].tolist()

            # Create an improved privacy-utility tradeoff visualization with privacy regions
            plot_privacy_utility_tradeoff_curve(epsilons, accuracies, f1_scores)

            # Generate visualization of theoretic membership inference risks
            if len(epsilons) > 0:
                # Generate a range of epsilon values for visualization
                epsilon_range = np.linspace(0.1, max(epsilons) * 1.5, 20)
                inference_risks = [simulate_membership_inference_risk(eps) for eps in epsilon_range]
                plot_membership_inference_risk(epsilon_range, inference_risks)

                # Generate privacy leakage reduction visualization
                leak_probs = [calculate_theoretical_leak_probability(eps) for eps in epsilon_range]
                plot_privacy_leakage_reduction(epsilon_range, leak_probs)

            # Generate improvement rate visualization if available
            if 'average_improvement_rate' in rounds_df.columns:
                plt.figure(figsize=(10, 6))
                plt.plot(rounds_df['round'], rounds_df['average_improvement_rate'], 'bo-', linewidth=2, markersize=8)
                plt.axhline(y=0, color='r', linestyle='--')
                plt.title('Average Accuracy Improvement Rate Over Rounds', fontsize=14)
                plt.xlabel('Round', fontsize=12)
                plt.ylabel('Accuracy Improvement from Previous Round', fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.savefig("./visualizations/average_improvement_rate.png")
                plt.close()

            # Generate client accuracy progression visualization
            if os.path.exists("./aggregated_metrics/client_performance.csv"):
                client_df = pd.read_csv("./aggregated_metrics/client_performance.csv")

                # Check if we have multiple clients
                if len(client_df['client_id'].unique()) > 1:
                    client_ids = sorted(client_df['client_id'].unique())

                    # Calculate improvement rates per client
                    client_improvements = []
                    for client_id in client_ids:
                        client_data = client_df[client_df['client_id'] == client_id].sort_values('round')
                        accuracies = client_data['accuracy'].tolist()
                        improvements = [0]  # First round has no improvement
                        for i in range(1, len(accuracies)):
                            improvements.append(accuracies[i] - accuracies[i - 1])

                        # Handle case where client might have missed rounds
                        if len(improvements) < args.num_rounds:
                            improvements += [0] * (args.num_rounds - len(improvements))
                        client_improvements.append(improvements)

                    # Plot improvement rates
                    plot_accuracy_improvement_rate(client_ids, client_improvements, is_dp=True)

                    # Plot client vs global accuracy if global accuracy is available
                    if 'global_accuracy' in rounds_df.columns:
                        plot_client_vs_global_accuracy_per_round(rounds_df)

            # Generate DP vs non-DP comparison if we ran multiple experiments
            # # This is a placeholder for research comparisons - you'll need to fill in actual data
            # if os.path.exists("./aggregated_metrics/dp_vs_non_dp_data.csv"):
            #     try:
            #         comparison_df = pd.read_csv("./aggregated_metrics/dp_vs_non_dp_data.csv")
            #         rounds = comparison_df['round'].unique().tolist()
            #         dp_data = comparison_df[comparison_df['training_type'] == 'dp']
            #         non_dp_data = comparison_df[comparison_df['training_type'] == 'non_dp']
            #
            #         dp_accuracies = dp_data.sort_values('round')['accuracy'].tolist()
            #         non_dp_accuracies = non_dp_data.sort_values('round')['accuracy'].tolist()
            #
            #         plot_dp_vs_non_dp_performance(rounds, dp_accuracies, non_dp_accuracies)
            #         logger.info("Generated DP vs non-DP performance comparison")
            #     except Exception as e:
            #         logger.error(f"Error generating DP vs non-DP comparison: {e}")

            # Generate noise multiplier impact visualization if we ran multiple experiments
            # This is a placeholder for research comparisons - you'll need to fill in actual data
            # if os.path.exists("./aggregated_metrics/noise_impact_data.csv"):
            #     try:
            #         noise_df = pd.read_csv("./aggregated_metrics/noise_impact_data.csv")
            #         plot_noise_impact_on_accuracy(
            #             noise_df['noise_multiplier'].tolist(),
            #             noise_df['accuracy'].tolist(),
            #             noise_df['f1_score'].tolist()
            #         )
            #         logger.info("Generated noise multiplier impact visualization")
            #     except Exception as e:
            #         logger.error(f"Error generating noise impact visualization: {e}")

            # Generate epsilon-delta tradeoff visualization
            visualize_epsilon_delta_tradeoff(np.linspace(0.5, 5.0, 20))

            # Generate IID vs non-IID performance comparison if we have data
            # if os.path.exists("./aggregated_metrics/distribution_comparison.csv"):
            #     try:
            #         dist_df = pd.read_csv("./aggregated_metrics/distribution_comparison.csv")
            #         rounds = dist_df['round'].unique().tolist()
            #         iid_data = dist_df[dist_df['distribution'] == 'iid']
            #         non_iid_data = dist_df[dist_df['distribution'] == 'non_iid']
            #
            #         iid_accuracies = iid_data.sort_values('round')['accuracy'].tolist()
            #         non_iid_accuracies = non_iid_data.sort_values('round')['accuracy'].tolist()
            #
            #         plot_iid_vs_non_iid_performance(rounds, iid_accuracies, non_iid_accuracies)
            #         logger.info("Generated IID vs non-IID performance comparison")
            #     except Exception as e:
            #         logger.error(f"Error generating distribution comparison: {e}")

            logger.info("Summary visualizations generated successfully")
        else:
            logger.warning("Could not generate summary visualizations: rounds_metadata.csv not found")
    except Exception as e:
        logger.error(f"Error generating summary visualizations: {e}")


# def generate_dp_experiments_data(args):
#     """
#     Generate data for DP vs non-DP comparison and noise impact analysis.
#
#     This function creates placeholder files for your research experiments.
#     Typically, you would need to run multiple experiments with different
#     parameters and collect the results.
#
#     For a real analysis, replace this function with actual experiment results.
#     """
#     logger.info("Creating placeholder files for privacy research experiments...")
#
#     try:
#         import pandas as pd
#         os.makedirs("./aggregated_metrics", exist_ok=True)
#
#         # Create placeholder for DP vs non-DP comparison
#         # RESEARCH NOTE: Replace this with actual experiment results
#         # by running your code with and without DP
#         rounds = list(range(1, args.num_rounds + 1))
#
#         # These values are placeholders - replace with actual results from experiments
#         dp_accuracies = [0.65 + 0.03 * i for i in range(len(rounds))]
#         non_dp_accuracies = [0.72 + 0.02 * i for i in range(len(rounds))]
#
#         dp_data = pd.DataFrame({
#             'round': rounds,
#             'accuracy': dp_accuracies,
#             'training_type': ['dp'] * len(rounds)
#         })
#
#         non_dp_data = pd.DataFrame({
#             'round': rounds,
#             'accuracy': non_dp_accuracies,
#             'training_type': ['non_dp'] * len(rounds)
#         })
#
#         comparison_df = pd.concat([dp_data, non_dp_data])
#         comparison_df.to_csv("./aggregated_metrics/dp_vs_non_dp_data.csv", index=False)
#
#         # Create placeholder for noise multiplier impact analysis
#         # RESEARCH NOTE: Replace this with actual experiment results
#         # by running your code with different noise multiplier values
#         noise_multipliers = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
#
#         # These values are placeholders - replace with actual results from experiments
#         accuracies = [0.85, 0.82, 0.78, 0.75, 0.70, 0.65]
#         f1_scores = [0.84, 0.80, 0.76, 0.73, 0.68, 0.63]
#
#         noise_df = pd.DataFrame({
#             'noise_multiplier': noise_multipliers,
#             'accuracy': accuracies,
#             'f1_score': f1_scores
#         })
#
#         noise_df.to_csv("./aggregated_metrics/noise_impact_data.csv", index=False)
#
#         # Create placeholder for IID vs non-IID distribution comparison
#         # RESEARCH NOTE: Replace this with actual experiment results
#         # by running your code with IID and non-IID distributions
#         iid_accuracies = [0.75 + 0.04 * i for i in range(len(rounds))]
#         non_iid_accuracies = [0.67 + 0.03 * i for i in range(len(rounds))]
#
#         iid_data = pd.DataFrame({
#             'round': rounds,
#             'accuracy': iid_accuracies,
#             'distribution': ['iid'] * len(rounds)
#         })
#
#         non_iid_data = pd.DataFrame({
#             'round': rounds,
#             'accuracy': non_iid_accuracies,
#             'distribution': ['non_iid'] * len(rounds)
#         })
#
#         distribution_df = pd.concat([iid_data, non_iid_data])
#         distribution_df.to_csv("./aggregated_metrics/distribution_comparison.csv", index=False)
#
#         logger.info("Created placeholder files for research experiments")
#         logger.info("IMPORTANT: Replace these placeholder values with actual experiment results")
#         logger.info("Run multiple experiments with different parameters and collect the results")
#     except Exception as e:
#         logger.error(f"Error creating placeholder files for research experiments: {e}")


def main(args):
    """
    Main function to coordinate server and client processes.

    Args:
        args: Command-line arguments
    """
    # Override from environment variables if present
    args.num_rounds = int(os.environ.get("FL_NUM_ROUNDS", args.num_rounds))
    args.num_clients = int(os.environ.get("FL_NUM_CLIENTS", args.num_clients))
    args.epochs = int(os.environ.get("FL_EPOCHS", args.epochs))
    args.distribution = os.environ.get("FL_DISTRIBUTION", args.distribution)
    args.alpha = float(os.environ.get("FL_ALPHA", args.alpha))

    # Set environment variables for server and clients to use
    os.environ["FL_NUM_ROUNDS"] = str(args.num_rounds)
    os.environ["FL_NUM_CLIENTS"] = str(args.num_clients)
    os.environ["FL_EPOCHS"] = str(args.epochs)
    os.environ["FL_DISTRIBUTION"] = args.distribution
    os.environ["FL_ALPHA"] = str(args.alpha)

    # Print system information
    logger.info(f"Python version: {multiprocessing.sys.version}")
    logger.info(f"Number of CPUs: {multiprocessing.cpu_count()}")

    # Check for CUDA availability
    import torch
    logger.info(f"Torch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Create directories for results
    create_directories()

    # Log experiment parameters
    logger.info(f"Experiment Parameters:")
    logger.info(f"  Number of rounds: {args.num_rounds}")
    logger.info(f"  Number of epochs per round: {args.epochs}")
    logger.info(f"  Number of clients: {args.num_clients}")
    logger.info(f"  Distribution type: {args.distribution}")
    if args.distribution == 'non_iid':
        logger.info(f"  Alpha parameter: {args.alpha}")
    logger.info(f"  Noise multiplier: {args.noise_multiplier}")
    logger.info(f"  Max gradient norm: {args.max_grad_norm}")
    logger.info(f"  Delta: 1e-5")

    # Generate research experiment placeholders
    # Comment this out if you don't want the placeholder files
    # if args.generate_research_placeholders:
    #     generate_dp_experiments_data(args)

    # Create processes
    processes = []

    # Start server process
    logger.info("Starting server process...")
    server_process = multiprocessing.Process(
        target=server_main,
        args=[args.num_rounds, args.num_clients]  # Pass number of rounds and min clients to server
    )
    processes.append(server_process)
    server_process.start()

    # Wait for server to start
    logger.info("Waiting for server to initialize...")
    time.sleep(5)  # Give server time to start

    # Start client processes with DP parameters
    num_clients = args.num_clients
    logger.info(f"Starting {num_clients} client processes with DP...")

    # Create DP parameters for each client
    dp_params = {
        "noise_multiplier": args.noise_multiplier,
        "max_grad_norm": args.max_grad_norm,
        "delta": 1e-5,
        "epochs": args.epochs
    }

    for client_id in range(num_clients):
        logger.info(f"Starting client {client_id} process")
        client_process = multiprocessing.Process(
            target=start_client,
            args=(client_id, dp_params)
        )
        processes.append(client_process)
        client_process.start()
        # Small delay between client starts to prevent connection issues
        time.sleep(1)

    # Wait for all processes to complete
    logger.info("All processes started. Waiting for completion...")
    for i, process in enumerate(processes):
        process_type = "Server" if i == 0 else f"Client {i - 1}"
        logger.info(f"Waiting for {process_type} process to complete...")
        process.join()
        logger.info(f"{process_type} process completed!")

    logger.info("All processes completed successfully!")

    # Generate final summary visualizations
    logger.info("Generating final summary visualizations...")
    generate_summary_visualizations(args)

    # Process completion
    logger.info("Federated learning with differential privacy completed successfully!")


if __name__ == "__main__":
    # Import torch to check CUDA availability
    import torch

    # Required for Windows multiprocessing
    multiprocessing.freeze_support()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Differential Privacy Federated Learning for Diabetic Retinopathy Diagnosis'
    )
    parser.add_argument('--num_clients', type=int, default=4, help='Number of clients to simulate')
    parser.add_argument('--num_rounds', type=int, default=1, help='Number of federated learning rounds')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs per round')
    parser.add_argument('--noise_multiplier', type=float, default=1.0, help='Noise multiplier for DP')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm for clipping')
    parser.add_argument('--distribution', type=str, default='iid', choices=['iid', 'non_iid'],
                        help='Data distribution type (iid or non_iid)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dirichlet concentration parameter for non-IID distribution')
    # parser.add_argument('--generate_research_placeholders', action='store_true',
    #                     help='Generate placeholder files for research experiments')
    args = parser.parse_args()

    # Start federation
    main(args)