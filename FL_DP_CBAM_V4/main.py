import multiprocessing
import time
import logging
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import warnings
import pandas as pd
import torch

# Import server and client functionality
from server import main as server_main
from client import start_client

# Import privacy evaluation functions
from utils import (
    plot_privacy_utility_tradeoff_curve,
    plot_membership_inference_risk,
    visualize_epsilon_delta_tradeoff,
    plot_privacy_leakage_reduction,
    simulate_membership_inference_risk,
    calculate_theoretical_leak_probability,
    compute_dp_sgd_privacy_budget,
    plot_epsilon_composition
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
        "./visualizations/privacy_comparison"
    ]:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Ensured directory exists: {dir_path}")


def generate_privacy_visualizations(args):
    """
    Generate privacy visualizations demonstrating how DP enhances privacy.

    Args:
        args: Command-line arguments containing DP parameters
    """
    logger.info("Generating privacy visualizations...")

    try:
        # Directory for privacy visualizations
        os.makedirs("./visualizations/privacy_analysis", exist_ok=True)

        # 1. Privacy budget composition over multiple rounds
        # Show how epsilon grows with the number of rounds for different noise values
        noise_values = [0.5, 0.8, 1.0, 1.5]
        for noise in noise_values:
            plot_epsilon_composition(
                num_rounds=args.num_rounds * 2,  # Look ahead to future rounds
                noise_multiplier=noise,
                sample_rate=0.01,  # Example sampling rate
                delta=1e-5
            )

        # 2. Generate theoretical membership inference risks at different epsilons
        epsilon_range = np.linspace(0.1, 10, 20)
        inference_risks = [simulate_membership_inference_risk(eps) for eps in epsilon_range]
        plot_membership_inference_risk(epsilon_range, inference_risks)

        # 3. Generate privacy leakage reduction visualization
        leak_probs = [calculate_theoretical_leak_probability(eps) for eps in epsilon_range]
        plot_privacy_leakage_reduction(epsilon_range, leak_probs)

        # 4. Privacy parameter tradeoff
        visualize_epsilon_delta_tradeoff(np.linspace(0.5, 3.0, 10))

        logger.info("Privacy visualizations generated successfully")
    except Exception as e:
        logger.error(f"Error generating privacy visualizations: {e}")


def generate_summary_visualizations(args):
    """Generate summary visualizations after all processes complete."""
    try:
        # Check if metadata exists
        if os.path.exists("./aggregated_metrics/rounds_metadata.csv"):
            # Load rounds metadata
            rounds_df = pd.read_csv("./aggregated_metrics/rounds_metadata.csv")

            # Create privacy budget progression visualization
            plt.figure(figsize=(10, 6))

            # Use cumulative epsilon values for better illustration
            if 'average_cumulative_epsilon' in rounds_df.columns:
                plt.plot(rounds_df['round'], rounds_df['average_cumulative_epsilon'], 'ro-', linewidth=2, markersize=8)
                plt.title('Cumulative Privacy Budget (ε) Progression Over Training Rounds', fontsize=14)
            else:
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
            # Plotting privacy-utility tradeoff - use cumulative epsilon if available
            epsilon_col = 'average_cumulative_epsilon' if 'average_cumulative_epsilon' in rounds_df.columns else 'average_epsilon'
            plt.scatter(rounds_df[epsilon_col], rounds_df['average_accuracy'],
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
            # Use cumulative epsilon if available
            epsilon_col = 'average_cumulative_epsilon' if 'average_cumulative_epsilon' in rounds_df.columns else 'average_epsilon'
            scatter = plt.scatter(rounds_df[epsilon_col],
                                  rounds_df['average_accuracy'],
                                  s=rounds_df['average_f1'] * 100,
                                  c=rounds_df['round'],
                                  cmap='viridis',
                                  alpha=0.8)

            # Add round numbers as labels
            for i, row in rounds_df.iterrows():
                plt.annotate(f"R{int(row['round'])}",
                             (row[epsilon_col], row['average_accuracy']),
                             xytext=(5, 5),
                             textcoords="offset points",
                             ha='center')

            plt.colorbar(scatter, label='Round')
            plt.title('Privacy-Utility Tradeoff (Size represents F1 Score)', fontsize=14)
            plt.xlabel('Privacy Budget (ε)', fontsize=12)
            plt.ylabel('Model Accuracy', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig("./visualizations/final_privacy_utility_tradeoff.png")
            plt.close()

            # Generate more advanced privacy-utility tradeoff visualization
            epsilon_col = 'average_cumulative_epsilon' if 'average_cumulative_epsilon' in rounds_df.columns else 'average_epsilon'
            epsilons = rounds_df[epsilon_col].tolist()
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

            # Generate epsilon-delta tradeoff visualization
            visualize_epsilon_delta_tradeoff(np.linspace(0.5, 5.0, 20))

            # Generate feature-specific privacy effectiveness visualization if available
            if 'feature_specific_effective' in rounds_df.columns:
                plt.figure(figsize=(10, 6))

                # Create bar chart showing feature-specific privacy status by round
                feature_specific_status = rounds_df['feature_specific_effective'].astype(int)
                bars = plt.bar(rounds_df['round'], feature_specific_status,
                               color=['green' if val else 'red' for val in feature_specific_status])

                # Add labels
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width() / 2., height / 2,
                             'Enabled' if height > 0 else 'Disabled',
                             ha='center', va='center', color='white', fontweight='bold')

                plt.title('Feature-Specific Privacy Status by Round', fontsize=14)
                plt.xlabel('Round', fontsize=12)
                plt.ylabel('Status', fontsize=12)
                plt.yticks([0, 1], ['Disabled', 'Enabled'])
                plt.grid(True, linestyle='--', alpha=0.7, axis='x')
                plt.savefig("./visualizations/feature_specific_privacy_status.png")
                plt.close()

            logger.info("Summary visualizations generated successfully")
        else:
            logger.warning("Could not generate summary visualizations: rounds_metadata.csv not found")
    except Exception as e:
        logger.error(f"Error generating summary visualizations: {e}")


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
    args.delta = float(os.environ.get("FL_DELTA", args.delta))
    args.strategy = os.environ.get("FL_STRATEGY", args.strategy)
    args.proximal_mu = float(os.environ.get("FL_PROX_MU", args.proximal_mu))
    args.feature_specific = os.environ.get("FL_FEATURE_SPECIFIC", str(args.feature_specific)).lower() == "true"
    args.model_type = os.environ.get("FL_MODEL_TYPE", args.model_type)
    args.model_name = os.environ.get("FL_MODEL_NAME", args.model_name)

    # Allow data paths to be configurable
    args.img_dir = os.environ.get("FL_IMG_DIR", args.img_dir)
    args.labels_path = os.environ.get("FL_LABELS_PATH", args.labels_path)

    # Set environment variables for server and clients to use
    os.environ["FL_NUM_ROUNDS"] = str(args.num_rounds)
    os.environ["FL_NUM_CLIENTS"] = str(args.num_clients)
    os.environ["FL_EPOCHS"] = str(args.epochs)
    os.environ["FL_DELTA"] = str(args.delta)
    os.environ["FL_DISTRIBUTION"] = args.distribution
    os.environ["FL_ALPHA"] = str(args.alpha)
    os.environ["FL_STRATEGY"] = args.strategy
    os.environ["FL_PROX_MU"] = str(args.proximal_mu)
    os.environ["FL_FEATURE_SPECIFIC"] = str(args.feature_specific)
    os.environ["FL_MODEL_TYPE"] = args.model_type
    os.environ["FL_MODEL_NAME"] = args.model_name
    os.environ["FL_IMG_DIR"] = args.img_dir
    os.environ["FL_LABELS_PATH"] = args.labels_path

    # Print system information
    logger.info(f"Python version: {multiprocessing.sys.version}")
    logger.info(f"Number of CPUs: {multiprocessing.cpu_count()}")

    # Check for CUDA availability
    logger.info(f"Torch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Create directories for results
    create_directories()

    # Generate privacy visualizations before training starts
    generate_privacy_visualizations(args)

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
    logger.info(f"  Strategy: {args.strategy}")
    if args.strategy == 'fedprox':
        logger.info(f"  Proximal term mu: {args.proximal_mu}")
    logger.info(f"  Delta: {args.delta}")
    logger.info(f"  Feature-specific privacy: {args.feature_specific}")
    logger.info(f"  Model type: {args.model_type}")
    logger.info(f"  Model name: {args.model_name}")
    logger.info(f"  Image directory: {args.img_dir}")
    logger.info(f"  Labels path: {args.labels_path}")

    # Create processes
    processes = []

    # Create model config for clients and server
    model_config = {
        "model_type": args.model_type,
        "model_name": args.model_name,
        "num_classes": 2
    }

    # Start server process
    logger.info("Starting server process...")
    server_process = multiprocessing.Process(
        target=server_main,
        args=[args.num_rounds, args.num_clients, args.strategy, args.proximal_mu, model_config]
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
        "delta": args.delta,
        "epochs": args.epochs,
        "feature_specific": args.feature_specific
    }

    for client_id in range(num_clients):
        logger.info(f"Starting client {client_id} process")
        client_process = multiprocessing.Process(
            target=start_client,
            args=(client_id, model_config, dp_params)
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
    # Required for Windows multiprocessing
    multiprocessing.freeze_support()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Feature-Specific Differential Privacy Federated Learning for Diabetic Retinopathy Diagnosis'
    )
    parser.add_argument('--num_clients', type=int, default=3, help='Number of clients to simulate')
    parser.add_argument('--num_rounds', type=int, default=5, help='Number of federated learning rounds')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs per round')
    parser.add_argument('--noise_multiplier', type=float, default=0.8, help='Noise multiplier for DP')
    parser.add_argument('--max_grad_norm', type=float, default=1, help='Maximum gradient norm for clipping')
    parser.add_argument('--delta', type=float, default=1e-5,
                        help='Delta parameter for differential privacy (probability of privacy guarantee breaking)')
    parser.add_argument('--distribution', type=str, default='iid', choices=['iid', 'non_iid'],
                        help='Data distribution type (iid or non_iid)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dirichlet concentration parameter for non-IID distribution')
    parser.add_argument('--strategy', type=str, default='fedavg', choices=['fedavg', 'fedprox'],
                        help='Federated learning strategy (fedavg or fedprox)')
    parser.add_argument('--proximal_mu', type=float, default=0.01,
                        help='Proximal term weight for FedProx (only used when strategy is fedprox)')
    parser.add_argument('--feature_specific', type=bool, default=True,
                        help='Enable feature-specific privacy using attention mechanisms')
    parser.add_argument('--model_type', type=str, default='densenet', choices=['resnet', 'densenet'],
                        help='Model architecture type (resnet or densenet)')
    parser.add_argument('--model_name', type=str, default='densenet121',
                        choices=['resnet18', 'resnet34', 'resnet50', 'densenet121', 'densenet169', 'densenet201'],
                        help='Specific model name within the architecture type')
    parser.add_argument('--img_dir', type=str, default='E:/IRP_dataset_new/IRP_Final_Images',
                        help='Directory containing the retinopathy images')
    parser.add_argument('--labels_path', type=str, default='E:/IRP_dataset_new/IRP_Final_Labels.csv',
                        help='Path to CSV file with image labels')
    args = parser.parse_args()

    # Start federation
    main(args)