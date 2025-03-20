import multiprocessing
import time
from server import main as server_main
from client import start_client
import logging
import os
import argparse
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import warnings
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`",
    category=FutureWarning
)

# Filter out the backward hook FutureWarning
warnings.filterwarnings(
    "ignore",
    message="Using a non-full backward hook when the forward contains multiple autograd Nodes is deprecated",
    category=FutureWarning
)

# General filter for Opacus warnings if needed
warnings.filterwarnings("ignore", module="opacus.*")


# Create a centralized logging configuration
def setup_logging():
    if not os.path.exists("./logs"):
        os.makedirs("./logs")

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


def main(args):
    """
    Main function to coordinate server and client processes with DP parameters.
    """
    # Print system information
    logger.info(f"Python version: {multiprocessing.sys.version}")
    logger.info(f"Number of CPUs: {multiprocessing.cpu_count()}")
    logger.info(f"Torch CUDA available: {torch.cuda.is_available() if 'torch' in globals() else 'N/A'}")
    if 'torch' in globals() and torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Create directories for results
    for dir_path in ["./logs", "./saved_models", "./visualizations", "./aggregated_metrics"]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"Created directory: {dir_path}")

    # Log DP parameters
    logger.info(f"Differential Privacy Parameters:")
    logger.info(f"  Number of rounds: {args.num_rounds}")
    logger.info(f"  Noise multiplier: {args.noise_multiplier}")
    logger.info(f"  Max gradient norm: {args.max_grad_norm}")
    logger.info(f"  Delta: 1e-5")

    # Create processes
    processes = []

    # Start server process
    logger.info("Starting server process...")
    server_process = multiprocessing.Process(target=server_main)
    processes.append(server_process)
    server_process.start()

    # Wait for server to start
    logger.info("Waiting for server to initialize...")
    time.sleep(5)  # Increased wait time to ensure server is fully initialized

    # Start client processes with DP parameters
    num_clients = args.num_clients
    logger.info(f"Starting {num_clients} client processes with DP...")

    # Create DP parameters for each client
    dp_params = {
        "noise_multiplier": args.noise_multiplier,
        "max_grad_norm": args.max_grad_norm,
        "delta": 1e-5
    }

    for client_id in range(num_clients):
        logger.info(f"Starting client {client_id} process with DP parameters: {dp_params}")
        client_process = multiprocessing.Process(
            target=start_client,
            args=(client_id, dp_params)
        )
        processes.append(client_process)
        client_process.start()
        # Small delay between client starts to prevent connection issues
        time.sleep(1)  # Increased delay for more reliable startup

    # Wait for all processes to complete
    logger.info("All processes started. Waiting for completion...")
    for i, process in enumerate(processes):
        process_type = "Server" if i == 0 else f"Client {i - 1}"
        logger.info(f"Waiting for {process_type} process to complete...")
        process.join()
        logger.info(f"{process_type} process completed!")

    logger.info("All processes completed successfully!")

    # Generate final summary plots and reports
    logger.info("Generating final summary visualizations...")
    generate_summary_visualizations()


def generate_summary_visualizations():
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

            # Generate summary report
            with open("./aggregated_metrics/final_summary_report.txt", "w") as f:
                f.write("===============================================\n")
                f.write("DIFFERENTIAL PRIVACY FEDERATED LEARNING REPORT\n")
                f.write("===============================================\n\n")

                f.write("PRIVACY METRICS:\n")
                f.write(f"Final Privacy Budget (ε): {rounds_df['average_epsilon'].iloc[-1]:.4f}\n")
                f.write(
                    f"Privacy Budget Increase Rate: {(rounds_df['average_epsilon'].iloc[-1] - rounds_df['average_epsilon'].iloc[0]) / (len(rounds_df) - 1):.4f} per round\n\n")

                f.write("UTILITY METRICS:\n")
                f.write(f"Final Accuracy: {rounds_df['average_accuracy'].iloc[-1]:.4f}\n")
                f.write(f"Final F1 Score: {rounds_df['average_f1'].iloc[-1]:.4f}\n")
                f.write(f"Final Loss: {rounds_df['average_loss'].iloc[-1]:.4f}\n\n")

                f.write("PRIVACY-UTILITY TRADEOFF:\n")
                f.write(
                    f"ε/Accuracy Ratio: {rounds_df['average_epsilon'].iloc[-1] / rounds_df['average_accuracy'].iloc[-1]:.4f}\n")
                f.write(
                    f"Accuracy gained per unit of privacy spent: {(rounds_df['average_accuracy'].iloc[-1] - rounds_df['average_accuracy'].iloc[0]) / (rounds_df['average_epsilon'].iloc[-1] - rounds_df['average_epsilon'].iloc[0] + 1e-8):.4f}\n\n")

                f.write("TRAINING PROGRESSION:\n")
                for i, row in rounds_df.iterrows():
                    f.write(
                        f"Round {int(row['round'])}: ε = {row['average_epsilon']:.4f}, Accuracy = {row['average_accuracy']:.4f}, F1 = {row['average_f1']:.4f}, Loss = {row['average_loss']:.4f}\n")

            logger.info("Summary report and visualizations generated successfully.")
        else:
            logger.warning("Could not generate summary visualizations: rounds_metadata.csv not found.")
    except Exception as e:
        logger.error(f"Error generating summary visualizations: {e}")


if __name__ == "__main__":
    # Import torch here to check CUDA availability in the main function
    import torch

    # Required for Windows multiprocessing
    multiprocessing.freeze_support()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Differential Privacy Federated Learning for Diabetic Retinopathy Diagnosis')
    parser.add_argument('--num_clients', type=int, default=3, help='Number of clients to simulate')
    parser.add_argument('--num_rounds', type=int, default=5, help='Number of federated learning rounds')
    parser.add_argument('--noise_multiplier', type=float, default=1.0, help='Noise multiplier for DP-SGD')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm for clipping')

    args = parser.parse_args()

    main(args)