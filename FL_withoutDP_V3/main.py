import multiprocessing
import time
from server import main as server_main
from client import start_client
import logging
import os

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


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


def main():
    """
    Main function to coordinate server and client processes.
    """
    # Print system information
    logger.info(f"Python version: {multiprocessing.sys.version}")
    logger.info(f"Number of CPUs: {multiprocessing.cpu_count()}")
    logger.info(f"Torch CUDA available: {torch.cuda.is_available() if 'torch' in globals() else 'N/A'}")
    if 'torch' in globals() and torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

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

    # Start client processes
    num_clients = 3
    logger.info(f"Starting {num_clients} client processes...")

    for client_id in range(num_clients):
        logger.info(f"Starting client {client_id} process...")
        client_process = multiprocessing.Process(
            target=start_client,
            args=(client_id,)
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


if __name__ == "__main__":
    # Import torch here to check CUDA availability in the main function
    import torch

    # Required for Windows multiprocessing
    multiprocessing.freeze_support()

    main()