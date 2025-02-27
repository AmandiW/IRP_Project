import multiprocessing
import time
from server import main as server_main
from client import start_client
import logging
import os

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Create logs directory
os.makedirs("logs", exist_ok=True)

# Configure logging to file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/federated_learning_{time.strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("Main")


def main():
    """
    Main function to coordinate server and client processes.
    """
    # Check if data paths exist
    img_dir = "D:/FYP_Data/combined_images"
    labels_path = "D:/FYP_Data/cleaned_valid_image_labels.csv"

    if not os.path.exists(img_dir):
        logger.error(f"Image directory not found: {img_dir}")
        return

    if not os.path.exists(labels_path):
        logger.error(f"Labels file not found: {labels_path}")
        return

    logger.info(f"Image directory: {img_dir}")
    logger.info(f"Labels file: {labels_path}")

    # CHANGE 1: Added log message about binary classification
    logger.info("Running with binary classification setup (2 classes)")

    # Create processes
    processes = []

    # Start server process
    logger.info("Starting server process...")
    server_process = multiprocessing.Process(target=server_main)
    processes.append(server_process)
    server_process.start()

    # Wait for server to start
    logger.info("Waiting for server to initialize...")
    time.sleep(5)  # Increased wait time

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
        # Increased delay between client starts to prevent connection issues
        time.sleep(2)

    # Wait for all processes to complete
    logger.info("All processes started. Waiting for completion...")
    for process in processes:
        process.join()

    logger.info("All processes completed successfully!")


if __name__ == "__main__":
    # Required for Windows multiprocessing
    multiprocessing.freeze_support()
    main()