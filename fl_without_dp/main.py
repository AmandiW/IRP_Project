import multiprocessing
import time
from server import main as server_main
from client import start_client
import logging

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("Main")


def main():
    """
    Main function to coordinate server and client processes.
    """
    # Create processes
    processes = []

    # Start server process
    logger.info("Starting server process...")
    server_process = multiprocessing.Process(target=server_main)
    processes.append(server_process)
    server_process.start()

    # Wait for server to start
    logger.info("Waiting for server to initialize...")
    time.sleep(3)

    # Start client processes
    for client_id in range(3):
        logger.info(f"Starting client {client_id} process...")
        client_process = multiprocessing.Process(
            target=start_client,
            args=(client_id,)
        )
        processes.append(client_process)
        client_process.start()
        # Small delay between client starts to prevent connection issues
        time.sleep(0.5)

    # Wait for all processes to complete
    logger.info("All processes started. Waiting for completion...")
    for process in processes:
        process.join()

    logger.info("All processes completed successfully!")


if __name__ == "__main__":
    # Required for Windows multiprocessing
    multiprocessing.freeze_support()
    main()