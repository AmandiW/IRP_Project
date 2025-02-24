import torch
from opacus.validators import ModuleValidator
from utils import get_model, create_data_loaders
from client import RetinopathyClient
import flwr as fl
from server import FederatedServer
import multiprocessing
import time
import logging
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Set environment variables
os.environ["GRPC_VERBOSITY"] = "debug"

def start_client(client_id, model, train_loader, test_loader, device):
    logging.info(f"Starting client {client_id}")

    # Convert BatchNorm to GroupNorm
    model = ModuleValidator.fix(model)  # This will handle BatchNorm and other issues

    client = RetinopathyClient(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        client_id=client_id,
        epochs=1,
        learning_rate=0.001,
        max_grad_norm=1.2,
        noise_multiplier=1.0,
        delta=1e-5
    )

    logging.info(f"Client {client_id} connecting to server...")
    # Update server address
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)
    logging.info(f"Client {client_id} finished")


def main():
    logging.info("Starting federated learning system")

    # Configuration
    image_dir = "D:/FYP_Data/combined_images"
    labels_file = "D:/FYP_Data/cleaned_valid_image_labels.csv"
    num_clients = 3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Create data loaders
    logging.info("Creating data loaders...")
    client_loaders, test_loader = create_data_loaders(
        image_dir=image_dir,
        labels_file=labels_file,
        num_clients=num_clients,
        batch_size=32,
        test_size=0.2
    )
    logging.info("Data loaders created successfully")

    try:
        # Start server in a separate process instead of thread
        import multiprocessing
        server = FederatedServer(num_rounds=5)
        server_process = multiprocessing.Process(target=server.start_server)
        server_process.start()

        # Wait for server to start
        logging.info("Waiting for server initialization...")
        time.sleep(5)  # Increased wait time

        # Start clients
        logging.info(f"Starting {num_clients} clients...")
        client_processes = []
        for i in range(num_clients):
            model = get_model().to(device)
            client_process = multiprocessing.Process(
                target=start_client,
                args=(i, model, client_loaders[i], test_loader, device)
            )
            client_processes.append(client_process)
            client_process.start()
            time.sleep(1)  # Add small delay between client starts

        # Wait for all clients to complete
        logging.info("Waiting for clients to complete...")
        for process in client_processes:
            process.join()

        # Terminate server process
        server_process.terminate()
        server_process.join()

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        # Ensure processes are terminated in case of error
        for process in client_processes:
            process.terminate()
        server_process.terminate()

    logging.info("Federated learning process completed")




if __name__ == "__main__":
    main()