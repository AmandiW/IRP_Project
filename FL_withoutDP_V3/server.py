import flwr as fl
import torch
from utils import create_model, load_data
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
import numpy as np
import logging
import os

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def get_evaluate_fn(test_loader):
    """Return an evaluation function for server-side evaluation."""
    logger = logging.getLogger("Server")

    def evaluate(server_round, parameters, config):
        model = create_model()
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
        criterion = torch.nn.CrossEntropyLoss()

        logger.info(f"Server-side evaluation - Round {server_round} - Testing on {len(test_loader.dataset)} samples")

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss += criterion(outputs, target).item()
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

        # Save model after each round
        if not os.path.exists("./saved_models"):
            os.makedirs("./saved_models")
        torch.save(model.state_dict(), f"./saved_models/model_round_{server_round}.pth")
        logger.info(f"Model saved after round {server_round}")

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

    # Define strategy
    strategy = fl.server.strategy.FedAvg(
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
    logger.info("Starting Flower server...")
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=2),
        strategy=strategy
    )


if __name__ == "__main__":
    # Use the logging setup from main
    from main import setup_logging

    setup_logging()

    main()