import flwr as fl
import torch
from utils import create_model, load_data
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def get_evaluate_fn(test_loader):
    """Return an evaluation function for server-side evaluation."""

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

        # Print evaluation metrics
        logging.info(f"\nServer-side evaluation - Round {server_round}")
        logging.info(f"Loss: {loss:.4f}")
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info("\nConfusion Matrix:")
        logging.info(confusion_matrix(y_true, y_pred))
        logging.info("\nClassification Report:")
        logging.info(classification_report(y_true, y_pred))

        return loss, {"accuracy": accuracy}

    return evaluate


def main():
    # Load data for server-side evaluation
    client_data = load_data(
        img_dir="D:/FYP_Data/combined_images",
        labels_path="D:/FYP_Data/cleaned_valid_image_labels.csv"
    )
    _, test_loader = client_data[0]  # Use first client's test data for server evaluation

    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Sample 100% of clients for training
        fraction_evaluate=1.0,  # Sample 100% of clients for evaluation
        min_fit_clients=3,  # All 3 clients should participate in training
        min_evaluate_clients=3,  # All 3 clients should participate in evaluation
        min_available_clients=3,
        evaluate_fn=get_evaluate_fn(test_loader)  # Pass the evaluation function
    )

    # Start server
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=2),  # 2 epochs as requested
        strategy=strategy
    )


if __name__ == "__main__":
    main()