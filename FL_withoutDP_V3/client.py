import flwr as fl
import torch
from utils import create_model, load_data, EarlyStopping
from collections import OrderedDict
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
import logging
import numpy as np

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class RetinopathyClient(fl.client.NumPyClient):
    def __init__(self, client_id, train_loader, test_loader, train_size, test_size):
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_size = train_size
        self.test_size = test_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_model().to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())

        # Early stopping initialization
        self.early_stopping = EarlyStopping(patience=3, min_delta=0.001)

        # Get logger
        self.logger = logging.getLogger(f"Client_{client_id}")
        self.logger.info(f"Client {self.client_id} initialized on device: {self.device}")
        self.logger.info(
            f"Client {self.client_id} - Training set size: {self.train_size}, Test set size: {self.test_size}")

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.logger.info(f"\nClient {self.client_id} - Starting training round")
        self.logger.info(f"Client {self.client_id} - Training on {self.train_size} samples")
        self.set_parameters(parameters)
        # Reset early stopping for this round
        self.early_stopping = EarlyStopping(patience=3, min_delta=0.001)
        stop_early = False

        # Training
        self.model.train()
        for epoch in range(2):
            running_loss = 0.0
            correct = 0
            total = 0
            all_targets = []
            all_predictions = []

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                all_targets.extend(target.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

                if batch_idx % 10 == 0:
                    self.logger.info(
                        f"Client {self.client_id} - Epoch {epoch} - Batch {batch_idx}/{len(self.train_loader)} - Loss: {loss.item():.4f}")

            epoch_loss = running_loss / len(self.train_loader)
            epoch_accuracy = correct / total

            # Calculate other metrics
            f1 = f1_score(all_targets, all_predictions, average='weighted')
            precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
            recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)

            # Count class predictions
            unique, counts = np.unique(all_predictions, return_counts=True)
            pred_distribution = dict(zip(unique, counts))

            self.logger.info(
                f"Client {self.client_id} - Epoch {epoch} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, "
                f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            self.logger.info(f"Client {self.client_id} - Epoch {epoch} - Prediction distribution: {pred_distribution}")

            # Evaluate on validation data for early stopping
            val_loss, _ = self.evaluate_for_early_stopping()

            # Check early stopping
            if self.early_stopping(val_loss, self.model):
                self.logger.info(f"Client {self.client_id} - Early stopping at epoch {epoch}")
                stop_early = True

                # Restore best weights if configured
                if self.early_stopping.restore_best_weights and self.early_stopping.best_weights is not None:
                    self.model.load_state_dict(self.early_stopping.best_weights)
                break

        return self.get_parameters(config={}), self.train_size, {}

    def evaluate_for_early_stopping(self):
        """Evaluate model for early stopping purposes"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:  # Using test set as validation
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        val_loss /= len(self.test_loader)
        val_accuracy = correct / total
        self.logger.info(f"Client {self.client_id} - Validation: Loss={val_loss:.4f}, Accuracy={val_accuracy:.4f}")

        return val_loss, val_accuracy

    def evaluate(self, parameters, config):
        self.logger.info(f"\nClient {self.client_id} - Starting evaluation")
        self.logger.info(f"Client {self.client_id} - Evaluating on {self.test_size} samples")
        self.set_parameters(parameters)

        # Evaluation
        self.model.eval()
        loss = 0.0
        correct = 0
        total = 0
        y_true = []
        y_pred = []

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += self.criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                y_true.extend(target.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        loss /= len(self.test_loader)
        accuracy = correct / total

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
        self.logger.info(f"\nClient {self.client_id} - Evaluation Results:")
        self.logger.info(f"Loss: {loss:.4f}")
        self.logger.info(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        self.logger.info(f"True class distribution: {true_distribution}")
        self.logger.info(f"Prediction distribution: {pred_distribution}")
        self.logger.info("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        self.logger.info(f"\n{cm}")
        self.logger.info("\nClassification Report:")
        cr = classification_report(y_true, y_pred)
        self.logger.info(f"\n{cr}")

        return loss, self.test_size, {
            "accuracy": accuracy,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }


def start_client(client_id):
    """Function to start a client with specific ID"""
    # Load data
    client_data = load_data(
        img_dir="D:/FYP_Data/combined_images",
        labels_path="D:/FYP_Data/cleaned_valid_image_labels.csv"
    )
    train_loader, test_loader, train_size, test_size = client_data[client_id]

    logger = logging.getLogger(f"Client_{client_id}")
    # Log client dataset sizes
    logger.info(f"Client {client_id} - Starting with {train_size} training samples, {test_size} testing samples")

    # Start client
    client = RetinopathyClient(client_id, train_loader, test_loader, train_size, test_size)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    import sys

    # Use the logging setup from main
    from main import setup_logging

    setup_logging()

    client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    start_client(client_id)
