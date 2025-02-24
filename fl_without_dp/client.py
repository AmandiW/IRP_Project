import flwr as fl
import torch
from utils import create_model, load_data
from collections import OrderedDict
from sklearn.metrics import confusion_matrix, classification_report
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class RetinopathyClient(fl.client.NumPyClient):
    def __init__(self, client_id, train_loader, test_loader):
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_model().to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        logging.info(f"\nClient {self.client_id} - Starting training")
        self.set_parameters(parameters)

        # Training
        self.model.train()
        for epoch in range(2):  # 2 epochs as requested
            running_loss = 0.0
            correct = 0
            total = 0

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

                if batch_idx % 10 == 0:
                    logging.info(
                        f"Client {self.client_id} - Epoch {epoch} - Batch {batch_idx} - Loss: {loss.item():.4f}")

            epoch_loss = running_loss / len(self.train_loader)
            epoch_accuracy = correct / total
            logging.info(
                f"Client {self.client_id} - Epoch {epoch} - Average Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        logging.info(f"\nClient {self.client_id} - Starting evaluation")
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

        # Print evaluation metrics
        logging.info(f"\nClient {self.client_id} - Evaluation Results:")
        logging.info(f"Loss: {loss:.4f}")
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info("\nConfusion Matrix:")
        logging.info(confusion_matrix(y_true, y_pred))
        logging.info("\nClassification Report:")
        logging.info(classification_report(y_true, y_pred))

        return loss, len(self.test_loader.dataset), {"accuracy": accuracy}


def start_client(client_id):
    """Function to start a client with specific ID"""
    # Load data
    client_data = load_data(
        img_dir="D:/FYP_Data/combined_images",
        labels_path="D:/FYP_Data/cleaned_valid_image_labels.csv"
    )
    train_loader, test_loader = client_data[client_id]

    # Start client
    client = RetinopathyClient(client_id, train_loader, test_loader)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    import sys

    client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    start_client(client_id)