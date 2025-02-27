import flwr as fl
import torch
import torch.nn.functional as F
from utils import create_model, load_data
from collections import OrderedDict
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score  # CHANGE 1: Added binary metrics
import numpy as np
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class RetinopathyClient(fl.client.NumPyClient):
    def __init__(self, client_id, train_loader, test_loader, num_classes=2):  # CHANGE 2: Default to 2 classes
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_model(num_classes=num_classes).to(self.device)

        # CHANGE 3: Added class weights for imbalanced dataset
        self.criterion = torch.nn.CrossEntropyLoss()

        # CHANGE 4: Adjusted optimizer parameters
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=0.001,
            momentum=0.9,
            weight_decay=1e-4
        )

        # Learning rate scheduler to reduce LR over time
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=5,
            gamma=0.5
        )

        # Track metrics
        self.best_accuracy = 0.0
        self.epochs_without_improvement = 0

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        logging.info(f"\nClient {self.client_id} - Starting training")
        self.set_parameters(parameters)

        # Extract config parameters
        epochs = config.get("epochs", 5)  # Default to 5 epochs if not specified

        # Training
        self.model.train()
        for epoch in range(epochs):
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

                # Store predictions and targets for metrics
                all_targets.extend(target.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

                if batch_idx % 10 == 0:
                    logging.info(
                        f"Client {self.client_id} - Epoch {epoch + 1}/{epochs} - Batch {batch_idx}/{len(self.train_loader)} - Loss: {loss.item():.4f}")

            # Apply learning rate scheduler
            self.scheduler.step()

            # Calculate metrics
            epoch_loss = running_loss / len(self.train_loader)
            epoch_accuracy = correct / total if total > 0 else 0

            # Log class distribution in predictions
            unique, counts = np.unique(all_predictions, return_counts=True)
            pred_distribution = dict(zip(unique, counts))

            logging.info(
                f"Client {self.client_id} - Epoch {epoch + 1}/{epochs} - Average Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
            logging.info(f"Predicted class distribution: {pred_distribution}")

            # Validate after each epoch
            val_loss, val_accuracy = self._validate()
            logging.info(
                f"Client {self.client_id} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

            # Early stopping check
            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= 3:
                logging.info(f"Client {self.client_id} - Early stopping triggered")
                break

        # Do one final validation to get metrics
        val_loss, val_accuracy = self._validate()

        return self.get_parameters(config={}), len(self.train_loader.dataset), {
            "loss": epoch_loss,
            "accuracy": epoch_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        }

    def _validate(self):
        """Internal validation during training"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        val_loss /= len(self.test_loader)
        val_accuracy = correct / total if total > 0 else 0

        self.model.train()
        return val_loss, val_accuracy

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
        all_probs = []

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += self.criterion(output, target).item()
                probs = F.softmax(output, dim=1)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                y_true.extend(target.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Calculate metrics
        loss /= len(self.test_loader)
        accuracy = correct / total if total > 0 else 0

        # Check if predictions are all the same class
        unique_preds = np.unique(y_pred)
        if len(unique_preds) == 1:
            logging.warning(f"Client {self.client_id} - WARNING: All predictions are class {unique_preds[0]}!")

        # Calculate confidence metrics
        mean_confidence = np.mean([probs[pred] for probs, pred in zip(all_probs, y_pred)])

        # CHANGE 5: Added binary classification metrics
        binary_metrics = {}
        try:
            # Calculate binary classification metrics
            if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
                binary_metrics["precision"] = precision_score(y_true, y_pred)
                binary_metrics["recall"] = recall_score(y_true, y_pred)
                binary_metrics["f1"] = f1_score(y_true, y_pred)

                # Calculate AUC if we have probability scores for positive class
                if len(all_probs) > 0 and len(all_probs[0]) >= 2:
                    # Get probability of positive class (class 1)
                    pos_probs = [prob[1] for prob in all_probs]
                    binary_metrics["auc"] = roc_auc_score(y_true, pos_probs)

                logging.info(f"Precision: {binary_metrics['precision']:.4f}")
                logging.info(f"Recall: {binary_metrics['recall']:.4f}")
                logging.info(f"F1 Score: {binary_metrics['f1']:.4f}")
                if "auc" in binary_metrics:
                    logging.info(f"AUC: {binary_metrics['auc']:.4f}")
        except Exception as e:
            logging.warning(f"Error calculating binary metrics: {e}")

        # Print detailed evaluation metrics
        logging.info(f"\nClient {self.client_id} - Evaluation Results:")
        logging.info(f"Loss: {loss:.4f}")
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Mean prediction confidence: {mean_confidence:.4f}")

        # Check if there are enough samples of each class for reliable metrics
        if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
            logging.info("\nConfusion Matrix:")
            cm = confusion_matrix(y_true, y_pred)
            logging.info(cm)
            logging.info("\nClassification Report:")
            cr = classification_report(y_true, y_pred)
            logging.info(cr)
        else:
            logging.warning(
                "Cannot compute confusion matrix or classification report - not enough class variety in predictions")

        # Merge binary metrics into the result
        result_metrics = {
            "loss": loss,
            "accuracy": accuracy,
            "mean_confidence": mean_confidence,
            **binary_metrics  # Add binary metrics to the results
        }

        return loss, len(self.test_loader.dataset), result_metrics


def start_client(client_id):
    """Function to start a client with specific ID"""
    # Load data
    client_data, num_classes = load_data(
        img_dir="D:/FYP_Data/combined_images",
        labels_path="D:/FYP_Data/cleaned_valid_image_labels.csv"
    )
    train_loader, test_loader = client_data[client_id]

    # CHANGE 6: Explicitly set num_classes to 2 for binary classification
    client = RetinopathyClient(client_id, train_loader, test_loader, num_classes=2)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    import sys

    client_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    start_client(client_id)