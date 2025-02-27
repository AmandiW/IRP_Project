import flwr as fl
import torch
import torch.nn.functional as F
from utils import create_model, load_data
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score  # CHANGE 1: Added binary metrics
import numpy as np
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Create directory for saving models
os.makedirs("model_checkpoints", exist_ok=True)


def get_evaluate_fn(test_loader, num_classes):
    """Return an evaluation function for server-side evaluation."""

    def evaluate(server_round, parameters, config):
        # CHANGE 2: Force num_classes to 2 for binary classification
        model = create_model(num_classes=2)
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
        all_probs = []
        criterion = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss += criterion(outputs, target).item()
                probs = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                y_true.extend(target.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        accuracy = correct / total if total > 0 else 0
        loss /= len(test_loader)

        # Check for suspicious accuracy
        if accuracy > 0.95 and server_round <= 3:
            logging.warning(
                "SUSPICIOUSLY HIGH ACCURACY DETECTED EARLY IN TRAINING! This may indicate data leakage or other issues."
            )

        # Calculate confidence metrics
        mean_confidence = np.mean([probs[pred] for probs, pred in zip(all_probs, y_pred)])

        # Check for class imbalance in predictions
        unique_preds, counts = np.unique(y_pred, return_counts=True)
        pred_distribution = dict(zip(unique_preds, counts))

        # Print evaluation metrics
        logging.info(f"\nServer-side evaluation - Round {server_round}")
        logging.info(f"Loss: {loss:.4f}")
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Mean prediction confidence: {mean_confidence:.4f}")
        logging.info(f"Prediction distribution: {pred_distribution}")

        # CHANGE 3: Added binary classification metrics
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

        # Only compute confusion matrix if there are multiple classes in both true and predicted
        if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
            logging.info("\nConfusion Matrix:")
            cm = confusion_matrix(y_true, y_pred)
            logging.info(cm)
            logging.info("\nClassification Report:")
            cr = classification_report(y_true, y_pred)
            logging.info(cr)
        else:
            logging.warning("Cannot compute confusion matrix or classification report - not enough class variety")

        # Save model checkpoint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(
            model.state_dict(),
            f"model_checkpoints/model_round_{server_round}_acc_{accuracy:.2f}_{timestamp}.pt"
        )

        # Merge binary metrics into the result
        result_metrics = {
            "accuracy": accuracy,
            "mean_confidence": mean_confidence,
            "pred_distribution": str(pred_distribution),
            **binary_metrics  # Add binary metrics to the results
        }

        return loss, result_metrics

    return evaluate


def main():
    # Load data for server-side evaluation
    client_data, num_classes = load_data(
        img_dir="D:/FYP_Data/combined_images",
        labels_path="D:/FYP_Data/cleaned_valid_image_labels.csv"
    )
    _, test_loader = client_data[0]  # Use first client's test data for server evaluation

    # Define strategy with more rounds and configurable epochs
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # Sample 100% of clients for training
        fraction_evaluate=1.0,  # Sample 100% of clients for evaluation
        min_fit_clients=3,  # All 3 clients should participate in training
        min_evaluate_clients=3,  # All 3 clients should participate in evaluation
        min_available_clients=3,
        evaluate_fn=get_evaluate_fn(test_loader, num_classes),  # Pass the evaluation function
        on_fit_config_fn=lambda server_round: {
            "epochs": min(5, 1 + server_round)  # Increase epochs with rounds, up to 5
        }
    )

    # Start server with more rounds
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=5),  # Increase to 5 rounds
        strategy=strategy
    )


if __name__ == "__main__":
    main()