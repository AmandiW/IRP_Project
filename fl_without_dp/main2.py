import flwr as fl
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import OrderedDict
import logging
import multiprocessing
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


# Dataset class
class RetinopathyDataset(Dataset):
    def __init__(self, img_dir, labels_df, transform=None):
        self.img_dir = img_dir
        self.labels_df = labels_df
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_id = self.labels_df.iloc[idx]['id_code']
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        image = Image.open(img_path).convert('RGB')
        label = self.labels_df.iloc[idx]['diagnosis']

        if self.transform:
            image = self.transform(image)

        return image, label


# Data loading function
def load_data(img_dir, labels_path, num_clients=3, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    labels_df = pd.read_csv(labels_path)
    client_dfs = []
    remaining_df = labels_df.copy()

    for i in range(num_clients - 1):
        client_size = len(remaining_df) // (num_clients - i)
        client_df = remaining_df.sample(n=client_size, random_state=42 + i)
        client_dfs.append(client_df)
        remaining_df = remaining_df.drop(client_df.index)

    client_dfs.append(remaining_df)

    client_data = []
    for client_df in client_dfs:
        train_df, test_df = train_test_split(client_df, test_size=0.2, random_state=42)

        train_dataset = RetinopathyDataset(img_dir, train_df, transform)
        test_dataset = RetinopathyDataset(img_dir, test_df, transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        client_data.append((train_loader, test_loader))

    return client_data


# Model creation function
def create_model():
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    return model


# Client implementation
class RetinopathyClient(fl.client.NumPyClient):
    def __init__(self, client_id, train_loader, test_loader):
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_model().to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.logger = logging.getLogger(f"Client_{client_id}")

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.logger.info("Starting training")
        self.set_parameters(parameters)

        self.model.train()
        for epoch in range(2):
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
                    self.logger.info(f"Epoch {epoch} - Batch {batch_idx} - Loss: {loss.item():.4f}")

            epoch_loss = running_loss / len(self.train_loader)
            epoch_accuracy = correct / total
            self.logger.info(f"Epoch {epoch} - Average Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.logger.info("Starting evaluation")
        self.set_parameters(parameters)

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

        self.logger.info(f"Evaluation Results:")
        self.logger.info(f"Loss: {loss:.4f}")
        self.logger.info(f"Accuracy: {accuracy:.4f}")
        self.logger.info("\nConfusion Matrix:")
        self.logger.info(confusion_matrix(y_true, y_pred))
        self.logger.info("\nClassification Report:")
        self.logger.info(classification_report(y_true, y_pred))

        return loss, len(self.test_loader.dataset), {"accuracy": accuracy}


# Server evaluation function
def get_evaluate_fn(test_loader):
    def evaluate(server_round, parameters, config):
        model = create_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)

        model.eval()
        loss = 0.0
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        criterion = torch.nn.CrossEntropyLoss()
        logger = logging.getLogger("Server")

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

        logger.info(f"\nServer-side evaluation - Round {server_round}")
        logger.info(f"Loss: {loss:.4f}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info("\nConfusion Matrix:")
        logger.info(confusion_matrix(y_true, y_pred))
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_true, y_pred))

        return loss, {"accuracy": accuracy}

    return evaluate


# Client process function
def start_client(client_id, client_data):
    train_loader, test_loader = client_data[client_id]
    client = RetinopathyClient(client_id, train_loader, test_loader)
    fl.client.start_numpy_client(server_address="[::]:8080", client=client)


# Server process function
def start_server(client_data):
    _, test_loader = client_data[0]

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        evaluate_fn=get_evaluate_fn(test_loader)
    )

    # Start server
    fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=2),
        strategy=strategy
    )


def main():
    # Load data
    client_data = load_data(
        img_dir="D:/FYP_Data/combined_images",
        labels_path="D:/FYP_Data/cleaned_valid_image_labels.csv"
    )

    # Create processes
    processes = []

    # Start server process
    server_process = multiprocessing.Process(target=start_server, args=(client_data,))
    processes.append(server_process)
    server_process.start()

    # Wait a bit for server to start
    time.sleep(3)

    # Start client processes
    for client_id in range(3):
        client_process = multiprocessing.Process(
            target=start_client,
            args=(client_id, client_data)
        )
        processes.append(client_process)
        client_process.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()


if __name__ == "__main__":
    # This is required for Windows
    multiprocessing.freeze_support()
    main()