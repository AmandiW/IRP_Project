import flwr as fl
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict


class RetinopathyClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device, client_id,
                 epochs=1, learning_rate=0.001, max_grad_norm=1.2,
                 noise_multiplier=1.0, delta=1e-5):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.client_id = client_id
        self.epochs = epochs

        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Initialize privacy engine
        self.privacy_engine = PrivacyEngine()

        self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm
        )

        self.criterion = nn.BCEWithLogitsLoss()

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        for epoch in range(self.epochs):
            with BatchMemoryManager(
                    data_loader=self.train_loader,
                    max_physical_batch_size=32,
                    optimizer=self.optimizer
            ) as memory_safe_loader:

                for data, target in memory_safe_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    self.optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.criterion(output.squeeze(), target)
                    loss.backward()
                    self.optimizer.step()

        # Get privacy spent
        epsilon = self.privacy_engine.get_epsilon(delta=1e-5)
        print(f"Client {self.client_id} - Privacy spent: (ε = {epsilon:.2f}, δ = {1e-5})")

        return self.get_parameters(config={}), len(self.train_loader.dataset), {"epsilon": epsilon}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss = 0
        correct = 0
        total = 0

        y_true = []
        y_pred = []

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += self.criterion(output.squeeze(), target).item()

                pred = torch.sigmoid(output.squeeze()) > 0.5
                y_true.extend(target.cpu().numpy())
                y_pred.extend(pred.cpu().numpy())

                correct += (pred == target).sum().item()
                total += target.size(0)

        accuracy = correct / total

        # Calculate metrics
        conf_matrix = confusion_matrix(y_true, y_pred)
        class_report = classification_report(y_true, y_pred)

        print(f"\nClient {self.client_id} Evaluation Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nConfusion Matrix:")
        print(conf_matrix)
        print("\nClassification Report:")
        print(class_report)

        return loss, total, {"accuracy": accuracy}