import flwr as fl
from typing import List, Tuple, Dict
import numpy as np
from collections import defaultdict


class FederatedServer:
    def __init__(self, num_rounds=5):
        self.num_rounds = num_rounds
        self.metrics_history = defaultdict(list)

    def weighted_average(self, metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
        # Aggregate metrics across all clients
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]

        return {"accuracy": sum(accuracies) / sum(examples)}

    def start_server(self):
        # Define strategy
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,  # Use all available clients for training
            fraction_evaluate=1.0,  # Use all available clients for evaluation
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
            evaluate_metrics_aggregation_fn=self.weighted_average,
        )

        # Start server
        fl.server.start_server(
            server_address="localhost:8080",
            config=fl.server.ServerConfig(num_rounds=self.num_rounds),
            strategy=strategy,
        )