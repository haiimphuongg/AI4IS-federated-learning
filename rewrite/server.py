from typing import List, Tuple

from flwr.common import Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg

from utils import *


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    examples = [num_examples for num_examples, _ in metrics]

    # Multiply accuracy of each client by number of examples used
    train_losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    train_accuracies = [num_examples * m["train_accuracy"] for num_examples, m in metrics]
    train_f1 = [num_examples * m["train_f1"] for num_examples, m in metrics]
    val_losses = [num_examples * m["val_loss"] for num_examples, m in metrics]
    val_accuracies = [num_examples * m["val_accuracy"] for num_examples, m in metrics]
    val_f1 = [num_examples * m["val_f1"] for num_examples, m in metrics]

    
    
    # Aggregate and return custom metric (weighted average)
    return {
        "train_loss": sum(train_losses) / sum(examples),
        "train_accuracy": sum(train_accuracies) / sum(examples),
        "train_f1": sum(train_f1) / sum(examples),
        "val_loss": sum(val_losses) / sum(examples),
        "val_accuracy": sum(val_accuracies) / sum(examples),
        "val_f1": sum(val_f1) / sum(examples),
    }

# Initialize model parameters
ndarrays = get_weights(SoftmaxRegression(512,10))
parameters = ndarrays_to_parameters(ndarrays)
test_dataset = load_dataset_from_csv("../data/test_dataset.csv")
test_dataloader = DataLoader(test_dataset, batch_size=256)

# Define custom strategy
class CustomFedAvg(FedAvg):
    def on_fit_end(self, parameters, rnd, **kwargs):
        
        print(f"Round {rnd} finished. Testing aggregated model on test set...")
        test_metrics = validate(model,test_dataloader)
        print(f"Test Metrics: {test_metrics}")
        
# Define strategy
strategy = FedAvg(
    fraction_fit=1.0,  # Select all available clients
    fraction_evaluate=0.0,  # Disable evaluation
    min_available_clients=2,
    fit_metrics_aggregation_fn=weighted_average,
    initial_parameters=parameters,
)


# Define config
config = ServerConfig(num_rounds=10)


# Legacy mode
if __name__ == "__main__":
    from flwr.server import start_server

    start_server(
        server_address="127.0.0.1:8080",
        config=config,
        strategy=strategy,
    )