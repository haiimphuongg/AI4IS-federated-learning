from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.client import start_client

from utils import *

DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU


print(f"Training on {DEVICE}")
print(f"Flower {flwr.__version__} / PyTorch {torch.__version__}")


class FlowerClient(NumPyClient):
    def __init__(self, partition_id, net, dataset_path, lr=0.01, weight_decay=0.0001, num_epochs=10):
        self.partition_id = partition_id
        self.net = net
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        
        full_dataset = load_dataset_from_csv(dataset_path)
        
        train_dataset, val_dataset = train_val_split(full_dataset)

        trainloader = DataLoader(train_dataset, batch_size=128, shuffle=False)
        valloader = DataLoader(val_dataset, batch_size=128, shuffle=False)

        self.trainloader = trainloader
        self.valloader = valloader

    def get_weights(self, config):
        print(f"[Client {self.partition_id}] get_weights")
        return get_weights(self.net)

    def fit(self, parameters, config):
        print(f"[Client {self.partition_id}] fit, config: {config}")
        set_weights(self.net, parameters)
        result = train(self.net, self.trainloader, self.valloader, num_epoch=self.num_epochs, lr=self.lr, weight_decay=self.weight_decay)
        return get_weights(self.net), len(self.trainloader), result 

    def evaluate(self, parameters, config):
        print(f"[Client {self.partition_id}] evaluate, config: {config}")
        set_weights(self.net, parameters)
        avg_val_loss, all_val_labels, all_val_predictions = validate(self.net, self.valloader)
        accuracy = accuracy_score(all_val_predictions, all_val_labels)
        return float(avg_val_loss), len(self.valloader), {"accuracy": float(accuracy)}


def main(client_id, num_epochs, lr, weight_decay):
    dataset_path = f"../data/client{client_id}_dataset.csv"

    softmax_regression = SoftmaxRegression(512, 10).to(DEVICE)    
    start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(client_id, softmax_regression, dataset_path, lr=lr, weight_decay=weight_decay, num_epochs=num_epochs).to_client(),
    )

# Legacy mode
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run federated learning client with configurable parameters.")
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080", help="Server address (default: 127.0.0.1:8080)")
    parser.add_argument("--client_id", type=int, choices=[1, 2, 3], help="Client ID (1, 2, or 3)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay for optimizer (default: 0.0001)")
    parser.add_argument("--num_epochs", type=int, default=10, help="number of model training epochs each round (default: 10)")
    args = parser.parse_args()

    main(args.client_id, args.num_epochs, args.lr, args.weight_decay)
