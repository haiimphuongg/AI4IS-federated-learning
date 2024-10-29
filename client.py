'''

LAB 2 – HỌC LIÊN KẾT (FEDERATE LEARNING) VÀ BẢO MẬT TẠI SERVER

Nhóm 11:
    21120143 - Vũ Minh Thư
    21120312 - Phan Nguyên Phương

'''

import socket
import random
import argparse

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam

from preprocess import get_dataloader
from model import Classifier, train

IP = socket.gethostbyname(socket.gethostname())
PORT = 5566
ADDR = (IP, PORT)
SIZE = 1024
FORMAT = "utf-8"
DISCONNECT_MSG = "!DISCONNECT"

def get_data(client_id=1, extract_method='hog'):
    train_dataset_extracted, train_loader, val_loader = get_dataloader(client_id, extract_method)
    return train_dataset_extracted, train_loader, val_loader

def client_train(train_dataset_extracted, train_loader, val_loader, lr, betas, weight_decay, num_epochs):
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Classifier(train_dataset_extracted[0][0].shape[0], 10).to(device)
    criterion = CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode = "min", factor = 0.2, patience = 5)

    loss, weights = train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, show_step = 10)

    for _ in range(5):
        print('Processing...')

    # if loss_ok:
    #   return None

    return " ".join(str(w) for _, w in weights.items())

def main(client_id, method_extract, lr, betas, weight_decay, num_epochs):
    train_dataset_extracted, train_dataloader, val_dataloader = get_data(client_id, method_extract)

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(ADDR)
    print(f"[CONNECTED] Client connected to server at {IP}:{PORT}")

    connected = 1
    while True:
        print(f'Loop {connected}')
        weights = client_train(train_dataset_extracted, train_dataloader, val_dataloader, lr, betas, weight_decay, num_epochs)
        
        print(len(weights))

        if weights is None or connected == 0:
            msg = "!DISCONNECT"
            client.close()
        else:
            msg = weights

        client.send(msg.encode(FORMAT))

        if msg == DISCONNECT_MSG:
            connected = False
        else:
            # Wait for the concatenated message from the server
            response = client.recv(SIZE).decode(FORMAT)
            print(f"[SERVER] {response}")
        
        connected -= 1

    client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run federated learning client with configurable parameters.")
    
    parser.add_argument("client_id", type=int, choices=[1, 2, 3], help="Client ID (1, 2, or 3)")
    parser.add_argument("--method_extract", type=str, choices=['hog', 'cnn'], default='hog', help="Feature extraction method (default: 'hog')")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs (default: 3)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.999), help="Betas for optimizer (default: (0.9, 0.999))")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay for optimizer (default: 0.0001)")

    args = parser.parse_args()

    main(args.client_id, args.method_extract, args.lr, args.betas, args.weight_decay, args.num_epochs)
