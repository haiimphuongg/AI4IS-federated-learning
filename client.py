'''

LAB 2 – HỌC LIÊN KẾT (FEDERATE LEARNING) VÀ BẢO MẬT TẠI SERVER

Nhóm 11:
    21120143 - Vũ Minh Thư
    21120312 - Phan Nguyên Phương

'''

import socket
import argparse
import pickle
import ast

import pandas as pd
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Adam

from preprocess import create_dataloader
from model import Classifier, train

IP = socket.gethostbyname(socket.gethostname())
PORT = 5566
ADDR = (IP, PORT)
SIZE = 1024
FORMAT = "utf-8"
DISCONNECT_MSG = "!DISCONNECT"

# def get_data(client_id=1, extract_method='hog'):
#     train_dataset_extracted, train_loader, val_loader = create_dataloader(client_id, extract_method)
#     return train_dataset_extracted, train_loader, val_loader

def client_train(model, train_loader, val_loader, lr, betas, weight_decay, num_epochs):
    torch.manual_seed(42)
    
    criterion = CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=lr,
        betas=betas,
        weight_decay=weight_decay,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode = "min", factor = 0.2, patience = 5)

    loss, weights = train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, show_step = 10)

    serialized_weights = pickle.dumps(weights)
    return serialized_weights


def main(client_id, method_extract, lr, betas, weight_decay, num_epochs, iterations):
    extension = 'dataset_HOG.csv' if method_extract == 'hog' else 'dataset_CNN.csv'
    train_filename = "client" + str(client_id) + "_train_" + extension
    val_filename = "client" + str(client_id) + "_val_" + extension

    train_dataset_extracted = pd.read_csv(train_filename)
    val_dataset_extracted = pd.read_csv(val_filename)
    train_dataloader = create_dataloader(train_dataset_extracted, batch_size=256)
    val_dataloader = create_dataloader(val_dataset_extracted, batch_size=256)
    
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(ADDR)
    print(f"[CONNECTED] Client connected to server at {IP}:{PORT}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Classifier(train_dataset_extracted.shape[1] - 1, 10).to(device) ##### CHECK AGAIN

    while True:
        # print(f'------ Loop {connected} -----')
        
        weights = client_train(model, train_dataloader, val_dataloader, lr, betas, weight_decay, num_epochs)
        
        if weights is None or iterations == 0:
            break
        else:
            weights_length = len(weights).to_bytes(4, 'big')  # 4-byte integer
            client.sendall(weights_length)                    # Send length first
            client.sendall(weights)                           # Send actual weights

        # Receive the averaged weights from the server
        response_length = int.from_bytes(client.recv(4), 'big')  # Receive length of response first
        response = client.recv(response_length)                  # Then receive the data based on length
        averaged_weights = pickle.loads(response)
        
        # Update model with new averaged weights
        model.load_state_dict(averaged_weights)

        # print(f"[SERVER SEND BACK] {averaged_weights}")
        
        iterations -= 1

    client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run federated learning client with configurable parameters.")
    
    parser.add_argument("client_id", type=int, choices=[1, 2, 3], help="Client ID (1, 2, or 3)")
    parser.add_argument("--method_extract", type=str, choices=['hog', 'cnn'], default='hog', help="Feature extraction method (default: 'hog')")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--betas", type=float, nargs='+', default=[0.9, 0.999], help="Betas for optimizer (default: 0.9 0.999)")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay for optimizer (default: 0.0001)")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs (default: 1)")
    parser.add_argument("--train_iterations", type=int, default=10, help="number of model training iterations (default: 10)")
    args = parser.parse_args()

    main(args.client_id, args.method_extract, args.lr, args.betas, args.weight_decay, args.num_epochs, args.train_iterations)

'''
Usage: 

    python client.py [client-id] --method_extract ['hog', 'cnn'] --lr [learning-rate] --betas [betas] --weight_decay [wd] --num_epochs [num] --train_iterations [num]

Example: 
    
    python client.py 1 --method_extract cnn --lr 0.01 --betas 0.9 0.999 --weight_decay 0.0005 --num_epochs 1 --train_iterations 10
    python client.py 2 --method_extract cnn --lr 0.01 --betas 0.9 0.999 --weight_decay 0.0005 --num_epochs 1 --train_iterations 10
    python client.py 3 --method_extract cnn --lr 0.01 --betas 0.9 0.999 --weight_decay 0.0005 --num_epochs 1 --train_iterations 10

    python client.py 2

    python client.py 1 --method_extract hog --lr 0.001 --betas 0.9 0.999 --weight_decay 0.0001 --num_epochs 1 --train_iterations 10
    python client.py 2 --method_extract hog --lr 0.001 --betas 0.9 0.999 --weight_decay 0.0001 --num_epochs 1 --train_iterations 10
    python client.py 3 --method_extract hog --lr 0.001 --betas 0.9 0.999 --weight_decay 0.0001 --num_epochs 1 --train_iterations 10
'''