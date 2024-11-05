'''

LAB 2 – HỌC LIÊN KẾT (FEDERATE LEARNING) VÀ BẢO MẬT TẠI SERVER

Nhóm 11:
    21120143 - Vũ Minh Thư
    21120312 - Phan Nguyên Phương

'''

import socket
import argparse
import threading
import pickle

import pandas as pd
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from sklearn.metrics import classification_report

from preprocess import create_dataloader
from model import Classifier, validate

IP = socket.gethostbyname(socket.gethostname())
PORT = 5566
ADDR = (IP, PORT)
SIZE = 1024
FORMAT = "utf-8"
DISCONNECT_MSG = "!DISCONNECT"

# Shared state
list_weights = []
clients = []
lock = threading.Lock()
disconnect_cnt = 0

def test(model, dataloader):
    
    avg_loss, all_labels, all_predictions = validate(model, dataloader)

    print(f'Average loss on test set: {avg_loss}')
    print(classification_report(all_labels, all_predictions, zero_division = True))

    return

def handle_client(conn, addr, model, test_dataloader):
    global disconnect_cnt
    print(f"[NEW CONNECTION] {addr} connected.")
    
    averaged_weights = {}
    connected = True
    while connected:
        # Read the length of the incoming message first (4 bytes)
        data_length_bytes = conn.recv(4)
        if not data_length_bytes:
            break
        data_length = int.from_bytes(data_length_bytes, 'big')

        # Now read the actual data based on the specified length
        data = conn.recv(data_length)
        
        if data == DISCONNECT_MSG.encode(FORMAT):
            connected = False
        else:
            weights = pickle.loads(data)  # Deserialize weights

            # Store the message and client in a thread-safe manner
            with lock:
                list_weights.append(weights)
                clients.append(conn)

                # If we have received weights from 3 clients, concatenate and send back
                if len(list_weights) == 3:
                    averaged_weights = {}
                    for key in list_weights[0].keys():
                        averaged_weights[key] = sum(d[key] for d in list_weights) / 3
                    
                    # Serialize averaged weights to send back to clients
                    serialized_averaged_weights = pickle.dumps(averaged_weights)
                    serialized_averaged_length = len(serialized_averaged_weights).to_bytes(4, 'big')
                    for client in clients:
                        client.sendall(serialized_averaged_length)       # Send length first
                        client.sendall(serialized_averaged_weights)      # Then send data
                    
                    # Clear lists for next round
                    list_weights.clear()
                    clients.clear()

    conn.close()
    print(f"[DISCONNECTED] {addr} disconnected.")

    with lock:
        disconnect_cnt += 1
        if disconnect_cnt % 3 == 0:
            model.load_state_dict(averaged_weights)
            test(model, test_dataloader)

def main(model, test_dataloader):
    print("[STARTING] Server is starting...")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(ADDR)
    server.listen()
    print(f"[LISTENING] Server is listening on {IP}:{PORT}")

    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr, model, test_dataloader))
        thread.start()
        print(f"[ACTIVE CONNECTIONS] {threading.activeCount() - 1}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run federated learning server with configurable parameters.")

    parser.add_argument("method_extract", type=str, choices=['hog', 'cnn'], default='hog', help="Feature extraction method (default: 'hog')")

    args = parser.parse_args()

    model = None
    test_dataset_extracted = None
    if args.method_extract == 'hog':
        test_dataset_extracted = pd.read_csv('test_dataset_HOG.csv')
    else:
        test_dataset_extracted = pd.read_csv('test_dataset_CNN.csv')
    
    test_dataloader = create_dataloader(test_dataset_extracted, batch_size=256)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Classifier(test_dataset_extracted.shape[1] - 1, 10).to(device)
    # criterion = CrossEntropyLoss()

    main(model, test_dataloader)