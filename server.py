'''

LAB 2 – HỌC LIÊN KẾT (FEDERATE LEARNING) VÀ BẢO MẬT TẠI SERVER

Nhóm 11:
    21120143 - Vũ Minh Thư
    21120312 - Phan Nguyên Phương

'''

import socket
import threading
import pickle

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

def handle_client(conn, addr):
    print(f"[NEW CONNECTION] {addr} connected.")
    
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

def main():
    print("[STARTING] Server is starting...")
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(ADDR)
    server.listen()
    print(f"[LISTENING] Server is listening on {IP}:{PORT}")

    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()
        print(f"[ACTIVE CONNECTIONS] {threading.activeCount() - 1}")

if __name__ == "__main__":
    main()
