'''

LAB 2 – HỌC LIÊN KẾT (FEDERATE LEARNING) VÀ BẢO MẬT TẠI SERVER

Nhóm 11:
    21120143 - Vũ Minh Thư
    21120312 - Phan Nguyên Phương

'''

import socket
import threading

IP = socket.gethostbyname(socket.gethostname())
PORT = 5566
ADDR = (IP, PORT)
SIZE = 1024
FORMAT = "utf-8"
DISCONNECT_MSG = "!DISCONNECT"

# Shared state
messages = []
clients = []
lock = threading.Lock()

def handle_client(conn, addr):
    print(f"[NEW CONNECTION] {addr} connected.")
    
    connected = True
    while connected:
        msg = conn.recv(SIZE).decode(FORMAT)
        if msg == DISCONNECT_MSG:
            connected = False
        else:
            print(f"[{addr}] {msg}")

            # Store the message and client in a thread-safe manner
            with lock:
                messages.append(msg)
                clients.append(conn)

                # If we have received messages from 3 clients, concatenate and send back
                if len(messages) == 3:
                    concatenated_msg = " | ".join(messages)
                    for client in clients:
                        client.send(f"Server response: {concatenated_msg}".encode(FORMAT))
                    
                    # Clear lists for next round
                    messages.clear()
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
