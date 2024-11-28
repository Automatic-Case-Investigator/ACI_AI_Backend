import subprocess
import socket
import os

HOST = '0.0.0.0'
PORT = 5555
MONITORING_DIR = "/root/gguf_shared/"
TARGET_EXTENSION = ".gguf"

def load_models():
    for path in os.listdir(MONITORING_DIR):
        basename = os.path.basename(path)
        filename, extension = os.path.splitext(basename)
        if os.path.isfile(MONITORING_DIR + path) and extension == TARGET_EXTENSION:
            print(f"Creating model {filename}")
            subprocess.run(["ollama", "create", filename, "-f", f"/root/modelfiles/{filename}"], capture_output=True, text=True)

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen()

        while True:
            client_socket, client_address = server_socket.accept()
            with client_socket:
                while True:
                    data = client_socket.recv(1024)
                    if not data:
                        break

                    decoded_data = data.decode().strip()
                    path = MONITORING_DIR + decoded_data + TARGET_EXTENSION

                    if os.path.exists(path) and os.path.isfile(path):
                        print(f"Creating model {decoded_data}")
                        subprocess.run(["ollama", "create", decoded_data, "-f", f"/root/modelfiles/{decoded_data}"], capture_output=True, text=True)
                    else:
                        print(f"Deleting model {decoded_data}")
                        subprocess.run(["ollama", "rm", decoded_data], capture_output=True, text=True)
                    
                    client_socket.sendall(b"Success")

if __name__ == "__main__":
    load_models()
    start_server()
