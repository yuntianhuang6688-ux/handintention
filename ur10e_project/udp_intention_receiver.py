import socket

UDP_IP = "0.0.0.0"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening for human intention on UDP port {UDP_PORT}...")

while True:
    data, addr = sock.recvfrom(1024)
    message = data.decode("utf-8").strip()

    print(f"Received from {addr}: {message}")
