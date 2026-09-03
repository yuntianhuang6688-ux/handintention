import socket
import time

WSL_IP = "172.26.149.220"   # 改成你 hostname -I 显示的 IP
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

targets = ["red", "blue", "green", "yellow"]

while True:
    for target in targets:
        message = target.encode("utf-8")
        sock.sendto(message, (WSL_IP, UDP_PORT))
        print(f"Sent: {target}")
        time.sleep(1.0)