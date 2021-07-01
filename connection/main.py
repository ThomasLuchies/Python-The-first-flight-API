import socket
import json

local_port = 9000
tello_drone_ip = "192.168.10.1"
tello_drone_port = 8889
connection_timeout_in_seconds = 5.0

print(local_port)
print(tello_drone_ip)
print(tello_drone_port)
print(connection_timeout_in_seconds)


def send_command(command:str) -> str:
    if local_port and tello_drone_ip and tello_drone_port and connection_timeout_in_seconds:
        conn = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        conn.bind(('', local_port))
        conn.connect((tello_drone_ip, tello_drone_port))
        conn.settimeout(connection_timeout_in_seconds)
        conn.send(command.encode('ASCII'))
        resp = conn.recv(1024)
        return resp.decode(errors='replace').strip()

    print("Invalid configuration.")
    return "Error"

send_command("takeoff")