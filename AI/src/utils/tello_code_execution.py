import socket
import json
import time

local_port = None
tello_drone_ip = None
tello_drone_port = None
timeout = None
s = None


local_port = 9000
tello_drone_ip = "192.168.10.1"
tello_drone_port = 8889
timeout = 5.0

def connect():
    conn = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    conn.bind(('', local_port))
    conn.connect((tello_drone_ip, tello_drone_port))
    conn.settimeout(timeout)
    conn.send(b'command')
    conn.send(b'streamon')
