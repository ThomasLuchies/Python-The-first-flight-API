import socket

def stream():
    conn = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    conn.bind(('', 9000))
    conn.connect(('192.168.10.1', 8889))
    conn.settimeout(5.0)
    conn.send(b'command')
    resp = conn.recv(1024)
    print(resp.decode(errors='replace').strip())
    conn.send(b'streamon')
    resp = conn.recv(1024)
    print ("yeet")