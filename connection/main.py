import socket
import controls

if(__name__ == '__main__'):
    conn = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    conn.bind(('', 9000))
    conn.connect(('192.168.10.1', 8889))
    conn.settimeout(5.0)
    conn.send(b'land')
    resp = conn.recv(1024)
    print(resp.decode(errors='replace').strip())
