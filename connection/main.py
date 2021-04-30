import socket
import controls
class main:
    __init__():
        conn = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        conn.bind(('', 9000))
        conn.connect(('192.168.10.1', 8889))
        conn.settimeout(5.0)
        controls = controls()
        convertCode()
    
    convertCode():


if(__name__ == '__main__'):
    main()