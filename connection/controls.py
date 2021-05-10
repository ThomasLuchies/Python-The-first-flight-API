class controls:
    def takeoff():
        conn.send(b'takeoff')
        resp = conn.recv(1024)
        print(resp.decode(errors='replace').strip())

    def land():
        conn.send(b'land')
        resp = conn.recv(1024)
        print(resp.decode(errors='replace').strip())

    def up(centimeter):
        conn.send(b('up' + str(centimeter)))
        resp = conn.recv(1024)
        print(resp.decode(errors='replace').strip())

    def down(centimeter):
        conn.send(b('down' + str(centimeter)))
        resp = conn.recv(1024)
        print(resp.decode(errors='replace').strip())

    def left(centimeter):
        conn.send(b('left' + str(centimeter)))
        resp = conn.recv(1024)
        print(resp.decode(errors='replace').strip())

    def right(centimeter):
        conn.send(b('right' + str(centimeter)))
        resp = conn.recv(1024)
        print(resp.decode(errors='replace').strip())

    def cw(degrees):
        conn.send(b('cw' + str(degrees)))
        resp = conn.recv(1024)
        print(resp.decode(errors='replace').strip())

    def ccw(degrees):
        conn.send(b('ccw' + str(degrees)))
        resp = conn.recv(1024)
        print(resp.decode(errors='replace').strip())