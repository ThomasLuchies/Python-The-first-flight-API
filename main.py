from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from threading import Lock

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"
socket = SocketIO(app)
thread = None
thread_lock = Lock()

@app.route("/onno", methods=["get"])
def index():
    return render_template("index.html", hallo="Welkom op de website!")

@socket.on("my_event")
def test_message():
    emit("my_response", {"data": "Connected."})

if __name__ == "__main__":
    socket.run(app, max_size=1024)

# AI-data post
# Student fetch/delete
# Drone-config update

# AI-frame post (Websocket)
# AI-frame get (Websocket)