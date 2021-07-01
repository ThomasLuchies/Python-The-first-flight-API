import argparse
import cv2
import torch
import torch.backends.cudnn as cudnn
import time
import json
from pathlib import Path
from flask import Flask, render_template, send_file
from flask_socketio import SocketIO, emit
from threading import Lock
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"
socket = SocketIO(app)
thread = None
thread_lock = Lock()

@app.route("/Direction", methods=["get"])
def index():
    with open("/shared-volume/direction.json", 'r') as j:
        data = json.loads(j.read())
    print(data)
    return data

@app.route("/Stream", methods=["get"])
def test_message():
    return send_file('/shared-volume/stream.png', mimetype='image/png')

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug = True)