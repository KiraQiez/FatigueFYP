import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import mediapipe as mp
from scipy.ndimage import gaussian_filter
from PIL import Image

# Configuration
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, '..','model','fatigue_fusion_model_generator_80.h5')
FACE_SIZE = 256
VIDEO_W, VIDEO_H = 320, 240
THRESHOLD = 0.6
CONF_MIN = 0.6
DURATION = 2.0
THUMB_SIZE = 64
CONSISTENCY = 0.8

# Load model (expects 936-dim landmark input)
model = tf.keras.models.load_model(MODEL_PATH)

# MediaPipe setup matching offline dataset preprocessing
mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
mp_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

recent_uploads = []
thumbnails = []



@app.route('/')
def index():
    return render_template('index.html', uploads=recent_uploads, thumbs=thumbnails)



if __name__ == '__main__':
    app.run(debug=True)