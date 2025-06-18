import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
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

def preprocess_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    det = mp_face.process(rgb)
    if not det.detections:
        return None, None, None

    bb = det.detections[0].location_data.relative_bounding_box
    h, w = frame.shape[:2]
    x1 = max(int(bb.xmin * w), 0)
    y1 = max(int(bb.ymin * h), 0)
    x2 = min(int((bb.xmin + bb.width) * w), w)
    y2 = min(int((bb.ymin + bb.height) * h), h)
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None, None, None

    face_resized = cv2.resize(face, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    gray_input = gray.reshape(1, FACE_SIZE, FACE_SIZE, 1).astype(np.float32) / 255.0

    mesh = mp_mesh.process(cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB))
    if not mesh.multi_face_landmarks:
        return None, None, None

    landmarks = mesh.multi_face_landmarks[0]
    dotmap = np.zeros((FACE_SIZE, FACE_SIZE), dtype=np.float32)
    coords = []

    for lm in landmarks.landmark:
        coords.append([lm.x, lm.y])
        px = int(lm.x * (FACE_SIZE - 1))
        py = int(lm.y * (FACE_SIZE - 1))
        if 0 <= px < FACE_SIZE and 0 <= py < FACE_SIZE:
            dotmap[py, px] = 1.0

    dotmap = gaussian_filter(dotmap, sigma=2)
    maxv = dotmap.max()
    dotmap = dotmap / maxv if maxv > 0 else dotmap
    dot_input = dotmap.reshape(1, FACE_SIZE, FACE_SIZE, 1).astype(np.float32)

    landmark_input = np.array(coords).flatten().reshape(1, -1).astype(np.float32)

    return gray_input, dot_input, landmark_input

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    file = request.files.get('video')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    cap = cv2.VideoCapture(filepath)
    results = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        face_in, dot_in, lmk_in = preprocess_frame(frame)
        if face_in is not None:
            pred = float(model.predict([face_in, dot_in, lmk_in])[0][0])
            conf = pred if pred > THRESHOLD else 1 - pred
            label = "Not Fatigued" if pred > THRESHOLD else "Fatigued"
            results.append({'label': label, 'confidence': round(conf * 100, 2)})
    cap.release()

    if not results:
        return jsonify({'error': 'No face detected'}), 200

    return jsonify(results[-1])

if __name__ == '__main__':
    app.run(debug=True)
