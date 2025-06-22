import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from tkinter import Tk, Frame, Label, Button, filedialog
from PIL import Image, ImageTk
from scipy.ndimage import gaussian_filter
import threading
import time
from collections import deque

# Configuration
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR,'..','model', 'fatigue_fusion_model_generator_80.h5')
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
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
connections = mp.solutions.face_mesh.FACEMESH_TESSELATION


def preprocess_frame(frame):
    # Convert to RGB & detect face
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    det = mp_face.process(rgb)
    if not det.detections:
        return None, None, None, None, None

    # Crop to detected bounding box (no margin)
    bb = det.detections[0].location_data.relative_bounding_box
    h, w = frame.shape[:2]
    x1 = max(int(bb.xmin * w), 0)
    y1 = max(int(bb.ymin * h), 0)
    x2 = min(int((bb.xmin + bb.width) * w), w)
    y2 = min(int((bb.ymin + bb.height) * h), h)
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None, None, None, None, None

    # Resize & grayscale
    face_resized = cv2.resize(face, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    gray_input = gray.reshape(1, FACE_SIZE, FACE_SIZE, 1).astype(np.float32) / 255.0

    # Facial landmarks
    mesh = mp_mesh.process(cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB))
    if not mesh.multi_face_landmarks:
        return None, None, None, None, None

    landmarks = mesh.multi_face_landmarks[0]
    dotmap = np.zeros((FACE_SIZE, FACE_SIZE), dtype=np.float32)
    coords = []

    for lm in landmarks.landmark:
        coords.append([lm.x, lm.y])
        px = int(lm.x * (FACE_SIZE - 1))
        py = int(lm.y * (FACE_SIZE - 1))
        if 0 <= px < FACE_SIZE and 0 <= py < FACE_SIZE:
            dotmap[py, px] = 1.0

    # Smooth & normalize dotmap with sigma=2
    dotmap = gaussian_filter(dotmap, sigma=2)
    maxv = dotmap.max()
    dotmap = dotmap / maxv if maxv > 0 else dotmap
    dot_input = dotmap.reshape(1, FACE_SIZE, FACE_SIZE, 1).astype(np.float32)

    # Flatten all 468 landmarks (936 dims)
    landmark_input = np.array(coords).flatten().reshape(1, -1).astype(np.float32)

    return gray_input, dot_input, landmark_input, face_resized, landmarks


class FatigueVideoGUI:
    def __init__(self):
        self.root = Tk()
        self.root.title("Fatigue Detection")
        self.root.geometry("1000x700")

        ctrl = Frame(self.root)
        ctrl.pack(pady=5)
        Button(ctrl, text="Run Video", command=self.load_video).pack(side='left', padx=5)
        Button(ctrl, text="Stop", command=self.stop).pack(side='left')

        disp = Frame(self.root)
        disp.pack()
        self.video_label = Label(disp)
        self.video_label.grid(row=0, column=0, padx=10)
        self.mesh_label = Label(disp)
        self.mesh_label.grid(row=0, column=1, padx=10)

        info = Frame(self.root)
        info.pack(pady=5)
        self.current_label = Label(info, text="Current: -", font=("Arial",14))
        self.current_label.grid(row=0, column=0, padx=10)
        self.confidence_label = Label(info, text="Confidence: -", font=("Arial",14))
        self.confidence_label.grid(row=0, column=1, padx=10)
        self.condition_label = Label(info, text="Condition: Not Fatigued", font=("Arial",16,"bold"))
        self.condition_label.grid(row=0, column=2, padx=10)

        self.window_label = Label(self.root, text="Fatigue/2 sec: 0/0 (0% of 2sec)", font=("Arial",14))
        self.window_label.pack(pady=2)

        thumb_frame = Frame(self.root)
        thumb_frame.pack(pady=10)
        self.thumbs = []
        for i in range(10):
            f = Frame(thumb_frame)
            f.grid(row=i//5, column=i%5, padx=5, pady=5)
            img_lbl = Label(f)
            img_lbl.pack()
            txt_lbl = Label(f, text="", font=("Arial",8))
            txt_lbl.pack()
            self.thumbs.append((img_lbl, txt_lbl))

        self.cap = None
        self.playing = False
        self.per_second_counts = deque(maxlen=int(DURATION))
        self.fatigued_counter = 0
        self.frame_index = 0
        self.accepted = deque(maxlen=10)

    def load_video(self):
        path = filedialog.askopenfilename(filetypes=[("Videos","*.mp4 *.avi *.mov")])
        if path:
            self.stop()
            self.cap = cv2.VideoCapture(path)
            self.playing = True
            threading.Thread(target=self.process, daemon=True).start()

    def process(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        fps_int = int(fps)
        window_frames = fps_int * int(DURATION)
        delay = 1.0 / fps

        while self.playing and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            face_input, dot_input, lmk_input, face_img, landmarks = preprocess_frame(frame)
            label, conf = "None", 0.0

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize((VIDEO_W, VIDEO_H))
            self.video_photo = ImageTk.PhotoImage(img)
            self.video_label.config(image=self.video_photo)
            self.video_label.image = self.video_photo

            if face_input is not None:
                pred = float(model.predict([face_input, dot_input, lmk_input])[0][0])
                if pred > THRESHOLD:
                    label, conf = "Not Fatigued", pred
                else:
                    label, conf = "Fatigued", 1 - pred
                    if conf >= CONF_MIN:
                        self.fatigued_counter += 1

                if conf >= CONF_MIN:
                    annotated = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    mp_drawing.draw_landmarks(
                        image=annotated,
                        landmark_list=landmarks,
                        connections=connections,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1),
                        connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style()
                    )
                    thumb = Image.fromarray(annotated).resize((THUMB_SIZE, THUMB_SIZE), Image.NEAREST)
                    self.accepted.append((thumb, label, conf))
                    self.update_thumbnails()

                    dotmap_display = (dot_input[0] * 255).astype('uint8')
                    dotmap_rgb = cv2.cvtColor(dotmap_display, cv2.COLOR_GRAY2RGB)
                    dotmap_img = Image.fromarray(dotmap_rgb).resize((VIDEO_W, VIDEO_H), Image.NEAREST)
                    self.dotmap_photo = ImageTk.PhotoImage(dotmap_img)
                    self.mesh_label.config(image=self.dotmap_photo)
                    self.mesh_label.image = self.dotmap_photo

            self.frame_index += 1
            if self.frame_index % fps_int == 0:
                self.per_second_counts.append(self.fatigued_counter)
                self.fatigued_counter = 0
                total = sum(self.per_second_counts)
                ratio = total / window_frames if window_frames else 0
                condition = "Fatigued" if ratio >= CONSISTENCY else "Not Fatigued"
                self.condition_label.config(text=f"Condition: {condition}")
                self.window_label.config(text=f"Fatigue/2 sec: {total}/{window_frames} ({int(ratio*100)}% of 2sec)")

            self.current_label.config(text=f"Current: {label}")
            self.confidence_label.config(text=f"Confidence: {conf:.2f}")
            self.root.update_idletasks()
            self.root.update()
            time.sleep(delay)

        self.stop()

    def update_thumbnails(self):
        for idx, (img_lbl, txt_lbl) in enumerate(self.thumbs):
            if idx < len(self.accepted):
                thumb_img, lbl, cf = self.accepted[idx]
                photo = ImageTk.PhotoImage(thumb_img)
                img_lbl.config(image=photo)
                img_lbl.image = photo
                txt_lbl.config(text=f"{lbl}\n{cf:.2f}")
            else:
                img_lbl.config(image=None)
                txt_lbl.config(text='')

    def stop(self):
        self.playing = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    FatigueVideoGUI().run()