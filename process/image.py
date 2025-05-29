import os
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from tkinter import Tk, Frame, Label, Button, filedialog, messagebox
from PIL import Image, ImageTk
from scipy.ndimage import gaussian_filter

# Configuration
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'fatigue_fusion_model_generator.h5')
FACE_SIZE = 256
DISPLAY_W, DISPLAY_H = 320, 240
THRESHOLD = 0.5
CONF_MIN = 0.5

# Load fusion model
eval_model = tf.keras.models.load_model(MODEL_PATH)

# MediaPipe setup
mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
mp_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
connections = mp.solutions.face_mesh.FACEMESH_TESSELATION


def preprocess_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    det = mp_face.process(rgb)
    if not det.detections:
        return None, None, None

    h, w = frame.shape[:2]
    bb = det.detections[0].location_data.relative_bounding_box
    x1 = max(int(bb.xmin * w), 0)
    y1 = max(int(bb.ymin * h), 0)
    x2 = min(int((bb.xmin + bb.width) * w), w)
    y2 = min(int((bb.ymin + bb.height) * h), h)
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        return None, None, None

    face_resized = cv2.resize(face, (FACE_SIZE, FACE_SIZE))
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
    if dotmap.max() > 0:
        dotmap /= dotmap.max()
    dot_input = dotmap.reshape(1, FACE_SIZE, FACE_SIZE, 1).astype(np.float32)

    landmark_input = np.array(coords).flatten().reshape(1, -1).astype(np.float32)
    return gray_input, dot_input, landmark_input


class FatigueImageGUI:
    def __init__(self):
        self.root = Tk()
        self.root.title("Fatigue Detection - Image")
        self.root.geometry("800x600")

        ctrl = Frame(self.root)
        ctrl.pack(pady=10)
        Button(ctrl, text="Load Image", command=self.load_image).pack(side='left', padx=5)

        disp = Frame(self.root)
        disp.pack()
        self.image_label = Label(disp)
        self.image_label.grid(row=0, column=0, padx=10)
        self.dot_label = Label(disp)
        self.dot_label.grid(row=0, column=1, padx=10)

        info = Frame(self.root)
        info.pack(pady=10)
        self.current_label = Label(info, text="Label: -", font=("Arial",14))
        self.current_label.grid(row=0, column=0, padx=10)
        self.confidence_label = Label(info, text="Confidence: -", font=("Arial",14))
        self.confidence_label.grid(row=0, column=1, padx=10)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images","*.jpg *.png *.bmp")])
        if not path:
            return
        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Error", "Cannot read image file.")
            return

        result = preprocess_frame(frame)
        if result[0] is None:
            messagebox.showwarning("No Face", "No face detected in the image.")
            return

        gray_in, dot_in, lmk_in = result
        pred = float(eval_model.predict([gray_in, dot_in, lmk_in])[0][0])
        if pred > THRESHOLD:
            label, conf = "Not Fatigued", pred
        else:
            label, conf = "Fatigued", 1 - pred

        # Display cropped face
        face_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(face_rgb).resize((DISPLAY_W, DISPLAY_H))
        self.photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.photo)
        self.image_label.image = self.photo

        # Display dotmap
        dotmap_disp = (dot_in[0] * 255).astype('uint8')
        dot_rgb = cv2.cvtColor(dotmap_disp, cv2.COLOR_GRAY2RGB)
        dot_img = Image.fromarray(dot_rgb).resize((DISPLAY_W, DISPLAY_H))
        self.dot_photo = ImageTk.PhotoImage(dot_img)
        self.dot_label.config(image=self.dot_photo)
        self.dot_label.image = self.dot_photo

        self.current_label.config(text=f"Label: {label}")
        self.confidence_label.config(text=f"Confidence: {conf:.2f}")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    FatigueImageGUI().run()