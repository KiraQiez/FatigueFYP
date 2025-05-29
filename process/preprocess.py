import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

# === Configuration ===
INPUT_DIRS = {
    'Drowsy':    'DDDFull/Drowsy',
    'NonDrowsy': 'DDDFull/NonDrowsy'
}
OUTPUT_DIR = 'Dataset/ProcessedFusionFull'
FACE_SIZE = 256

# === MediaPipe Setup ===
mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)
mp_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True   
)

# Create output folders
for cls in INPUT_DIRS:
    for sub in ['face', 'dotmap', 'landmark']:
        os.makedirs(os.path.join(OUTPUT_DIR, cls, sub), exist_ok=True)

# Processing Loop
for label, in_dir in INPUT_DIRS.items():
    out_face = os.path.join(OUTPUT_DIR, label, 'face')
    out_dot  = os.path.join(OUTPUT_DIR, label, 'dotmap')
    out_lmk  = os.path.join(OUTPUT_DIR, label, 'landmark')

    for fname in tqdm(os.listdir(in_dir), desc=f'Processing {label}'):
        fpath = os.path.join(in_dir, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue

        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        det = mp_face.process(rgb)
        if not det.detections:
            continue

        # Crop face
        bb = det.detections[0].location_data.relative_bounding_box
        x1 = max(int(bb.xmin * w), 0)
        y1 = max(int(bb.ymin * h), 0)
        x2 = min(int((bb.xmin + bb.width) * w), w)
        y2 = min(int((bb.ymin + bb.height) * h), h)
        face = img[y1:y2, x1:x2]
        if face.size == 0:
            continue

        # Prepare face
        face_resized = cv2.resize(face, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_LINEAR)
        gray_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

        # Face mesh (now refined)
        mesh = mp_mesh.process(cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB))
        if not mesh.multi_face_landmarks:
            continue

        landmark_all = []
        dotmap = np.zeros((FACE_SIZE, FACE_SIZE), dtype=np.float32)

        # Collect all points (now 478) and build dotmap
        for lm in mesh.multi_face_landmarks[0].landmark:
            landmark_all.append([lm.x, lm.y])
            px = int(lm.x * (FACE_SIZE - 1))
            py = int(lm.y * (FACE_SIZE - 1))
            if 0 <= px < FACE_SIZE and 0 <= py < FACE_SIZE:
                dotmap[py, px] = 1.0

        landmark_all = np.array(landmark_all)   # shape (478,2)
        dotmap = gaussian_filter(dotmap, sigma=2)
        if dotmap.max() > 0:
            dotmap /= dotmap.max()

        base = os.path.splitext(fname)[0]
        # Save
        cv2.imwrite(os.path.join(out_face, base + '.png'),    gray_face)
        cv2.imwrite(os.path.join(out_dot,  base + '.png'),   (dotmap * 255).astype('uint8'))
        np.save(  os.path.join(out_lmk,  base + '.npy'),     landmark_all)
