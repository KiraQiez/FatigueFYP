import cv2
import os

# Load pre-trained Haar cascade face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# === Input Image Path ===
image_path = 'D:\FYP Coding\FatigueFYP\process\da.png'  

# Read the image
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

# Convert to grayscale for detection
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces using Viola-Jones
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display result
cv2.imshow('Viola-Jones Face Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
