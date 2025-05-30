# FatigueFYP# 💤 FatigueFYP: Real-Time Fatigue Detection Using CNN + Facial Landmarks

Welcome to **FatigueFYP**, a smart fatigue detection system that leverages **Convolutional Neural Networks (CNNs)** and **facial landmark analysis** for real-time monitoring of driver drowsiness. This project aims to prevent accidents by providing an accurate and lightweight fatigue recognition pipeline based on visual cues.

--

## 🚀 Features

- 🎥 **Real-Time Face Detection** with MediaPipe
- 🧠 **Multi-Input Fusion Model** combining:
  - Grayscale Face Images
  - Facial Landmark Dotmaps
  - 478-Point Landmark Coordinates
- 📊 **Robust Performance**: Over **98% accuracy**
- 🖥️ **Tkinter GUI Interface** with dual video feed + condition display
- 🔄 **Temporal Consistency Logic** to avoid false alarms
- 💾 Offline & lightweight — no external server required

---

## 🧠 Model Architecture

**Fusion-based Neural Network**

- **Input 1**: CNN branch for 256×256 grayscale face images  
- **Input 2**: CNN branch for 256×256 facial landmark dotmaps  
- **Input 3**: MLP for 478-point (flattened) landmark vectors  
- **Output**: Fatigue Prediction (`Drowsy` / `NonDrowsy`)

All inputs are fused and trained together to form a multi-modal classifier with high generalization and precision.

---

## 🗃 Dataset

Preprocessed from **RLDD (Real-Life Drowsiness Dataset)**:

- Faces extracted using **Viola-Jones**
- Facial landmarks extracted using **MediaPipe**
- Dotmaps generated with Gaussian blur from landmark points
- Balanced dataset with:
  - 📁 `face/`
  - 📁 `dotmap/`
  - 📁 `landmark/` (`.npy` format)
  - 📄 `labels.csv`

---

## 📈 Performance Snapshot

| Metric       | Value     |
|--------------|-----------|
| Accuracy     | **98.83%** |
| Precision    | 99.51%    |
| Recall       | 98.52%    |
| F1 Score     | 99.01%    |

🌀 Model evaluated using real-world drowsy vs non-drowsy video frames.

---

## 💡 How It Works

1. Load a video (live or recorded)
2. Extract face and landmarks
3. Generate input vectors
4. Predict fatigue level with trained fusion model
5. Display results in real-time (including thumbnails for evidence)

---

## 🛠️ Requirements

```bash
pip install tensorflow opencv-python mediapipe pillow numpy
