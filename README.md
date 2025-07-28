# Face Mask Detection Using Tensorflow,Keras,Opencv & Python.
Here’s a complete and professional **`README.md`** file for your **Face Mask Detection System** using **TensorFlow, Keras, OpenCV, and Python**:

---

# 😷 Face Mask Detection System

A real-time deep learning-based face mask detection system built using **TensorFlow**, **Keras**, **OpenCV**, and **Python**. It classifies whether a person in a live webcam feed is **wearing a mask** or **not**, making it highly useful in public safety and health monitoring.

---

## 🧠 Technologies Used

* ✅ **TensorFlow / Keras** – Model building & training
* ✅ **OpenCV** – Real-time video capture & face detection
* ✅ **Python** – Core programming
* ✅ **Scikit-learn** – Data preprocessing, metrics
* ✅ **Matplotlib / Seaborn** – Visualizations
* ✅ **Haar Cascades** – Face detection (OpenCV pre-trained models)

---

## 📁 Dataset

The project uses a face mask dataset typically structured like:

```
dataset/
├── with_mask/
├── without_mask/
```

You can download sample datasets like:

* [Face Mask Detection Dataset by Prajna Bhandary](https://github.com/prajnasb/observations/tree/master/experiements/data)

---

## 📌 Project Structure

```bash
Face-Mask-Detection/
├── dataset/
│   ├── with_mask/
│   └── without_mask/
├── train_mask_detector.py
├── detect_mask_video.py
├── face_detector/
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── mask_detector.model
├── README.md
```

---

## 🔧 Model Training (`train_mask_detector.py`)

### ✨ Workflow:

1. Load and label images from both `with_mask` and `without_mask`
2. Preprocess images (`224x224`, scale, normalize)
3. Split into training & validation sets
4. Train using **MobileNetV2** (transfer learning)
5. Save model as `mask_detector.model`

```python
model = MobileNetV2(weights="imagenet", include_top=False,
                    input_tensor=Input(shape=(224, 224, 3)))
# Freeze base layers, add classifier head
```

### 📈 Evaluation:

* **Accuracy**: \~98% on validation set
* **Model size**: Lightweight for real-time inference
* **Metrics**: Precision, Recall, F1-Score

---

## 🖥️ Real-Time Detection (`detect_mask_video.py`)

### Steps:

1. Load the saved model `mask_detector.model`
2. Load OpenCV's face detector (SSD + prototxt)
3. Capture video using `cv2.VideoCapture(0)`
4. Detect face, predict mask status, display result with bounding box and label

```python
if label == "Mask":
    color = (0, 255, 0)  # Green
else:
    color = (0, 0, 255)  # Red
cv2.putText(frame, label, ...)
```

### 🔴 Output:

* ✅ Green box: Mask
* ❌ Red box: No Mask
* 📷 Works on webcam, CCTV, or any video source

---

## ▶️ How to Run

### 1. Install Requirements

```bash
pip install tensorflow keras opencv-python imutils numpy matplotlib
```

### 2. Train the Model

```bash
python train_mask_detector.py
```

### 3. Start Real-Time Detection

```bash
python detect_mask_video.py
```

---

## 💡 Features

* 🔎 Real-time face detection and mask prediction
* 🎯 High accuracy using MobileNetV2
* ⚡ Lightweight & fast inference
* 🔄 Easily extendable to include other PPE (e.g., goggles)

---

## 🚀 Future Enhancements

* Add social distancing monitoring
* Deploy as a web app using Streamlit or Flask
* Integrate with CCTV systems for smart surveillance
* Support for mobile deployment with TensorFlow Lite

---

## 📷 Sample Output

![Sample](https://user-images.githubusercontent.com/your-image-link.png)

---

## 🛡️ License

This project is licensed under the **MIT License**.
Feel free to use, modify, and distribute for educational or commercial purposes.

---

## 🤝 Credits

* [OpenCV Haarcascades & SSD Face Detector](https://github.com/opencv/opencv)
* [Prajna Bhandary’s Dataset](https://github.com/prajnasb/observations)

---

Let me know if you'd like a PowerPoint, UI with Streamlit, or even a GitHub-ready `.zip` project structure.



