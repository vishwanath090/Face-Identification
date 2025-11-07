# 🧠 Face Identification using Deep Learning

This repository contains **two distinct face identification approaches**, both trained and evaluated on a **custom dataset of faces** — including classes such as **Karthik**, **Vishwanath**, **Jafar**, and **Unknown**.

---

## 🔹 1. EfficientNet-B0 Based Face Identification

A **Convolutional Neural Network (CNN)**–based model using **EfficientNet-B0**, fine-tuned on a small, custom dataset for multi-class face identification.

### 🧩 Key Highlights:
- ✅ **Model:** EfficientNet-B0 (pretrained on ImageNet, fine-tuned on custom dataset)
- 🎯 **Task:** Face Identification (Multi-Class Classification)
- ⚙️ **Augmentations:** Random crop, rotation, blur, color jitter, horizontal flip, random erasing
- 📈 **Optimizer:** AdamW with OneCycleLR scheduler for smooth convergence
- 💾 **Loss:** CrossEntropy with label smoothing
- 🧠 **Framework:** PyTorch
- ⚡ **Confidence Score:** > 60%
- 👤 **Purpose:** Identify which known person (among trained classes) appears in the uploaded image

---

## 🔹 2. MTCNN + Face Embedding + Logistic Regression

A **hybrid approach** combining a deep embedding extractor with a lightweight logistic regression classifier for high-accuracy recognition.

### 🧩 Key Highlights:
- 🔍 **Face Detection:** MTCNN (Multi-task Cascaded Convolutional Networks)
- 🧬 **Embedding Model:** 512-dimensional face embeddings extracted using a pretrained deep network (FaceNet / Inception-ResNet)
- 🧮 **Classifier:** Logistic Regression trained on embeddings for identity classification
- ⚡ **Confidence Score:** > 80%
- 🚀 **Advantages:** Robust against lighting, pose, and background variations
- 💡 **Ideal for:** Small datasets — avoids retraining large CNNs by leveraging embeddings

---

## 📊 Performance Summary

| Model | Detection | Embedding / Architecture | Classifier | Confidence | Use Case |
|--------|------------|--------------------------|-------------|-------------|-----------|
| **EfficientNet-B0** | YOLOv8 / Manual crop | CNN feature extractor | Softmax | >60% | General face recognition |
| **MTCNN + Logistic Regression** | MTCNN | 512-D FaceNet embedding | Logistic Regression | >80% | Lightweight, accurate verification |

---

## 🧾 Technologies Used

- 🐍 **Python**, **PyTorch**, **torchvision**
- ⚙️ **EfficientNet-B0** (ImageNet Pretrained)
- 👁️ **MTCNN** for face detection  
- 📊 **Scikit-learn** Logistic Regression  
- 💻 **Streamlit** for deployment and image upload interface  
- 📈 **Matplotlib** / **Seaborn** for performance visualization  

---

## 🧩 Applications
- ✅ Identity verification  
- 🕵️ Attendance systems  
- 🔐 Personalized access control  
- 🧑‍💼 Small-scale recognition and authentication systems  

---

## 🚀 Deployment

A **Streamlit web application** is included for interactive face identification.

### Features:
- 🖼️ Upload an image to classify the person  
- 📊 Displays **original and predicted** images side-by-side  
- 🎯 Shows **predicted class** and **confidence score**  
- ⚡ Real-time inference using the trained EfficientNet-B0 model  

---

## 📂 Project Structure

face-identification/
│
├── dataset/
│ ├── train/
│ │ ├── karthik/
│ │ ├── vishwanath/
│ │ ├── jafar/
│ │ └── unknown/
│ └── val/
│
├── model/
│ ├── best_efficientnetb0_faceid.pth
│ └── best_mtcnn_logreg.pkl
│
├── app.py # Streamlit deployment file
├── train_efficientnet.py # Training script for EfficientNet-B0
├── inference.py # Image testing / prediction script
└── README.md
