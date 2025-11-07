🧠 Face Identification using Deep Learning

This repository contains two distinct face identification approaches, both trained and evaluated on a custom dataset of faces (including classes like Karthik, Vishwanath, Jafar, and Unknown).

🔹 1. EfficientNet-B0 Based Face Identification

A convolutional neural network (CNN)–based model using EfficientNet-B0, fine-tuned for facial identity classification.

Key Highlights:

✅ Model: EfficientNet-B0 (pretrained on ImageNet, fine-tuned on custom dataset)

🎯 Task: Face Identification (Multi-Class Classification)

⚙️ Augmentations: Random crop, rotation, blur, color jitter, horizontal flip, random erasing

📈 Optimizer: AdamW with OneCycleLR scheduler for stable learning

💾 Loss: CrossEntropy with label smoothing

🧩 Framework: PyTorch

⚡ Confidence Score: > 60%

🧠 Used for identifying which known person (among 3–4 trained classes) appears in the uploaded image.

🔹 2. MTCNN + Face Embedding + Logistic Regression

A hybrid approach combining a deep embedding extractor with a lightweight classifier for higher precision.

Key Highlights:

🔍 Face Detection: MTCNN (Multi-task Cascaded Convolutional Networks)

🧬 Embedding Model: 512-dimensional face embeddings extracted using a pretrained deep network (e.g., FaceNet / Inception-ResNet)

🧮 Classifier: Logistic Regression trained on embeddings for identity classification

⚡ Confidence Score: > 80%

🚀 Faster and more robust against lighting and background variations

💡 Ideal for small datasets — uses embeddings instead of retraining a full CNN.

📊 Performance Summary
Model	Detection	Embedding / Architecture	Classifier	Confidence	Use Case
EfficientNet-B0	YOLOv8 / Manual crop	CNN feature extractor	Softmax	>60%	General face recognition
MTCNN + Logistic Regression	MTCNN	512-D FaceNet embedding	Logistic Regression	>80%	Lightweight, accurate verification
🧾 Technologies Used

Python, PyTorch, torchvision

EfficientNet-B0 (ImageNet Pretrained)

MTCNN for face detection

Scikit-learn Logistic Regression

Streamlit for deployment and image upload interface

Matplotlib / Seaborn for visualization

🧩 Applications

Identity verification

Attendance systems

Personalized access control

Small-scale recognition systems

🚀 Deployment

A Streamlit web app allows users to upload an image and get real-time predictions:

Displays both the original and predicted face label

Shows confidence score and class probabilities
