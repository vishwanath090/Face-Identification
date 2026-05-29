# Face Identification

A face identification project built around a FaceNet-based Streamlit application and notebook-based experiments on a custom face dataset.

## Overview

This repository contains two face identification workflows trained and evaluated on a custom dataset:

- A Streamlit app for uploading a face image and predicting the most likely identity
- Notebook experiments used to train and compare models

The app follows a simple pipeline:

1. Detect and align the face
2. Generate a 512-dimensional embedding
3. Classify the embedding
4. Reject the prediction when confidence is below a chosen threshold

## Features

- Face upload through a clean Streamlit interface
- Face detection and alignment with MTCNN
- FaceNet embeddings for identification
- Unknown face rejection with a configurable threshold
- Class probability display for model interpretation
- Support for GPU if available, with CPU fallback

## Repository Structure

```text
Face-Identification/
├── app.py
├── augment_faces.py
├── facenet.ipynb
├── effinetb0.ipynb
├── facenet_artifacts/
│   ├── clf.pkl
│   └── label_encoder.pkl
├── best_effnetb0_fixed.pth
├── requirements.txt
└── .gitignore
```

## Requirements

The project depends on Python and the packages listed in `requirements.txt`.  
The Streamlit app itself uses:

- streamlit
- torch
- facenet-pytorch
- pillow
- numpy
- scikit-learn

Some notebook experiments may require additional packages such as OpenCV, TensorFlow, InsightFace, or ONNX Runtime.

## Installation

Clone the repository:

```bash
git clone https://github.com/vishwanath090/Face-Identification.git
cd Face-Identification
```

Create and activate a virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

If you are running the Streamlit app and your environment does not already include the FaceNet stack, install the missing packages manually:

```bash
pip install streamlit torch facenet-pytorch pillow numpy scikit-learn
```

## Setup

Before running the app, make sure the following model artifacts are available:

- `facenet_artifacts/clf.pkl`
- `facenet_artifacts/label_encoder.pkl`

The current `app.py` uses a hardcoded local path for `ART_DIR`. Update that path so it points to the `facenet_artifacts` folder in your local clone.

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

Then:

1. Upload a face image
2. Adjust the unknown rejection threshold from the sidebar
3. Click Predict
4. Review the predicted identity and class probabilities

## Model Workflow

The main application uses:

- `MTCNN` for face detection and alignment
- `InceptionResnetV1` pretrained on VGGFace2 for embeddings
- A saved classifier from `clf.pkl`
- A saved label encoder from `label_encoder.pkl`

If the top prediction confidence is below the threshold, the app marks the face as unknown.

## Training and Experiments

The repository also includes notebook files for experimentation and model development:

- `facenet.ipynb`
- `effinetb0.ipynb`

These notebooks can be used to explore training, evaluation, and data preparation on the custom face dataset.

## Notes

- The application is designed for identification, not face verification.
- Performance depends on image quality, lighting, pose, and dataset coverage.
- For consistent results, use clear front-facing images.
- The `augment_faces.py` script can be used to expand the training data.


