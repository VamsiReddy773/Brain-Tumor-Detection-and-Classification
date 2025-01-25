# Brain Tumor Detection Using Deep Learning Methods

![GitHub](https://img.shields.io/badge/License-MIT-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)

This project focuses on automating the detection and classification of brain tumors from MRI scans using deep learning techniques. The system leverages Convolutional Neural Networks (CNNs) and transfer learning to achieve high accuracy in tumor detection. A user-friendly web interface is also provided for real-time predictions.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Methodology](#methodology)
   - [Data Collection](#data-collection)
   - [Data Preprocessing](#data-preprocessing)
   - [Model Architecture](#model-architecture)
   - [Evaluation Metrics](#evaluation-metrics)
6. [Results](#results)
7. [Future Work](#future-work)
8. [Contributing](#contributing)
9. [License](#license)

---

## Project Overview
Brain tumors are one of the most serious medical conditions, requiring early and accurate detection for effective treatment. Traditional diagnostic methods, such as manual analysis of MRI scans, are time-consuming and prone to human error. This project aims to automate the process using deep learning models, providing a reliable and efficient tool for brain tumor detection.

The system uses a combination of **Convolutional Neural Networks (CNNs)** and **transfer learning** to classify MRI images into different tumor types (e.g., glioma, meningioma, pituitary tumor). A **Streamlit-based web interface** is also developed to allow users to upload MRI images and receive real-time predictions.

---

## Features
- **Automated Tumor Detection**: Uses deep learning models to detect and classify brain tumors from MRI scans.
- **High Accuracy**: Achieves an accuracy of **96.83%** on the test dataset.
- **User-Friendly Interface**: Provides a web-based interface for real-time predictions.
- **Data Augmentation**: Enhances model performance by generating diverse training data.
- **Transfer Learning**: Leverages pre-trained models like VGG16 and ResNet for improved feature extraction.

---

## Installation
To set up the project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/VamsiReddy773/brain-tumor-detection.git
   cd brain-tumor-detection
   ```
# Install Dependencies:
pip install -r requirements.txt


#Run the Web Interface:
streamlit run app.py

## Usage

# Training the Model:
Run the training script to train the deep learning model:
bash
Copy
python train.py
Using the Web Interface:
Start the Streamlit app:
bash
streamlit run app.py
Upload an MRI image through the web interface to get real-time predictions.
Methodology

# Data Collection

The dataset used in this project is the BraTS dataset, which contains MRI images of brain tumors. The dataset is divided into training, validation, and testing sets.

# Data Preprocessing

Resizing: Images are resized to 224x224 pixels.
Normalization: Pixel values are scaled to the range [0, 1].
Data Augmentation: Techniques like rotation, flipping, and scaling are applied to increase dataset diversity.
Model Architecture

The model uses a CNN-based architecture with transfer learning. Pre-trained models like VGG16 and ResNet are fine-tuned on the brain tumor dataset.

python
Copy
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten

# Load pre-trained VGG16 model
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Add custom layers
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation="relu")(x)
predictions = Dense(4, activation="softmax")(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
Evaluation Metrics

# The model's performance is evaluated using:

Accuracy: 96.83%
Precision: 93.13%
Recall: 94.8%
F1-Score: 93.9%
Confusion Matrix: Visualizes the model's predictions.
Results

# The model achieved the following results on the test dataset:

Accuracy: 96.83%
Precision: 93.13%
Recall: 94.8%
F1-Score: 93.9%
The confusion matrix and classification report provide detailed insights into the model's performance.

# Future Work

Multi-Modal Imaging: Integrate MRI with CT or PET scans for improved accuracy.
Model Interpretability: Use explainable AI techniques like Grad-CAM to provide insights into model predictions.
Real-Time Deployment: Deploy the system in clinical settings for real-time tumor detection.
Contributing

# Contributions are welcome! If you'd like to contribute, please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature/YourFeature).
Commit your changes (git commit -m 'Add some feature').
Push to the branch (git push origin feature/YourFeature).
Open a pull request.
