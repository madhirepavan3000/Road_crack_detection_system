﻿# Road_crack_detection_system
Road Crack Detection System
An automated road crack detection system leveraging deep learning techniques (CNNs & Vision Transformers) to identify and classify road surface cracks from images. Includes a user-friendly Streamlit web interface for real-time analysis.

Features
Detects and classifies cracks in road images using advanced deep learning models (CNN & ViT)

High accuracy (up to 99.7%) on a large, real-world dataset (40,000+ images)

Robust data preprocessing and augmentation for real-world deployment

Streamlit web app for instant image upload and prediction

Visualizations for model performance and predictions

Tech Stack
Python 3.10

TensorFlow / Keras

OpenCV

Streamlit

Scikit-learn

Usage
Clone the repository:

bash
git clone https://github.com/<your-username>/Road_crack_detection_system.git
Install dependencies:

bash
pip install -r requirements.txt
Run the Streamlit app:

bash
streamlit run app.py
Project Structure
app.py — Main Streamlit application

models/ — Saved model files (CNN and ViT)

data/ — Scripts for data preprocessing and augmentation

notebooks/ — Jupyter/Colab notebooks for model training and evaluation

README.md — Project overview and instructions
